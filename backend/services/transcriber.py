"""
Tiered audio-to-MIDI pipeline.

Tier 1 (<30 s)  – solo / simple instrument  → direct Basic Pitch
Tier 2 (<2 min) – vocals + light backing    → center-channel removal + parallel transcription
Tier 3 (<5 min) – complex mix               → Demucs + parallel transcription

Mode override:
    "quick"   → force Tier 1 regardless of analysis
    "quality" → force Tier 3 regardless of analysis
    "auto"    → trust the pre-analysis (default)
"""

import logging
import os
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Optional

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Public entry point
# ──────────────────────────────────────────────────────────────────────────────

def run_full_pipeline(
    audio_path: str,
    output_dir: str,
    progress_callback: Optional[Callable[[str, Optional[dict]], None]] = None,
    mode: str = "auto",
) -> dict:
    """
    Run the full transcription pipeline and return a midi_data dict.
    progress_callback(stage_name, meta_dict_or_None) is called at each stage transition.
    meta_dict may contain: tier, tier_reason, estimated_seconds, stage_timings.
    """
    timings: dict = {}

    def stage(name: str, meta: dict = None):
        logger.info(f"[{name}]" + (f" {meta}" if meta else ""))
        if progress_callback:
            progress_callback(name, meta)

    # ── Preprocessing: trim silence, normalize, limit to 5 min ───────────────
    t0 = time.time()
    proc_audio = _preprocess_audio(audio_path, output_dir)
    timings["preprocessing"] = round(time.time() - t0, 1)

    # ── Pre-analysis ──────────────────────────────────────────────────────────
    stage("analyzing", {"stage_timings": timings})
    t0 = time.time()
    from services.audio_analyzer import analyze_audio
    analysis = analyze_audio(proc_audio)
    timings["analyzing"] = round(time.time() - t0, 1)

    # Mode override
    if mode == "quick":
        analysis["tier"] = 1
        analysis["reason"] = "Quick mode — skipping source separation"
        analysis["estimated_seconds"] = max(15, analysis["estimated_seconds"] // 3)
    elif mode == "quality":
        analysis["tier"] = 3
        analysis["reason"] = "Quality mode — running full Demucs separation"

    tier = analysis["tier"]
    meta = {
        "tier": tier,
        "tier_reason": analysis["reason"],
        "estimated_seconds": analysis["estimated_seconds"],
        "stage_timings": timings,
    }
    logger.info(f"Tier {tier}: {analysis['reason']}")
    stage("separating", meta)

    # ── Tier dispatch ─────────────────────────────────────────────────────────
    if tier == 1:
        result = _tier1_fast(proc_audio, output_dir, stage, timings)
    elif tier == 2:
        result = _tier2_medium(proc_audio, output_dir, stage, timings)
    else:
        # Tier 3 uses original audio for Demucs (higher quality), proc_audio for rest
        result = _tier3_full(audio_path, proc_audio, output_dir, stage, timings)

    result["tier"] = tier
    result["tier_reason"] = analysis["reason"]
    result["estimated_seconds"] = analysis["estimated_seconds"]
    result["stage_timings"] = timings
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Tier implementations
# ──────────────────────────────────────────────────────────────────────────────

def _tier1_fast(proc_audio: str, output_dir: str, stage, timings: dict) -> dict:
    """No separation — run Basic Pitch directly on the full preprocessed audio."""
    logger.info("Tier 1: Direct Basic Pitch transcription")

    stage("transcribing", {"stage_timings": timings})
    t0 = time.time()
    from services.stem_transcriber import transcribe_polyphonic_stem
    instrument_notes = transcribe_polyphonic_stem(proc_audio)
    timings["transcribing"] = round(time.time() - t0, 1)

    if not instrument_notes:
        raise RuntimeError(
            "No notes detected. The audio may not contain music "
            "or the recording quality is too low."
        )
    return _finish([], [], instrument_notes, proc_audio, stage, timings)


def _tier2_medium(proc_audio: str, output_dir: str, stage, timings: dict) -> dict:
    """Center-channel vocal removal → parallel mono + poly transcription."""
    logger.info("Tier 2: Center-channel separation + parallel transcription")

    t0 = time.time()
    vocal_path, inst_path = _center_channel_split(proc_audio, output_dir)
    timings["separating"] = round(time.time() - t0, 1)

    stage("transcribing", {"stage_timings": timings})
    t0 = time.time()
    vocals_notes, instrument_notes = _parallel_two(vocal_path, inst_path)
    timings["transcribing"] = round(time.time() - t0, 1)

    if not any([vocals_notes, instrument_notes]):
        raise RuntimeError("No notes detected after center-channel separation")
    return _finish(vocals_notes, [], instrument_notes, proc_audio, stage, timings)


def _tier3_full(audio_path: str, proc_audio: str, output_dir: str,
                stage, timings: dict) -> dict:
    """Full Demucs → parallel per-stem transcription. Falls back to Tier 1 on timeout."""
    logger.info("Tier 3: Full Demucs source separation")

    t0 = time.time()
    from services.source_separator import separate_audio_sources
    try:
        stems = separate_audio_sources(audio_path, output_dir)
        timings["separating"] = round(time.time() - t0, 1)
    except Exception as e:
        timings["separating"] = round(time.time() - t0, 1)
        logger.warning(f"Demucs failed ({e}) — falling back to Tier 1")
        return _tier1_fast(proc_audio, output_dir, stage, timings)

    stage("transcribing", {"stage_timings": timings})
    t0 = time.time()
    vocals_notes, bass_notes, instrument_notes = _parallel_stems(stems)
    timings["transcribing"] = round(time.time() - t0, 1)

    if not any([vocals_notes, bass_notes, instrument_notes]):
        logger.warning("No notes from stems — falling back to Tier 1")
        return _tier1_fast(proc_audio, output_dir, stage, timings)

    return _finish(vocals_notes, bass_notes, instrument_notes, proc_audio, stage, timings)


# ──────────────────────────────────────────────────────────────────────────────
# Shared finishing steps (quantise → arrange → key)
# ──────────────────────────────────────────────────────────────────────────────

def _finish(vocals_notes, bass_notes, instrument_notes,
            proc_audio, stage, timings: dict) -> dict:
    stage("quantizing", {"stage_timings": timings})
    t0 = time.time()
    from services.beat_grid import detect_beat_grid, quantize_notes_to_grid

    # Pass all transcribed notes as hints so tempo scoring can use them
    notes_hint = vocals_notes + bass_notes + instrument_notes
    beat_grid = detect_beat_grid(proc_audio, notes_hint=notes_hint if notes_hint else None)
    logger.info(
        f"Tempo: {beat_grid['tempo']:.1f} BPM  "
        f"{beat_grid['timeSignature'][0]}/{beat_grid['timeSignature'][1]}"
    )
    vocals_notes = quantize_notes_to_grid(vocals_notes, beat_grid)
    bass_notes = quantize_notes_to_grid(bass_notes, beat_grid)
    instrument_notes = quantize_notes_to_grid(instrument_notes, beat_grid)
    timings["quantizing"] = round(time.time() - t0, 1)

    stage("arranging", {"stage_timings": timings})
    t0 = time.time()
    from services.piano_arranger import arrange_for_piano, detect_key_signature
    arr = arrange_for_piano(vocals_notes, bass_notes, instrument_notes)
    treble, bass = arr["treble"], arr["bass"]

    # Post-process: merge close notes, smooth contour, fill bass, fix repetitions
    try:
        from services.post_processor import post_process_notes
        treble, bass = post_process_notes(
            treble, bass, beat_grid["tempo"], beat_grid["timeSignature"]
        )
    except Exception as e:
        logger.warning(f"Post-processing failed ({e}) — using unprocessed notes")

    key_sig = detect_key_signature(treble + bass)
    timings["arranging"] = round(time.time() - t0, 1)

    all_notes = sorted(treble + bass, key=lambda n: (n["startTime"], n["pitch"]))
    return {
        "notes": all_notes,
        "treble_notes": treble,
        "bass_notes": bass,
        "tempo": beat_grid["tempo"],
        "timeSignature": beat_grid["timeSignature"],
        "keySignature": key_sig,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Audio preprocessing
# ──────────────────────────────────────────────────────────────────────────────

def _preprocess_audio(audio_path: str, output_dir: str) -> str:
    """
    Return a 22 050 Hz mono WAV: silence-trimmed, normalized, max 5 minutes.
    Used for analysis, beat tracking, and non-Demucs transcription.
    """
    import librosa
    import soundfile as sf

    y, sr = librosa.load(audio_path, sr=22050, mono=True)

    # Trim leading/trailing silence
    y, _ = librosa.effects.trim(y, top_db=40)

    # Enforce 5-minute maximum
    MAX_SAMPLES = 5 * 60 * sr
    if len(y) > MAX_SAMPLES:
        logger.info(f"  Audio capped at 5 minutes (was {len(y)/sr:.0f}s)")
        y = y[:MAX_SAMPLES]

    # Normalize peak to −1 dB
    peak = float(np.max(np.abs(y)))
    if peak > 0:
        y = (y / peak * 0.9).astype(np.float32)

    out = os.path.join(output_dir, "processed.wav")
    sf.write(out, y, sr)
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Center-channel separation (Tier 2)
# ──────────────────────────────────────────────────────────────────────────────

def _center_channel_split(audio_path: str, output_dir: str) -> tuple:
    """
    L+R → center (vocals), L-R → side (instruments).
    If audio is already mono, returns (audio_path, audio_path) unchanged.
    """
    import soundfile as sf

    y, sr = sf.read(audio_path)

    if y.ndim == 1:
        logger.info("  Mono audio — center-channel split skipped")
        return audio_path, audio_path

    left = y[:, 0].astype(np.float32)
    right = y[:, 1].astype(np.float32)
    center = (left + right) / 2
    side = (left - right) / 2

    for sig in (center, side):
        p = float(np.max(np.abs(sig)))
        if p > 0:
            sig /= p
            sig *= 0.9

    vocal_path = os.path.join(output_dir, "vocals_cc.wav")
    inst_path = os.path.join(output_dir, "inst_cc.wav")
    sf.write(vocal_path, center, sr)
    sf.write(inst_path, side, sr)
    return vocal_path, inst_path


# ──────────────────────────────────────────────────────────────────────────────
# Parallel transcription helpers
# ──────────────────────────────────────────────────────────────────────────────

def _parallel_two(vocal_path: str, inst_path: str) -> tuple:
    """Transcribe vocals (mono) and instruments (poly) in parallel threads."""
    from services.stem_transcriber import transcribe_monophonic_stem, transcribe_polyphonic_stem

    results: dict = {}

    def _vocals():
        results["vocals"] = transcribe_monophonic_stem(vocal_path, "vocals")

    def _instruments():
        results["instruments"] = transcribe_polyphonic_stem(inst_path)

    with ThreadPoolExecutor(max_workers=2) as ex:
        futures = [ex.submit(_vocals), ex.submit(_instruments)]
        for f in futures:
            try:
                f.result(timeout=240)
            except Exception as e:
                logger.warning(f"Transcription thread error: {e}")

    return results.get("vocals", []), results.get("instruments", [])


def _parallel_stems(stems: dict) -> tuple:
    """Transcribe vocals, bass, and other stems in parallel threads."""
    from services.stem_transcriber import transcribe_monophonic_stem, transcribe_polyphonic_stem

    results: dict = {"vocals": [], "bass": [], "other": []}
    tasks: dict = {}

    with ThreadPoolExecutor(max_workers=3) as ex:
        if "vocals" in stems:
            tasks["vocals"] = ex.submit(
                transcribe_monophonic_stem, stems["vocals"], "vocals")
        if "bass" in stems:
            tasks["bass"] = ex.submit(
                transcribe_monophonic_stem, stems["bass"], "bass")
        if "other" in stems:
            tasks["other"] = ex.submit(transcribe_polyphonic_stem, stems["other"])

        for key, future in tasks.items():
            try:
                results[key] = future.result(timeout=240)
            except Exception as e:
                logger.warning(f"Stem '{key}' transcription failed: {e}")

    return results["vocals"], results["bass"], results["other"]


# ──────────────────────────────────────────────────────────────────────────────
# Legacy alias
# ──────────────────────────────────────────────────────────────────────────────

def transcribe_audio(audio_path: str) -> dict:
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        return run_full_pipeline(audio_path, tmp)
