"""
Full audio-to-MIDI pipeline orchestrator.

Stage 1  separating   – Demucs source separation
Stage 2  transcribing – Per-stem pitch detection
Stage 3  quantizing   – Beat grid detection + rhythmic quantisation
Stage 4  arranging    – Two-hand piano arrangement + key detection
"""

import logging
import os
from typing import Callable, Optional

logger = logging.getLogger(__name__)


def run_full_pipeline(
    audio_path: str,
    output_dir: str,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> dict:
    """
    Runs the five-stage transcription pipeline.

    Returns a dict with:
        notes         – all notes combined (for frontend preview, backward-compat)
        treble_notes  – right-hand notes
        bass_notes    – left-hand notes
        tempo         – float BPM
        timeSignature – [num, denom]
        keySignature  – int 0-11
    """
    def _log(stage: str, msg: str = ""):
        logger.info(f"[{stage}] {msg}")
        if progress_callback:
            progress_callback(stage)

    # ── Stage 1: Source separation ──────────────────────────────────────────
    _log("separating", "Separating audio sources with Demucs htdemucs_ft…")
    from services.source_separator import separate_audio_sources
    stems = separate_audio_sources(audio_path, output_dir)

    # ── Stage 2: Per-stem transcription ─────────────────────────────────────
    _log("transcribing", "Transcribing note content from each stem…")
    from services.stem_transcriber import (
        transcribe_monophonic_stem,
        transcribe_polyphonic_stem,
    )

    vocals_notes = []
    bass_notes = []
    instrument_notes = []

    if "vocals" in stems:
        logger.info("Transcribing vocal melody…")
        vocals_notes = transcribe_monophonic_stem(stems["vocals"], "vocals")

    if "bass" in stems:
        logger.info("Transcribing bass line…")
        bass_notes = transcribe_monophonic_stem(stems["bass"], "bass")

    if "other" in stems:
        logger.info("Transcribing instrument chords…")
        instrument_notes = transcribe_polyphonic_stem(stems["other"])

    if not any([vocals_notes, bass_notes, instrument_notes]):
        raise RuntimeError(
            "No musical notes detected in any stem. "
            "The audio may not contain music or the quality is too low."
        )

    # ── Stage 3: Beat detection + quantisation ───────────────────────────────
    _log("quantizing",
         "Detecting tempo and beat grid from full mix…")
    from services.beat_grid import detect_beat_grid, quantize_notes_to_grid

    beat_grid = detect_beat_grid(audio_path)
    logger.info(
        f"Tempo: {beat_grid['tempo']:.1f} BPM  "
        f"Time: {beat_grid['timeSignature'][0]}/{beat_grid['timeSignature'][1]}"
    )

    logger.info("Quantising notes to beat grid (16th-note resolution)…")
    vocals_notes = quantize_notes_to_grid(vocals_notes, beat_grid)
    bass_notes = quantize_notes_to_grid(bass_notes, beat_grid)
    instrument_notes = quantize_notes_to_grid(instrument_notes, beat_grid)

    # ── Stage 4: Piano arrangement ───────────────────────────────────────────
    _log("arranging", "Building two-hand piano arrangement…")
    from services.piano_arranger import arrange_for_piano, detect_key_signature

    arrangement = arrange_for_piano(vocals_notes, bass_notes, instrument_notes)
    treble = arrangement["treble"]
    bass = arrangement["bass"]

    key_sig = detect_key_signature(treble + bass)
    logger.info(f"Key signature detected: pitch class {key_sig}")

    all_notes = treble + bass
    all_notes.sort(key=lambda n: (n["startTime"], n["pitch"]))

    logger.info(
        f"Pipeline complete — {len(treble)} treble, {len(bass)} bass notes"
    )

    return {
        "notes": all_notes,          # backward-compat for frontend preview
        "treble_notes": treble,
        "bass_notes": bass,
        "tempo": beat_grid["tempo"],
        "timeSignature": beat_grid["timeSignature"],
        "keySignature": key_sig,
    }


# Legacy alias kept so any direct callers still work
def transcribe_audio(audio_path: str) -> dict:
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        return run_full_pipeline(audio_path, tmp)
