"""
Tempo / beat-grid detection and rhythmic quantisation.

Fix 1: Tempo doubling problem
  Many trackers lock onto sub-beat pulses and report 2× or 1.5× the true tempo.
  We generate multiple tempo hypotheses (T, T/2, T×2/3, T×3/4) and score each
  against note-onset alignment if notes are available.

Fix 1b: Time-signature detection
  DBNDownBeatTrackingProcessor now considers [2, 3, 4, 6] beats per bar.
  We also validate using a note-density-per-beat-position histogram.
"""

import numpy as np
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)

_STANDARD_BEATS = [4.0, 3.0, 2.0, 1.5, 1.0, 0.75, 0.5, 0.375, 0.25, 0.125]

# Musically meaningful tempo ratios to try around the raw detection
_TEMPO_RATIOS = [1.0, 0.5, 2/3, 3/4, 4/3, 3/2, 2.0]


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def detect_beat_grid(audio_path: str, notes_hint: Optional[List[dict]] = None) -> dict:
    """
    Detect tempo, beat positions, and time signature from audio.

    notes_hint – already-transcribed notes; used to cross-validate tempo
                 hypotheses. Pass None when notes aren't yet available.
    """
    try:
        return _detect_madmom(audio_path, notes_hint)
    except ImportError:
        logger.info("madmom not installed — using librosa")
    except Exception as e:
        logger.warning(f"madmom failed ({e}) — librosa fallback")

    return _detect_librosa(audio_path, notes_hint)


def quantize_notes_to_grid(notes: List[dict], beat_grid: dict) -> List[dict]:
    """Snap every note onset to the nearest 16th-note and quantise duration."""
    if not notes:
        return []

    tempo = beat_grid["tempo"]
    beats = beat_grid.get("beats", [])
    beat_dur = 60.0 / tempo
    subdivision = beat_dur / 4

    if beats and len(beats) >= 2:
        return _quantize_to_beats(notes, np.array(beats, dtype=float),
                                  subdivision, beat_dur)
    return _quantize_to_tempo(notes, beat_dur, subdivision)


# ──────────────────────────────────────────────────────────────────────────────
# Detection backends
# ──────────────────────────────────────────────────────────────────────────────

def _detect_madmom(audio_path: str, notes_hint: Optional[List[dict]]) -> dict:
    from madmom.features.beats import RNNBeatProcessor, BeatTrackingProcessor
    from madmom.features.downbeats import (RNNDownBeatProcessor,
                                           DBNDownBeatTrackingProcessor)

    logger.info("  Detecting beats with madmom…")
    beat_act = RNNBeatProcessor()(audio_path)
    beats = BeatTrackingProcessor(fps=100)(beat_act)

    raw_tempo = float(60.0 / float(np.median(np.diff(beats)))) if len(beats) >= 2 else 120.0

    # ── Time signature ────────────────────────────────────────────────────────
    time_sig = [4, 4]
    downbeats: list = []
    try:
        dbeat_act = RNNDownBeatProcessor()(audio_path)
        # Allow 2, 3, 4, 6 — critical to catch 3/4 waltzes
        dbeat_data = DBNDownBeatTrackingProcessor(
            beats_per_bar=[2, 3, 4, 6], fps=100)(dbeat_act)
        max_beat = int(np.max(dbeat_data[:, 1]))
        time_sig = {2: [2, 4], 3: [3, 4], 4: [4, 4], 6: [6, 8]}.get(max_beat, [4, 4])
        downbeats = dbeat_data[dbeat_data[:, 1] == 1, 0].tolist()
    except Exception as ex:
        logger.debug(f"  Downbeat detection skipped: {ex}")

    # ── Tempo hypothesis selection ────────────────────────────────────────────
    candidates = sorted(set(
        round(raw_tempo * r, 2)
        for r in _TEMPO_RATIOS
        if 40 <= raw_tempo * r <= 220
    ))
    best_tempo = _pick_best_tempo(candidates, beats, time_sig, notes_hint, raw_tempo)

    # ── Validate time signature against note density ──────────────────────────
    if notes_hint and len(notes_hint) >= 8:
        time_sig = _validate_time_sig(time_sig, best_tempo, notes_hint)

    logger.info(
        f"  Beat grid: {best_tempo:.1f} BPM  {time_sig[0]}/{time_sig[1]}"
        f"  (raw={raw_tempo:.1f})"
    )
    return {
        "tempo": round(best_tempo, 1),
        "beats": beats.tolist(),
        "downbeats": downbeats,
        "timeSignature": time_sig,
    }


def _detect_librosa(audio_path: str, notes_hint: Optional[List[dict]]) -> dict:
    import librosa

    logger.info("  Detecting beats with librosa…")
    y, sr = librosa.load(audio_path, sr=22050, mono=True)
    tempo_arr, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    raw_tempo = float(np.squeeze(tempo_arr))
    beat_times = librosa.frames_to_time(beat_frames, sr=sr).tolist()

    candidates = sorted(set(
        round(raw_tempo * r, 2)
        for r in _TEMPO_RATIOS
        if 40 <= raw_tempo * r <= 220
    ))
    best_tempo = _pick_best_tempo(candidates, beat_times, [4, 4], notes_hint, raw_tempo)

    time_sig = [4, 4]
    if notes_hint:
        time_sig = _validate_time_sig([4, 4], best_tempo, notes_hint)

    logger.info(f"  Beat grid: {best_tempo:.1f} BPM (librosa)")
    return {
        "tempo": round(best_tempo, 1),
        "beats": beat_times,
        "downbeats": [],
        "timeSignature": time_sig,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Tempo hypothesis scoring
# ──────────────────────────────────────────────────────────────────────────────

def _pick_best_tempo(candidates: list, beats: list, time_sig: list,
                     notes_hint: Optional[List[dict]], raw_tempo: float) -> float:
    """
    Score each tempo candidate against note-onset alignment and musical
    duration distribution. Return the best candidate.
    """
    if not notes_hint or len(notes_hint) < 4:
        return raw_tempo   # can't validate; trust raw detection

    beats_arr = np.array(beats) if beats else np.array([])

    def score(tempo: float) -> float:
        beat_dur = 60.0 / tempo

        # ── Alignment: how many note onsets land near a beat subdivision ──
        align_scores = []
        for n in notes_hint:
            t = n["startTime"]
            # Distance to nearest 16th note
            sub = beat_dur / 4
            offset = t % sub
            dist = min(offset, sub - offset) / sub     # 0=perfect, 0.5=worst
            align_scores.append(1.0 - 2 * dist)
        align = float(np.mean(align_scores))

        # ── Musicality: prefer tempos where note durations are "standard" ──
        dur_scores = []
        std_beats = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0]
        for n in notes_hint:
            dur_b = n["duration"] / beat_dur
            nearest = min(std_beats, key=lambda v: abs(v - dur_b))
            dev = abs(dur_b - nearest) / max(nearest, 0.01)
            dur_scores.append(max(0.0, 1.0 - dev))
        musicality = float(np.mean(dur_scores))

        # ── Range preference: human performance range ──────────────────────
        range_ok = 1.0 if 50 <= tempo <= 200 else 0.6

        # ── Prefer candidates close to the raw detection (less aggressive) ─
        proximity = 1.0 / (1.0 + abs(tempo - raw_tempo) / max(raw_tempo, 1))

        return 0.45 * align + 0.30 * musicality + 0.15 * range_ok + 0.10 * proximity

    scored = [(t, score(t)) for t in candidates]
    scored.sort(key=lambda x: -x[1])

    logger.debug(f"  Tempo hypotheses: { {round(t,1): round(s,3) for t,s in scored[:5]} }")
    return scored[0][0]


# ──────────────────────────────────────────────────────────────────────────────
# Time-signature validation
# ──────────────────────────────────────────────────────────────────────────────

def _validate_time_sig(time_sig: list, tempo: float,
                       notes: List[dict]) -> list:
    """
    Re-examine time signature by building a beat-position density histogram.
    If note density peaks at positions 1, 2, 3 (not 4), 3/4 is more likely.
    """
    beat_dur = 60.0 / tempo

    # Build beat-position histogram for 3 vs 4 beats per measure
    def measure_fit(n_beats: int) -> float:
        measure_dur = n_beats * beat_dur
        counts = np.zeros(n_beats)
        for note in notes:
            pos = (note["startTime"] % measure_dur) / beat_dur
            beat_idx = int(pos) % n_beats
            counts[beat_idx] += 1
        # Good fit = beats 1 and 2 have more notes than beat n (for n=3 or 4)
        if counts.sum() == 0:
            return 0.0
        # Score: concentration on beat 1 (index 0) vs spread
        normed = counts / counts.sum()
        beat1_weight = float(normed[0])
        return beat1_weight

    fit3 = measure_fit(3)
    fit4 = measure_fit(4)

    logger.debug(f"  Time sig fit: 3/4={fit3:.3f}  4/4={fit4:.3f}")

    # Only override if the evidence is clear (>20% difference)
    if time_sig == [4, 4] and fit3 > fit4 * 1.2:
        logger.info("  Note density suggests 3/4 — overriding detected 4/4")
        return [3, 4]
    if time_sig == [3, 4] and fit4 > fit3 * 1.2:
        logger.info("  Note density suggests 4/4 — overriding detected 3/4")
        return [4, 4]

    return time_sig


# ──────────────────────────────────────────────────────────────────────────────
# Quantisation
# ──────────────────────────────────────────────────────────────────────────────

def _quantize_to_beats(notes: List[dict], beats: np.ndarray,
                       subdivision: float, beat_dur: float) -> List[dict]:
    quantized = []
    for note in notes:
        onset = note["startTime"]
        nearest_beat = float(beats[int(np.argmin(np.abs(beats - onset)))])
        offset = onset - nearest_beat
        snapped_offset = round(offset / subdivision) * subdivision
        new_onset = max(0.0, nearest_beat + snapped_offset)
        q_dur = _snap_duration(note["duration"], beat_dur)
        q_dur = max(q_dur, subdivision)
        quantized.append({**note, "startTime": round(new_onset, 3),
                          "duration": round(q_dur, 3)})
    return sorted(quantized, key=lambda n: (n["startTime"], n["pitch"]))


def _quantize_to_tempo(notes: List[dict], beat_dur: float,
                       subdivision: float) -> List[dict]:
    quantized = []
    for note in notes:
        snapped = round(note["startTime"] / subdivision) * subdivision
        q_dur = max(_snap_duration(note["duration"], beat_dur), subdivision)
        quantized.append({**note, "startTime": round(snapped, 3),
                          "duration": round(q_dur, 3)})
    return sorted(quantized, key=lambda n: (n["startTime"], n["pitch"]))


def _snap_duration(duration_sec: float, beat_dur: float) -> float:
    beats = duration_sec / beat_dur
    nearest = min(_STANDARD_BEATS, key=lambda v: abs(v - beats))
    return nearest * beat_dur
