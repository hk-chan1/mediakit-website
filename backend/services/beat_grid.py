"""
Tempo / beat-grid detection and rhythmic quantisation.

Beat tracking: madmom RNN (preferred) → librosa (fallback).
Quantisation: snap every note onset to the nearest 16th-note subdivision.
"""

import numpy as np
import logging
from typing import List

logger = logging.getLogger(__name__)

# Standard rhythmic values in beats (whole → 32nd), including dots
_STANDARD_BEATS = [4.0, 3.0, 2.0, 1.5, 1.0, 0.75, 0.5, 0.375, 0.25, 0.125]


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def detect_beat_grid(audio_path: str) -> dict:
    """
    Detect tempo, beat positions, and time signature from the FULL audio mix.

    Returns:
        tempo        – float BPM
        beats        – list of beat onset times (seconds)
        downbeats    – list of downbeat (bar-start) times (seconds)
        timeSignature – [numerator, denominator]
    """
    try:
        return _detect_madmom(audio_path)
    except ImportError:
        logger.info("madmom not installed — using librosa beat tracker")
    except Exception as e:
        logger.warning(f"madmom failed ({e}) — falling back to librosa")

    return _detect_librosa(audio_path)


def quantize_notes_to_grid(notes: List[dict], beat_grid: dict) -> List[dict]:
    """
    Snap every note onset to the nearest 16th-note position and quantise
    its duration to the nearest standard rhythmic value.
    """
    if not notes:
        return []

    tempo = beat_grid["tempo"]
    beats = beat_grid.get("beats", [])
    beat_dur = 60.0 / tempo
    subdivision = beat_dur / 4   # 16th note

    if beats and len(beats) >= 2:
        return _quantize_to_beats(notes, np.array(beats, dtype=float),
                                  subdivision, beat_dur)
    return _quantize_to_tempo(notes, beat_dur, subdivision)


# ──────────────────────────────────────────────────────────────────────────────
# Beat detection backends
# ──────────────────────────────────────────────────────────────────────────────

def _detect_madmom(audio_path: str) -> dict:
    from madmom.features.beats import RNNBeatProcessor, BeatTrackingProcessor
    from madmom.features.downbeats import (RNNDownBeatProcessor,
                                           DBNDownBeatTrackingProcessor)

    logger.info("  Detecting beats with madmom RNN…")
    beat_act = RNNBeatProcessor()(audio_path)
    beats = BeatTrackingProcessor(fps=100)(beat_act)

    if len(beats) >= 2:
        tempo = float(60.0 / float(np.median(np.diff(beats))))
    else:
        tempo = 120.0

    # Downbeat / time-signature detection
    time_sig = [4, 4]
    downbeats: list = []
    try:
        dbeat_act = RNNDownBeatProcessor()(audio_path)
        dbeat_data = DBNDownBeatTrackingProcessor(
            beats_per_bar=[3, 4], fps=100)(dbeat_act)
        max_beat = int(np.max(dbeat_data[:, 1]))
        time_sig = [3, 4] if max_beat == 3 else [4, 4]
        downbeats = dbeat_data[dbeat_data[:, 1] == 1, 0].tolist()
    except Exception as ex:
        logger.debug(f"  Downbeat detection skipped: {ex}")

    logger.info(f"  Detected: {tempo:.1f} BPM  {time_sig[0]}/{time_sig[1]}")
    return {
        "tempo": round(tempo, 1),
        "beats": beats.tolist(),
        "downbeats": downbeats,
        "timeSignature": time_sig,
    }


def _detect_librosa(audio_path: str) -> dict:
    import librosa

    logger.info("  Detecting beats with librosa…")
    y, sr = librosa.load(audio_path, sr=22050, mono=True)
    tempo_arr, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    tempo = float(np.squeeze(tempo_arr))
    beat_times = librosa.frames_to_time(beat_frames, sr=sr).tolist()

    logger.info(f"  Detected: {tempo:.1f} BPM (librosa)")
    return {
        "tempo": round(tempo, 1),
        "beats": beat_times,
        "downbeats": [],
        "timeSignature": [4, 4],
    }


# ──────────────────────────────────────────────────────────────────────────────
# Quantisation helpers
# ──────────────────────────────────────────────────────────────────────────────

def _quantize_to_beats(notes: List[dict], beats: np.ndarray,
                       subdivision: float, beat_dur: float) -> List[dict]:
    quantized = []
    for note in notes:
        onset = note["startTime"]
        nearest_idx = int(np.argmin(np.abs(beats - onset)))
        nearest_beat = float(beats[nearest_idx])

        offset = onset - nearest_beat
        snapped_offset = round(offset / subdivision) * subdivision
        new_onset = max(0.0, nearest_beat + snapped_offset)

        q_dur = _snap_duration(note["duration"], beat_dur)
        q_dur = max(q_dur, subdivision)

        quantized.append({**note,
                          "startTime": round(new_onset, 3),
                          "duration": round(q_dur, 3)})

    return sorted(quantized, key=lambda n: (n["startTime"], n["pitch"]))


def _quantize_to_tempo(notes: List[dict], beat_dur: float,
                       subdivision: float) -> List[dict]:
    quantized = []
    for note in notes:
        onset = note["startTime"]
        snapped = round(onset / subdivision) * subdivision
        q_dur = _snap_duration(note["duration"], beat_dur)
        q_dur = max(q_dur, subdivision)
        quantized.append({**note,
                          "startTime": round(snapped, 3),
                          "duration": round(q_dur, 3)})
    return sorted(quantized, key=lambda n: (n["startTime"], n["pitch"]))


def _snap_duration(duration_sec: float, beat_dur: float) -> float:
    """Snap duration (seconds) to nearest standard rhythmic value."""
    beats = duration_sec / beat_dur
    nearest = min(_STANDARD_BEATS, key=lambda v: abs(v - beats))
    return nearest * beat_dur
