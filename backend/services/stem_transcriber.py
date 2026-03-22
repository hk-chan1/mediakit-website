"""
Per-stem pitch transcription.

- Monophonic stems (vocals, bass): CREPE (preferred) → librosa pYIN (fallback)
- Polyphonic stem (other/instruments): Spotify Basic Pitch
"""

import numpy as np
import logging
from typing import List

logger = logging.getLogger(__name__)

_SILENCE_RMS = 0.004        # RMS threshold for silence detection
_MIN_NOTE_SECS = 0.05       # 50 ms minimum note length
_PIANO_LOW = 21             # A0
_PIANO_HIGH = 108           # C8


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def is_stem_silent(audio_path: str) -> bool:
    """Return True when the stem is near-silence and should be skipped."""
    try:
        import soundfile as sf
        y, _ = sf.read(audio_path)
        if y.ndim > 1:
            y = y.mean(axis=1)
        rms = float(np.sqrt(np.mean(y.astype(np.float32) ** 2)))
        logger.info(f"  Stem RMS level: {rms:.5f}")
        return rms < _SILENCE_RMS
    except Exception:
        return False


def transcribe_monophonic_stem(audio_path: str, stem_name: str) -> List[dict]:
    """
    Transcribe a monophonic stem (vocals or bass line).
    Tries CREPE first; falls back to librosa pYIN.
    Returns list of {pitch, startTime, duration, velocity} dicts.
    """
    logger.info(f"  Transcribing {stem_name} (monophonic)...")

    if is_stem_silent(audio_path):
        logger.info(f"  {stem_name} stem is silent — skipped")
        return []

    try:
        return _transcribe_crepe(audio_path, stem_name)
    except ImportError:
        logger.info(f"  CREPE not installed — using librosa pYIN for {stem_name}")
    except Exception as e:
        logger.warning(f"  CREPE failed ({e}) — falling back to pYIN for {stem_name}")

    return _transcribe_pyin(audio_path, stem_name)


def transcribe_polyphonic_stem(audio_path: str) -> List[dict]:
    """
    Transcribe the 'other' (instruments) stem using Spotify Basic Pitch.
    Returns list of {pitch, startTime, duration, velocity} dicts.
    """
    logger.info("  Transcribing instruments stem (polyphonic, Basic Pitch)...")

    if is_stem_silent(audio_path):
        logger.info("  Instruments stem is silent — skipped")
        return []

    from basic_pitch.inference import predict
    from basic_pitch import ICASSP_2022_MODEL_PATH

    _, _, note_events = predict(
        audio_path,
        model_or_model_path=ICASSP_2022_MODEL_PATH,
        onset_threshold=0.5,
        frame_threshold=0.3,
        minimum_note_length=58,
        midi_tempo=120,
    )

    notes = []
    for ev in note_events:
        start = float(ev[0])
        end = float(ev[1])
        pitch = int(ev[2])
        vel = int(ev[3] * 127) if ev[3] <= 1.0 else int(ev[3])
        notes.append({
            "pitch": pitch,
            "startTime": round(start, 3),
            "duration": round(end - start, 3),
            "velocity": min(127, max(1, vel)),
        })

    logger.info(f"  Detected {len(notes)} instrument notes")
    return notes


# ──────────────────────────────────────────────────────────────────────────────
# Private helpers
# ──────────────────────────────────────────────────────────────────────────────

def _transcribe_crepe(audio_path: str, stem_name: str) -> List[dict]:
    import crepe
    import soundfile as sf

    y, sr = sf.read(audio_path)
    if y.ndim > 1:
        y = y.mean(axis=1)
    y = y.astype(np.float32)

    times, freqs, confidences, _ = crepe.predict(
        y, sr,
        viterbi=True,
        step_size=10,           # 10 ms frames
        model_capacity="full",
    )

    notes = _pitch_track_to_notes(times, freqs, confidences, confidence_threshold=0.6)
    logger.info(f"  {stem_name}: {len(notes)} notes via CREPE")
    return notes


def _transcribe_pyin(audio_path: str, stem_name: str) -> List[dict]:
    import librosa

    y, sr = librosa.load(audio_path, sr=22050, mono=True)

    f0, _, voiced_probs = librosa.pyin(
        y,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C7"),
        sr=sr,
        hop_length=512,
    )
    times = librosa.times_like(f0, sr=sr, hop_length=512)

    notes = _pitch_track_to_notes(times, f0, voiced_probs, confidence_threshold=0.5)
    logger.info(f"  {stem_name}: {len(notes)} notes via pYIN")
    return notes


def _pitch_track_to_notes(times, freqs, confidences,
                           confidence_threshold: float = 0.5) -> List[dict]:
    """Convert a frame-level pitch track to discrete MIDI note events."""
    notes: List[dict] = []
    current: dict | None = None

    for t, f, c in zip(times, freqs, confidences):
        t = float(t)
        c = float(c) if c is not None else 0.0
        f = float(f) if (f is not None and not np.isnan(f)) else 0.0

        voiced = f > 0 and c >= confidence_threshold

        if not voiced:
            if current is not None:
                dur = t - current["startTime"]
                if dur >= _MIN_NOTE_SECS:
                    current["duration"] = round(dur, 3)
                    notes.append(current)
                current = None
            continue

        midi_pitch = int(round(69.0 + 12.0 * np.log2(f / 440.0)))
        midi_pitch = max(_PIANO_LOW, min(_PIANO_HIGH, midi_pitch))

        if current is None:
            current = {"pitch": midi_pitch, "startTime": t,
                       "duration": 0.0, "velocity": 80}
        elif current["pitch"] != midi_pitch:
            dur = t - current["startTime"]
            if dur >= _MIN_NOTE_SECS:
                current["duration"] = round(dur, 3)
                notes.append(current)
            current = {"pitch": midi_pitch, "startTime": t,
                       "duration": 0.0, "velocity": 80}

    # Close the last open note
    if current is not None and len(times) > 0:
        dur = float(times[-1]) - current["startTime"]
        if dur >= _MIN_NOTE_SECS:
            current["duration"] = round(dur, 3)
            notes.append(current)

    return notes
