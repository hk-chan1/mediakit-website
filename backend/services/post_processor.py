"""
Post-processing refinements for transcribed notes.

Fix 3: Melodic contour smoothing — remove pitch outliers > 1 octave from neighbors
Fix 5: Repetition detection — copy the denser of two similar phrases to the sparser
Also:   Note merging (close same-pitch events), sparse bass infill
"""

import numpy as np
import logging
from typing import List

logger = logging.getLogger(__name__)


def post_process_notes(treble: List[dict], bass: List[dict],
                       tempo: float, time_sig: list) -> tuple:
    """Main entry point. Returns (treble, bass) after all refinements."""
    treble = merge_close_notes(treble)
    bass   = merge_close_notes(bass)

    treble = smooth_melodic_contour(treble)

    bass = fill_sparse_bass(bass, treble, tempo, time_sig)

    treble, bass = fix_repetitions(treble, bass, tempo, time_sig)

    return treble, bass


# ──────────────────────────────────────────────────────────────────────────────
# Step 1: Merge same-pitch notes separated by a short gap
# ──────────────────────────────────────────────────────────────────────────────

def merge_close_notes(notes: List[dict], max_gap_sec: float = 0.12) -> List[dict]:
    """Merge same-pitch notes whose gap is ≤ max_gap_sec (default 120 ms)."""
    if not notes:
        return notes

    sorted_notes = sorted(notes, key=lambda n: (n["pitch"], n["startTime"]))
    merged = []
    prev = dict(sorted_notes[0])

    for note in sorted_notes[1:]:
        prev_end = prev["startTime"] + prev["duration"]
        same_pitch = note["pitch"] == prev["pitch"]
        gap = note["startTime"] - prev_end
        if same_pitch and gap <= max_gap_sec:
            new_end = max(prev_end, note["startTime"] + note["duration"])
            prev["duration"] = round(new_end - prev["startTime"], 3)
            prev["velocity"] = max(prev.get("velocity", 80), note.get("velocity", 80))
        else:
            merged.append(prev)
            prev = dict(note)
    merged.append(prev)

    return sorted(merged, key=lambda n: (n["startTime"], n["pitch"]))


# ──────────────────────────────────────────────────────────────────────────────
# Step 2: Smooth melodic contour (treble only)
# ──────────────────────────────────────────────────────────────────────────────

def smooth_melodic_contour(notes: List[dict], window: int = 5) -> List[dict]:
    """
    Discard notes whose pitch is > 1 octave from the median of their neighbors.
    Applied to the treble voice to remove transcription artifacts.
    """
    if len(notes) < window:
        return notes

    pitches = [n["pitch"] for n in notes]
    keep = [True] * len(notes)
    half = window // 2

    for i in range(len(notes)):
        lo = max(0, i - half)
        hi = min(len(notes), i + half + 1)
        neighbors = pitches[lo:i] + pitches[i + 1:hi]
        if not neighbors:
            continue
        med = float(np.median(neighbors))
        if abs(pitches[i] - med) > 12:
            keep[i] = False

    result = [n for n, k in zip(notes, keep) if k]
    removed = len(notes) - len(result)
    if removed:
        logger.debug(f"  Contour smoothing removed {removed} outlier notes")
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Step 3: Fill sparse bass
# ──────────────────────────────────────────────────────────────────────────────

def fill_sparse_bass(bass: List[dict], treble: List[dict],
                     tempo: float, time_sig: list) -> List[dict]:
    """
    When bass covers < 30 % of piece duration, infer root notes from the
    treble harmony at each measure downbeat.
    """
    if not treble:
        return bass

    beat_dur    = 60.0 / tempo
    measure_dur = time_sig[0] * beat_dur
    treble_end  = max(n["startTime"] + n["duration"] for n in treble)

    if treble_end <= 0:
        return bass

    bass_covered = sum(n["duration"] for n in bass)
    coverage = bass_covered / treble_end

    if coverage >= 0.30:
        logger.debug(f"  Bass coverage {coverage:.1%} — no infill needed")
        return bass

    logger.info(f"  Bass sparse ({coverage:.1%}) — inferring root notes from treble chords")

    inferred = []
    t = 0.0
    while t < treble_end:
        measure_end = t + measure_dur

        # Notes active during this measure
        active = [n for n in treble
                  if n["startTime"] < measure_end and
                     n["startTime"] + n["duration"] > t]

        # Is there already a bass note near the downbeat?
        has_bass = any(
            n["startTime"] < t + beat_dur and
            n["startTime"] + n["duration"] > t
            for n in bass
        )

        if active and not has_bass:
            lowest_pitch = min(n["pitch"] for n in active)
            bass_pitch = lowest_pitch
            while bass_pitch >= 48:
                bass_pitch -= 12
            while bass_pitch < 28:
                bass_pitch += 12

            inferred.append({
                "pitch": bass_pitch,
                "startTime": round(t, 3),
                "duration": round(min(measure_dur * 0.8, beat_dur * 2), 3),
                "velocity": 60,
            })

        t = measure_end

    if inferred:
        logger.info(f"  Added {len(inferred)} inferred bass notes")
        bass = sorted(bass + inferred, key=lambda n: (n["startTime"], n["pitch"]))

    return bass


# ──────────────────────────────────────────────────────────────────────────────
# Step 4: Repetition detection and repair
# ──────────────────────────────────────────────────────────────────────────────

def fix_repetitions(treble: List[dict], bass: List[dict],
                    tempo: float, time_sig: list) -> tuple:
    """
    Compare consecutive 4-bar phrases by pitch-class histogram cosine similarity.
    When similarity ≥ 0.85 but one phrase is ≤ 60 % as dense as the other,
    copy notes from the denser phrase into the sparse one.
    """
    beat_dur   = 60.0 / tempo
    phrase_dur = time_sig[0] * beat_dur * 4   # 4-bar phrase in seconds

    all_notes = treble + bass
    if not all_notes:
        return treble, bass

    max_time = max(n["startTime"] + n["duration"] for n in all_notes)
    n_phrases = int(max_time / phrase_dur)

    if n_phrases < 2:
        return treble, bass

    def phrase_histogram(notes: List[dict], idx: int) -> np.ndarray:
        t0, t1 = idx * phrase_dur, (idx + 1) * phrase_dur
        hist = np.zeros(12)
        for n in notes:
            if t0 <= n["startTime"] < t1:
                hist[n["pitch"] % 12] += n["duration"]
        return hist

    def cosine_sim(h1: np.ndarray, h2: np.ndarray) -> float:
        if h1.sum() == 0 or h2.sum() == 0:
            return 0.0
        n1, n2 = h1 / (h1.sum() + 1e-9), h2 / (h2.sum() + 1e-9)
        denom = np.linalg.norm(n1) * np.linalg.norm(n2)
        return float(np.dot(n1, n2) / (denom + 1e-9))

    def density(idx: int) -> int:
        t0, t1 = idx * phrase_dur, (idx + 1) * phrase_dur
        return sum(1 for n in treble + bass if t0 <= n["startTime"] < t1)

    changes = 0

    for i in range(n_phrases):
        for j in range(i + 1, n_phrases):
            h_i = phrase_histogram(treble, i)
            h_j = phrase_histogram(treble, j)
            if cosine_sim(h_i, h_j) < 0.85:
                continue

            den_i, den_j = density(i), density(j)
            if den_i == 0 and den_j == 0:
                continue

            dense_idx  = i if den_i >= den_j else j
            sparse_idx = j if den_i >= den_j else i
            sparse_den = min(den_i, den_j)
            dense_den  = max(den_i, den_j)

            if dense_den == 0 or sparse_den / dense_den > 0.60:
                continue

            dt = (sparse_idx - dense_idx) * phrase_dur
            t0_dense, t1_dense = dense_idx * phrase_dur, (dense_idx + 1) * phrase_dur
            t0_sparse, t1_sparse = sparse_idx * phrase_dur, (sparse_idx + 1) * phrase_dur

            copied_treble, copied_bass = [], []
            for n in treble:
                if t0_dense <= n["startTime"] < t1_dense:
                    new_n = dict(n)
                    new_n["startTime"] = round(n["startTime"] + dt, 3)
                    copied_treble.append(new_n)
            for n in bass:
                if t0_dense <= n["startTime"] < t1_dense:
                    new_n = dict(n)
                    new_n["startTime"] = round(n["startTime"] + dt, 3)
                    copied_bass.append(new_n)

            if not (copied_treble or copied_bass):
                continue

            logger.info(
                f"  Repetition fix: phrase {dense_idx}→{sparse_idx} "
                f"(sim={cosine_sim(h_i,h_j):.2f}, "
                f"dense={dense_den}, sparse={sparse_den})"
            )
            treble = sorted(
                [n for n in treble if not (t0_sparse <= n["startTime"] < t1_sparse)]
                + copied_treble,
                key=lambda n: (n["startTime"], n["pitch"]),
            )
            bass = sorted(
                [n for n in bass if not (t0_sparse <= n["startTime"] < t1_sparse)]
                + copied_bass,
                key=lambda n: (n["startTime"], n["pitch"]),
            )
            changes += 1

    if changes:
        logger.info(f"  Repetition detection: repaired {changes} phrase(s)")

    return treble, bass
