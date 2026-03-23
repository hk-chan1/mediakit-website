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
    # Merge only identical-pitch notes with tiny gap (≤30ms) — never merge different pitches
    treble = merge_close_notes(treble, max_gap_sec=0.03)
    bass   = merge_close_notes(bass,   max_gap_sec=0.03)

    treble = smooth_melodic_contour(treble)

    # Fill waltz left-hand pattern when time_sig is 3/4
    if time_sig[0] == 3:
        bass = fill_waltz_pattern(bass, treble, tempo, time_sig)
    else:
        bass = fill_sparse_bass(bass, treble, tempo, time_sig)

    treble, bass = fix_repetitions(treble, bass, tempo, time_sig)

    return treble, bass


# ──────────────────────────────────────────────────────────────────────────────
# Step 1: Merge same-pitch notes separated by a short gap
# ──────────────────────────────────────────────────────────────────────────────

def merge_close_notes(notes: List[dict], max_gap_sec: float = 0.03) -> List[dict]:
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
# Step 3b: Waltz pattern fill (3/4 time)
# ──────────────────────────────────────────────────────────────────────────────

# Chord tones for each scale degree root (relative semitones for major triad)
_MAJOR_TRIAD = [0, 4, 7]
# Common chord roots by scale degree (0=I, 2=II, 4=III, 5=IV, 7=V, 9=VI, 11=VII)
_CHORD_TONE_OFFSETS = {
    0: [0, 4, 7],   # I major
    2: [2, 5, 9],   # II minor
    4: [4, 7, 11],  # III minor
    5: [5, 9, 0],   # IV major
    7: [7, 11, 2],  # V major
    9: [9, 0, 4],   # VI minor
    11: [11, 2, 5], # VII diminished
}


def fill_waltz_pattern(bass: List[dict], treble: List[dict],
                       tempo: float, time_sig: list) -> List[dict]:
    """
    Enforce / repair the waltz "oom-pah-pah" left-hand pattern for 3/4 pieces.

    For each measure:
      Beat 1 — single bass note (root of the implied chord)
      Beat 2 — two-note mid-range chord
      Beat 3 — two-note mid-range chord (same chord as beat 2)

    Strategy:
    1. Identify existing beat-1 bass notes (they anchor the harmony)
    2. Fill missing beat-1 notes from the treble's lowest pitch in that measure
    3. Fill missing beats 2+3 from the inferred chord of the beat-1 note
    """
    if not treble:
        return bass

    beat_dur    = 60.0 / tempo
    measure_dur = time_sig[0] * beat_dur
    piece_end   = max(
        (n["startTime"] + n["duration"] for n in treble + bass),
        default=0.0,
    )
    if piece_end <= 0:
        return bass

    # Index existing bass notes by measure
    def measure_idx(t: float) -> int:
        return int(t / measure_dur)

    bass_by_measure: dict = {}
    for n in bass:
        idx = measure_idx(n["startTime"])
        bass_by_measure.setdefault(idx, []).append(n)

    treble_by_measure: dict = {}
    for n in treble:
        idx = measure_idx(n["startTime"])
        treble_by_measure.setdefault(idx, []).append(n)

    total_measures = int(piece_end / measure_dur) + 1
    new_notes: List[dict] = list(bass)  # start from existing bass

    for m in range(total_measures):
        t_beat1 = m * measure_dur
        t_beat2 = t_beat1 + beat_dur
        t_beat3 = t_beat1 + 2 * beat_dur
        existing = bass_by_measure.get(m, [])

        # ── Determine beat-1 root ──────────────────────────────────────────
        beat1_notes = [n for n in existing
                       if abs(n["startTime"] - t_beat1) < beat_dur * 0.4]
        if beat1_notes:
            root_pitch = min(n["pitch"] for n in beat1_notes)
        else:
            # Infer from lowest treble note in this measure
            t_notes = treble_by_measure.get(m, [])
            if not t_notes:
                continue
            lowest_treble = min(n["pitch"] for n in t_notes)
            root_pitch = lowest_treble
            # Transpose to bass register (MIDI 36-52)
            while root_pitch > 52:
                root_pitch -= 12
            while root_pitch < 28:
                root_pitch += 12
            # Add inferred beat-1 note if missing
            new_notes.append({
                "pitch": root_pitch,
                "startTime": round(t_beat1, 3),
                "duration": round(beat_dur * 0.85, 3),
                "velocity": 65,
            })

        # ── Fill beats 2 and 3 if missing ─────────────────────────────────
        beat23_notes = [n for n in existing
                        if n["startTime"] > t_beat1 + beat_dur * 0.3]
        if beat23_notes:
            continue  # beats 2+3 already present

        # Build a two-note mid-range chord from the root
        # Chord tones: major third above root + perfect fifth above root
        root_pc = root_pitch % 12
        # Find the closest scale-degree chord offsets
        best_key = min(_CHORD_TONE_OFFSETS.keys(),
                       key=lambda k: min((root_pc - k) % 12,
                                         (k - root_pc) % 12))
        offsets = _CHORD_TONE_OFFSETS[best_key][1:]  # skip root, take 3rd and 5th
        chord_pitches = []
        for off in offsets:
            p = (root_pitch // 12) * 12 + off
            if p <= root_pitch:
                p += 12
            # Keep in mid-range (48-64)
            while p > 64:
                p -= 12
            while p < 48:
                p += 12
            chord_pitches.append(p)

        if len(chord_pitches) < 2:
            continue

        for beat_t in (t_beat2, t_beat3):
            if beat_t >= piece_end:
                break
            for cp in chord_pitches:
                new_notes.append({
                    "pitch": cp,
                    "startTime": round(beat_t, 3),
                    "duration": round(beat_dur * 0.80, 3),
                    "velocity": 55,
                })

    # Deduplicate: remove generated notes that clash with existing ones (<50ms apart, same pitch)
    existing_set = {(round(n["startTime"], 2), n["pitch"]) for n in bass}
    filtered = []
    for n in new_notes:
        key = (round(n["startTime"], 2), n["pitch"])
        if key not in existing_set or n in bass:
            filtered.append(n)
            existing_set.add(key)

    result = sorted(filtered, key=lambda n: (n["startTime"], n["pitch"]))
    added = len(result) - len(bass)
    if added > 0:
        logger.info(f"  Waltz pattern: added {added} left-hand notes")
    return result


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
