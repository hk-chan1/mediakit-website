"""
Intelligent two-hand piano arrangement.

Takes note lists from three stems (vocals, bass, instruments) and produces
separate treble (right-hand) and bass (left-hand) voices.

Rules:
  - Vocals → treble, priority 1
  - Bass stem → bass clef, priority 1
  - Instrument notes >= C4 (MIDI 60) → treble, priority 2
  - Instrument notes <  C4            → bass,   priority 2
  - Max 5 simultaneous notes per hand
  - No chord span wider than a 10th (~15 semitones); excess redistributed
  - Key signature: Krumhansl–Schmuckler pitch-class profile
"""

import numpy as np
import logging
from typing import List, Tuple

logger = logging.getLogger(__name__)

MIDDLE_C = 60
MAX_NOTES_PER_HAND = 5
MAX_SPAN_SEMITONES = 15     # roughly a 10th
_GROUP_TOL = 0.05           # seconds — notes within this window are "simultaneous"

# Krumhansl–Kessler major key profile
_MAJOR_PROFILE = np.array(
    [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
)


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def arrange_for_piano(vocals_notes: List[dict],
                      bass_notes: List[dict],
                      instrument_notes: List[dict]) -> dict:
    """
    Merge three stem note-lists into a two-hand piano arrangement.
    Returns {"treble": [...], "bass": [...]}.
    """
    logger.info(
        f"  Arranging: {len(vocals_notes)} vocal, "
        f"{len(bass_notes)} bass, {len(instrument_notes)} instrument notes"
    )

    treble: List[dict] = []
    bass: List[dict] = []

    for n in vocals_notes:
        treble.append({**n, "_src": "vocal", "_pri": 1})

    for n in bass_notes:
        bass.append({**n, "_src": "bass", "_pri": 1})

    for n in instrument_notes:
        if n["pitch"] >= MIDDLE_C:
            treble.append({**n, "_src": "inst", "_pri": 2})
        else:
            bass.append({**n, "_src": "inst", "_pri": 2})

    # When there are no vocals, the instrument melody IS the top voice —
    # it's already in treble, nothing extra needed.

    treble = _deduplicate(treble)
    bass = _deduplicate(bass)

    treble = _limit_density(treble, MAX_NOTES_PER_HAND)
    bass = _limit_density(bass, MAX_NOTES_PER_HAND)

    treble, overflow = _enforce_span(treble, MAX_SPAN_SEMITONES, prefer_high=True)
    bass.extend(overflow)
    bass, overflow2 = _enforce_span(bass, MAX_SPAN_SEMITONES, prefer_high=False)
    treble.extend(overflow2)

    treble.sort(key=lambda n: (n["startTime"], n["pitch"]))
    bass.sort(key=lambda n: (n["startTime"], n["pitch"]))

    logger.info(f"  Result: {len(treble)} treble notes, {len(bass)} bass notes")
    return {"treble": treble, "bass": bass}


def detect_key_signature(all_notes: List[dict]) -> int:
    """
    Return the best-matching major key (0–11) using Krumhansl–Schmuckler.
    0=C, 1=C#/Db, 2=D, …, 11=B.
    """
    if not all_notes:
        return 0

    counts = np.zeros(12)
    for n in all_notes:
        counts[n["pitch"] % 12] += n.get("duration", 0.5)

    total = counts.sum()
    if total == 0:
        return 0

    counts = counts / total
    profile = _MAJOR_PROFILE / _MAJOR_PROFILE.sum()

    best_key, best_corr = 0, -2.0
    for root in range(12):
        corr = float(np.corrcoef(counts, np.roll(profile, root))[0, 1])
        if corr > best_corr:
            best_corr = corr
            best_key = root

    return best_key


# ──────────────────────────────────────────────────────────────────────────────
# Grouping helpers
# ──────────────────────────────────────────────────────────────────────────────

def _group_simultaneous(notes: List[dict]) -> List[List[dict]]:
    """Group notes whose startTimes are within _GROUP_TOL of each other."""
    if not notes:
        return []
    sorted_n = sorted(notes, key=lambda n: n["startTime"])
    groups: List[List[dict]] = []
    cur = [sorted_n[0]]
    t0 = sorted_n[0]["startTime"]
    for n in sorted_n[1:]:
        if abs(n["startTime"] - t0) <= _GROUP_TOL:
            cur.append(n)
        else:
            groups.append(cur)
            cur = [n]
            t0 = n["startTime"]
    groups.append(cur)
    return groups


def _deduplicate(notes: List[dict]) -> List[dict]:
    """Remove notes sharing (startTime, pitch), keeping lowest priority number."""
    groups = _group_simultaneous(notes)
    result: List[dict] = []
    for grp in groups:
        seen: set = set()
        for n in sorted(grp, key=lambda x: x.get("_pri", 9)):
            if n["pitch"] not in seen:
                seen.add(n["pitch"])
                result.append(n)
    return result


def _limit_density(notes: List[dict], max_n: int) -> List[dict]:
    """Cap simultaneous notes at max_n, keeping high-priority and extreme pitches."""
    groups = _group_simultaneous(notes)
    result: List[dict] = []
    for grp in groups:
        if len(grp) <= max_n:
            result.extend(grp)
            continue

        by_pri = sorted(grp, key=lambda x: (x.get("_pri", 9), 0))
        by_pitch = sorted(grp, key=lambda x: x["pitch"])
        kept_ids: set = set()
        selected: List[dict] = []

        def add(n: dict):
            if id(n) not in kept_ids and len(selected) < max_n:
                kept_ids.add(id(n))
                selected.append(n)

        # Always keep the top-priority note (melody)
        add(by_pri[0])
        # Keep extreme pitches for good voicing
        add(by_pitch[0])
        add(by_pitch[-1])
        # Fill remaining slots in priority order
        for n in by_pri:
            add(n)

        result.extend(selected)
    return result


def _enforce_span(notes: List[dict], max_span: int,
                  prefer_high: bool) -> Tuple[List[dict], List[dict]]:
    """
    Ensure no simultaneous chord spans more than max_span semitones.
    Notes that exceed the span are returned as overflow.
    """
    groups = _group_simultaneous(notes)
    kept: List[dict] = []
    overflow: List[dict] = []

    for grp in groups:
        if len(grp) <= 1:
            kept.extend(grp)
            continue

        pitches = sorted(n["pitch"] for n in grp)
        if pitches[-1] - pitches[0] <= max_span:
            kept.extend(grp)
            continue

        sorted_grp = sorted(grp, key=lambda n: n["pitch"])
        if prefer_high:
            anchor = pitches[-1]
            in_r = [n for n in sorted_grp if anchor - n["pitch"] <= max_span]
            out_r = [n for n in sorted_grp if anchor - n["pitch"] > max_span]
        else:
            anchor = pitches[0]
            in_r = [n for n in sorted_grp if n["pitch"] - anchor <= max_span]
            out_r = [n for n in sorted_grp if n["pitch"] - anchor > max_span]

        kept.extend(in_r)
        overflow.extend(out_r)

    return kept, overflow
