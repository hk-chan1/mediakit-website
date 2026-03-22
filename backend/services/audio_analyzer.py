"""
Quick spectral pre-analysis (<3 seconds) to decide which processing tier to use.

Tier 1 – solo / simple instrument  → skip separation, direct Basic Pitch
Tier 2 – vocals + light backing    → center-channel removal, parallel transcription
Tier 3 – complex multi-source mix  → full Demucs separation
"""

import numpy as np
import logging
import time

logger = logging.getLogger(__name__)


def analyze_audio(audio_path: str) -> dict:
    """
    Analyze up to 30 s of audio and return a tier decision plus metadata.

    Returns:
        tier              – 1 | 2 | 3
        has_vocals        – bool
        has_percussion    – bool
        duration          – float (full file length in seconds)
        estimated_seconds – int
        reason            – human-readable explanation
    """
    import librosa

    t0 = time.time()
    # Load only 30 s at 22 050 Hz for fast analysis
    y, sr = librosa.load(audio_path, sr=22050, mono=True, duration=30.0)
    duration = float(librosa.get_duration(path=audio_path))

    # ── Harmonic / percussive split ───────────────────────────────────────────
    y_harm, y_perc = librosa.effects.hpss(y)
    total_e = float(np.mean(y ** 2)) + 1e-8
    perc_ratio = float(np.mean(y_perc ** 2)) / total_e
    has_percussion = perc_ratio > 0.12

    # ── Vocal activity via mid-frequency energy concentration ─────────────────
    S_harm = np.abs(librosa.stft(y_harm))
    freqs = librosa.fft_frequencies(sr=sr)
    vocal_mask = (freqs >= 300) & (freqs <= 3000)
    all_e = np.mean(S_harm ** 2) + 1e-8
    vocal_e = np.mean(S_harm[vocal_mask] ** 2)
    vocal_ratio = float(vocal_e / all_e)

    # ── Spectral flatness (0 = pure tone, 1 = noise) ──────────────────────────
    flatness = float(np.mean(librosa.feature.spectral_flatness(y=y)))

    # ── Chroma entropy (proxy for harmonic complexity) ────────────────────────
    chroma = librosa.feature.chroma_cqt(y=y_harm, sr=sr)
    chroma_norm = chroma / (chroma.sum(axis=0, keepdims=True) + 1e-8)
    entropy = float(np.mean(
        -np.sum(chroma_norm * np.log2(chroma_norm + 1e-8), axis=0)
    ))

    # A piano/guitar solo has very low flatness (< 0.015) and low entropy
    # Vocals tend to concentrate energy in the mid band AND have moderate flatness
    has_vocals = (vocal_ratio > 0.45) and (flatness > 0.003) and (entropy < 3.0)
    is_simple = (flatness < 0.015) and (not has_percussion)

    # ── Tier selection ────────────────────────────────────────────────────────
    if is_simple:
        tier = 1
        reason = "Solo/simple instrument detected — skipping source separation"
        est = max(15, int(duration * 0.25))
    elif has_vocals and has_percussion:
        tier = 3
        reason = "Vocals + drums/complex mix — running full Demucs separation"
        est = max(60, int(duration * 2.5))
    elif has_vocals:
        tier = 2
        reason = "Vocals with light instrumentation — using center-channel removal"
        est = max(30, int(duration * 0.8))
    elif has_percussion:
        if entropy < 2.8:
            tier = 1
            reason = "Instrumental with percussion, simple harmony — direct transcription"
            est = max(20, int(duration * 0.35))
        else:
            tier = 3
            reason = "Complex instrumental mix — running full Demucs separation"
            est = max(60, int(duration * 2.0))
    else:
        tier = 1
        reason = "Instrumental without vocals — direct polyphonic transcription"
        est = max(20, int(duration * 0.3))

    elapsed = time.time() - t0
    logger.info(
        f"Pre-analysis ({elapsed:.1f}s): flatness={flatness:.5f} "
        f"perc={perc_ratio:.3f} vocal={vocal_ratio:.3f} "
        f"entropy={entropy:.2f} → Tier {tier}"
    )
    logger.info(f"  {reason} (est. {est}s)")

    return {
        "tier": tier,
        "has_vocals": has_vocals,
        "has_percussion": has_percussion,
        "duration": duration,
        "estimated_seconds": est,
        "reason": reason,
        # debug fields
        "flatness": round(flatness, 6),
        "perc_ratio": round(perc_ratio, 4),
        "vocal_ratio": round(vocal_ratio, 4),
        "chroma_entropy": round(entropy, 3),
    }
