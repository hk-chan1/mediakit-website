"""
Demucs audio source separation.

Optimisations vs. the previous version:
  - Uses htdemucs (base model) instead of htdemucs_ft — faster, good enough
  - Auto-detects GPU and passes --device accordingly
  - 120-second hard timeout; raises RuntimeError on expiry (caller falls back)
  - --segment 30 for chunked, lower-memory processing
"""

import subprocess
import os
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_DEMUCS_TIMEOUT = 120   # seconds before we give up and fall back


def separate_audio_sources(audio_path: str, output_dir: str) -> dict:
    """
    Separate audio into vocals, bass, other stems using Demucs htdemucs.
    Drums are discarded.

    Falls back to {"other": audio_path} if Demucs is unavailable or times out.
    Raises RuntimeError (with explanatory message) so the caller can fall back.
    """
    logger.info("Separating audio sources with Demucs htdemucs…")
    return _run_demucs(audio_path, output_dir)


def _run_demucs(audio_path: str, output_dir: str) -> dict:
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"  Demucs device: {device}")

    stems_dir = os.path.join(output_dir, "stems")
    os.makedirs(stems_dir, exist_ok=True)

    cmd = [
        "python", "-m", "demucs",
        "--name", "htdemucs",       # base model — faster than htdemucs_ft
        "--out", stems_dir,
        "--device", device,
        "--segment", "30",          # 30-second chunks: lower memory, predictable time
        "--overlap", "0.1",
        audio_path,
    ]

    logger.info(f"  CMD: {' '.join(cmd)}")
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=_DEMUCS_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError(
            f"Demucs timed out after {_DEMUCS_TIMEOUT}s — "
            "falling back to direct transcription"
        )

    if result.returncode != 0:
        raise RuntimeError(
            f"Demucs exit {result.returncode}: {result.stderr[-400:]}"
        )

    # Output layout: {stems_dir}/htdemucs/{track_stem}/{part}.wav
    model_out = Path(stems_dir) / "htdemucs"
    track_dirs = list(model_out.glob("*")) if model_out.exists() else []
    if not track_dirs:
        raise RuntimeError(f"Demucs ran but produced no output under {model_out}")

    # Pick the best-matching track dir
    audio_stem = Path(audio_path).stem
    track_dir = track_dirs[0]
    for d in track_dirs:
        if d.name == audio_stem:
            track_dir = d
            break

    stems: dict = {}
    for stem_name in ("vocals", "bass", "other"):
        p = track_dir / f"{stem_name}.wav"
        if p.exists():
            stems[stem_name] = str(p)
            logger.info(f"  ✓ {stem_name}")
        else:
            logger.warning(f"  ✗ {stem_name} not found")

    if not stems:
        raise RuntimeError("Demucs produced no usable stems")

    return stems
