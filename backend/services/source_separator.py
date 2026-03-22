import subprocess
import os
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def separate_audio_sources(audio_path: str, output_dir: str) -> dict:
    """
    Separate audio into stems (vocals, bass, other) using Demucs htdemucs_ft.
    Drums are discarded.  Falls back to the original mix if Demucs is
    unavailable or fails.

    Returns a dict mapping stem name → absolute file path.
    """
    logger.info("Stage 1: Separating audio sources with Demucs htdemucs_ft...")
    try:
        return _run_demucs(audio_path, output_dir)
    except Exception as e:
        logger.warning(f"Demucs unavailable or failed ({e}). "
                       "Using full mix for instrument transcription.")
        return {"other": audio_path}


def _run_demucs(audio_path: str, output_dir: str) -> dict:
    stems_dir = os.path.join(output_dir, "stems")
    os.makedirs(stems_dir, exist_ok=True)

    cmd = [
        "python", "-m", "demucs",
        "--name", "htdemucs_ft",
        "--out", stems_dir,
        audio_path,
    ]

    logger.info(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=1200)

    if result.returncode != 0:
        raise RuntimeError(f"Demucs failed (exit {result.returncode}): "
                           f"{result.stderr[-600:]}")

    # Demucs outputs to: {stems_dir}/htdemucs_ft/{audio_stem}/{part}.wav
    audio_stem = Path(audio_path).stem
    model_out = Path(stems_dir) / "htdemucs_ft"

    # Locate the per-track directory (might be 'audio' or a sanitised name)
    track_dirs = list(model_out.glob("*")) if model_out.exists() else []
    if not track_dirs:
        raise RuntimeError(
            f"Demucs ran but produced no output under {model_out}")

    track_dir = track_dirs[0]
    if len(track_dirs) > 1:
        # Pick the best match by name
        matches = [d for d in track_dirs if d.name == audio_stem]
        if matches:
            track_dir = matches[0]

    stems: dict = {}
    for stem_name in ("vocals", "bass", "other"):
        stem_path = track_dir / f"{stem_name}.wav"
        if stem_path.exists():
            stems[stem_name] = str(stem_path)
            logger.info(f"  ✓ {stem_name}: {stem_path.name}")
        else:
            logger.warning(f"  ✗ {stem_name} stem not found at {stem_path}")

    if not stems:
        raise RuntimeError("Demucs produced no usable stems")

    return stems
