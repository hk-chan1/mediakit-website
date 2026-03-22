"""
Simple file-based result cache keyed by an MD5 fingerprint of the audio content.
Prevents re-processing the same audio twice (e.g., same YouTube URL or same file).
"""

import hashlib
import json
import os
import shutil
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_CACHE_DIR = Path("cache")
_CACHE_DIR.mkdir(exist_ok=True)


def get_audio_fingerprint(audio_path: str) -> str:
    """Hash the first 8 MB of the audio file for a fast but stable fingerprint."""
    h = hashlib.md5()
    with open(audio_path, "rb") as f:
        h.update(f.read(8 * 1024 * 1024))
    return h.hexdigest()


def get_cached(fingerprint: str) -> dict | None:
    """Return cached (midi_data, pdf_path) or None on miss."""
    meta_file = _CACHE_DIR / f"{fingerprint}.json"
    pdf_file = _CACHE_DIR / f"{fingerprint}.pdf"

    if meta_file.exists() and pdf_file.exists():
        try:
            with open(meta_file) as f:
                midi_data = json.load(f)
            logger.info(f"Cache hit: {fingerprint[:8]}…")
            return {"midi_data": midi_data, "pdf_path": str(pdf_file)}
        except Exception as e:
            logger.warning(f"Cache read failed: {e}")
    return None


def save_to_cache(fingerprint: str, midi_data: dict, pdf_path: str):
    """Persist midi_data + PDF for future cache hits."""
    try:
        meta_file = _CACHE_DIR / f"{fingerprint}.json"
        pdf_dest = _CACHE_DIR / f"{fingerprint}.pdf"
        # Strip large internal fields before caching
        cacheable = {k: v for k, v in midi_data.items()
                     if k not in ("treble_notes", "bass_notes")}
        cacheable["treble_notes"] = midi_data.get("treble_notes", [])[:500]
        cacheable["bass_notes"] = midi_data.get("bass_notes", [])[:500]
        with open(meta_file, "w") as f:
            json.dump(cacheable, f)
        shutil.copy2(pdf_path, pdf_dest)
        logger.info(f"Cached: {fingerprint[:8]}…")
    except Exception as e:
        logger.warning(f"Cache write failed: {e}")
