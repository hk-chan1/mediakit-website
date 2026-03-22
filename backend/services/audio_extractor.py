import subprocess
import os
from pathlib import Path


def extract_audio_from_file(video_path: str, output_dir: str) -> str:
    """Extract audio from a local video file using FFmpeg."""
    output_path = os.path.join(output_dir, "audio.wav")

    cmd = [
        "ffmpeg",
        "-i", video_path,
        "-vn",                    # No video
        "-acodec", "pcm_s16le",   # WAV format
        "-ar", "44100",           # 44100 Hz — required for Demucs source separation
        "-y",                     # Overwrite
        output_path,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg failed: {result.stderr[:500]}")

    if not Path(output_path).exists() or Path(output_path).stat().st_size == 0:
        raise RuntimeError("FFmpeg produced no output. The video may not contain audio.")

    return output_path


def extract_audio_from_url(url: str, output_dir: str) -> str:
    """Download audio from a video URL using yt-dlp."""
    output_path = os.path.join(output_dir, "audio.wav")

    cmd = [
        "yt-dlp",
        "--no-playlist",
        "--extract-audio",
        "--audio-format", "wav",
        "--postprocessor-args", "ffmpeg:-ar 44100",
        "--output", os.path.join(output_dir, "audio.%(ext)s"),
        "--max-filesize", "500M",
        url,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

    if result.returncode != 0:
        raise RuntimeError(f"yt-dlp failed: {result.stderr[:500]}")

    # yt-dlp might output to slightly different path
    if not Path(output_path).exists():
        # Look for any wav file in the output dir
        wav_files = list(Path(output_dir).glob("*.wav"))
        if wav_files:
            os.rename(str(wav_files[0]), output_path)
        else:
            raise RuntimeError("Failed to extract audio from the URL. Check the URL and try again.")

    return output_path
