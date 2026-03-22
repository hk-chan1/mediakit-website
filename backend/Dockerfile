FROM python:3.11-slim

# ── System dependencies ───────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    lilypond \
    libsndfile1 \
    libsndfile1-dev \
    build-essential \
    cython3 \
    git \
    && rm -rf /var/lib/apt/lists/*

# ── Python tooling ────────────────────────────────────────────────────────────
RUN pip install --no-cache-dir --upgrade pip setuptools wheel cython

# ── yt-dlp ───────────────────────────────────────────────────────────────────
RUN pip install --no-cache-dir yt-dlp

# ── PyTorch (CPU build — smaller image; swap for CUDA wheel if you have a GPU)
RUN pip install --no-cache-dir \
    torch==2.2.2+cpu \
    torchaudio==2.2.2+cpu \
    --index-url https://download.pytorch.org/whl/cpu

WORKDIR /app

# ── Python dependencies ───────────────────────────────────────────────────────
COPY requirements.txt .

# Install all deps except madmom first (it needs numpy to already be present)
RUN pip install --no-cache-dir numpy>=1.23.0
RUN pip install --no-cache-dir \
    fastapi==0.110.0 \
    "uvicorn[standard]==0.29.0" \
    python-multipart==0.0.9 \
    pydantic==2.6.4 \
    aiofiles==23.2.1 \
    "yt-dlp>=2025.1.0" \
    "basic-pitch==0.3.3" \
    "librosa>=0.10.0" \
    "soundfile>=0.12.1" \
    demucs \
    "crepe>=0.0.13" \
    reportlab \
    midiutil

# madmom requires Cython + numpy to already be installed
RUN pip install --no-cache-dir madmom || \
    echo "madmom install failed — beat_grid.py will use librosa fallback"

# ── Pre-download Demucs model weights (avoids slow first-run download) ────────
RUN python -c "from demucs.pretrained import get_model; get_model('htdemucs_ft')" \
    || echo "Demucs model pre-download failed — will download on first use"

COPY . .

RUN mkdir -p temp

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
