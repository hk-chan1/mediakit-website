FROM python:3.11-slim

# ── System dependencies ───────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    lilypond \
    libsndfile1 \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip setuptools wheel cython

# ── yt-dlp ───────────────────────────────────────────────────────────────────
RUN pip install --no-cache-dir yt-dlp

# ── Step 1: numpy first (madmom needs it at build time) ──────────────────────
RUN pip install --no-cache-dir "numpy>=1.23.0,<2.0"

# ── Step 2: PyTorch CPU (install before anything that depends on torch) ───────
RUN pip install --no-cache-dir \
    torch==2.2.2+cpu \
    torchaudio==2.2.2+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# ── Step 3: Demucs (depends on torch, must come after Step 2) ────────────────
RUN pip install --no-cache-dir demucs

# ── Step 4: Basic Pitch + its TensorFlow stack ───────────────────────────────
RUN pip install --no-cache-dir "basic-pitch>=0.3.3"

# ── Step 5: Audio / science stack ────────────────────────────────────────────
RUN pip install --no-cache-dir \
    "librosa>=0.10.0" \
    "soundfile>=0.12.1"

# ── Step 6: CREPE (optional — monophonic pitch; falls back to pYIN if absent) -
RUN pip install --no-cache-dir crepe || \
    echo "CREPE install failed — stem_transcriber will use librosa pYIN fallback"

# ── Step 7: madmom (optional — beat tracking; falls back to librosa) ──────────
RUN pip install --no-cache-dir madmom || \
    echo "madmom install failed — beat_grid will use librosa fallback"

# ── Step 8: Web framework + utilities ────────────────────────────────────────
RUN pip install --no-cache-dir \
    fastapi==0.110.0 \
    "uvicorn[standard]==0.29.0" \
    python-multipart==0.0.9 \
    "pydantic==2.6.4" \
    aiofiles==23.2.1 \
    reportlab \
    midiutil

WORKDIR /app

COPY requirements.txt .

# ── Pre-download Demucs model weights ────────────────────────────────────────
RUN python -c "from demucs.pretrained import get_model; get_model('htdemucs_ft')" \
    || echo "Demucs model pre-download failed — will download on first use"

COPY . .

RUN mkdir -p temp

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
