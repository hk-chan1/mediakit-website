from __future__ import annotations

import os
import uuid
import asyncio
import shutil
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from services.audio_extractor import extract_audio_from_file, extract_audio_from_url
from services.transcriber import transcribe_audio
from services.sheet_generator import generate_sheet_music_pdf

TEMP_DIR = Path("temp")
TEMP_DIR.mkdir(exist_ok=True)

# In-memory job store
jobs: dict[str, dict] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    # Cleanup temp files on shutdown
    if TEMP_DIR.exists():
        shutil.rmtree(TEMP_DIR, ignore_errors=True)


app = FastAPI(title="MediaKit API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class UrlRequest(BaseModel):
    url: str


class JobStatus(BaseModel):
    job_id: str
    status: str
    stage: Optional[str] = None
    error: Optional[str] = None
    midi_data: Optional[dict] = None


def get_job_dir(job_id: str) -> Path:
    d = TEMP_DIR / job_id
    d.mkdir(exist_ok=True)
    return d


async def process_job(job_id: str, audio_path: str):
    """Background task that runs the full pipeline."""
    try:
        # Stage 2: Transcribe audio to MIDI
        jobs[job_id]["stage"] = "analyzing"
        midi_data = await asyncio.to_thread(transcribe_audio, audio_path)

        jobs[job_id]["midi_data"] = midi_data

        # Stage 3: Generate sheet music PDF
        jobs[job_id]["stage"] = "generating"
        job_dir = get_job_dir(job_id)
        pdf_path = str(job_dir / "sheet_music.pdf")
        await asyncio.to_thread(generate_sheet_music_pdf, midi_data, pdf_path)

        jobs[job_id]["stage"] = "done"
        jobs[job_id]["status"] = "done"
        jobs[job_id]["pdf_path"] = pdf_path

    except Exception as e:
        jobs[job_id]["status"] = "error"
        jobs[job_id]["stage"] = "error"
        jobs[job_id]["error"] = str(e)


@app.post("/api/process/upload")
async def process_upload(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(400, "No file provided")

    ext = Path(file.filename).suffix.lower()
    if ext not in {".mp4", ".mov", ".webm"}:
        raise HTTPException(400, f"Unsupported format: {ext}. Use MP4, MOV, or WebM.")

    job_id = str(uuid.uuid4())
    job_dir = get_job_dir(job_id)

    # Save uploaded file
    video_path = str(job_dir / f"input{ext}")
    with open(video_path, "wb") as f:
        content = await file.read()
        f.write(content)

    # Start processing
    jobs[job_id] = {"status": "processing", "stage": "extracting"}

    try:
        audio_path = await asyncio.to_thread(extract_audio_from_file, video_path, str(job_dir))
    except Exception as e:
        jobs[job_id] = {"status": "error", "stage": "error", "error": f"Audio extraction failed: {e}"}
        raise HTTPException(500, f"Audio extraction failed: {e}")

    # Continue pipeline in background
    asyncio.create_task(process_job(job_id, audio_path))

    return {"job_id": job_id}


@app.post("/api/process/url")
async def process_url(req: UrlRequest):
    if not req.url.strip():
        raise HTTPException(400, "URL is required")

    job_id = str(uuid.uuid4())
    job_dir = get_job_dir(job_id)

    jobs[job_id] = {"status": "processing", "stage": "extracting"}

    try:
        audio_path = await asyncio.to_thread(extract_audio_from_url, req.url, str(job_dir))
    except Exception as e:
        jobs[job_id] = {"status": "error", "stage": "error", "error": f"Failed to download video: {e}"}
        raise HTTPException(500, f"Failed to download video: {e}")

    asyncio.create_task(process_job(job_id, audio_path))

    return {"job_id": job_id}


@app.get("/api/status/{job_id}")
async def get_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")

    job = jobs[job_id]
    return JobStatus(
        job_id=job_id,
        status=job.get("status", "unknown"),
        stage=job.get("stage"),
        error=job.get("error"),
        midi_data=job.get("midi_data"),
    )


@app.get("/api/download/{job_id}")
async def download_pdf(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")

    job = jobs[job_id]
    pdf_path = job.get("pdf_path")

    if not pdf_path or not Path(pdf_path).exists():
        raise HTTPException(404, "PDF not ready yet")

    return FileResponse(
        pdf_path,
        media_type="application/pdf",
        filename="piano_sheet_music.pdf",
    )


@app.delete("/api/cleanup/{job_id}")
async def cleanup_job(job_id: str):
    job_dir = TEMP_DIR / job_id
    if job_dir.exists():
        shutil.rmtree(job_dir, ignore_errors=True)
    jobs.pop(job_id, None)
    return {"status": "cleaned"}
