from __future__ import annotations

import os
import uuid
import asyncio
import shutil
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from services.audio_extractor import extract_audio_from_file, extract_audio_from_url
from services.transcriber import run_full_pipeline
from services.sheet_generator import generate_sheet_music_pdf

TEMP_DIR = Path("temp")
TEMP_DIR.mkdir(exist_ok=True)

jobs: dict[str, dict] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
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


# ── Models ────────────────────────────────────────────────────────────────────

class UrlRequest(BaseModel):
    url: str
    mode: str = "auto"   # "auto" | "quick" | "quality"


class JobStatus(BaseModel):
    job_id: str
    status: str
    stage: Optional[str] = None
    error: Optional[str] = None
    midi_data: Optional[dict] = None
    tier: Optional[int] = None
    tier_reason: Optional[str] = None
    estimated_seconds: Optional[int] = None
    stage_timings: Optional[dict] = None


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_job_dir(job_id: str) -> Path:
    d = TEMP_DIR / job_id
    d.mkdir(exist_ok=True)
    return d


# ── Background worker ─────────────────────────────────────────────────────────

async def process_job(job_id: str, audio_path: str, mode: str = "auto"):
    """Run the full pipeline as a background asyncio task."""
    try:
        job_dir = get_job_dir(job_id)

        # Check content-based cache before doing any work
        from services.cache_manager import get_audio_fingerprint, get_cached, save_to_cache
        fp = get_audio_fingerprint(audio_path)
        cached = get_cached(fp)
        if cached:
            # Copy cached PDF into this job's directory so the download route works
            pdf_path = str(job_dir / "sheet_music.pdf")
            shutil.copy2(cached["pdf_path"], pdf_path)
            jobs[job_id].update({
                "status": "done",
                "stage": "done",
                "midi_data": cached["midi_data"],
                "pdf_path": pdf_path,
                "tier": cached["midi_data"].get("tier"),
                "tier_reason": "Served from cache",
                "estimated_seconds": 0,
                "stage_timings": {"cache": 0.0},
            })
            return

        # Stage/meta callback — safe to call from a worker thread (GIL)
        def update_stage(stage: str, meta: dict = None):
            jobs[job_id]["stage"] = stage
            if meta:
                for k in ("tier", "tier_reason", "estimated_seconds", "stage_timings"):
                    if k in meta:
                        jobs[job_id][k] = meta[k]

        # Stages 1-4 inside run_full_pipeline
        midi_data = await asyncio.to_thread(
            run_full_pipeline,
            audio_path,
            str(job_dir),
            update_stage,
            mode,
        )

        jobs[job_id]["midi_data"] = midi_data
        jobs[job_id]["tier"] = midi_data.get("tier")
        jobs[job_id]["tier_reason"] = midi_data.get("tier_reason")
        jobs[job_id]["stage_timings"] = midi_data.get("stage_timings", {})

        # Stage 5: engrave PDF
        jobs[job_id]["stage"] = "generating"
        pdf_path = str(job_dir / "sheet_music.pdf")
        await asyncio.to_thread(generate_sheet_music_pdf, midi_data, pdf_path)

        # Cache the result
        save_to_cache(fp, midi_data, pdf_path)

        jobs[job_id].update({
            "status": "done",
            "stage": "done",
            "pdf_path": pdf_path,
        })

    except Exception as e:
        jobs[job_id]["status"] = "error"
        jobs[job_id]["stage"] = "error"
        jobs[job_id]["error"] = str(e)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.post("/api/process/upload")
async def process_upload(
    file: UploadFile = File(...),
    mode: str = Form(default="auto"),
):
    if not file.filename:
        raise HTTPException(400, "No file provided")

    ext = Path(file.filename).suffix.lower()
    if ext not in {".mp4", ".mov", ".webm"}:
        raise HTTPException(400, f"Unsupported format: {ext}. Use MP4, MOV, or WebM.")

    job_id = str(uuid.uuid4())
    job_dir = get_job_dir(job_id)
    jobs[job_id] = {"status": "processing", "stage": "extracting", "mode": mode}

    video_path = str(job_dir / f"input{ext}")
    with open(video_path, "wb") as f:
        content = await file.read()
        f.write(content)

    try:
        audio_path = await asyncio.to_thread(
            extract_audio_from_file, video_path, str(job_dir))
    except Exception as e:
        jobs[job_id] = {"status": "error", "stage": "error",
                        "error": f"Audio extraction failed: {e}"}
        raise HTTPException(500, str(e))

    asyncio.create_task(process_job(job_id, audio_path, mode))
    return {"job_id": job_id}


@app.post("/api/process/url")
async def process_url(req: UrlRequest):
    if not req.url.strip():
        raise HTTPException(400, "URL is required")

    job_id = str(uuid.uuid4())
    job_dir = get_job_dir(job_id)
    jobs[job_id] = {"status": "processing", "stage": "extracting", "mode": req.mode}

    try:
        audio_path = await asyncio.to_thread(
            extract_audio_from_url, req.url, str(job_dir))
    except Exception as e:
        jobs[job_id] = {"status": "error", "stage": "error",
                        "error": f"Failed to download video: {e}"}
        raise HTTPException(500, str(e))

    asyncio.create_task(process_job(job_id, audio_path, req.mode))
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
        tier=job.get("tier"),
        tier_reason=job.get("tier_reason"),
        estimated_seconds=job.get("estimated_seconds"),
        stage_timings=job.get("stage_timings"),
    )


@app.get("/api/download/{job_id}")
async def download_pdf(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    pdf_path = jobs[job_id].get("pdf_path")
    if not pdf_path or not Path(pdf_path).exists():
        raise HTTPException(404, "PDF not ready yet")
    return FileResponse(pdf_path, media_type="application/pdf",
                        filename="piano_sheet_music.pdf")


@app.delete("/api/cleanup/{job_id}")
async def cleanup_job(job_id: str):
    job_dir = TEMP_DIR / job_id
    if job_dir.exists():
        shutil.rmtree(job_dir, ignore_errors=True)
    jobs.pop(job_id, None)
    return {"status": "cleaned"}
