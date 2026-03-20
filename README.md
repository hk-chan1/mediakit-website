# MediaKit - Video to Piano Sheet Music

Turn any video into downloadable piano sheet music. Upload a video file or paste a YouTube/Vimeo URL, and MediaKit automatically extracts the audio, detects musical notes, and generates a clean piano sheet music PDF.

## Features

- **Upload video files** (MP4, MOV, WebM) via drag-and-drop or file picker
- **Paste video URLs** from YouTube, Vimeo, and other platforms
- **AI-powered music transcription** using Spotify's Basic Pitch
- **Piano arrangement** — melody in treble clef, chords/bass in bass clef
- **In-browser preview** of sheet music (VexFlow)
- **PDF download** with proper notation (LilyPond engraving)
- **Responsive UI** built with Next.js, Tailwind CSS, and shadcn/ui

## Quick Start (Docker)

**One command to run everything:**

```bash
docker compose up --build
```

Then open [http://localhost:3000](http://localhost:3000).

## Local Development Setup

### Prerequisites

- Node.js 20+
- Python 3.11+
- FFmpeg
- yt-dlp
- LilyPond (optional — fallback PDF generation works without it)

### Backend

```bash
cd backend
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install reportlab       # for fallback PDF generation
uvicorn main:app --reload --port 8000
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000). The frontend expects the backend at `http://localhost:8000`.

## Architecture

```
┌─────────────┐     ┌──────────────────────────────────────────┐
│   Browser    │────>│  Next.js Frontend (port 3000)            │
│              │<────│  - Upload / URL input                     │
│              │     │  - Progress tracking                      │
│              │     │  - VexFlow sheet music preview             │
│              │     │  - PDF download                            │
└─────────────┘     └───────────────┬──────────────────────────┘
                                    │ API calls
                    ┌───────────────▼──────────────────────────┐
                    │  FastAPI Backend (port 8000)              │
                    │                                           │
                    │  1. Audio Extraction                      │
                    │     - FFmpeg (local files)                │
                    │     - yt-dlp (URLs)                       │
                    │                                           │
                    │  2. Music Transcription                   │
                    │     - Spotify Basic Pitch (audio → MIDI)  │
                    │                                           │
                    │  3. Sheet Music Generation                │
                    │     - LilyPond (PDF engraving)            │
                    │     - reportlab fallback                  │
                    └──────────────────────────────────────────┘
```

## Processing Pipeline

1. **Audio extraction** — FFmpeg strips audio from video files; yt-dlp downloads audio from URLs
2. **Pitch detection** — Spotify's Basic Pitch ML model converts audio to MIDI note events
3. **Piano arrangement** — Notes are split into treble (right hand) and bass (left hand) based on pitch
4. **Sheet music rendering** — LilyPond engraves professional notation; reportlab provides a fallback
5. **Cleanup** — Temporary files are removed after download

## Tech Stack

| Layer    | Technology                              |
| -------- | --------------------------------------- |
| Frontend | Next.js 14, TypeScript, Tailwind, shadcn/ui, VexFlow |
| Backend  | FastAPI, Python 3.11                    |
| Audio    | FFmpeg, yt-dlp                          |
| AI/ML    | Spotify Basic Pitch                     |
| Notation | LilyPond, reportlab                    |
| Deploy   | Docker, docker-compose                  |

## Environment Variables

| Variable              | Default                  | Description           |
| --------------------- | ------------------------ | --------------------- |
| `NEXT_PUBLIC_API_URL` | `http://localhost:8000`  | Backend API URL       |

## License

MIT
