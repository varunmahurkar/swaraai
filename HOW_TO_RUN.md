# SwaraAI — How to Run

## Quick Start

### 1. Start the Backend
```
python start_backend.py
```
Or manually:
```
cd codebase/backend
python -m uvicorn source.main:app --reload --port 17493
```

### 2. Start the Frontend (separate terminal)
```
start_frontend.bat
```
Or manually:
```
cd codebase/frontend/source
npm run dev
```

### 3. Open in Browser
- **Frontend:** http://localhost:5173
- **Backend API:** http://localhost:17493
- **API Docs:** http://localhost:17493/docs

---

## Project Structure

```
SwaraAI/
├── start_backend.py           ← Backend startup script
├── start_frontend.bat         ← Frontend startup script
├── HOW_TO_RUN.md             ← This file
│
├── codebase/
│   ├── backend/
│   │   └── source/           ← FastAPI backend (Python)
│   │       ├── main.py        ← API server (30+ endpoints)
│   │       ├── database.py    ← SQLite database (SQLAlchemy ORM)
│   │       ├── profiles.py    ← Voice profile management
│   │       ├── tts.py         ← TTS model abstraction
│   │       ├── transcribe.py  ← Speech-to-text
│   │       ├── history.py     ← Generation history
│   │       ├── backends/
│   │       │   ├── pytorch_backend.py  ← PyTorch/CUDA (Windows/Linux)
│   │       │   └── mlx_backend.py      ← MLX (Apple Silicon)
│   │       ├── utils/
│   │       │   ├── audio.py   ← Audio processing
│   │       │   └── cache.py   ← Voice prompt cache
│   │       └── data/
│   │           └── swaraai.db  ← SQLite database (auto-created)
│   │
│   └── frontend/
│       └── source/            ← React + TypeScript frontend
│           └── src/
│               ├── main.tsx           ← React entry point
│               ├── App.tsx            ← Main app + routing
│               ├── index.css          ← Global styles
│               ├── components/        ← Reusable UI components
│               ├── pages/             ← VoiceProfiles, SpeechGeneration, History, Models
│               ├── hooks/             ← useVoiceProfiles, useSynthesis, etc.
│               ├── lib/api.ts         ← API client
│               ├── stores/            ← Zustand global state
│               └── types/             ← TypeScript type definitions
```

---

## SQLite Database

The database is automatically created at `codebase/backend/data/swaraai.db` when the backend starts.

**Tables:**
- `profiles` — Voice profiles (name, description, language, avatar)
- `profile_samples` — Audio samples for each profile
- `generations` — Generation history (text, audio path, duration, seed)
- `stories` — Multi-voice story compositions
- `story_items` — Items in a story (linked to generations)
- `audio_channels` — Audio output channels
- `channel_device_mappings` — OS device assignments
- `profile_channel_mappings` — Profile-to-channel assignments

---

## Local Model Downloads (HuggingFace)

Models are downloaded automatically from HuggingFace Hub on first use.

**TTS Models:**
- `Qwen/Qwen3-TTS-12Hz-1.7B-Base` (~3.4 GB) — Best quality
- `Qwen/Qwen3-TTS-12Hz-0.6B-Base` (~1.2 GB) — Faster, smaller

**Transcription (Whisper) Models:**
- `openai/whisper-large` — Most accurate
- `openai/whisper-medium` — Good balance
- `openai/whisper-base` — Fastest

**Download location:** `~/.cache/huggingface/hub/`

**Pre-download via API:**
```bash
curl -X POST http://localhost:17493/models/download \
  -H "Content-Type: application/json" \
  -d '{"model_name": "qwen-tts-1.7B"}'
```

Track progress via SSE:
```bash
curl http://localhost:17493/models/progress/qwen-tts-1.7B
```

---

## Install Dependencies

### Backend
```bash
cd codebase/backend
pip install -r source/requirements.txt

# If qwen-tts fails from PyPI, install from GitHub:
pip install git+https://github.com/QwenLM/Qwen3-TTS.git
```

### Frontend
```bash
cd codebase/frontend/source
npm install
```

---

## API Quick Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Server health check |
| `/profiles` | GET/POST | List/create voice profiles |
| `/profiles/{id}` | GET/PUT/DELETE | Manage a profile |
| `/profiles/{id}/samples` | GET/POST | List/add audio samples |
| `/generate` | POST | Generate speech |
| `/audio/{id}` | GET | Download generated audio |
| `/transcribe` | POST | Speech to text |
| `/history` | GET | Generation history |
| `/models/status` | GET | Model download status |
| `/models/download` | POST | Trigger model download |
| `/models/progress/{name}` | GET | SSE download progress |
| `/docs` | GET | Interactive API documentation |

---

## Workflow

1. **Create a Voice Profile** — Name, description, language
2. **Add Audio Samples** — Record (10-30 sec) or upload audio files
3. **Auto-transcribe** — Click "Auto-transcribe" to get reference text
4. **Generate Speech** — Type text, select profile, click Generate
5. **Listen & Download** — Use the audio player to preview and download

---

## GPU Support

- **CUDA (NVIDIA):** Automatically detected, uses `bfloat16` for efficiency
- **CPU:** Falls back to CPU with `float32` (slower but works)
- **Apple Silicon:** Uses MLX backend with Metal acceleration

Check backend logs on startup for GPU detection info.
