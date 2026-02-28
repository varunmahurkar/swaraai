# SwaraAI

Local voice cloning and text-to-speech synthesis, powered by **Qwen3-TTS** and **Faster-Whisper**.
Runs entirely on your machine — no cloud, no subscriptions.

---

## Features

- **Voice Cloning** — Upload a 10–300 second reference audio clip to clone any voice
- **High-Quality TTS** — Qwen3-TTS 1.7B (best quality) or 0.6B (faster)
- **Speech-to-Text** — Transcribe audio with Faster-Whisper (base / small / medium / large)
- **Generation History** — Browse, replay, and download all generated audio
- **Model Manager** — Download and manage models with real-time progress
- **GPU Accelerated** — CUDA (NVIDIA), CPU fallback, MLX on Apple Silicon

---

## Quick Start

### 1. Backend
```bash
python start_backend.py
```
Backend runs at: **http://localhost:17493**
API docs: **http://localhost:17493/docs**

### 2. Frontend (separate terminal)
```bash
start_frontend.bat
```
Frontend runs at: **http://localhost:5173**

---

## Project Structure

```
SwaraAI/
├── start_backend.py           ← Start the backend server
├── start_frontend.bat         ← Start the frontend dev server
├── HOW_TO_RUN.md             ← Detailed setup instructions
│
└── codebase/
    ├── backend/
    │   └── source/            ← FastAPI backend (Python)
    │       ├── main.py        ← All API endpoints
    │       ├── profiles.py    ← Voice profile management
    │       ├── tts.py         ← TTS model abstraction
    │       ├── transcribe.py  ← Whisper transcription
    │       ├── history.py     ← Generation history
    │       ├── backends/
    │       │   ├── pytorch_backend.py  ← PyTorch/CUDA (Windows/Linux)
    │       │   └── mlx_backend.py      ← MLX (Apple Silicon)
    │       └── utils/         ← Audio, cache, progress utilities
    │
    └── frontend/
        └── source/            ← React + TypeScript (Vite)
            └── src/
                ├── pages/     ← VoiceProfiles, SpeechGeneration, History, Models
                ├── components/ ← Header, AudioPlayer, ProfileForm, SampleUpload
                ├── hooks/     ← useVoiceProfiles, useSynthesis, useTranscription
                └── lib/api.ts ← API client
```

---

## Install Dependencies

### Backend
```bash
cd codebase/backend
pip install -r source/requirements.txt

# If qwen-tts is not on PyPI, install from GitHub:
pip install git+https://github.com/QwenLM/Qwen3-TTS.git
```

### Frontend
```bash
cd codebase/frontend/source
npm install
```

---

## Models

Models are downloaded from HuggingFace Hub and cached at `~/.cache/huggingface/hub/`.

| Model | Size | Notes |
|-------|------|-------|
| Qwen3-TTS 1.7B | ~3.4 GB | Best voice quality |
| Qwen3-TTS 0.6B | ~1.2 GB | Faster generation |
| Whisper Large | ~3 GB | Most accurate transcription |
| Whisper Medium | ~1.5 GB | Good balance |
| Whisper Small | ~480 MB | Fast |
| Whisper Base | ~140 MB | Fastest |

Use the **Models** page in the UI to download and manage them.

---

## GPU Support

| Platform | Backend | Compute |
|----------|---------|---------|
| NVIDIA GPU | PyTorch | CUDA + bfloat16 |
| CPU (any) | PyTorch | float32 |
| Apple Silicon | MLX | Metal |

Detection is automatic — no configuration needed.

---

## Database

SQLite database at `codebase/backend/data/swaraai.db` (auto-created on first run).

| Table | Contents |
|-------|----------|
| `profiles` | Voice profiles |
| `profile_samples` | Audio reference clips |
| `generations` | Generation history |
| `stories` | Multi-voice compositions |
| `audio_channels` | Output channel config |
