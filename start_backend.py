"""
SwaraAI Backend Startup Script
Run from the O:/SwaraAI directory:
    python start_backend.py
"""
import sys
import subprocess
from pathlib import Path

BACKEND_DIR = Path(__file__).parent / "codebase" / "backend"
HOST = "0.0.0.0"
PORT = 17493

def main():
    print("=" * 60)
    print("  SwaraAI Backend — Local Voice Synthesis")
    print("=" * 60)
    print(f"\n  URL:  http://localhost:{PORT}")
    print(f"  Docs: http://localhost:{PORT}/docs")
    print(f"  Dir:  {BACKEND_DIR}\n")
    print("  Note: First generation will download the Qwen3-TTS model")
    print("        (~3GB). This may take 10-30 minutes.\n")
    print("  Ctrl+C to stop\n")
    print("=" * 60)

    cmd = [
        sys.executable, "-m", "uvicorn",
        "source.main:app",
        "--host", HOST,
        "--port", str(PORT),
        "--reload",
        "--reload-dir", str(BACKEND_DIR / "source"),
        "--log-level", "info",
    ]

    subprocess.run(cmd, cwd=str(BACKEND_DIR))


if __name__ == "__main__":
    main()
