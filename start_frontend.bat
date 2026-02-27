@echo off
echo ============================================================
echo   SwaraAI Frontend - Local Voice Synthesis
echo ============================================================
echo.
echo   Starting development server...
echo   Open: http://localhost:5173
echo.
echo   Ctrl+C to stop
echo ============================================================

cd /d "%~dp0codebase\frontend\source"
npm run dev
