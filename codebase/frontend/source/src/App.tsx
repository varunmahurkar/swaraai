import { useEffect, useState } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import Header from './components/Header';
import VoiceProfiles from './pages/VoiceProfiles';
import SpeechGeneration from './pages/SpeechGeneration';
import History from './pages/History';
import Models from './pages/Models';

const SERVER_URL = import.meta.env.VITE_SERVER_URL || 'http://localhost:17493';

function ServerStatus() {
  const [status, setStatus] = useState<'connecting' | 'online' | 'offline'>('connecting');

  useEffect(() => {
    const check = async () => {
      try {
        const r = await fetch(`${SERVER_URL}/health`, { signal: AbortSignal.timeout(3000) });
        setStatus(r.ok ? 'online' : 'offline');
      } catch {
        setStatus('offline');
      }
    };
    check();
    const timer = setInterval(check, 15_000);
    return () => clearInterval(timer);
  }, []);

  if (status === 'online') return null;

  return (
    <div className={`text-center py-2 text-xs font-medium ${status === 'connecting' ? 'bg-amber-500/20 text-amber-400' : 'bg-red-500/20 text-red-400'}`}>
      {status === 'connecting'
        ? '⏳ Connecting to backend…'
        : `⚠️ Backend offline — make sure the server is running at ${SERVER_URL}`}
    </div>
  );
}

export default function App() {
  return (
    <Router>
      <div className="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-indigo-950 flex flex-col">
        <Header />
        <ServerStatus />

        <main className="flex-1 px-6 py-8">
          <Routes>
            <Route path="/" element={<VoiceProfiles />} />
            <Route path="/generate" element={<SpeechGeneration />} />
            <Route path="/history" element={<History />} />
            <Route path="/models" element={<Models />} />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        </main>

        <footer className="text-center py-4 text-xs text-white/20">
          SwaraAI — Local Voice Synthesis powered by Qwen3-TTS
        </footer>
      </div>
    </Router>
  );
}
