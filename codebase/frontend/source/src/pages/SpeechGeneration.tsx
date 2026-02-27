import { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { useQuery } from '@tanstack/react-query';
import { useSynthesis } from '../hooks/useSynthesis';
import AudioPlayer from '../components/AudioPlayer';
import { api } from '../lib/api';
import type { VoiceProfile } from '../types';

const LANGUAGES = [
  { code: 'en', label: 'English' },
  { code: 'zh', label: 'Chinese' },
  { code: 'ja', label: 'Japanese' },
  { code: 'ko', label: 'Korean' },
  { code: 'de', label: 'German' },
  { code: 'fr', label: 'French' },
  { code: 'es', label: 'Spanish' },
  { code: 'it', label: 'Italian' },
  { code: 'pt', label: 'Portuguese' },
  { code: 'ru', label: 'Russian' },
];

export default function SpeechGeneration() {
  const [profileId, setProfileId] = useState('');
  const [text, setText] = useState('');
  const [language, setLanguage] = useState('en');
  const [modelSize, setModelSize] = useState<'1.7B' | '0.6B'>('1.7B');
  const [seed, setSeed] = useState('');
  const [instruct, setInstruct] = useState('');
  const [showAdvanced, setShowAdvanced] = useState(false);

  const { data: profiles = [] } = useQuery<VoiceProfile[]>({
    queryKey: ['profiles'],
    queryFn: () => api.listProfiles(),
  });

  const { data: modelStatus } = useQuery({
    queryKey: ['modelStatus'],
    queryFn: () => api.getModelStatus(),
    refetchInterval: 5000,
  });

  const { isGenerating, generatedAudio, lastGeneration, error, generate, reset } = useSynthesis();

  // Sync language when profile changes
  useEffect(() => {
    const profile = profiles.find((p) => p.id === profileId);
    if (profile) setLanguage(profile.language);
  }, [profileId, profiles]);

  const ttsModel = modelStatus?.models?.find((m: import('../types').ModelStatus) => m.model_name === `qwen-tts-${modelSize}`);
  const modelReady = ttsModel?.downloaded;

  const handleGenerate = async () => {
    if (!profileId || !text.trim()) return;
    await generate({
      profile_id: profileId,
      text: text.trim(),
      language,
      model_size: modelSize,
      seed: seed ? Number(seed) : undefined,
      instruct: instruct.trim() || undefined,
    });
  };

  return (
    <div className="max-w-2xl mx-auto">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-white">Generate Speech</h1>
        <p className="text-white/50 mt-1">Clone voices and synthesize speech using Qwen3-TTS</p>
      </div>

      {/* Model status banner */}
      {ttsModel && !modelReady && (
        <div className="bg-amber-500/10 border border-amber-500/30 rounded-xl px-4 py-3 mb-6 flex items-center gap-3">
          <span className="text-amber-400 text-lg">⚠️</span>
          <div className="flex-1 min-w-0">
            <p className="text-amber-400 text-sm font-medium">Model not downloaded</p>
            <p className="text-amber-400/70 text-xs">
              Download the model first, or the first generation will trigger it automatically.
            </p>
          </div>
          <div className="flex gap-2 flex-shrink-0">
            <button
              onClick={() => api.downloadModel(`qwen-tts-${modelSize}`)}
              className="text-xs bg-amber-500/20 hover:bg-amber-500/30 text-amber-400 px-3 py-1.5 rounded-lg transition-colors"
            >
              Download Now
            </button>
            <Link
              to="/models"
              className="text-xs bg-white/5 hover:bg-white/10 text-white/60 hover:text-white px-3 py-1.5 rounded-lg transition-colors"
            >
              View Models
            </Link>
          </div>
        </div>
      )}

      <div className="bg-white/5 border border-white/10 rounded-xl p-6 space-y-5">
        {/* Profile selector */}
        <div>
          <label className="block text-sm font-medium text-white/70 mb-1">
            Voice Profile <span className="text-red-400">*</span>
          </label>
          <select
            value={profileId}
            onChange={(e) => setProfileId(e.target.value)}
            required
            className="w-full bg-slate-800 border border-white/10 rounded-lg px-3 py-2.5 text-white focus:outline-none focus:border-violet-500 focus:ring-1 focus:ring-violet-500 transition-colors"
          >
            <option value="">— Select a profile —</option>
            {profiles.map((p) => (
              <option key={p.id} value={p.id}>
                {p.name} ({p.language.toUpperCase()})
              </option>
            ))}
          </select>
          {profiles.length === 0 && (
            <p className="text-xs text-white/40 mt-1">
              No profiles found.{' '}
              <a href="/" className="text-violet-400 hover:text-violet-300">
                Create one first.
              </a>
            </p>
          )}
        </div>

        {/* Text input */}
        <div>
          <label className="block text-sm font-medium text-white/70 mb-1">
            Text to Synthesize <span className="text-red-400">*</span>
          </label>
          <textarea
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder="Enter the text you want to convert to speech…"
            rows={4}
            maxLength={5000}
            className="w-full bg-white/5 border border-white/10 rounded-lg px-3 py-2.5 text-white placeholder-white/30 focus:outline-none focus:border-violet-500 focus:ring-1 focus:ring-violet-500 transition-colors resize-none"
          />
          <p className="text-xs text-white/30 mt-1 text-right">{text.length}/5000</p>
        </div>

        {/* Language */}
        <div>
          <label className="block text-sm font-medium text-white/70 mb-1">Language</label>
          <select
            value={language}
            onChange={(e) => setLanguage(e.target.value)}
            className="w-full bg-slate-800 border border-white/10 rounded-lg px-3 py-2.5 text-white focus:outline-none focus:border-violet-500 focus:ring-1 focus:ring-violet-500 transition-colors"
          >
            {LANGUAGES.map((l) => (
              <option key={l.code} value={l.code}>{l.label}</option>
            ))}
          </select>
        </div>

        {/* Model size */}
        <div>
          <label className="block text-sm font-medium text-white/70 mb-1">Model Size</label>
          <div className="flex gap-2">
            {(['1.7B', '0.6B'] as const).map((size) => (
              <button
                key={size}
                type="button"
                onClick={() => setModelSize(size)}
                className={`flex-1 py-2 rounded-lg text-sm font-medium border transition-colors ${
                  modelSize === size
                    ? 'bg-violet-600 border-violet-500 text-white'
                    : 'border-white/10 text-white/50 hover:bg-white/5 hover:text-white'
                }`}
              >
                {size}
                {size === '1.7B' ? ' (Best Quality)' : ' (Faster)'}
              </button>
            ))}
          </div>
        </div>

        {/* Advanced options toggle */}
        <button
          type="button"
          onClick={() => setShowAdvanced(!showAdvanced)}
          className="text-xs text-white/40 hover:text-white/70 transition-colors"
        >
          {showAdvanced ? '▲ Hide' : '▼ Show'} Advanced Options
        </button>

        {showAdvanced && (
          <div className="space-y-4 border-t border-white/5 pt-4">
            <div>
              <label className="block text-sm font-medium text-white/70 mb-1">
                Style Instruction
              </label>
              <input
                type="text"
                value={instruct}
                onChange={(e) => setInstruct(e.target.value)}
                placeholder="e.g. Speak slowly and clearly"
                maxLength={500}
                className="w-full bg-white/5 border border-white/10 rounded-lg px-3 py-2 text-white placeholder-white/30 focus:outline-none focus:border-violet-500 focus:ring-1 focus:ring-violet-500 transition-colors"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-white/70 mb-1">
                Seed (for reproducibility)
              </label>
              <input
                type="number"
                value={seed}
                onChange={(e) => setSeed(e.target.value)}
                placeholder="Leave empty for random"
                min={0}
                className="w-full bg-white/5 border border-white/10 rounded-lg px-3 py-2 text-white placeholder-white/30 focus:outline-none focus:border-violet-500 focus:ring-1 focus:ring-violet-500 transition-colors"
              />
            </div>
          </div>
        )}

        {/* Generate button */}
        <button
          onClick={handleGenerate}
          disabled={isGenerating || !profileId || !text.trim()}
          className="w-full bg-violet-600 hover:bg-violet-500 disabled:opacity-50 disabled:cursor-not-allowed text-white font-semibold py-3 rounded-lg transition-colors flex items-center justify-center gap-2"
        >
          {isGenerating ? (
            <>
              <span className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
              Generating Speech…
            </>
          ) : (
            '🎙 Generate Speech'
          )}
        </button>

        {/* Error */}
        {error && (
          <div className="bg-red-500/10 border border-red-500/30 text-red-400 rounded-lg px-4 py-3 text-sm">
            {error}
            {(error.includes('202') || error.includes('download')) && (
              <p className="mt-1 text-xs">The model is downloading in the background. Please wait a few minutes and try again.</p>
            )}
          </div>
        )}
      </div>

      {/* Result */}
      {generatedAudio && lastGeneration && (
        <div className="mt-6 bg-white/5 border border-white/10 rounded-xl p-5">
          <div className="flex items-center justify-between mb-4">
            <h3 className="font-medium text-white">Generated Audio</h3>
            <div className="flex gap-3 text-xs text-white/40">
              <span>{lastGeneration.duration.toFixed(1)}s</span>
              <span>{lastGeneration.language.toUpperCase()}</span>
            </div>
          </div>
          <AudioPlayer src={generatedAudio} title={lastGeneration.text.slice(0, 50)} />
          <p className="text-xs text-white/30 mt-3 line-clamp-2">{lastGeneration.text}</p>
          <button
            onClick={reset}
            className="mt-3 text-xs text-white/40 hover:text-white/70 transition-colors"
          >
            Clear
          </button>
        </div>
      )}
    </div>
  );
}
