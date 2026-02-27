import { useState, useEffect, useRef } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { api } from '../lib/api';
import type { ModelStatus } from '../types';

interface DownloadProgress {
  model_name: string;
  current: number;
  total: number;
  progress: number; // 0-100
  filename: string | null;
  status: 'downloading' | 'extracting' | 'complete' | 'error';
  error?: string;
}

const MODEL_META: Record<string, { description: string; approxSize: string; type: 'tts' | 'whisper' }> = {
  'qwen-tts-1.7B':  { description: 'Best quality voice cloning',        approxSize: '3.4 GB', type: 'tts'     },
  'qwen-tts-0.6B':  { description: 'Faster, lighter voice model',        approxSize: '1.2 GB', type: 'tts'     },
  'whisper-large':  { description: 'Most accurate transcription',         approxSize: '3 GB',   type: 'whisper' },
  'whisper-medium': { description: 'Good balance of speed and accuracy',  approxSize: '1.5 GB', type: 'whisper' },
  'whisper-small':  { description: 'Fast transcription',                  approxSize: '480 MB', type: 'whisper' },
  'whisper-base':   { description: 'Fastest transcription',               approxSize: '140 MB', type: 'whisper' },
};

function fmtBytes(bytes: number): string {
  if (bytes <= 0) return '0 B';
  if (bytes >= 1024 ** 3) return `${(bytes / 1024 ** 3).toFixed(2)} GB`;
  if (bytes >= 1024 ** 2) return `${(bytes / 1024 ** 2).toFixed(1)} MB`;
  return `${(bytes / 1024).toFixed(0)} KB`;
}

function fmtSize(sizeMb: number | null): string {
  if (sizeMb == null) return '';
  if (sizeMb >= 1024) return `${(sizeMb / 1024).toFixed(1)} GB`;
  return `${sizeMb.toFixed(0)} MB`;
}

// ── Model card ───────────────────────────────────────────────────────────────
function ModelCard({
  model,
  onDownload,
  onCancel,
  onLoad,
  onUnload,
}: {
  model: ModelStatus;
  onDownload: () => void;
  onCancel: () => void;
  onLoad: () => void;
  onUnload: () => void;
}) {
  const meta = MODEL_META[model.model_name];
  const isTTS = meta?.type === 'tts';

  const [progress, setProgress] = useState<DownloadProgress | null>(null);
  const esRef = useRef<EventSource | null>(null);

  // Open SSE only while the model is actively downloading
  useEffect(() => {
    if (!model.downloading) {
      esRef.current?.close();
      esRef.current = null;
      setProgress(null);
      return;
    }

    const es = new EventSource(`${api.baseURL}/models/progress/${model.model_name}`);
    esRef.current = es;

    es.onmessage = (e) => {
      try {
        const data: DownloadProgress = JSON.parse(e.data);
        setProgress(data);
        if (data.status === 'complete' || data.status === 'error') {
          es.close();
          esRef.current = null;
        }
      } catch { /* ignore parse errors */ }
    };

    es.onerror = () => { es.close(); esRef.current = null; };
    return () => { es.close(); esRef.current = null; };
  }, [model.model_name, model.downloading]);

  // ── Derived progress values ──────────────────────────────────────────────
  const pct        = progress?.progress ?? 0;
  const knownTotal = !!(progress && progress.total > 0);

  // What file / path is being downloaded right now
  const fileLabel = (() => {
    if (!progress) return null;
    if (progress.status === 'extracting') return 'Extracting files…';
    const f = progress.filename;
    if (!f) return null;
    // If it looks like a file name (has an extension), show it; otherwise treat as path
    return f;
  })();

  // Byte counter string
  const byteLabel = (() => {
    if (!progress) return null;
    if (knownTotal) {
      return `${fmtBytes(progress.current)} / ${fmtBytes(progress.total)}`;
    }
    if (progress.current > 0) {
      return `${fmtBytes(progress.current)} downloaded`;
    }
    return null;
  })();

  // The cache/download destination path to display
  // model.local_path is set for all states: not-downloaded (expected path),
  // downloading (cache root), downloaded (snapshots dir)
  const displayPath = model.local_path;

  const border = model.downloaded
    ? 'border-green-500/20'
    : model.downloading
    ? 'border-violet-500/40'
    : 'border-white/10';

  return (
    <div className={`bg-white/5 border ${border} rounded-xl p-5 transition-all`}>

      {/* ── Header row ── */}
      <div className="flex items-start justify-between gap-4">
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 flex-wrap">
            <h3 className="font-semibold text-white">{model.display_name}</h3>

            {/* Size badge */}
            <span className="text-xs px-2 py-0.5 rounded-full bg-white/10 text-white/40">
              {model.downloaded && model.size_mb
                ? fmtSize(model.size_mb)
                : meta?.approxSize ?? '?'}
            </span>

            {model.loaded && (
              <span className="text-xs px-2 py-0.5 rounded-full bg-violet-500/20 text-violet-300 font-medium">
                In Memory
              </span>
            )}
          </div>
          <p className="text-sm text-white/40 mt-0.5">{meta?.description}</p>
        </div>

        {/* Status pill */}
        <div className="flex-shrink-0 text-xs font-medium mt-0.5">
          {model.downloaded ? (
            <span className="text-green-400">✓ Downloaded</span>
          ) : model.downloading ? (
            <span className="text-violet-400 flex items-center gap-1.5">
              <span className="inline-block w-1.5 h-1.5 rounded-full bg-violet-400 animate-pulse" />
              {progress?.status === 'extracting' ? 'Extracting' : 'Downloading'}
            </span>
          ) : (
            <span className="text-white/25">Not downloaded</span>
          )}
        </div>
      </div>

      {/* ── Progress block (while downloading) ── */}
      {model.downloading && (
        <div className="mt-4 space-y-2">

          {/* File being downloaded */}
          <div className="flex justify-between items-start gap-3 text-xs">
            <span className="text-white/40 truncate min-w-0 font-mono">
              {fileLabel ?? 'Connecting to HuggingFace…'}
            </span>
            <span className="text-white/60 flex-shrink-0 tabular-nums font-medium">
              {byteLabel
                ? knownTotal
                  ? `${byteLabel} · ${pct.toFixed(0)}%`
                  : byteLabel
                : pct > 0
                  ? `${pct.toFixed(0)}%`
                  : '—'}
            </span>
          </div>

          {/* Progress bar */}
          <div className="w-full bg-white/10 rounded-full h-2 overflow-hidden">
            {knownTotal ? (
              <div
                className="bg-violet-500 h-full rounded-full transition-all duration-500"
                style={{ width: `${Math.max(1, pct)}%` }}
              />
            ) : (
              /* Indeterminate shimmer when total is unknown */
              <div className="h-full w-full relative overflow-hidden rounded-full">
                <div className="absolute inset-0 bg-violet-500/40 rounded-full" />
                <div
                  className="absolute top-0 bottom-0 w-1/3 bg-violet-400 rounded-full animate-pulse"
                  style={{
                    left: progress?.current
                      ? `${Math.min(70, (progress.current / (1024 ** 3 * 4)) * 100)}%`
                      : '0%',
                    transition: 'left 2s ease-out',
                  }}
                />
              </div>
            )}
          </div>

          {/* Download destination path */}
          {displayPath && (
            <div className="flex items-start gap-2 mt-1">
              <span className="text-white/25 text-xs flex-shrink-0 mt-0.5">Saving to</span>
              <span className="text-xs text-white/40 font-mono break-all">{displayPath}</span>
            </div>
          )}
        </div>
      )}

      {/* SSE error feedback */}
      {progress?.status === 'error' && progress.error !== 'Cancelled by user' && (
        <p className="mt-3 text-xs text-red-400 bg-red-500/10 rounded-lg px-3 py-2">
          {progress.error ?? 'Download failed'}
        </p>
      )}

      {/* ── Path row (downloaded state) ── */}
      {model.downloaded && displayPath && (
        <div className="mt-3 flex items-start gap-2 bg-white/3 border border-white/5 rounded-lg px-3 py-2">
          <span className="text-white/30 text-xs mt-0.5 flex-shrink-0">Saved to</span>
          <span className="text-xs text-white/50 font-mono break-all">{displayPath}</span>
        </div>
      )}

      {/* ── Path row (not downloaded — shows expected location) ── */}
      {!model.downloaded && !model.downloading && displayPath && (
        <div className="mt-3 flex items-start gap-2">
          <span className="text-white/20 text-xs mt-0.5 flex-shrink-0">Will save to</span>
          <span className="text-xs text-white/30 font-mono break-all">{displayPath}</span>
        </div>
      )}

      {/* ── Action buttons ── */}
      <div className="flex gap-2 mt-4">
        {!model.downloaded && !model.downloading && (
          <button
            onClick={onDownload}
            className="text-sm bg-violet-600 hover:bg-violet-500 text-white px-4 py-1.5 rounded-lg transition-colors font-medium"
          >
            Download
          </button>
        )}

        {model.downloading && (
          <>
            <button disabled className="text-sm bg-violet-600/30 text-violet-400/60 px-4 py-1.5 rounded-lg cursor-not-allowed font-medium">
              Downloading…
            </button>
            <button
              onClick={onCancel}
              className="text-sm bg-red-500/10 hover:bg-red-500/20 text-red-400 hover:text-red-300 border border-red-500/20 px-3 py-1.5 rounded-lg transition-colors"
            >
              Cancel
            </button>
          </>
        )}

        {model.downloaded && isTTS && !model.loaded && (
          <button onClick={onLoad} className="text-sm bg-white/10 hover:bg-white/20 text-white px-4 py-1.5 rounded-lg transition-colors">
            Load into Memory
          </button>
        )}

        {model.downloaded && isTTS && model.loaded && (
          <button onClick={onUnload} className="text-sm text-white/40 hover:text-white/70 hover:bg-white/5 px-4 py-1.5 rounded-lg transition-colors">
            Unload
          </button>
        )}

        {model.downloaded && !isTTS && (
          <span className="text-xs text-white/25 self-center">Loaded automatically during transcription</span>
        )}
      </div>
    </div>
  );
}

// ── Page ─────────────────────────────────────────────────────────────────────
export default function Models() {
  const queryClient = useQueryClient();
  const [actionError, setActionError] = useState<string | null>(null);

  const { data: modelStatus, isLoading } = useQuery({
    queryKey: ['modelStatus'],
    queryFn: () => api.getModelStatus(),
    refetchInterval: 3000,
  });

  const models = modelStatus?.models ?? [];
  const ttsModels     = models.filter((m) => m.model_name.startsWith('qwen-tts'));
  const whisperModels = models.filter((m) => m.model_name.startsWith('whisper'));

  const invalidate = () => queryClient.invalidateQueries({ queryKey: ['modelStatus'] });

  const handleDownload = async (modelName: string) => {
    try { setActionError(null); await api.downloadModel(modelName); invalidate(); }
    catch (e) { setActionError(e instanceof Error ? e.message : 'Download failed'); }
  };

  const handleCancel = async (modelName: string) => {
    try { setActionError(null); await api.cancelDownload(modelName); invalidate(); }
    catch (e) { setActionError(e instanceof Error ? e.message : 'Cancel failed'); }
  };

  const handleLoad = async (modelName: string) => {
    try { setActionError(null); await api.loadModel(modelName.replace('qwen-tts-', '')); invalidate(); }
    catch (e) { setActionError(e instanceof Error ? e.message : 'Failed to load model'); }
  };

  const handleUnload = async () => {
    try { setActionError(null); await api.unloadModel(); invalidate(); }
    catch (e) { setActionError(e instanceof Error ? e.message : 'Failed to unload model'); }
  };

  const downloadedCount = models.filter((m) => m.downloaded).length;

  return (
    <div className="max-w-3xl mx-auto">
      <div className="mb-8 flex items-end justify-between gap-4">
        <div>
          <h1 className="text-3xl font-bold text-white">Models</h1>
          <p className="text-white/50 mt-1">Download and manage AI models for voice synthesis and transcription</p>
        </div>
        {!isLoading && models.length > 0 && (
          <span className="text-sm text-white/30 flex-shrink-0">
            {downloadedCount} / {models.length} downloaded
          </span>
        )}
      </div>

      {actionError && (
        <div className="bg-red-500/10 border border-red-500/30 text-red-400 rounded-lg px-4 py-3 mb-6 text-sm flex items-center justify-between">
          <span>{actionError}</span>
          <button onClick={() => setActionError(null)} className="ml-3 opacity-60 hover:opacity-100">✕</button>
        </div>
      )}

      {isLoading ? (
        <div className="text-center py-16 text-white/40">Loading model status…</div>
      ) : (
        <div className="space-y-8">

          <section>
            <div className="flex items-center gap-3 mb-3">
              <h2 className="text-xs font-semibold text-white/40 uppercase tracking-wider">Voice Synthesis — TTS</h2>
              <div className="flex-1 h-px bg-white/5" />
            </div>
            <div className="space-y-3">
              {ttsModels.map((m) => (
                <ModelCard key={m.model_name} model={m}
                  onDownload={() => handleDownload(m.model_name)}
                  onCancel={() => handleCancel(m.model_name)}
                  onLoad={() => handleLoad(m.model_name)}
                  onUnload={handleUnload}
                />
              ))}
            </div>
          </section>

          <section>
            <div className="flex items-center gap-3 mb-3">
              <h2 className="text-xs font-semibold text-white/40 uppercase tracking-wider">Transcription — Whisper</h2>
              <div className="flex-1 h-px bg-white/5" />
            </div>
            <div className="space-y-3">
              {whisperModels.map((m) => (
                <ModelCard key={m.model_name} model={m}
                  onDownload={() => handleDownload(m.model_name)}
                  onCancel={() => handleCancel(m.model_name)}
                  onLoad={() => handleLoad(m.model_name)}
                  onUnload={handleUnload}
                />
              ))}
            </div>
          </section>

          <div className="bg-white/3 border border-white/5 rounded-xl p-4 text-xs text-white/30 space-y-1">
            <p>Models are cached by HuggingFace Hub at <code className="text-white/50">~/.cache/huggingface/hub/</code></p>
            <p>TTS models must be loaded into memory before generating speech. Whisper models load on demand during transcription.</p>
          </div>
        </div>
      )}
    </div>
  );
}
