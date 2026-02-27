import { useState, useRef } from 'react';
import { useAudioRecording } from '../hooks/useAudioRecording';
import { useTranscription } from '../hooks/useTranscription';

interface SampleUploadProps {
  profileId: string;
  onAdd: (audio: Blob, referenceText: string) => Promise<unknown>;
  onDone: () => void;
}

export default function SampleUpload({ onAdd, onDone }: SampleUploadProps) {
  const [referenceText, setReferenceText] = useState('');
  const [audioBlob, setAudioBlob] = useState<Blob | null>(null);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [mode, setMode] = useState<'upload' | 'record'>('record');
  const fileInputRef = useRef<HTMLInputElement>(null);

  const { isRecording, durationMs, audioBlob: recordedBlob, startRecording, stopRecording, reset: resetRecording } = useAudioRecording();
  const { isTranscribing, transcribe } = useTranscription();

  const currentBlob = audioBlob ?? recordedBlob;

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setAudioBlob(file);
      setError(null);
    }
  };

  const handleAutoTranscribe = async () => {
    if (!currentBlob) return;
    const text = await transcribe(currentBlob);
    if (text) setReferenceText(text);
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!currentBlob || !referenceText.trim()) return;

    setUploading(true);
    setError(null);
    try {
      await onAdd(currentBlob, referenceText.trim());
      setReferenceText('');
      setAudioBlob(null);
      resetRecording();
      onDone();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Upload failed');
    } finally {
      setUploading(false);
    }
  };

  const formatDuration = (ms: number) => {
    const s = Math.floor(ms / 1000);
    return `${Math.floor(s / 60)}:${(s % 60).toString().padStart(2, '0')}`;
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-4">
      {/* Mode selector */}
      <div className="flex rounded-lg overflow-hidden border border-white/10">
        <button
          type="button"
          onClick={() => setMode('record')}
          className={`flex-1 py-2 text-sm font-medium transition-colors ${mode === 'record' ? 'bg-violet-600 text-white' : 'text-white/50 hover:text-white hover:bg-white/5'}`}
        >
          Record
        </button>
        <button
          type="button"
          onClick={() => setMode('upload')}
          className={`flex-1 py-2 text-sm font-medium transition-colors ${mode === 'upload' ? 'bg-violet-600 text-white' : 'text-white/50 hover:text-white hover:bg-white/5'}`}
        >
          Upload File
        </button>
      </div>

      {mode === 'record' ? (
        <div className="space-y-3">
          <div className="flex items-center gap-3">
            {!isRecording ? (
              <button
                type="button"
                onClick={startRecording}
                className="flex items-center gap-2 bg-red-600 hover:bg-red-500 text-white px-4 py-2 rounded-lg text-sm font-medium transition-colors"
              >
                <span className="w-2 h-2 rounded-full bg-white animate-pulse" />
                Start Recording
              </button>
            ) : (
              <button
                type="button"
                onClick={stopRecording}
                className="flex items-center gap-2 bg-slate-600 hover:bg-slate-500 text-white px-4 py-2 rounded-lg text-sm font-medium transition-colors"
              >
                <span className="w-2 h-2 rounded-full bg-white" />
                Stop — {formatDuration(durationMs)}
              </button>
            )}
            {recordedBlob && !isRecording && (
              <span className="text-sm text-green-400">Recording ready</span>
            )}
          </div>
          <p className="text-xs text-white/40">Speak clearly for 10–300 seconds (up to 5 min) for best quality.</p>
        </div>
      ) : (
        <div>
          <input
            ref={fileInputRef}
            type="file"
            accept="audio/*"
            onChange={handleFileChange}
            className="hidden"
          />
          <button
            type="button"
            onClick={() => fileInputRef.current?.click()}
            className="w-full border-2 border-dashed border-white/20 hover:border-violet-500 rounded-lg p-4 text-center text-white/50 hover:text-white transition-colors text-sm"
          >
            {audioBlob ? (audioBlob as File).name ?? 'Audio selected' : 'Click to select audio file (WAV, MP3, M4A)'}
          </button>
        </div>
      )}

      {/* Reference text */}
      <div>
        <div className="flex items-center justify-between mb-1">
          <label className="text-sm font-medium text-white/70">
            Reference Text <span className="text-red-400">*</span>
          </label>
          {currentBlob && (
            <button
              type="button"
              onClick={handleAutoTranscribe}
              disabled={isTranscribing}
              className="text-xs text-violet-400 hover:text-violet-300 disabled:opacity-50"
            >
              {isTranscribing ? 'Transcribing...' : 'Auto-transcribe'}
            </button>
          )}
        </div>
        <textarea
          value={referenceText}
          onChange={(e) => setReferenceText(e.target.value)}
          placeholder="Type exactly what is said in the recording..."
          required
          rows={3}
          maxLength={1000}
          className="w-full bg-white/5 border border-white/10 rounded-lg px-3 py-2 text-white placeholder-white/30 focus:outline-none focus:border-violet-500 focus:ring-1 focus:ring-violet-500 transition-colors resize-none"
        />
      </div>

      {error && (
        <div className="text-red-400 text-sm bg-red-400/10 rounded-lg px-3 py-2">{error}</div>
      )}

      <div className="flex gap-3">
        <button
          type="submit"
          disabled={uploading || !currentBlob || !referenceText.trim()}
          className="flex-1 bg-violet-600 hover:bg-violet-500 disabled:opacity-50 disabled:cursor-not-allowed text-white font-medium py-2 px-4 rounded-lg transition-colors"
        >
          {uploading ? 'Uploading...' : 'Add Sample'}
        </button>
        <button
          type="button"
          onClick={onDone}
          className="px-4 py-2 rounded-lg border border-white/10 text-white/70 hover:bg-white/5 transition-colors"
        >
          Cancel
        </button>
      </div>
    </form>
  );
}
