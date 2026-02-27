import { useState } from 'react';
import { api } from '../lib/api';

export function useTranscription() {
  const [isTranscribing, setIsTranscribing] = useState(false);
  const [transcript, setTranscript] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const transcribe = async (audio: Blob, language?: string): Promise<string | null> => {
    setIsTranscribing(true);
    setError(null);
    try {
      const result = await api.transcribeAudio(audio, language);
      setTranscript(result.text);
      return result.text;
    } catch (err) {
      const msg = err instanceof Error ? err.message : 'Transcription failed';
      setError(msg);
      // Return 202 hint if model downloading
      if (msg.includes('202') || msg.includes('downloading')) {
        setError('Whisper model is downloading. Please wait and try again.');
      }
      return null;
    } finally {
      setIsTranscribing(false);
    }
  };

  return { isTranscribing, transcript, error, transcribe };
}
