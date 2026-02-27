import { useState } from 'react';
import { useQueryClient } from '@tanstack/react-query';
import { api } from '../lib/api';
import type { GenerationRequest, Generation } from '../types';

export function useSynthesis() {
  const queryClient = useQueryClient();
  const [isGenerating, setIsGenerating] = useState(false);
  const [generatedAudio, setGeneratedAudio] = useState<string | null>(null);
  const [lastGeneration, setLastGeneration] = useState<Generation | null>(null);
  const [error, setError] = useState<string | null>(null);

  const generate = async (data: GenerationRequest): Promise<Generation | null> => {
    setIsGenerating(true);
    setError(null);
    setGeneratedAudio(null);

    try {
      const result = await api.generate(data);
      const audioUrl = api.getAudioUrl(result.id);
      setGeneratedAudio(audioUrl);
      setLastGeneration(result);
      // Refresh history
      queryClient.invalidateQueries({ queryKey: ['history'] });
      return result;
    } catch (err) {
      const msg = err instanceof Error ? err.message : 'Generation failed';
      setError(msg);
      return null;
    } finally {
      setIsGenerating(false);
    }
  };

  const reset = () => {
    setGeneratedAudio(null);
    setLastGeneration(null);
    setError(null);
  };

  return {
    isGenerating,
    generatedAudio,
    lastGeneration,
    error,
    generate,
    reset,
  };
}
