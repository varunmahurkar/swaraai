import { create } from 'zustand';
import type { VoiceProfile, Generation } from '../types';

interface VoiceStore {
  profiles: VoiceProfile[];
  activeProfile: VoiceProfile | null;
  generations: Generation[];
  serverUrl: string;
  loading: boolean;

  setProfiles: (profiles: VoiceProfile[]) => void;
  setActiveProfile: (profile: VoiceProfile | null) => void;
  addGeneration: (generation: Generation) => void;
  setGenerations: (generations: Generation[]) => void;
  setLoading: (loading: boolean) => void;
  setServerUrl: (url: string) => void;
}

export const useVoiceStore = create<VoiceStore>((set) => ({
  profiles: [],
  activeProfile: null,
  generations: [],
  serverUrl: import.meta.env.VITE_SERVER_URL || 'http://localhost:17493',
  loading: false,

  setProfiles: (profiles) => set({ profiles }),
  setActiveProfile: (profile) => set({ activeProfile: profile }),
  addGeneration: (generation) =>
    set((state) => ({ generations: [generation, ...state.generations] })),
  setGenerations: (generations) => set({ generations }),
  setLoading: (loading) => set({ loading }),
  setServerUrl: (url) => set({ serverUrl: url }),
}));
