// TypeScript types matching the backend Pydantic models

export interface VoiceProfile {
  id: string;
  name: string;
  description: string | null;
  language: string;
  avatar_path: string | null;
  created_at: string;
  updated_at: string;
}

export interface ProfileSample {
  id: string;
  profile_id: string;
  audio_path: string;
  reference_text: string;
}

export interface GenerationRequest {
  profile_id: string;
  text: string;
  language?: string;
  seed?: number;
  model_size?: '1.7B' | '0.6B';
  instruct?: string;
}

export interface Generation {
  id: string;
  profile_id: string;
  text: string;
  language: string;
  audio_path: string;
  duration: number;
  seed: number | null;
  instruct: string | null;
  created_at: string;
}

export interface HistoryItem {
  id: string;
  profile_id: string;
  profile_name: string;
  text: string;
  language: string;
  audio_path: string;
  duration: number;
  seed: number | null;
  instruct: string | null;
  created_at: string;
}

export interface HistoryListResponse {
  items: HistoryItem[];
  total: number;
}

export interface HealthResponse {
  status: string;
  model_loaded: boolean;
  model_downloaded: boolean | null;
  model_size: string | null;
  gpu_available: boolean;
  gpu_type: string | null;
  vram_used_mb: number | null;
  backend_type: string | null;
}

export interface ModelStatus {
  model_name: string;
  display_name: string;
  downloaded: boolean;
  downloading: boolean;
  size_mb: number | null;
  loaded: boolean;
  local_path: string | null;
}

export interface ModelStatusListResponse {
  models: ModelStatus[];
}

export interface TranscriptionResponse {
  text: string;
  duration: number;
}

export interface ApiError {
  detail: string;
}
