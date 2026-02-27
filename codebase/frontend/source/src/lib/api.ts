// API client for the SwaraAI backend

const API_BASE_URL = import.meta.env.VITE_SERVER_URL || 'http://localhost:17493';

class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
    this.name = 'ApiError';
  }
}

async function request<T>(endpoint: string, options: RequestInit = {}): Promise<T> {
  const url = `${API_BASE_URL}${endpoint}`;
  const response = await fetch(url, {
    headers: {
      'Content-Type': 'application/json',
      ...options.headers,
    },
    ...options,
  });

  if (!response.ok) {
    let message = `HTTP ${response.status}`;
    try {
      const err = await response.json();
      message = err.detail || message;
    } catch {
      // ignore JSON parse error
    }
    throw new ApiError(response.status, message);
  }

  return response.json() as Promise<T>;
}

export const api = {
  baseURL: API_BASE_URL,

  // Health
  health() {
    return request<import('../types').HealthResponse>('/health');
  },

  // Models
  getModelStatus() {
    return request<import('../types').ModelStatusListResponse>('/models/status');
  },
  downloadModel(modelName: string) {
    return request('/models/download', {
      method: 'POST',
      body: JSON.stringify({ model_name: modelName }),
    });
  },
  loadModel(modelSize = '1.7B') {
    return request('/models/load', {
      method: 'POST',
      body: JSON.stringify({ model_size: modelSize }),
    });
  },
  unloadModel() {
    return request('/models/unload', { method: 'POST' });
  },
  cancelDownload(modelName: string) {
    return request('/models/cancel', {
      method: 'POST',
      body: JSON.stringify({ model_name: modelName }),
    });
  },

  // Profiles
  listProfiles() {
    return request<import('../types').VoiceProfile[]>('/profiles');
  },
  getProfile(id: string) {
    return request<import('../types').VoiceProfile>(`/profiles/${id}`);
  },
  createProfile(data: { name: string; description?: string; language?: string }) {
    return request<import('../types').VoiceProfile>('/profiles', {
      method: 'POST',
      body: JSON.stringify(data),
    });
  },
  updateProfile(id: string, data: { name: string; description?: string; language?: string }) {
    return request<import('../types').VoiceProfile>(`/profiles/${id}`, {
      method: 'PUT',
      body: JSON.stringify(data),
    });
  },
  deleteProfile(id: string) {
    return request(`/profiles/${id}`, { method: 'DELETE' });
  },

  // Profile Samples
  getProfileSamples(profileId: string) {
    return request<import('../types').ProfileSample[]>(`/profiles/${profileId}/samples`);
  },
  addProfileSample(profileId: string, audio: Blob, referenceText: string) {
    const formData = new FormData();
    formData.append('file', audio, 'sample.wav');
    formData.append('reference_text', referenceText);
    return request<import('../types').ProfileSample>(`/profiles/${profileId}/samples`, {
      method: 'POST',
      headers: {},  // Let browser set Content-Type with boundary
      body: formData,
    });
  },
  updateSample(sampleId: string, referenceText: string) {
    return request<import('../types').ProfileSample>(`/profiles/samples/${sampleId}`, {
      method: 'PUT',
      body: JSON.stringify({ reference_text: referenceText }),
    });
  },
  deleteSample(sampleId: string) {
    return request(`/profiles/samples/${sampleId}`, { method: 'DELETE' });
  },

  // Avatar
  uploadAvatar(profileId: string, file: File) {
    const formData = new FormData();
    formData.append('avatar', file);
    return request<import('../types').VoiceProfile>(`/profiles/${profileId}/avatar`, {
      method: 'POST',
      headers: {},
      body: formData,
    });
  },
  getAvatarUrl(profileId: string) {
    return `${API_BASE_URL}/profiles/${profileId}/avatar`;
  },
  deleteAvatar(profileId: string) {
    return request(`/profiles/${profileId}/avatar`, { method: 'DELETE' });
  },

  // Generation
  generate(data: import('../types').GenerationRequest) {
    return request<import('../types').Generation>('/generate', {
      method: 'POST',
      body: JSON.stringify(data),
    });
  },
  getAudioUrl(generationId: string) {
    return `${API_BASE_URL}/audio/${generationId}`;
  },

  // History
  getHistory(params?: { profile_id?: string; search?: string; limit?: number; offset?: number }) {
    const query = new URLSearchParams();
    if (params?.profile_id) query.set('profile_id', params.profile_id);
    if (params?.search) query.set('search', params.search);
    if (params?.limit) query.set('limit', String(params.limit));
    if (params?.offset) query.set('offset', String(params.offset));
    const qs = query.toString();
    return request<import('../types').HistoryListResponse>(`/history${qs ? `?${qs}` : ''}`);
  },
  deleteGeneration(id: string) {
    return request(`/history/${id}`, { method: 'DELETE' });
  },
  getGenerationAudioUrl(generationId: string) {
    return `${API_BASE_URL}/audio/${generationId}`;
  },

  // Transcription
  transcribeAudio(audio: Blob, language?: string) {
    const formData = new FormData();
    formData.append('file', audio, 'recording.wav');
    if (language) formData.append('language', language);
    return request<import('../types').TranscriptionResponse>('/transcribe', {
      method: 'POST',
      headers: {},
      body: formData,
    });
  },

  // Cache
  clearCache() {
    return request('/cache/clear', { method: 'POST' });
  },

  // Sample audio URL
  getSampleAudioUrl(sampleId: string) {
    return `${API_BASE_URL}/samples/${sampleId}`;
  },
};
