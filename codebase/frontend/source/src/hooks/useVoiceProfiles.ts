import { useState } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { api } from '../lib/api';
import type { VoiceProfile } from '../types';

export function useVoiceProfiles() {
  const queryClient = useQueryClient();
  const [error, setError] = useState<string | null>(null);

  const { data: profiles = [], isLoading } = useQuery({
    queryKey: ['profiles'],
    queryFn: () => api.listProfiles(),
    retry: 1,
  });

  const createMutation = useMutation({
    mutationFn: (data: { name: string; description?: string; language?: string }) =>
      api.createProfile(data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['profiles'] });
      setError(null);
    },
    onError: (err: Error) => setError(err.message),
  });

  const updateMutation = useMutation({
    mutationFn: ({ id, data }: { id: string; data: { name: string; description?: string; language?: string } }) =>
      api.updateProfile(id, data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['profiles'] });
      setError(null);
    },
    onError: (err: Error) => setError(err.message),
  });

  const deleteMutation = useMutation({
    mutationFn: (id: string) => api.deleteProfile(id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['profiles'] });
      setError(null);
    },
    onError: (err: Error) => setError(err.message),
  });

  const addSampleMutation = useMutation({
    mutationFn: ({
      profileId,
      audio,
      referenceText,
    }: {
      profileId: string;
      audio: Blob;
      referenceText: string;
    }) => api.addProfileSample(profileId, audio, referenceText),
    onSuccess: (_data: unknown, variables: { profileId: string; audio: Blob; referenceText: string }) => {
      queryClient.invalidateQueries({ queryKey: ['samples', variables.profileId] });
      setError(null);
    },
    onError: (err: Error) => setError(err.message),
  });

  const deleteSampleMutation = useMutation({
    mutationFn: ({ sampleId }: { sampleId: string; profileId: string }) =>
      api.deleteSample(sampleId),
    onSuccess: (_data: unknown, variables: { sampleId: string; profileId: string }) => {
      queryClient.invalidateQueries({ queryKey: ['samples', variables.profileId] });
      setError(null);
    },
    onError: (err: Error) => setError(err.message),
  });

  const uploadAvatarMutation = useMutation({
    mutationFn: ({ profileId, file }: { profileId: string; file: File }) =>
      api.uploadAvatar(profileId, file),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['profiles'] });
      setError(null);
    },
    onError: (err: Error) => setError(err.message),
  });

  const deleteAvatarMutation = useMutation({
    mutationFn: (profileId: string) => api.deleteAvatar(profileId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['profiles'] });
      setError(null);
    },
    onError: (err: Error) => setError(err.message),
  });

  return {
    profiles: profiles as VoiceProfile[],
    loading: isLoading,
    error,
    createProfile: createMutation.mutateAsync,
    updateProfile: (id: string, data: { name: string; description?: string; language?: string }) =>
      updateMutation.mutateAsync({ id, data }),
    deleteProfile: deleteMutation.mutateAsync,
    addSample: (profileId: string, audio: Blob, referenceText: string) =>
      addSampleMutation.mutateAsync({ profileId, audio, referenceText }),
    deleteSample: (sampleId: string, profileId: string) =>
      deleteSampleMutation.mutateAsync({ sampleId, profileId }),
    uploadAvatar: (profileId: string, file: File) =>
      uploadAvatarMutation.mutateAsync({ profileId, file }),
    deleteAvatar: deleteAvatarMutation.mutateAsync,
    refetch: () => queryClient.invalidateQueries({ queryKey: ['profiles'] }),
  };
}

export function useProfileSamples(profileId: string | null) {
  return useQuery({
    queryKey: ['samples', profileId],
    queryFn: () => api.getProfileSamples(profileId!),
    enabled: !!profileId,
  });
}
