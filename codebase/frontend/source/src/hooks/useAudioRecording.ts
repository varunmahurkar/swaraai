import { useState, useRef, useCallback } from 'react';

const MAX_RECORDING_MS = 300_000; // 5 minutes

export function useAudioRecording() {
  const [isRecording, setIsRecording] = useState(false);
  const [durationMs, setDurationMs] = useState(0);
  const [audioBlob, setAudioBlob] = useState<Blob | null>(null);
  const [error, setError] = useState<string | null>(null);

  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const startRecording = useCallback(async () => {
    try {
      setError(null);
      setAudioBlob(null);
      setDurationMs(0);
      chunksRef.current = [];

      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

      // Prefer WAV-compatible formats
      const mimeType = MediaRecorder.isTypeSupported('audio/webm;codecs=opus')
        ? 'audio/webm;codecs=opus'
        : 'audio/webm';

      const recorder = new MediaRecorder(stream, { mimeType });

      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunksRef.current.push(e.data);
      };

      recorder.onstop = () => {
        const blob = new Blob(chunksRef.current, { type: mimeType });
        setAudioBlob(blob);
        stream.getTracks().forEach((t) => t.stop());
      };

      recorder.start(100);
      mediaRecorderRef.current = recorder;
      setIsRecording(true);

      let elapsed = 0;
      timerRef.current = setInterval(() => {
        elapsed += 100;
        setDurationMs(elapsed);
        if (elapsed >= MAX_RECORDING_MS) stopRecording();
      }, 100);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Microphone access denied');
    }
  }, []);

  const stopRecording = useCallback(() => {
    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }
    if (mediaRecorderRef.current?.state !== 'inactive') {
      mediaRecorderRef.current?.stop();
    }
    setIsRecording(false);
  }, []);

  const reset = useCallback(() => {
    stopRecording();
    setAudioBlob(null);
    setDurationMs(0);
    setError(null);
    chunksRef.current = [];
  }, [stopRecording]);

  return {
    isRecording,
    durationMs,
    audioBlob,
    error,
    startRecording,
    stopRecording,
    reset,
  };
}
