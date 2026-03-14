import { useState, useEffect, useRef, useCallback } from 'react';
import type { Replay, ReplayFrame } from '../types';

export const SPEEDS = [0.5, 1, 2, 4, 8] as const;

const BASE_INTERVAL_MS = 500;

export interface ReplayPlayer {
  frameIndex: number;
  currentFrame: ReplayFrame | null;
  playing: boolean;
  speedIndex: number;
  speeds: typeof SPEEDS;
  togglePlay: () => void;
  stepBack: () => void;
  stepForward: () => void;
  seek: (i: number) => void;
  cycleSpeed: () => void;
  reset: () => void;
  totalFrames: number;
}

export function useReplayPlayer(replay: Replay | null): ReplayPlayer {
  const [frameIndex, setFrameIndex] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speedIndex, setSpeedIndex] = useState(1);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const clearTimer = () => {
    if (intervalRef.current !== null) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
  };

  // Reset when replay changes
  useEffect(() => {
    setFrameIndex(0);
    setPlaying(false);
    clearTimer();
  }, [replay]);

  // Tick forward while playing
  useEffect(() => {
    if (!playing || !replay) {
      clearTimer();
      return;
    }
    const delay = BASE_INTERVAL_MS / SPEEDS[speedIndex];
    intervalRef.current = setInterval(() => {
      setFrameIndex(i => {
        if (i >= replay.frames.length - 1) {
          setPlaying(false);
          return i;
        }
        return i + 1;
      });
    }, delay);
    return clearTimer;
  }, [playing, speedIndex, replay]);

  const togglePlay = useCallback(() => setPlaying(p => !p), []);

  const stepBack = useCallback(() => {
    setPlaying(false);
    setFrameIndex(i => Math.max(0, i - 1));
  }, []);

  const stepForward = useCallback(() => {
    if (!replay) return;
    setPlaying(false);
    setFrameIndex(i => Math.min(replay.frames.length - 1, i + 1));
  }, [replay]);

  const seek = useCallback((i: number) => {
    setPlaying(false);
    setFrameIndex(i);
  }, []);

  const cycleSpeed = useCallback(
    () => setSpeedIndex(i => (i + 1) % SPEEDS.length),
    [],
  );

  const reset = useCallback(() => {
    setFrameIndex(0);
    setPlaying(false);
  }, []);

  return {
    frameIndex,
    currentFrame: replay ? replay.frames[frameIndex] ?? null : null,
    playing,
    speedIndex,
    speeds: SPEEDS,
    togglePlay,
    stepBack,
    stepForward,
    seek,
    cycleSpeed,
    reset,
    totalFrames: replay?.frames.length ?? 0,
  };
}
