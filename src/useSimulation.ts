import { useCallback, useEffect, useState } from 'react';
import { RewardConfig } from './engine/environment';
import { ScenarioDef } from './engine/scenarios';
import {
  Params,
  SimState,
  advance,
  advanceAction,
  advanceEpisode,
  createInitialState,
  resetSimulation,
  restartEpisode,
} from './engine/simulation';
import { Pos } from './engine/types';

/** Delay between microsteps while playing, from "very slow" to "fast". */
export const SPEEDS = [
  { label: 'Very slow', ms: 3200 },
  { label: 'Slow', ms: 2000 },
  { label: 'Relaxed', ms: 1300 },
  { label: 'Normal', ms: 800 },
  { label: 'Brisk', ms: 450 },
  { label: 'Fast', ms: 200 },
  { label: 'Turbo', ms: 60 },
];

/**
 * React wrapper around the pure simulation engine. Owns Play/Pause timing.
 * Every state change goes through an engine function; the UI never computes
 * Q-values itself.
 */
export function useSimulation() {
  const [state, setState] = useState<SimState>(() => createInitialState());
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(3);
  const [inspected, setInspected] = useState<Pos | null>(null);
  const delay = SPEEDS[speed].ms;

  useEffect(() => {
    if (!playing) return;
    const id = window.setInterval(() => setState((s) => advance(s)), delay);
    return () => window.clearInterval(id);
  }, [playing, delay]);

  const pausedDo = useCallback((fn: (s: SimState) => SimState) => {
    setPlaying(false);
    setState(fn);
  }, []);

  return {
    state,
    playing,
    speed,
    delay,
    inspected,
    setInspected,
    setSpeed,
    play: () => setPlaying(true),
    pause: () => setPlaying(false),
    nextStep: () => pausedDo(advance),
    nextAction: () => pausedDo(advanceAction),
    nextEpisode: () => pausedDo(advanceEpisode),
    train: (n: number) =>
      pausedDo((s) => {
        let t = s;
        for (let i = 0; i < n; i++) t = advanceEpisode(t);
        return t;
      }),
    restartEpisode: () => pausedDo(restartEpisode),
    reset: () => {
      setInspected(null);
      pausedDo((s) => resetSimulation(s));
    },
    newSeed: () => {
      setInspected(null);
      pausedDo((s) => resetSimulation(s, Math.floor(Math.random() * 1_000_000)));
    },
    loadScenario: (sc: ScenarioDef) => {
      setInspected(null);
      pausedDo((s) => sc.build(s));
    },
    exitScenario: () => {
      setInspected(null);
      pausedDo((s) => ({ ...resetSimulation(s), rewards: s.rewards }));
    },
    setParams: (p: Partial<Params>) => setState((s) => ({ ...s, params: { ...s.params, ...p } })),
    setRewards: (r: Partial<RewardConfig>) => setState((s) => ({ ...s, rewards: { ...s.rewards, ...r } })),
  };
}

export type SimController = ReturnType<typeof useSimulation>;
