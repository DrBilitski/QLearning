import { useCallback, useEffect, useState } from 'react';
import {
  DqnParams,
  DqnState,
  createDqnState,
  dqnAdvance,
  dqnNextAction,
  dqnNextEpisode,
  dqnNextUpdate,
  dqnReset,
  dqnRestartEpisode,
} from './engine/dqn';
import { DqnScenarioDef } from './engine/dqnScenarios';
import { RewardConfig } from './engine/environment';
import { SPEEDS } from './useSimulation';

export type DqnView = 'game' | 'training';

/** React wrapper around the pure DQN engine (Play/Pause timing + UI-only state). */
export function useDqnSimulation() {
  const [state, setState] = useState<DqnState>(() => createDqnState());
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(3);
  const [view, setView] = useState<DqnView>('game');
  const [detailed, setDetailed] = useState(false);
  /** Which sampled experience is inspected in detail (UI only; the whole batch always trains). */
  const [focus, setFocus] = useState(0);
  const delay = SPEEDS[speed].ms;

  useEffect(() => {
    if (!playing) return;
    const id = window.setInterval(() => setState((s) => dqnAdvance(s)), delay);
    return () => window.clearInterval(id);
  }, [playing, delay]);

  // New mini-batch → inspect its first sample.
  const batchId = state.train?.samples.map((s) => s.exp.id).join(',') ?? '';
  useEffect(() => setFocus(0), [batchId]);

  const pausedDo = useCallback((fn: (s: DqnState) => DqnState) => {
    setPlaying(false);
    setState(fn);
  }, []);

  return {
    state,
    playing,
    speed,
    delay,
    view,
    setView,
    detailed,
    setDetailed,
    focus: Math.min(focus, Math.max(0, (state.train?.samples.length ?? 1) - 1)),
    setFocus,
    setSpeed,
    play: () => setPlaying(true),
    pause: () => setPlaying(false),
    nextStep: () => pausedDo(dqnAdvance),
    nextAction: () => pausedDo(dqnNextAction),
    nextUpdate: () => pausedDo(dqnNextUpdate),
    nextEpisode: () => pausedDo(dqnNextEpisode),
    train: (n: number) =>
      pausedDo((s) => {
        let t = s;
        for (let i = 0; i < n; i++) t = dqnNextEpisode(t);
        return t;
      }),
    restartEpisode: () => pausedDo(dqnRestartEpisode),
    reset: () => pausedDo((s) => dqnReset(s)),
    newSeed: () => pausedDo((s) => dqnReset(s, Math.floor(Math.random() * 1_000_000))),
    loadScenario: (sc: DqnScenarioDef) => {
      pausedDo((s) => sc.build(s));
      if (sc.view) setView(sc.view);
      setDetailed(!!sc.detailed);
    },
    exitScenario: () => pausedDo((s) => ({ ...dqnReset(s), rewards: s.rewards })),
    setParams: (p: Partial<DqnParams>) => setState((s) => ({ ...s, params: { ...s.params, ...p } })),
    setRewards: (r: Partial<RewardConfig>) => setState((s) => ({ ...s, rewards: { ...s.rewards, ...r } })),
  };
}

export type DqnController = ReturnType<typeof useDqnSimulation>;
