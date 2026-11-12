import { Action, Pos } from './engine/types';
import { SimState, phaseIndex } from './engine/simulation';

/**
 * Which things should be highlighted right now? Derived purely from the
 * engine state so the board, the Q-table and the panels always agree.
 */
export interface Highlights {
  current: Pos | null;
  /** The four Q-values of s are being looked at (inspect → select). */
  inspectingCurrent: boolean;
  selectedAction: Action | null;
  next: Pos | null;
  nextRow: boolean;
  nextMax: Action[];
  updated: { s: Pos; a: Action } | null;
}

const between = (s: SimState, a: string, b: string) => {
  const i = phaseIndex(s.phase);
  return i >= phaseIndex(a as never) && i <= phaseIndex(b as never);
};

export function highlights(s: SimState): Highlights {
  const tr = s.transition;
  return {
    current: between(s, 'observe', 'update') ? s.current : null,
    inspectingCurrent: between(s, 'inspect', 'select'),
    selectedAction: between(s, 'select', 'update') ? s.decision?.action ?? null : null,
    next: tr && between(s, 'reward', 'update') ? tr.to : null,
    nextRow: !!tr && !tr.terminal && between(s, 'inspectNext', 'bellman'),
    nextMax: s.nextInfo && between(s, 'inspectNext', 'bellman') ? s.nextInfo.best : [],
    updated: s.phase === 'update' && s.lastUpdate ? { s: s.lastUpdate.s, a: s.lastUpdate.a } : null,
  };
}
