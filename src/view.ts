import { DqnState, dqnPhaseIndex, dqnReached, POSITION_ENCODER } from './engine/dqn';
import { Transition } from './engine/environment';
import { predict } from './engine/nn';
import { Decision } from './engine/policy';
import { SimState, phaseIndex, reached } from './engine/simulation';
import { Action, Pos, posKey } from './engine/types';

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

/** Everything the game board needs, independent of the learning algorithm. */
export interface BoardModel {
  pos: Pos;
  /** Q-values shown on each tile (table lookup or network prediction). */
  values: (p: Pos) => number[] | undefined;
  hl: Highlights;
  transition: Transition | null;
  afterExecute: boolean;
  look: Action | null;
  bumping: boolean;
  showReward: boolean;
  tick: number;
  updateKey: number;
  arrowKey: string;
}

export function boardModelFromSim(s: SimState): BoardModel {
  return {
    pos: s.pos,
    values: (p) => s.q[posKey(p)],
    hl: highlights(s),
    transition: s.transition,
    afterExecute: !!s.transition && reached(s.phase, 'execute'),
    look: s.decision?.action !== undefined && reached(s.phase, 'select') ? s.decision.action : null,
    bumping: s.phase === 'execute' && s.transition?.outcome === 'bump',
    showReward: s.phase === 'reward',
    tick: s.tick,
    updateKey: s.lastUpdate?.tick ?? 0,
    arrowKey: `arr${s.decision?.action}${s.step}`,
  };
}

const dBetween = (s: DqnState, a: string, b: string) => {
  const i = dqnPhaseIndex(s.phase);
  return i >= dqnPhaseIndex(a as never) && i <= dqnPhaseIndex(b as never);
};

/** Main-network predictions for every non-terminal tile (cached per network object). */
const tileCache = new WeakMap<object, Map<string, number[]>>();
export function networkTileValues(net: DqnState['main']): (p: Pos) => number[] | undefined {
  let m = tileCache.get(net);
  if (!m) {
    m = new Map();
    tileCache.set(net, m);
  }
  const cache = m;
  return (p) => {
    const k = posKey(p);
    if (!cache.has(k)) cache.set(k, predict(net, POSITION_ENCODER.encode(p)));
    return cache.get(k);
  };
}

export function boardModelFromDqn(s: DqnState): BoardModel {
  const tr = s.transition;
  const live = s.phase !== 'start' && s.phase !== 'continue';
  return {
    pos: s.pos,
    values: networkTileValues(s.main),
    hl: {
      current: live || s.scenario?.illustrative ? s.current : null,
      inspectingCurrent: dBetween(s, 'predict', 'select'),
      selectedAction: dBetween(s, 'select', 'store') ? s.decision?.action ?? null : s.scenario?.illustrative && tr ? tr.action : null,
      next: tr && (dBetween(s, 'reward', 'store') || !!s.scenario?.illustrative) ? tr.to : null,
      nextRow: false,
      nextMax: [],
      updated: null,
    },
    transition: tr,
    afterExecute: !!tr && dqnReached(s.phase, 'execute') && s.phase !== 'continue',
    look: s.decision?.action !== undefined && dBetween(s, 'select', 'store') ? s.decision.action : null,
    bumping: s.phase === 'execute' && tr?.outcome === 'bump',
    showReward: s.phase === 'reward',
    tick: s.tick,
    updateKey: s.updates,
    arrowKey: `arr${s.decision?.action}${s.totalSteps}`,
  };
}

/** Inputs for the shared epsilon-greedy panel. */
export interface EpsilonModel {
  epsilon: number;
  decision: Decision | null;
  /** Q-values to show on the four action chips (null = not known yet). */
  values: number[] | null;
  /** True once the action has been chosen. */
  selected: Action | null;
  /** Changes only when a new decision is drawn (restarts the marker animation). */
  animKey: string;
}

export function epsilonModelFromSim(s: SimState): EpsilonModel {
  const d = reached(s.phase, 'decide') && s.phase !== 'continue' ? s.decision : null;
  return {
    epsilon: d ? d.epsilon : s.params.epsilon,
    decision: d,
    values: d?.qValues ?? (reached(s.phase, 'inspect') && s.phase !== 'continue' ? s.q[posKey(s.current)] : null),
    selected: d && reached(s.phase, 'select') ? d.action ?? null : null,
    animKey: `${s.episode}-${s.step}`,
  };
}

export function epsilonModelFromDqn(s: DqnState): EpsilonModel {
  const d = dBetween(s, 'decide', 'sync') ? s.decision : null;
  return {
    epsilon: d ? d.epsilon : s.params.epsilon,
    decision: d,
    values: d?.qValues ?? (dBetween(s, 'predict', 'sync') ? s.predicted : null),
    selected: d && dqnReached(s.phase, 'select') ? d.action ?? null : null,
    animKey: `${s.episode}-${s.totalSteps}`,
  };
}
