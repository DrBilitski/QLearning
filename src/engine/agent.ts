import { GridWorld, RewardConfig } from './environment';
import { ACTIONS, Action, Pos, posKey } from './types';

/**
 * Anything that can answer "what are the Q-values of the four actions in this
 * state?". The visualisation only depends on this interface, so a future DQN
 * (a neural network producing four outputs) can plug into the same UI.
 */
export interface ActionValueProvider {
  readonly kind: 'tabular' | 'dqn';
  getQValues(state: Pos): number[];
}

/** Plain-data Q-table: "x,y" -> [Q(up), Q(down), Q(left), Q(right)]. Immutable by convention. */
export type QTableData = Record<string, number[]>;

export function emptyQTable(world: GridWorld): QTableData {
  const q: QTableData = {};
  for (const s of world.states()) q[posKey(s)] = [0, 0, 0, 0];
  return q;
}

export class TabularQ implements ActionValueProvider {
  readonly kind = 'tabular' as const;
  constructor(private readonly data: QTableData) {}
  getQValues(state: Pos): number[] {
    return this.data[posKey(state)] ?? [0, 0, 0, 0];
  }
}

const TIE_TOLERANCE = 1e-9;

/** All actions whose value equals the maximum (ties are kept so they can be broken randomly). */
export function bestActions(values: number[]): Action[] {
  const max = Math.max(...values);
  return ACTIONS.filter((a) => Math.abs(values[a] - max) <= TIE_TOLERANCE);
}

export function maxValue(values: number[]): number {
  return Math.max(...values);
}

export interface BellmanResult {
  /** r + γ·max Q(s',a'), or just r when s' is terminal. */
  target: number;
  /** target − Q(s,a) */
  tdError: number;
  /** Q(s,a) + α·tdError */
  newQ: number;
}

/**
 * The Q-learning update:
 *   Q(s,a) ← Q(s,a) + α [ r + γ·max_a' Q(s',a') − Q(s,a) ]
 * When s' is terminal there is no future, so the γ·max term is dropped.
 */
export function bellmanUpdate(
  oldQ: number,
  reward: number,
  maxNext: number,
  alpha: number,
  gamma: number,
  terminal: boolean,
): BellmanResult {
  const target = terminal ? reward : reward + gamma * maxNext;
  const tdError = target - oldQ;
  return { target, tdError, newQ: oldQ + alpha * tdError };
}

export function withUpdatedQ(q: QTableData, s: Pos, a: Action, value: number): QTableData {
  const key = posKey(s);
  const row = [...(q[key] ?? [0, 0, 0, 0])];
  row[a] = value;
  return { ...q, [key]: row };
}

/**
 * Synchronous Q-value iteration on the (known, deterministic) environment.
 * Used only to prepare Q-table snapshots for the guided scenarios:
 *   sweeps = small  -> a "partially trained" table
 *   sweeps = large  -> the converged table a long training run approaches
 */
export function qValueIteration(
  world: GridWorld,
  rewards: RewardConfig,
  gamma: number,
  sweeps: number,
): QTableData {
  let q = emptyQTable(world);
  for (let k = 0; k < sweeps; k++) {
    const next: QTableData = {};
    for (const s of world.states()) {
      next[posKey(s)] = ACTIONS.map((a) => {
        const tr = world.step(s, a, rewards);
        const future = tr.terminal ? 0 : maxValue(q[posKey(tr.to)]);
        return tr.reward + gamma * future;
      });
    }
    q = next;
  }
  return q;
}
