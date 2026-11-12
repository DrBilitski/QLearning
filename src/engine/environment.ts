import { ACTION_DELTAS, Action, Pos, samePos } from './types';

/**
 * The game environment. It knows the board, the rules of movement and the
 * rewards. It knows nothing about Q-values, so the same environment can later
 * be used by a DQN agent.
 */

export interface RewardConfig {
  goal: number;
  trap: number;
  move: number;
  bump: number;
}

export const DEFAULT_REWARDS: RewardConfig = {
  goal: 10,
  trap: -10,
  move: -0.1,
  bump: -1,
};

export type CellType = 'empty' | 'start' | 'goal' | 'trap' | 'wall';
export type Outcome = 'move' | 'bump' | 'goal' | 'trap';

export interface Transition {
  from: Pos;
  action: Action;
  to: Pos;
  reward: number;
  outcome: Outcome;
  /** True when s' is the goal or the trap. */
  terminal: boolean;
  /** For a bump: was it a wall or the edge of the board? */
  blockedBy?: 'wall' | 'edge';
}

export class GridWorld {
  readonly cols = 4;
  readonly rows = 3;
  readonly start: Pos = { x: 0, y: 2 };
  readonly goal: Pos = { x: 3, y: 0 };
  readonly trap: Pos = { x: 1, y: 1 };
  readonly walls: readonly Pos[] = [{ x: 2, y: 1 }];

  inBounds(p: Pos): boolean {
    return p.x >= 0 && p.x < this.cols && p.y >= 0 && p.y < this.rows;
  }

  isWall(p: Pos): boolean {
    return this.walls.some((w) => samePos(w, p));
  }

  isTerminal(p: Pos): boolean {
    return samePos(p, this.goal) || samePos(p, this.trap);
  }

  cellType(p: Pos): CellType {
    if (samePos(p, this.goal)) return 'goal';
    if (samePos(p, this.trap)) return 'trap';
    if (this.isWall(p)) return 'wall';
    if (samePos(p, this.start)) return 'start';
    return 'empty';
  }

  /** Every cell, row by row from the top. */
  allCells(): Pos[] {
    const out: Pos[] = [];
    for (let y = 0; y < this.rows; y++) for (let x = 0; x < this.cols; x++) out.push({ x, y });
    return out;
  }

  /** States the agent can act from: not a wall, not terminal. These are the Q-table rows. */
  states(): Pos[] {
    return this.allCells().filter((p) => !this.isWall(p) && !this.isTerminal(p));
  }

  /** Deterministic transition function: (s, a) -> (s', r). */
  step(from: Pos, action: Action, rewards: RewardConfig): Transition {
    const [dx, dy] = ACTION_DELTAS[action];
    const target = { x: from.x + dx, y: from.y + dy };
    if (!this.inBounds(target) || this.isWall(target)) {
      return {
        from,
        action,
        to: from,
        reward: rewards.bump,
        outcome: 'bump',
        terminal: false,
        blockedBy: this.inBounds(target) ? 'wall' : 'edge',
      };
    }
    if (samePos(target, this.goal)) {
      return { from, action, to: target, reward: rewards.goal, outcome: 'goal', terminal: true };
    }
    if (samePos(target, this.trap)) {
      return { from, action, to: target, reward: rewards.trap, outcome: 'trap', terminal: true };
    }
    return { from, action, to: target, reward: rewards.move, outcome: 'move', terminal: false };
  }

  /**
   * Numeric encoding of a state, for a future DQN: (x, y) scaled to [0, 1].
   * Not used by tabular Q-learning.
   */
  encodeState(p: Pos): number[] {
    return [p.x / (this.cols - 1), p.y / (this.rows - 1)];
  }
}

export const WORLD = new GridWorld();
