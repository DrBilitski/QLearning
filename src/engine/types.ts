/** Shared primitive types for the grid world and the learning agent. */

export type Action = 0 | 1 | 2 | 3;

export const ACTIONS: readonly Action[] = [0, 1, 2, 3];
export const ACTION_NAMES = ['UP', 'DOWN', 'LEFT', 'RIGHT'] as const;
export const ACTION_ARROWS = ['↑', '↓', '←', '→'] as const;
/** [dx, dy] — y grows downward, so row 0 is the top of the board. */
export const ACTION_DELTAS: readonly [number, number][] = [
  [0, -1],
  [0, 1],
  [-1, 0],
  [1, 0],
];

/** A board position: x = column, y = row. Written (x,y) everywhere in the UI. */
export interface Pos {
  x: number;
  y: number;
}

export const posKey = (p: Pos): string => `${p.x},${p.y}`;
export const fmtPos = (p: Pos): string => `(${p.x},${p.y})`;
export const samePos = (a: Pos | null | undefined, b: Pos | null | undefined): boolean =>
  !!a && !!b && a.x === b.x && a.y === b.y;
