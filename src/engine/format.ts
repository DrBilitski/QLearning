/** Number formatting shared by the engine's explanations and the UI. */

const clean = (n: number, digits: number): number => {
  const r = Number(n.toFixed(digits));
  return Object.is(r, -0) ? 0 : r;
};

/** Compact number for equations: up to 3 decimals, trailing zeros dropped (4.4, 2.2, -0.1). */
export function fmtNum(n: number): string {
  return String(clean(n, 3));
}

/** Fixed 2-decimal value, as used in the Q-table. */
export function fmtQ(n: number): string {
  return clean(n, 2).toFixed(2);
}

/** Signed reward, e.g. +10, -0.1. */
export function fmtReward(n: number): string {
  const v = clean(n, 3);
  return v > 0 ? `+${v}` : String(v);
}

/** Probability-style value with 3 decimals; adds digits when needed so a < b never looks like equality. */
export function fmtProb(n: number, compareTo?: number): string {
  let digits = 3;
  if (compareTo !== undefined) {
    while (digits < 8 && n !== compareTo && n.toFixed(digits) === compareTo.toFixed(digits)) digits++;
  }
  return n.toFixed(digits);
}
