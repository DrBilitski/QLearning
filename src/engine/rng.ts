/**
 * Deterministic, seedable random number generator (mulberry32).
 *
 * It is written as a pure function — (state) => [value, nextState] — so the
 * RNG state lives inside the simulation state. Nothing random can happen
 * unless the simulation is explicitly advanced, which is what guarantees that
 * pausing never changes a decision.
 */
export function rngNext(state: number): [number, number] {
  const next = (state + 0x6d2b79f5) | 0;
  let t = next;
  t = Math.imul(t ^ (t >>> 15), t | 1);
  t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
  const value = ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  return [value, next];
}

/**
 * Random numbers shown to students are quantised to 3 decimals so the number
 * on screen is exactly the number used in the comparison.
 */
export function quantise(raw: number): number {
  return Math.floor(raw * 1000) / 1000;
}
