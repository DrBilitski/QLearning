import { emptyQTable, qValueIteration, withUpdatedQ } from './agent';
import { WORLD } from './environment';
import { Params, ScenarioInfo, SimState, createInitialState } from './simulation';
import { Pos, fmtPos } from './types';

/**
 * Guided demonstration scenarios.
 *
 * Each scenario is an ordinary SimState: the same engine and the same Bellman
 * code run it, so every number on screen is computed, not typed in. What a
 * scenario prepares is the starting point: robot position, Q-table snapshot,
 * epsilon, and (optionally) a short list of predetermined random numbers so
 * the demo plays out the same way every time. Once the script runs out, the
 * seeded RNG takes over.
 */

export interface ScenarioDef extends ScenarioInfo {
  /** One-line summary for the scenario picker. */
  summary: string;
  build: (current: SimState) => SimState;
}

function scenarioState(
  info: ScenarioInfo,
  opts: {
    current: SimState;
    pos?: Pos;
    epsilon: number;
    q?: SimState['q'];
    script?: number[];
    seed?: number;
    explanation: string;
  },
): SimState {
  const params: Partial<Params> = {
    alpha: opts.current.params.alpha,
    gamma: opts.current.params.gamma,
    epsilonDecay: opts.current.params.epsilonDecay,
    minEpsilon: opts.current.params.minEpsilon,
    maxSteps: opts.current.params.maxSteps,
    initialEpsilon: opts.epsilon,
    epsilon: opts.epsilon,
  };
  const base = createInitialState({ seed: opts.seed ?? 7, params, rewards: opts.current.rewards });
  const pos = opts.pos ?? WORLD.start;
  return {
    ...base,
    pos,
    current: pos,
    episodeStart: pos,
    q: opts.q ?? base.q,
    script: opts.script ?? [],
    scenario: info,
    explanation: opts.explanation,
    tick: opts.current.tick + 1,
  };
}

export const SCENARIOS: ScenarioDef[] = [
  {
    id: 'A',
    letter: 'A',
    title: 'Initial Exploration',
    summary: 'All Q-values are 0 and ε = 1. Every move is random.',
    description:
      'The robot knows nothing. All Q-values are 0 and ε = 1.0, so every decision is exploration. Random numbers come from a fixed seed, so this replays identically each time.',
    build: (current) =>
      scenarioState(SCENARIOS[0], {
        current,
        epsilon: 1.0,
        seed: 42,
        explanation:
          'SCENARIO A: All Q-values are 0, so there is nothing to exploit yet. Every action looks equally good (or equally unknown). With ε = 1.0 the robot explores on every step. That is the only way it can ever discover the treasure. Step through and watch it wander.',
      }),
  },
  {
    id: 'B',
    letter: 'B',
    title: 'First Reward',
    summary: 'Robot placed next to the goal. Watch +10 arrive.',
    description:
      'The robot is placed at (3,1), directly below the treasure, with an all-zero Q-table. The first two random numbers are predetermined so it explores UP into the goal.',
    build: (current) =>
      scenarioState(SCENARIOS[1], {
        current,
        pos: { x: 3, y: 1 },
        epsilon: 1.0,
        // 0.42 < ε=1 → explore;  0.10 × 4 = 0.4 → slot 0 → UP
        script: [0.42, 0.1],
        explanation: `SCENARIO B: For this demo the robot starts at ${fmtPos({ x: 3, y: 1 })}, directly below the treasure. Every Q-value is still 0. Step through to see the first +10 reward and the update it causes. The goal is terminal, so no future value is added.`,
      }),
  },
  {
    id: 'C',
    letter: 'C',
    title: 'Reward Propagation',
    summary: 'Q((3,1),UP) = 5 already. Watch value flow back one step.',
    description:
      'Snapshot: Q((3,1), UP) = 5.0, as if the robot had reached the goal from (3,1) once before (0 + 0.5×10). The robot starts at (3,2). Random numbers are predetermined for the first two actions.',
    build: (current) =>
      scenarioState(SCENARIOS[2], {
        current,
        pos: { x: 3, y: 2 },
        epsilon: 0.5,
        q: withUpdatedQ(emptyQTable(WORLD), { x: 3, y: 1 }, 0, 5),
        // action 1: 0.20 < 0.5 → explore; 0.05 × 4 = 0.2 → UP   → (3,1), learns from max Q = 5
        // action 2: 0.70 ≥ 0.5 → exploit → UP (Q = 5)           → goal
        script: [0.2, 0.05, 0.7],
        explanation:
          'SCENARIO C: One cell already holds value: Q((3,1), UP) = 5.00, learned earlier when UP from (3,1) reached the goal. The robot starts at (3,2), one step further back. When it moves into (3,1), its update uses the 5.00 in (3,1)’s row. That is how the goal’s value spreads backward, one experience at a time.',
      }),
  },
  {
    id: 'D',
    letter: 'D',
    title: 'Exploration vs. Exploitation',
    summary: 'Partly-trained table, ε = 0.5. One exploit, then one explore.',
    description:
      'Snapshot: a partially trained Q-table (4 sweeps of Bellman backups from zero). The robot starts at (2,2) with ε = 0.5. The first decision exploits; the second explores and picks a worse action.',
    build: (current) =>
      scenarioState(SCENARIOS[3], {
        current,
        pos: { x: 2, y: 2 },
        epsilon: 0.5,
        q: qValueIteration(WORLD, current.rewards, current.params.gamma, 4),
        // action 1: 0.81 ≥ 0.5 → exploit (best action)
        // action 2: 0.22 < 0.5 → explore; 0.61 × 4 = 2.44 → slot 2 → LEFT
        script: [0.81, 0.22, 0.61],
        explanation:
          'SCENARIO D: The Q-values now differ, so the two choices give different results. EXPLOIT always takes the highest Q-value. EXPLORE picks uniformly from all four actions, including bad ones. Step through two decisions and compare.',
      }),
  },
  {
    id: 'E',
    letter: 'E',
    title: 'Trained Agent',
    summary: 'Converged Q-table, ε = 0. The robot follows its policy.',
    description:
      'Snapshot: the Q-table that long training converges to (Bellman backups repeated until values stop changing, using the current rewards and γ). ε = 0, so the robot never explores. Learning is still on.',
    build: (current) =>
      scenarioState(SCENARIOS[4], {
        current,
        epsilon: 0,
        q: qValueIteration(WORLD, current.rewards, current.params.gamma, 200),
        explanation:
          'SCENARIO E: This is a fully trained Q-table. ε = 0 turns off random exploration, so the robot always exploits and follows the arrows to the treasure. Learning is NOT switched off: every step still runs the Bellman update. The values already match their targets, so they barely change. (Two equally short routes exist; ties are broken randomly.)',
      }),
  },
];
