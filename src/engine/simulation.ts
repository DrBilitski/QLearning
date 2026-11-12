import {
  QTableData,
  bellmanUpdate,
  bestActions,
  emptyQTable,
  maxValue,
  withUpdatedQ,
} from './agent';
import { DEFAULT_REWARDS, RewardConfig, Transition, WORLD } from './environment';
import { fmtNum, fmtProb, fmtQ, fmtReward } from './format';
import { Decision, Draw, epsilonDecision, needsActionDraw, selectAction } from './policy';
import { quantise, rngNext } from './rng';
import { ACTION_NAMES, Action, Pos, fmtPos, posKey } from './types';

/**
 * The simulation controller's core: a pure state machine.
 *
 * `advance(state)` performs exactly ONE microstep of Q-learning and returns a
 * new state. Every random number, movement and Q-value change happens inside
 * advance(), and nowhere else. The UI only calls advance() in response to an
 * explicit command (or a Play tick), so a paused simulation cannot change.
 */

export type Phase =
  | 'start'
  | 'observe'
  | 'inspect'
  | 'decide'
  | 'select'
  | 'execute'
  | 'reward'
  | 'inspectNext'
  | 'bellman'
  | 'update'
  | 'continue';

export const MICROSTEPS: { phase: Phase; label: string }[] = [
  { phase: 'observe', label: 'Observe State' },
  { phase: 'inspect', label: 'Inspect Q-values' },
  { phase: 'decide', label: 'Epsilon Decision' },
  { phase: 'select', label: 'Select Action' },
  { phase: 'execute', label: 'Execute Action' },
  { phase: 'reward', label: 'Receive Reward' },
  { phase: 'inspectNext', label: 'Inspect Next State' },
  { phase: 'bellman', label: 'Bellman Calculation' },
  { phase: 'update', label: 'Update Q-Table' },
  { phase: 'continue', label: 'Continue' },
];

const PHASE_ORDER: Phase[] = ['start', ...MICROSTEPS.map((m) => m.phase)];
export const phaseIndex = (p: Phase): number => PHASE_ORDER.indexOf(p);
/** True when the current phase is at or after `p` within this action. */
export const reached = (current: Phase, p: Phase): boolean => phaseIndex(current) >= phaseIndex(p);

export interface Params {
  alpha: number;
  gamma: number;
  /** Epsilon used when the simulation is reset. */
  initialEpsilon: number;
  /** Epsilon in effect right now (decays, or is set by the instructor). */
  epsilon: number;
  epsilonDecay: number;
  minEpsilon: number;
  maxSteps: number;
}

export const DEFAULT_PARAMS: Params = {
  alpha: 0.5,
  gamma: 0.9,
  initialEpsilon: 1.0,
  epsilon: 1.0,
  epsilonDecay: 0.95,
  minEpsilon: 0.05,
  maxSteps: 30,
};

export const DEFAULT_SEED = 1783;

export type { Decision } from './policy';

export interface NextStateInfo {
  terminal: boolean;
  timeout: boolean;
  values: number[] | null;
  maxNext: number;
  best: Action[];
}

export interface BellmanCalc {
  s: Pos;
  a: Action;
  oldQ: number;
  r: number;
  sNext: Pos;
  terminal: boolean;
  maxNext: number;
  alpha: number;
  gamma: number;
  target: number;
  tdError: number;
  newQ: number;
}

export type EpisodeOutcome = 'goal' | 'trap' | 'timeout';

export interface EpisodeRecord {
  episode: number;
  totalReward: number;
  steps: number;
  outcome: EpisodeOutcome;
  epsilonUsed: number;
  epsilonAfter: number;
}

export interface ScenarioInfo {
  id: string;
  letter: string;
  title: string;
  description: string;
}

export interface SimState {
  phase: Phase;
  /** Robot position (moves at the "execute" microstep). */
  pos: Pos;
  /** State s of the action in progress. */
  current: Pos;
  /** Where this episode began (the start tile, unless a scenario placed the robot elsewhere). */
  episodeStart: Pos;
  q: QTableData;
  params: Params;
  rewards: RewardConfig;
  seed: number;
  rngState: number;
  /** Predetermined random numbers (guided scenarios only). Used before the seeded RNG. */
  script: number[];
  episode: number;
  step: number;
  episodeReward: number;
  /** True once the current episode has ended (goal, trap or step limit). */
  episodeDone: boolean;
  decision: Decision | null;
  transition: Transition | null;
  timeout: boolean;
  nextInfo: NextStateInfo | null;
  bellman: BellmanCalc | null;
  lastUpdate: { s: Pos; a: Action; oldQ: number; newQ: number; tick: number } | null;
  lastEpisode: EpisodeRecord | null;
  history: EpisodeRecord[];
  successes: number;
  traps: number;
  timeouts: number;
  scenario: ScenarioInfo | null;
  explanation: string;
  /** Increments on every advance; handy as an animation key. */
  tick: number;
}

export function createInitialState(
  opts: { seed?: number; params?: Partial<Params>; rewards?: RewardConfig } = {},
): SimState {
  const seed = opts.seed ?? DEFAULT_SEED;
  const params = { ...DEFAULT_PARAMS, ...opts.params };
  if (opts.params?.epsilon === undefined) params.epsilon = params.initialEpsilon;
  return {
    phase: 'start',
    pos: WORLD.start,
    current: WORLD.start,
    episodeStart: WORLD.start,
    q: emptyQTable(WORLD),
    params,
    rewards: { ...(opts.rewards ?? DEFAULT_REWARDS) },
    seed,
    rngState: seed,
    script: [],
    episode: 1,
    step: 0,
    episodeReward: 0,
    episodeDone: false,
    decision: null,
    transition: null,
    timeout: false,
    nextInfo: null,
    bellman: null,
    lastUpdate: null,
    lastEpisode: null,
    history: [],
    successes: 0,
    traps: 0,
    timeouts: 0,
    scenario: null,
    explanation: startExplanation(1, WORLD.start, null),
    tick: 0,
  };
}

function startExplanation(episode: number, start: Pos, last: EpisodeRecord | null): string {
  const intro =
    episode === 1 && !last
      ? 'The robot knows nothing yet: every Q-value is 0. It has never been told which moves are good. It will only receive rewards.'
      : `Episode ${episode} begins. The Q-table is kept, so everything learned so far carries over.`;
  return `${intro} The robot is placed at ${fmtPos(start)}. Press Next Step to watch it make its first decision.`;
}

/** Take the next random number: a scripted one if a guided scenario provides it, otherwise the seeded RNG. */
function draw(s: SimState): [Draw, Pick<SimState, 'rngState' | 'script'>] {
  if (s.script.length > 0) {
    return [{ value: s.script[0], scripted: true }, { rngState: s.rngState, script: s.script.slice(1) }];
  }
  const [raw, rngState] = rngNext(s.rngState);
  return [{ value: quantise(raw), scripted: false }, { rngState, script: s.script }];
}

const names = (acts: Action[]) => acts.map((a) => ACTION_NAMES[a]).join(', ');

const outcomeText = (tr: Transition): string => {
  switch (tr.outcome) {
    case 'goal':
      return 'reached the treasure (goal)';
    case 'trap':
      return 'fell into the trap';
    case 'bump':
      return tr.blockedBy === 'wall' ? 'bumped into the wall' : 'bumped into the edge of the board';
    default:
      return 'made a normal move';
  }
};

/** Advance exactly one microstep. Pure: returns a new state, never mutates. */
export function advance(s: SimState): SimState {
  const base = { ...s, tick: s.tick + 1 };

  switch (s.phase) {
    case 'start':
    case 'continue': {
      if (s.phase === 'continue' && s.episodeDone) {
        // Begin the next episode: robot back to the start, Q-table kept.
        const episode = s.episode + 1;
        return {
          ...base,
          phase: 'start',
          episode,
          pos: WORLD.start,
          current: WORLD.start,
          episodeStart: WORLD.start,
          step: 0,
          episodeReward: 0,
          episodeDone: false,
          decision: null,
          transition: null,
          timeout: false,
          nextInfo: null,
          bellman: null,
          explanation: startExplanation(episode, WORLD.start, s.lastEpisode),
        };
      }
      return {
        ...base,
        phase: 'observe',
        current: s.pos,
        decision: null,
        transition: null,
        timeout: false,
        nextInfo: null,
        bellman: null,
        explanation: `OBSERVE: The robot is at ${fmtPos(s.pos)}. This is its current state, s. The matching row of the Q-table is outlined in purple.`,
      };
    }

    case 'observe': {
      const vals = s.q[posKey(s.current)];
      const list = vals.map((v, i) => `${ACTION_NAMES[i]} ${fmtQ(v)}`).join(', ');
      const allZero = vals.every((v) => v === 0);
      return {
        ...base,
        phase: 'inspect',
        explanation:
          `INSPECT: The Q-values for ${fmtPos(s.current)} are ${list}. Each one is the robot's current estimate of the total discounted reward it will collect by taking that action and continuing from there.` +
          (allZero ? ' All of them are 0, so the robot has no preference yet.' : ''),
      };
    }

    case 'inspect': {
      const [d, rng] = draw(s);
      const { decision, explanation } = epsilonDecision(s.q[posKey(s.current)], s.params.epsilon, d);
      return { ...base, ...rng, phase: 'decide', decision, explanation };
    }

    case 'decide': {
      const dec = s.decision!;
      let d: Draw | null = null;
      let rng: Partial<SimState> = {};
      if (needsActionDraw(dec)) [d, rng] = draw(s);
      const { decision, explanation } = selectAction(dec, d);
      return { ...base, ...rng, phase: 'select', decision, explanation };
    }

    case 'select': {
      const action = s.decision!.action!;
      const tr = WORLD.step(s.current, action, s.rewards);
      const step = s.step + 1;
      const timeout = !tr.terminal && step >= s.params.maxSteps;
      let explanation: string;
      if (tr.outcome === 'bump') {
        explanation = `EXECUTE: The robot tries to move ${ACTION_NAMES[action]}, but ${tr.blockedBy === 'wall' ? 'the wall blocks it' : 'that is off the edge of the board'}. It stays at ${fmtPos(tr.to)}.`;
      } else {
        explanation = `EXECUTE: The robot moves ${ACTION_NAMES[action]} from ${fmtPos(tr.from)} to ${fmtPos(tr.to)}.`;
        if (tr.outcome === 'goal') explanation += ' It found the treasure!';
        if (tr.outcome === 'trap') explanation += ' Oh no, that is the trap!';
      }
      return {
        ...base,
        phase: 'execute',
        pos: tr.to,
        step,
        transition: tr,
        timeout,
        episodeDone: tr.terminal || timeout,
        explanation,
      };
    }

    case 'execute': {
      const tr = s.transition!;
      const ruleName: Record<Transition['outcome'], string> = {
        goal: 'reaching the goal',
        trap: 'entering the trap',
        move: 'a normal move',
        bump: 'hitting a wall or edge',
      };
      return {
        ...base,
        phase: 'reward',
        episodeReward: s.episodeReward + tr.reward,
        explanation:
          `RECEIVE REWARD: The robot ${outcomeText(tr)} and receives r = ${fmtReward(tr.reward)}. The new state is s′ = ${fmtPos(tr.to)}. ` +
          `The reward rules (${fmtReward(tr.reward)} for ${ruleName[tr.outcome]}) were set by the developer. Nobody told the robot which action was correct.`,
      };
    }

    case 'reward': {
      const tr = s.transition!;
      let nextInfo: NextStateInfo;
      let explanation: string;
      if (tr.terminal) {
        nextInfo = { terminal: true, timeout: false, values: null, maxNext: 0, best: [] };
        explanation = `INSPECT NEXT STATE: s′ = ${fmtPos(tr.to)} is a terminal state (${tr.outcome === 'goal' ? 'the goal' : 'the trap'}). The episode is over and there are no future rewards to collect, so the future value is 0. We do not look up Q-values for s′.`;
      } else {
        const values = [...s.q[posKey(tr.to)]];
        const maxNext = maxValue(values);
        const best = bestActions(values);
        nextInfo = { terminal: false, timeout: s.timeout, values, maxNext, best };
        explanation =
          `INSPECT NEXT STATE: In s′ = ${fmtPos(tr.to)}, the highest Q-value is ${fmtQ(maxNext)} (${names(best)}${best.length === 4 ? ': all tied' : ''}), outlined in yellow. ` +
          'This is the robot’s best estimate of what it can still earn from s′. Q-learning always uses this maximum, even if the robot later explores instead.' +
          (s.timeout
            ? ` The ${s.params.maxSteps}-step limit ends this episode, but s′ is not a goal or trap, so its future value still counts.`
            : '');
      }
      return { ...base, phase: 'inspectNext', nextInfo, explanation };
    }

    case 'inspectNext': {
      const tr = s.transition!;
      const ni = s.nextInfo!;
      const oldQ = s.q[posKey(s.current)][tr.action];
      const { alpha, gamma } = s.params;
      const res = bellmanUpdate(oldQ, tr.reward, ni.maxNext, alpha, gamma, ni.terminal);
      const bellman: BellmanCalc = {
        s: s.current,
        a: tr.action,
        oldQ,
        r: tr.reward,
        sNext: tr.to,
        terminal: ni.terminal,
        maxNext: ni.maxNext,
        alpha,
        gamma,
        ...res,
      };
      const targetText = ni.terminal
        ? `the target is just the reward, ${fmtNum(tr.reward)}, because s′ is terminal`
        : `the target is r + γ·max = ${fmtNum(tr.reward)} + ${fmtNum(gamma)} × ${fmtNum(ni.maxNext)} = ${fmtNum(res.target)}`;
      return {
        ...base,
        phase: 'bellman',
        bellman,
        explanation: `BELLMAN CALCULATION: The old estimate is ${fmtNum(oldQ)}, and ${targetText}. The gap (target − old) is ${fmtNum(res.tdError)}. With α = ${fmtNum(alpha)}, we move the estimate ${fmtNum(alpha * 100)}% of the way toward the target: New Q = ${fmtNum(res.newQ)}.`,
      };
    }

    case 'bellman': {
      const b = s.bellman!;
      return {
        ...base,
        phase: 'update',
        q: withUpdatedQ(s.q, b.s, b.a, b.newQ),
        lastUpdate: { s: b.s, a: b.a, oldQ: b.oldQ, newQ: b.newQ, tick: s.tick + 1 },
        explanation:
          `UPDATE Q-TABLE: Q(${fmtPos(b.s)}, ${ACTION_NAMES[b.a]}) changes from ${fmtQ(b.oldQ)} to ${fmtQ(b.newQ)}. ` +
          'Only this one cell changes. Earlier states on the robot’s path do not update now. They learn from this value only when the robot visits them again and takes the action that leads here.',
      };
    }

    case 'update': {
      if (!s.episodeDone) {
        return {
          ...base,
          phase: 'continue',
          explanation: `CONTINUE: The episode goes on. The next action starts from ${fmtPos(s.pos)} (step ${s.step} of at most ${s.params.maxSteps}).`,
        };
      }
      const tr = s.transition!;
      const outcome: EpisodeOutcome = tr.outcome === 'goal' ? 'goal' : tr.outcome === 'trap' ? 'trap' : 'timeout';
      const p = s.params;
      // Multiplicative decay, applied once per episode. If the instructor has set
      // epsilon at or below the minimum (e.g. 0 for a trained-policy demo), leave it.
      const epsilonAfter = p.epsilon <= p.minEpsilon ? p.epsilon : Math.max(p.minEpsilon, p.epsilon * p.epsilonDecay);
      const record: EpisodeRecord = {
        episode: s.episode,
        totalReward: s.episodeReward,
        steps: s.step,
        outcome,
        epsilonUsed: p.epsilon,
        epsilonAfter,
      };
      const result = { goal: 'reached the goal', trap: 'fell into the trap', timeout: `ran out of steps (${p.maxSteps})` }[outcome];
      return {
        ...base,
        phase: 'continue',
        params: { ...p, epsilon: epsilonAfter },
        lastEpisode: record,
        history: [...s.history, record],
        successes: s.successes + (outcome === 'goal' ? 1 : 0),
        traps: s.traps + (outcome === 'trap' ? 1 : 0),
        timeouts: s.timeouts + (outcome === 'timeout' ? 1 : 0),
        explanation:
          `EPISODE ${s.episode} OVER: The robot ${result} after ${s.step} moves, with a total reward of ${fmtNum(s.episodeReward)}. ` +
          (epsilonAfter !== p.epsilon
            ? `Epsilon decays: ${fmtProb(p.epsilon)} × ${fmtNum(p.epsilonDecay)} → ${fmtProb(epsilonAfter)}${epsilonAfter === p.minEpsilon ? ' (the minimum)' : ''}. `
            : `Epsilon stays at ${fmtProb(p.epsilon)}. `) +
          'The Q-table is kept. Next, the robot returns to the start.',
      };
    }
  }
}

const MAX_LOOP = 100_000;

/** Next Action: finish the current action (including its Q update) and stop before the next action is chosen. */
export function advanceAction(s: SimState): SimState {
  let t = s;
  let updated = false;
  for (let i = 0; i < MAX_LOOP; i++) {
    t = advance(t);
    if (t.phase === 'update') updated = true;
    if (updated && (t.phase === 'observe' || t.phase === 'start')) break;
  }
  return t;
}

/** Next Episode: finish this episode and pause at the beginning of the next one. */
export function advanceEpisode(s: SimState): SimState {
  let t = s;
  for (let i = 0; i < MAX_LOOP; i++) {
    t = advance(t);
    if (t.phase === 'start') break;
  }
  return t;
}

/** Restart Episode: robot back to the start; Q-table, epsilon and statistics are kept. */
export function restartEpisode(s: SimState): SimState {
  return {
    ...s,
    tick: s.tick + 1,
    phase: 'start',
    pos: WORLD.start,
    current: WORLD.start,
    episodeStart: WORLD.start,
    step: 0,
    episodeReward: 0,
    episodeDone: false,
    decision: null,
    transition: null,
    timeout: false,
    nextInfo: null,
    bellman: null,
    explanation: `Episode ${s.episode} restarted. The robot is back at ${fmtPos(WORLD.start)}. The learned Q-values are kept.`,
  };
}

/**
 * Reset: everything back to the initial values. Live training keeps its seed so
 * the run is reproducible; leaving a guided scenario returns to the default seed.
 */
export function resetSimulation(s: SimState, seed: number = s.scenario ? DEFAULT_SEED : s.seed): SimState {
  return { ...createInitialState({ seed }), tick: s.tick + 1 };
}
