import { bestActions, maxValue } from './agent';
import { DEFAULT_REWARDS, RewardConfig, Transition, WORLD } from './environment';
import { fmtNum, fmtProb, fmtQ, fmtReward } from './format';
import {
  AdamState,
  DQN_ARCHITECTURE,
  Net,
  Params,
  adamStep,
  backwardInto,
  cloneNet,
  forward,
  initAdam,
  initNet,
  predict,
  zerosLike,
} from './nn';
import { Decision, Draw, epsilonDecision, needsActionDraw, selectAction } from './policy';
import { quantise, rngNext } from './rng';
import { EpisodeOutcome, EpisodeRecord } from './simulation';
import { ACTION_NAMES, Action, Pos, fmtPos } from './types';

/**
 * Deep Q-Network (DQN) simulation — a pure state machine like the tabular one.
 *
 * advance(state) performs exactly ONE teaching stage. Networks are immutable
 * values, so a training step produces a new main network and the previous one
 * is still available for before/after comparison. Nothing random or numerical
 * happens outside advance(), so a paused simulation never changes.
 */

// ---------------------------------------------------------------- state encoding

/**
 * Replaceable state encoder: grid position → network inputs.
 * (x / 3, y / 2) — normalised coordinates. A richer game could swap this out.
 */
export interface StateEncoder {
  readonly labels: string[];
  encode(p: Pos): number[];
}

export const POSITION_ENCODER: StateEncoder = {
  labels: ['X', 'Y'],
  encode: (p) => WORLD.encodeState(p),
};

const enc = (p: Pos) => POSITION_ENCODER.encode(p);

// ---------------------------------------------------------------- types

export type DqnPhase =
  | 'start'
  | 'observe'
  | 'predict'
  | 'decide'
  | 'select'
  | 'execute'
  | 'reward'
  | 'store'
  | 'sample'
  | 'mainPredict'
  | 'targetPredict'
  | 'bellman'
  | 'loss'
  | 'backprop'
  | 'sync'
  | 'continue';

export interface DqnStage {
  phase: DqnPhase;
  label: string;
  event: string;
  training?: boolean;
}

export const DQN_STAGES: DqnStage[] = [
  { phase: 'observe', label: 'Observe State', event: 'STATE_OBSERVED' },
  { phase: 'predict', label: 'Main Net Predicts', event: 'Q_VALUES_PREDICTED' },
  { phase: 'decide', label: 'Epsilon Decision', event: 'EPSILON_DECIDED' },
  { phase: 'select', label: 'Select Action', event: 'ACTION_SELECTED' },
  { phase: 'execute', label: 'Execute Action', event: 'ACTION_EXECUTED' },
  { phase: 'reward', label: 'Receive Reward', event: 'REWARD_RECEIVED' },
  { phase: 'store', label: 'Store Experience', event: 'EXPERIENCE_STORED' },
  { phase: 'sample', label: 'Sample Mini-batch', event: 'BATCH_SAMPLED', training: true },
  { phase: 'mainPredict', label: 'Main: Q(s,a)', event: 'MAIN_PREDICTED', training: true },
  { phase: 'targetPredict', label: 'Target: max Q(s′)', event: 'TARGET_VALUES_CALCULATED', training: true },
  { phase: 'bellman', label: 'Bellman Target', event: 'BELLMAN_TARGET_CALCULATED', training: true },
  { phase: 'loss', label: 'Loss', event: 'LOSS_CALCULATED', training: true },
  { phase: 'backprop', label: 'Backpropagation', event: 'BACKPROPAGATION_COMPLETED', training: true },
  { phase: 'sync', label: 'Target Sync?', event: 'TARGET_NETWORK_SYNCHRONIZED', training: true },
  { phase: 'continue', label: 'Continue', event: 'STEP_COMPLETED' },
];

const ORDER: DqnPhase[] = ['start', ...DQN_STAGES.map((s) => s.phase)];
export const dqnPhaseIndex = (p: DqnPhase) => ORDER.indexOf(p);
export const dqnReached = (cur: DqnPhase, p: DqnPhase) => dqnPhaseIndex(cur) >= dqnPhaseIndex(p);

export interface DqnParams {
  gamma: number;
  /** Learning rate of the main network's optimizer (Adam). Not the tabular α. */
  lr: number;
  initialEpsilon: number;
  epsilon: number;
  epsilonDecay: number;
  minEpsilon: number;
  maxSteps: number;
  bufferCapacity: number;
  batchSize: number;
  /** No training until the buffer holds at least this many experiences. */
  minBuffer: number;
  /** Train once every N environment steps. */
  trainEvery: number;
  /** Copy main → target every N training updates. */
  targetInterval: number;
}

export const DEFAULT_DQN_PARAMS: DqnParams = {
  gamma: 0.9,
  lr: 0.01,
  initialEpsilon: 1.0,
  epsilon: 1.0,
  epsilonDecay: 0.97,
  minEpsilon: 0.05,
  maxSteps: 30,
  bufferCapacity: 1000,
  batchSize: 8,
  minBuffer: 32,
  trainEvery: 1,
  targetInterval: 20,
};

export const DQN_DEFAULT_SEED = 2024;

export interface Experience {
  id: number;
  s: Pos;
  a: Action;
  r: number;
  sNext: Pos;
  /** True only when s′ is terminal (goal or trap). A step-limit ending is NOT terminal. */
  done: boolean;
  /** Made-up numbers for the worked example; never used in live training. */
  illustrative?: boolean;
}

export interface TrainSample {
  exp: Experience;
  bufferIndex: number;
  /** Main network output for s (all four actions). */
  qMain?: number[];
  /** Q_main(s, a) — the prediction being trained. */
  pred?: number;
  /** Target network output for s′ (null when terminal). */
  qTarget?: number[] | null;
  /** Main network's opinion of s′ — shown for comparison only, never used. */
  qMainNext?: number[] | null;
  maxNext?: number;
  target?: number;
  loss?: number;
  /** After backprop: the main network's new prediction for (s, a). */
  newPred?: number;
  newLoss?: number;
}

export interface TrainCalc {
  samples: TrainSample[];
  batchLoss?: number;
  newBatchLoss?: number;
  /** Gradient of the batch loss (for the detailed view). */
  grads?: Params;
  /** Actual weight changes applied by Adam. */
  delta?: Params;
  /** The main network before this update (for before/after comparison). */
  mainBefore?: Net;
  /** The target network when the batch was sampled — to prove backprop never touches it. */
  targetBefore?: Net;
}

export interface DqnScenarioInfo {
  id: string;
  letter: string;
  title: string;
  description: string;
  /** Uses made-up numbers (worked example). Stops instead of continuing into live play. */
  illustrative?: boolean;
  /** Show the Q-table vs network comparison card. */
  compareWithTable?: boolean;
  /** Suggested view when loaded. */
  view?: 'game' | 'training';
}

export interface DqnState {
  phase: DqnPhase;
  pos: Pos;
  current: Pos;
  main: Net;
  target: Net;
  adam: AdamState;
  buffer: Experience[];
  nextExpId: number;
  params: DqnParams;
  rewards: RewardConfig;
  seed: number;
  rngState: number;
  script: number[];
  episode: number;
  step: number;
  totalSteps: number;
  episodeReward: number;
  episodeDone: boolean;
  /** Main network output for the current state (stage 2). */
  predicted: number[] | null;
  decision: Decision | null;
  transition: Transition | null;
  timeout: boolean;
  stored: Experience | null;
  train: TrainCalc | null;
  /** Why this step did not train (shown when training stages are skipped). */
  skipReason: string | null;
  updates: number;
  syncs: number;
  sinceSync: number;
  /** True at the 'sync' stage when a copy actually happened. */
  syncedNow: boolean;
  lossHistory: { update: number; loss: number }[];
  history: EpisodeRecord[];
  successes: number;
  traps: number;
  timeouts: number;
  lastEpisode: EpisodeRecord | null;
  scenario: DqnScenarioInfo | null;
  explanation: string;
  tick: number;
}

// ---------------------------------------------------------------- creation

export function createDqnState(opts: { seed?: number; params?: Partial<DqnParams>; rewards?: RewardConfig } = {}): DqnState {
  const seed = opts.seed ?? DQN_DEFAULT_SEED;
  const params = { ...DEFAULT_DQN_PARAMS, ...opts.params };
  if (opts.params?.epsilon === undefined) params.epsilon = params.initialEpsilon;
  const [main, rngState] = initNet(DQN_ARCHITECTURE, seed);
  return {
    phase: 'start',
    pos: WORLD.start,
    current: WORLD.start,
    main,
    target: cloneNet(main), // the target network starts as an exact copy
    adam: initAdam(main),
    buffer: [],
    nextExpId: 1,
    params,
    rewards: { ...(opts.rewards ?? DEFAULT_REWARDS) },
    seed,
    rngState,
    script: [],
    episode: 1,
    step: 0,
    totalSteps: 0,
    episodeReward: 0,
    episodeDone: false,
    predicted: null,
    decision: null,
    transition: null,
    timeout: false,
    stored: null,
    train: null,
    skipReason: null,
    updates: 0,
    syncs: 0,
    sinceSync: 0,
    syncedNow: false,
    lossHistory: [],
    history: [],
    successes: 0,
    traps: 0,
    timeouts: 0,
    lastEpisode: null,
    scenario: null,
    explanation:
      'DQN mode. The Q-table is gone. A neural network (the MAIN network) now predicts the four Q-values from the robot’s position. Its weights start random, so its first predictions are meaningless. A second network, the TARGET network, starts as an exact copy. Press Next Step to begin.',
    tick: 0,
  };
}

// ---------------------------------------------------------------- randomness

/** Decision random numbers: scripted ones first (guided scenarios), then the seeded RNG. */
function drawDecision(s: DqnState): [Draw, Pick<DqnState, 'rngState' | 'script'>] {
  if (s.script.length > 0) return [{ value: s.script[0], scripted: true }, { rngState: s.rngState, script: s.script.slice(1) }];
  const [raw, rngState] = rngNext(s.rngState);
  return [{ value: quantise(raw), scripted: false }, { rngState, script: s.script }];
}

/** Sample k distinct buffer indices with the seeded RNG. */
function sampleIndices(n: number, k: number, rngState: number): [number[], number] {
  const pool = Array.from({ length: n }, (_, i) => i);
  let st = rngState;
  const out: number[] = [];
  for (let j = 0; j < k; j++) {
    const [u, next] = rngNext(st);
    st = next;
    const pick = j + Math.floor(u * (n - j));
    [pool[j], pool[pick]] = [pool[pick], pool[j]];
    out.push(pool[j]);
  }
  return [out, st];
}

// ---------------------------------------------------------------- the state machine

function canTrain(s: DqnState): string | null {
  const p = s.params;
  if (s.buffer.length < p.minBuffer)
    return `No training yet: the replay buffer holds ${s.buffer.length} experience${s.buffer.length === 1 ? '' : 's'}, and training starts once it has ${p.minBuffer}. Until then the robot only collects experiences.`;
  if (s.totalSteps % p.trainEvery !== 0) return `No training this step: training happens once every ${p.trainEvery} steps.`;
  return null;
}

function enterContinue(s: DqnState, base: DqnState, lead: string): DqnState {
  if (!s.episodeDone) {
    return {
      ...base,
      phase: 'continue',
      explanation: `${lead}CONTINUE: The next action starts from ${fmtPos(s.pos)} (step ${s.step} of at most ${s.params.maxSteps}).`,
    };
  }
  const tr = s.transition!;
  const outcome: EpisodeOutcome = tr.outcome === 'goal' ? 'goal' : tr.outcome === 'trap' ? 'trap' : 'timeout';
  const p = s.params;
  const epsilonAfter = p.epsilon <= p.minEpsilon ? p.epsilon : Math.max(p.minEpsilon, p.epsilon * p.epsilonDecay);
  const record: EpisodeRecord = { episode: s.episode, totalReward: s.episodeReward, steps: s.step, outcome, epsilonUsed: p.epsilon, epsilonAfter };
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
      `${lead}EPISODE ${s.episode} OVER: The robot ${result} after ${s.step} moves, with a total reward of ${fmtNum(s.episodeReward)}. ` +
      (epsilonAfter !== p.epsilon ? `Epsilon decays to ${fmtProb(epsilonAfter)}. ` : '') +
      'Both networks and the replay buffer are kept.',
  };
}

const outcomeText = (tr: Transition) =>
  ({ goal: 'reached the treasure', trap: 'fell into the trap', bump: tr.blockedBy === 'wall' ? 'bumped into the wall' : 'bumped into the edge', move: 'made a normal move' })[tr.outcome];

const expText = (e: Experience) => `(${fmtPos(e.s)}, ${ACTION_NAMES[e.a]}, ${fmtReward(e.r)}, ${fmtPos(e.sNext)}, ${e.done ? 'done' : 'not done'})`;

export function dqnAdvance(s: DqnState): DqnState {
  const base: DqnState = { ...s, tick: s.tick + 1 };

  switch (s.phase) {
    case 'start':
    case 'continue': {
      if (s.phase === 'continue' && s.scenario?.illustrative) {
        return {
          ...s,
          explanation:
            'This worked example is finished. Its numbers were illustrative, so it does not continue into live play. Load another scenario, or press “Back to live DQN”.',
        };
      }
      if (s.phase === 'continue' && s.episodeDone) {
        const episode = s.episode + 1;
        return {
          ...base,
          phase: 'start',
          episode,
          pos: WORLD.start,
          current: WORLD.start,
          step: 0,
          episodeReward: 0,
          episodeDone: false,
          predicted: null,
          decision: null,
          transition: null,
          timeout: false,
          stored: null,
          train: null,
          skipReason: null,
          syncedNow: false,
          explanation: `Episode ${episode} begins at ${fmtPos(WORLD.start)}. The networks and the replay buffer carry over from earlier episodes.`,
        };
      }
      const x = enc(s.pos);
      return {
        ...base,
        phase: 'observe',
        current: s.pos,
        predicted: null,
        decision: null,
        transition: null,
        timeout: false,
        stored: null,
        train: null,
        skipReason: null,
        syncedNow: false,
        explanation: `OBSERVE: The robot is at ${fmtPos(s.pos)}. The network cannot see the board. It only receives two numbers: X = ${s.pos.x}/3 = ${fmtNum(x[0])} and Y = ${s.pos.y}/2 = ${fmtNum(x[1])}.`,
      };
    }

    case 'observe': {
      const q = predict(s.main, enc(s.current));
      const best = bestActions(q);
      return {
        ...base,
        phase: 'predict',
        predicted: q,
        explanation: `MAIN NETWORK PREDICTS: The inputs flow through two hidden layers to four output neurons. Each output is a Q-value: UP ${fmtQ(q[0])}, DOWN ${fmtQ(q[1])}, LEFT ${fmtQ(q[2])}, RIGHT ${fmtQ(q[3])}. These are estimates of future reward, NOT probabilities; they do not add up to 1. The highest is ${best.map((a) => ACTION_NAMES[a]).join('/')}.`,
      };
    }

    case 'predict': {
      const [d, rng] = drawDecision(s);
      const { decision, explanation } = epsilonDecision(s.predicted!, s.params.epsilon, d);
      return { ...base, ...rng, phase: 'decide', decision, explanation: explanation + ' This works exactly as in tabular Q-learning, except the Q-values came from the network.' };
    }

    case 'decide': {
      const dec = s.decision!;
      let d: Draw | null = null;
      let rng: Partial<DqnState> = {};
      if (needsActionDraw(dec)) [d, rng] = drawDecision(s);
      const { decision, explanation } = selectAction(dec, d);
      return { ...base, ...rng, phase: 'select', decision, explanation };
    }

    case 'select': {
      const a = s.decision!.action!;
      const tr = WORLD.step(s.current, a, s.rewards);
      const step = s.step + 1;
      const timeout = !tr.terminal && step >= s.params.maxSteps;
      return {
        ...base,
        phase: 'execute',
        pos: tr.to,
        step,
        totalSteps: s.totalSteps + 1,
        transition: tr,
        timeout,
        episodeDone: tr.terminal || timeout,
        explanation:
          tr.outcome === 'bump'
            ? `EXECUTE: The robot tries ${ACTION_NAMES[a]} but is blocked, so it stays at ${fmtPos(tr.to)}.`
            : `EXECUTE: The robot moves ${ACTION_NAMES[a]} from ${fmtPos(tr.from)} to ${fmtPos(tr.to)}.`,
      };
    }

    case 'execute': {
      const tr = s.transition!;
      return {
        ...base,
        phase: 'reward',
        episodeReward: s.episodeReward + tr.reward,
        explanation: `RECEIVE REWARD: The robot ${outcomeText(tr)}. The game gives an actual, observed reward r = ${fmtReward(tr.reward)}, and the new state is s′ = ${fmtPos(tr.to)}. The reward comes from the game rules, not from either network.`,
      };
    }

    case 'reward': {
      const tr = s.transition!;
      const exp: Experience = { id: s.nextExpId, s: tr.from, a: tr.action, r: tr.reward, sNext: tr.to, done: tr.terminal };
      const buffer = [...s.buffer, exp];
      const overflow = buffer.length > s.params.bufferCapacity;
      if (overflow) buffer.splice(0, buffer.length - s.params.bufferCapacity);
      return {
        ...base,
        phase: 'store',
        buffer,
        nextExpId: s.nextExpId + 1,
        stored: exp,
        explanation:
          `STORE EXPERIENCE: The step becomes one experience, (s, a, r, s′, done) = ${expText(exp)}, and goes into the replay buffer (${buffer.length}/${s.params.bufferCapacity}).` +
          (overflow ? ' The buffer was full, so the oldest experience was dropped.' : '') +
          (s.timeout ? ' The step limit ended the episode, but s′ is not a goal or trap, so done = false.' : '') +
          ' The robot does NOT learn from this step directly. It learns from random samples of stored experiences.',
      };
    }

    case 'store': {
      const why = canTrain(s);
      if (why) return enterContinue(s, { ...base, skipReason: why, train: null }, `${why} `);
      const k = Math.min(s.params.batchSize, s.buffer.length);
      const [idx, rngState] = sampleIndices(s.buffer.length, k, s.rngState);
      const samples: TrainSample[] = idx.map((i) => ({ exp: s.buffer[i], bufferIndex: i }));
      return {
        ...base,
        phase: 'sample',
        rngState,
        skipReason: null,
        train: { samples, targetBefore: s.target },
        explanation: `SAMPLE MINI-BATCH: ${k} experiences are picked at random from the ${s.buffer.length} in the buffer (highlighted in blue). Random sampling breaks up runs of similar, back-to-back steps, and it lets old experiences be reused many times. The whole batch trains the network. Click a sampled card to inspect it in detail.`,
      };
    }

    case 'sample': {
      const samples = s.train!.samples.map((sm) => {
        const qMain = predict(s.main, enc(sm.exp.s));
        return { ...sm, qMain, pred: qMain[sm.exp.a] };
      });
      const f = samples[0];
      return {
        ...base,
        phase: 'mainPredict',
        train: { ...s.train!, samples },
        explanation: `MAIN NETWORK — PREDICTION: Each sampled state s goes into the main network. We keep only the output for the action that was actually taken. For ${fmtPos(f.exp.s)}, ${ACTION_NAMES[f.exp.a]} that is Q_main = ${fmtQ(f.pred!)}. The other three outputs get no training signal from this experience. We never invent targets for actions that were not taken.`,
      };
    }

    case 'mainPredict': {
      const samples = s.train!.samples.map((sm) => {
        if (sm.exp.done) return { ...sm, qTarget: null, qMainNext: null, maxNext: 0 };
        const qTarget = predict(s.target, enc(sm.exp.sNext));
        return { ...sm, qTarget, qMainNext: predict(s.main, enc(sm.exp.sNext)), maxNext: maxValue(qTarget) };
      });
      const f = samples[0];
      return {
        ...base,
        phase: 'targetPredict',
        train: { ...s.train!, samples },
        explanation: f.exp.done
          ? `TARGET NETWORK: For sample #1, s′ = ${fmtPos(f.exp.sNext)} is terminal (done = true). There is no future, so the future value is 0 and the target network is not consulted. Non-terminal samples still use it.`
          : `TARGET NETWORK: Each next state s′ goes into the FROZEN target network, which estimates how good s′ is. For ${fmtPos(f.exp.sNext)} its highest output is ${fmtQ(f.maxNext!)}. This is only an estimate, not ground truth. The target network supplies a number. It never sends weights or gradients to the main network.`,
      };
    }

    case 'targetPredict': {
      const g = s.params.gamma;
      const samples = s.train!.samples.map((sm) => ({ ...sm, target: sm.exp.r + (sm.exp.done ? 0 : g * sm.maxNext!) }));
      const f = samples[0];
      return {
        ...base,
        phase: 'bellman',
        train: { ...s.train!, samples },
        explanation: f.exp.done
          ? `BELLMAN TARGET: s′ is terminal, so the target is just the observed reward: ${fmtNum(f.target!)}.`
          : `BELLMAN TARGET: target = r + γ × max Q_target(s′) = ${fmtNum(f.exp.r)} + ${fmtNum(g)} × ${fmtNum(f.maxNext!)} = ${fmtNum(f.target!)}. It mixes a REAL observed reward with an ESTIMATED future value.`,
      };
    }

    case 'bellman': {
      const samples = s.train!.samples.map((sm) => ({ ...sm, loss: (sm.target! - sm.pred!) ** 2 }));
      const batchLoss = samples.reduce((a, sm) => a + sm.loss!, 0) / samples.length;
      const f = samples[0];
      return {
        ...base,
        phase: 'loss',
        train: { ...s.train!, samples, batchLoss },
        explanation: `LOSS: The main network predicted ${fmtNum(f.pred!)}, but the target is ${fmtNum(f.target!)}. Squared error = (target − prediction)² = (${fmtNum(f.target!)} − ${fmtNum(f.pred!)})² = ${fmtNum(f.loss!)}. Averaged over the ${samples.length} samples, the batch loss (MSE) is ${fmtNum(batchLoss)}. Nothing has been trained yet. The next step runs backpropagation.`,
      };
    }

    case 'loss': {
      const t = s.train!;
      const n = t.samples.length;
      const grads = zerosLike(s.main);
      for (const sm of t.samples) {
        const fr = forward(s.main, enc(sm.exp.s));
        const dOut = [0, 0, 0, 0];
        dOut[sm.exp.a] = (2 * (sm.pred! - sm.target!)) / n; // d/dpred of mean squared error
        backwardInto(s.main, fr, dOut, grads);
      }
      const { net: newMain, adam, delta } = adamStep(s.main, grads, s.adam, s.params.lr);
      const samples = t.samples.map((sm) => {
        const newPred = predict(newMain, enc(sm.exp.s))[sm.exp.a];
        return { ...sm, newPred, newLoss: (sm.target! - newPred) ** 2 };
      });
      const newBatchLoss = samples.reduce((a, sm) => a + sm.newLoss!, 0) / n;
      const f = samples[0];
      const updates = s.updates + 1;
      return {
        ...base,
        phase: 'backprop',
        main: newMain,
        adam,
        updates,
        sinceSync: s.sinceSync + 1,
        lossHistory: [...s.lossHistory, { update: updates, loss: t.batchLoss! }].slice(-2000),
        train: { ...t, samples, grads, delta, mainBefore: s.main, newBatchLoss },
        explanation:
          `BACKPROPAGATION: The loss gradient flows backward through the MAIN network, and Adam nudges all of its weights and biases. For sample #1 the prediction moved from ${fmtNum(f.pred!)} to ${fmtNum(f.newPred!)} (target ${fmtNum(f.target!)}). ` +
          `Batch loss: ${fmtNum(t.batchLoss!)} → ${fmtNum(newBatchLoss)}. One update does not always lower the loss. The TARGET network was not touched.`,
      };
    }

    case 'backprop': {
      const due = s.sinceSync >= s.params.targetInterval;
      if (due) {
        return {
          ...base,
          phase: 'sync',
          target: cloneNet(s.main),
          syncs: s.syncs + 1,
          sinceSync: 0,
          syncedNow: true,
          explanation: `TARGET SYNCHRONIZATION: ${s.params.targetInterval} training updates have happened since the last copy. All weights and biases are COPIED from the main network into the target network, so the two networks are now identical. The target then stays frozen again for the next ${s.params.targetInterval} updates. This keeps the training target from shifting every step.`,
        };
      }
      return {
        ...base,
        phase: 'sync',
        syncedNow: false,
        explanation: `TARGET NETWORK STAYS FROZEN: ${s.sinceSync} of ${s.params.targetInterval} updates since the last copy. Keeping the target network fixed for a while gives the main network a steadier target to learn toward.`,
      };
    }

    case 'sync':
      return enterContinue(s, base, '');
  }
}

// ---------------------------------------------------------------- multi-step commands

const LOOP_LIMIT = 200_000;

function loopUntil(s: DqnState, stop: (t: DqnState, seen: Set<DqnPhase>) => boolean): DqnState {
  let t = s;
  const seen = new Set<DqnPhase>();
  for (let i = 0; i < LOOP_LIMIT; i++) {
    const n = dqnAdvance(t);
    if (n === t || n.tick === t.tick) return n; // locked (finished worked example)
    t = n;
    seen.add(t.phase);
    if (stop(t, seen)) break;
  }
  return t;
}

/** Next Action: finish this action (including any training) and stop before the next one is chosen. */
export const dqnNextAction = (s: DqnState) => loopUntil(s, (t, seen) => seen.has('store') && (t.phase === 'observe' || t.phase === 'start'));
/** Next Training Update: run until one full training update (incl. sync check) has finished. */
export const dqnNextUpdate = (s: DqnState) => loopUntil(s, (t) => t.phase === 'sync');
/** Next Episode: finish this episode and pause at the beginning of the next one. */
export const dqnNextEpisode = (s: DqnState) => loopUntil(s, (t) => t.phase === 'start');

export function dqnRestartEpisode(s: DqnState): DqnState {
  return {
    ...s,
    tick: s.tick + 1,
    phase: 'start',
    pos: WORLD.start,
    current: WORLD.start,
    step: 0,
    episodeReward: 0,
    episodeDone: false,
    predicted: null,
    decision: null,
    transition: null,
    timeout: false,
    stored: null,
    train: null,
    skipReason: null,
    syncedNow: false,
    explanation: `Episode ${s.episode} restarted at ${fmtPos(WORLD.start)}. Networks and replay buffer are kept.`,
  };
}

export function dqnReset(s: DqnState, seed: number = s.scenario ? DQN_DEFAULT_SEED : s.seed): DqnState {
  return { ...createDqnState({ seed }), tick: s.tick + 1 };
}
