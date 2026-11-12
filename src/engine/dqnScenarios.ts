import {
  DqnParams,
  DqnScenarioInfo,
  DqnState,
  Experience,
  POSITION_ENCODER,
  createDqnState,
  dqnAdvance,
  dqnNextEpisode,
} from './dqn';
import { WORLD } from './environment';
import { Net, cloneNet, predict } from './nn';
import { Pos } from './types';

/**
 * Guided DQN demonstrations. Except for F (the worked example, which is
 * clearly marked illustrative), every scenario is a real DQN state: the
 * networks were produced by actually training on the game with a fixed seed,
 * and the replay buffers contain real experiences. Some scenarios add a few
 * predetermined random numbers so the decisions replay identically.
 */

export interface DqnScenarioDef extends DqnScenarioInfo {
  summary: string;
  detailed?: boolean;
  build: (current: DqnState) => DqnState;
}

const SNAP_SEED = 2024;

function baseParams(current: DqnState, extra: Partial<DqnParams> = {}): Partial<DqnParams> {
  const { gamma, lr, epsilonDecay, minEpsilon, maxSteps, bufferCapacity, batchSize, minBuffer, trainEvery, targetInterval } = current.params;
  return { gamma, lr, epsilonDecay, minEpsilon, maxSteps, bufferCapacity, batchSize, minBuffer, trainEvery, targetInterval, ...extra };
}

/** Real training run with a fixed seed (memoised per reward/parameter setup). */
const memo = new Map<string, DqnState>();
function trainedRun(current: DqnState, episodes: number): DqnState {
  const params = baseParams(current, { epsilon: 1, initialEpsilon: 1 });
  const key = JSON.stringify([current.rewards, params, episodes]);
  const hit = memo.get(key);
  if (hit) return hit;
  let s = createDqnState({ seed: SNAP_SEED, params, rewards: current.rewards });
  for (let i = 0; i < episodes; i++) s = dqnNextEpisode(s);
  memo.set(key, s);
  return s;
}

/** Advance until the robot has just stored an experience mid-episode and training is about to begin. */
function untilReadyToTrain(s: DqnState): DqnState {
  let t = s;
  for (let i = 0; i < 20000; i++) {
    t = dqnAdvance(t);
    if (t.phase === 'store' && !t.episodeDone && t.buffer.length >= t.params.minBuffer) return t;
  }
  return t;
}

function finish(
  info: DqnScenarioInfo,
  current: DqnState,
  s: DqnState,
  explanation: string,
  extra: Partial<DqnState> = {},
): DqnState {
  return { ...s, ...extra, scenario: info, explanation, tick: current.tick + 1, history: [], lossHistory: [], successes: 0, traps: 0, timeouts: 0, lastEpisode: null };
}

/** Fresh state that keeps the learned networks + buffer of `from` but restarts the episode bookkeeping. */
function fromSnapshot(current: DqnState, from: DqnState, seed: number, epsilon: number, pos: Pos = WORLD.start): DqnState {
  const fresh = createDqnState({ seed, params: baseParams(current, { epsilon, initialEpsilon: epsilon }), rewards: current.rewards });
  return {
    ...fresh,
    main: cloneNet(from.main),
    target: cloneNet(from.target),
    adam: from.adam,
    buffer: from.buffer,
    nextExpId: from.nextExpId,
    updates: from.updates,
    syncs: from.syncs,
    sinceSync: from.sinceSync,
    pos,
    current: pos,
  };
}

/** Shift the output-layer biases so the network outputs exactly `values` at state p. */
function pinOutputs(net: Net, p: Pos, values: Partial<Record<number, number>>): Net {
  const out = cloneNet(net);
  const now = predict(out, POSITION_ENCODER.encode(p));
  const last = out.layers[out.layers.length - 1];
  for (const [k, v] of Object.entries(values)) last.b[+k] += (v as number) - now[+k];
  return out;
}

export const DQN_SCENARIOS: DqnScenarioDef[] = [
  {
    id: 'A',
    letter: 'A',
    title: 'DQN vs. Q-Learning',
    summary: 'Same state, two ways to get four Q-values.',
    description: 'Fresh, untrained networks. Compare the Q-table row for (0,2) with the main network’s four outputs for the same state.',
    compareWithTable: true,
    view: 'game',
    build(current) {
      const s = createDqnState({ seed: 11, params: baseParams(current), rewards: current.rewards });
      return finish(this, current, s, 'SCENARIO A: In Q-learning, the four Q-values for a state are LOOKED UP in a table row. In DQN they are COMPUTED: the state (X, Y) goes into the main network, and four output neurons produce the Q-values. The game, rewards, actions and epsilon-greedy choice all stay the same. The comparison card shows both side by side. These weights are still random, so the outputs are meaningless until training.');
    },
  },
  {
    id: 'B',
    letter: 'B',
    title: 'Action Selection',
    summary: 'Trained network, ε = 0.5: one exploit, then one explore.',
    description: 'Networks trained for 120 real episodes (fixed seed). ε = 0.5. The first decisions are predetermined: exploit first, then explore.',
    view: 'game',
    build(current) {
      const snap = trainedRun(current, 120);
      // action 1: 0.81 ≥ 0.5 → exploit.  action 2: 0.22 < 0.5 → explore; 0.61 × 4 = 2.44 → LEFT
      const s = { ...fromSnapshot(current, snap, 21, 0.5), script: [0.81, 0.22, 0.61] };
      return finish(this, current, s, 'SCENARIO B: The main network has been trained, so its four outputs now differ. Epsilon-greedy works exactly as before. A random number ≥ ε means EXPLOIT: take the highest predicted Q-value. A random number < ε means EXPLORE: take any of the four at random. Step through two decisions.');
    },
  },
  {
    id: 'C',
    letter: 'C',
    title: 'Collecting Experiences',
    summary: 'Empty buffer, ε = 1. Watch (s, a, r, s′, done) pile up.',
    description: 'Fresh networks and an empty replay buffer. ε = 1, and training waits until the buffer holds the minimum number of experiences.',
    view: 'game',
    build(current) {
      const s = createDqnState({ seed: 33, params: baseParams(current, { epsilon: 1, initialEpsilon: 1 }), rewards: current.rewards });
      return finish(this, current, s, 'SCENARIO C: Every move produces one experience (s, a, r, s′, done), which is stored in the replay buffer. Training is skipped until the buffer holds enough experiences to sample from. Use Next Action to watch the cards pile up.');
    },
  },
  {
    id: 'D',
    letter: 'D',
    title: 'Training from Replay',
    summary: 'Prepared buffer of real experiences; sample a mini-batch.',
    description: 'The buffer was filled by real random play (fixed seed). The robot has just stored an experience, and the next step samples a mini-batch.',
    view: 'training',
    build(current) {
      const collect = createDqnState({ seed: 44, params: baseParams(current, { epsilon: 1, initialEpsilon: 1 }), rewards: current.rewards });
      const s = untilReadyToTrain(collect);
      return finish(this, current, s, 'SCENARIO D: The replay buffer already holds real experiences from random play. Press Next Step to draw a random mini-batch. Each sampled experience becomes one training example: its state goes to the main network, and its next state goes to the target network.');
    },
  },
  {
    id: 'E',
    letter: 'E',
    title: 'Target Network',
    summary: 'Main and target disagree. The target supplies max Q(s′).',
    description: 'The target network is a copy taken 25 episodes earlier, so it disagrees with the main network. Both came from real training.',
    view: 'training',
    build(current) {
      const early = trainedRun(current, 25);
      const later = trainedRun(current, 50);
      const s0 = { ...fromSnapshot(current, later, 55, 0.3), target: cloneNet(early.main), sinceSync: 0 };
      const s = untilReadyToTrain(s0);
      return finish(this, current, s, 'SCENARIO E: The main and target networks currently give DIFFERENT predictions, because the target is an older copy. Step to the target-network stage. The future value max Q(s′) comes from the TARGET network. The main network’s opinion of s′ is shown greyed out for comparison and is not used.');
    },
  },
  {
    id: 'F',
    letter: 'F',
    title: 'Bellman Target & Loss',
    summary: 'Worked example: 2 + 0.9 × 10 = 11, loss (11 − 3)² = 64.',
    description: 'ILLUSTRATIVE numbers, not from game play: Q_main(A, RIGHT) = 3, r = +2, target network outputs [4, 6, 3, 10] for B. Real networks are pinned to produce exactly these values.',
    illustrative: true,
    view: 'training',
    build(current) {
      const A: Pos = { x: 0, y: 2 };
      const B: Pos = { x: 1, y: 2 };
      const fresh = createDqnState({ seed: 66, params: baseParams(current, { gamma: 0.9, batchSize: 1, epsilon: 1, initialEpsilon: 1 }), rewards: current.rewards });
      const main = pinOutputs(fresh.main, A, { 3: 3 });
      const target = pinOutputs(fresh.target, B, { 0: 4, 1: 6, 2: 3, 3: 10 });
      const exp: Experience = { id: 1, s: A, a: 3, r: 2, sNext: B, done: false, illustrative: true };
      return finish(this, current, fresh, 'SCENARIO F (illustrative numbers): State A = (0,2), action RIGHT, reward +2 (made up for the arithmetic; this game never pays +2), next state B = (1,2). The two networks were adjusted to output exactly the textbook values. Step through: main prediction 3, target max 10, Bellman target 2 + 0.9 × 10 = 11, loss (11 − 3)² = 64, then a REAL backpropagation step.', {
        main,
        target,
        pos: B,
        current: A,
        buffer: [exp],
        nextExpId: 2,
        phase: 'sample',
        train: { samples: [{ exp, bufferIndex: 0 }], targetBefore: target },
        transition: { from: A, action: 3, to: B, reward: 2, outcome: 'move', terminal: false },
      });
    },
  },
  {
    id: 'G',
    letter: 'G',
    title: 'Backpropagation',
    summary: 'Detailed view: gradients and weight changes, main net only.',
    description: 'Networks after 15 real episodes. The detailed view shows gradients and actual weight changes. The target network stays unchanged.',
    view: 'training',
    detailed: true,
    build(current) {
      const snap = trainedRun(current, 15);
      const s = untilReadyToTrain(fromSnapshot(current, snap, 77, 0.5));
      return finish(this, current, s, 'SCENARIO G: Step until just before backpropagation (the Loss stage) and explain the error. Then advance one more step. The main network’s connections light up with their weight changes, and the target network stays exactly the same.');
    },
  },
  {
    id: 'H',
    letter: 'H',
    title: 'Target Synchronization',
    summary: 'One update before a sync. Watch the copy happen.',
    description: 'The target network is an older copy (it differs from main), and the sync counter is one update away from the interval.',
    view: 'training',
    build(current) {
      const early = trainedRun(current, 20);
      const later = trainedRun(current, 40);
      const s0 = { ...fromSnapshot(current, later, 88, 0.3), target: cloneNet(early.main), sinceSync: current.params.targetInterval - 1 };
      const s = untilReadyToTrain(s0);
      return finish(this, current, s, `SCENARIO H: The sync counter is at ${current.params.targetInterval - 1}/${current.params.targetInterval}. The main and target networks differ right now (see the “difference” readout). Step through one training update. At the sync stage every weight is copied, and the difference drops to exactly 0.`);
    },
  },
  {
    id: 'I',
    letter: 'I',
    title: 'Trained Agent',
    summary: 'Network trained for 200 episodes, ε = 0.',
    description: 'Main and target networks after 200 real training episodes (fixed seed). ε = 0, so the robot always exploits. Training still continues.',
    view: 'game',
    build(current) {
      const snap = trainedRun(current, 200);
      const s = { ...fromSnapshot(current, snap, 99, 0), target: cloneNet(snap.main) };
      return finish(this, current, s, 'SCENARIO I: This network learned from 200 episodes of real play. With ε = 0 the robot always picks the highest predicted Q-value. Watch it head for the treasure. Look at the board too: states it rarely visited may still have imperfect values, because DQN generalizes but is not guaranteed to be perfect everywhere.');
    },
  },
];
