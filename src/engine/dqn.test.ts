import { describe, expect, it } from 'vitest';
import { WORLD } from './environment';
import { DqnState, createDqnState, dqnAdvance, dqnNextAction, dqnNextEpisode, dqnNextUpdate } from './dqn';
import { DQN_SCENARIOS } from './dqnScenarios';
import { DQN_ARCHITECTURE, backwardInto, forward, initNet, maxParamDiff, predict, zerosLike } from './nn';

const run = (s: DqnState, n: number) => {
  for (let i = 0; i < n; i++) s = dqnAdvance(s);
  return s;
};
const scenario = (id: string) => DQN_SCENARIOS.find((s) => s.id === id)!.build(createDqnState());

describe('neural network', () => {
  it('has 2-16-16-4 shape and linear outputs', () => {
    const [net] = initNet(DQN_ARCHITECTURE, 1);
    expect(net.layers.map((l) => [l.nIn, l.nOut])).toEqual([[2, 16], [16, 16], [16, 4]]);
    expect(predict(net, [0.5, 0.5])).toHaveLength(4);
  });

  it('backprop matches a numerical gradient', () => {
    const [net] = initNet(DQN_ARCHITECTURE, 3);
    const x = [2 / 3, 0.5];
    const a = 1;
    const target = 4;
    const lossOf = (n: typeof net) => (predict(n, x)[a] - target) ** 2;
    const fr = forward(net, x);
    const dOut = [0, 0, 0, 0];
    dOut[a] = 2 * (fr.output[a] - target);
    const g = zerosLike(net);
    backwardInto(net, fr, dOut, g);
    const h = 1e-6;
    for (const [li, key, k] of [[0, 'W', 3], [1, 'W', 40], [2, 'W', 17], [2, 'b', 1], [1, 'b', 5]] as const) {
      const plus = structuredClone(net);
      plus.layers[li][key][k] += h;
      const minus = structuredClone(net);
      minus.layers[li][key][k] -= h;
      const num = (lossOf(plus) - lossOf(minus)) / (2 * h);
      expect(g[li][key][k]).toBeCloseTo(num, 4);
    }
  });
});

describe('DQN state machine', () => {
  it('main and target start identical', () => {
    const s = createDqnState();
    expect(maxParamDiff(s.main, s.target)).toBe(0);
  });

  it('skips training until the buffer has enough experiences', () => {
    const s = dqnNextAction(createDqnState());
    expect(s.buffer).toHaveLength(1);
    expect(s.updates).toBe(0);
  });

  it('runs every training stage in order once the buffer is ready', () => {
    let s = createDqnState({ params: { minBuffer: 1 } });
    const phases: string[] = [];
    for (let i = 0; i < 15; i++) {
      s = dqnAdvance(s);
      phases.push(s.phase);
    }
    expect(phases).toEqual(['observe', 'predict', 'decide', 'select', 'execute', 'reward', 'store', 'sample', 'mainPredict', 'targetPredict', 'bellman', 'loss', 'backprop', 'sync', 'continue']);
  });

  it('only the main network changes at backprop; nothing changes before it', () => {
    let s = createDqnState({ params: { minBuffer: 1 } });
    s = run(s, 12); // at 'loss'
    expect(s.phase).toBe('loss');
    const main = s.main;
    const target = s.target;
    s = dqnAdvance(s);
    expect(s.phase).toBe('backprop');
    expect(maxParamDiff(s.main, main)).toBeGreaterThan(0);
    expect(s.target).toBe(target);
  });

  it('copies main → target exactly at the sync interval', () => {
    let s = createDqnState({ params: { minBuffer: 1, targetInterval: 3 } });
    s = dqnNextUpdate(dqnNextUpdate(s));
    expect(s.syncs).toBe(0);
    expect(maxParamDiff(s.main, s.target)).toBeGreaterThan(0);
    s = dqnNextUpdate(s);
    expect(s.syncedNow).toBe(true);
    expect(s.syncs).toBe(1);
    expect(maxParamDiff(s.main, s.target)).toBe(0);
  });

  it('uses zero future value for terminal experiences', () => {
    let s = createDqnState({ params: { minBuffer: 1, batchSize: 64 } });
    for (let i = 0; i < 400 && !s.buffer.some((e) => e.done); i++) s = dqnAdvance(s);
    s = dqnNextUpdate(s);
    for (const sm of s.train!.samples) {
      if (sm.exp.done) expect(sm.target).toBe(sm.exp.r);
      else expect(sm.target).toBeCloseTo(sm.exp.r + 0.9 * Math.max(...sm.qTarget!));
    }
  });

  it('is reproducible for a seed', () => {
    const a = dqnNextEpisode(dqnNextEpisode(createDqnState({ seed: 5 })));
    const b = dqnNextEpisode(dqnNextEpisode(createDqnState({ seed: 5 })));
    expect(a.main).toEqual(b.main);
    expect(a.buffer).toEqual(b.buffer);
  });

  it('learns to reach the goal', () => {
    let s = createDqnState();
    for (let i = 0; i < 200; i++) s = dqnNextEpisode(s);
    expect(s.history.slice(-20).filter((e) => e.outcome === 'goal').length).toBeGreaterThanOrEqual(17);
  });
});

describe('DQN scenarios', () => {
  it('F: worked example gives 3 → target 11 → loss 64, target frozen', () => {
    let s = scenario('F');
    s = run(s, 4); // mainPredict, targetPredict, bellman, loss
    const f = s.train!.samples[0];
    expect(f.pred).toBeCloseTo(3, 9);
    expect(f.qTarget!.map((v) => Math.round(v * 1e9) / 1e9)).toEqual([4, 6, 3, 10]);
    expect(f.target).toBeCloseTo(11, 9);
    expect(f.loss).toBeCloseTo(64, 6);
    const target = s.target;
    s = dqnAdvance(s); // backprop
    expect(s.train!.samples[0].newPred).toBeGreaterThan(3);
    expect(s.target).toBe(target);
    s = run(s, 3);
    expect(s.phase).toBe('continue'); // illustrative example stops here
  });

  it('H: sync makes the networks identical', () => {
    let s = scenario('H');
    expect(maxParamDiff(s.main, s.target)).toBeGreaterThan(0);
    s = dqnNextUpdate(s);
    expect(s.syncedNow).toBe(true);
    expect(maxParamDiff(s.main, s.target)).toBe(0);
  });

  it('B: exploit then explore LEFT', () => {
    let s = run(scenario('B'), 4);
    expect(s.decision).toMatchObject({ explore: false });
    s = dqnNextAction(s);
    s = run(s, 3);
    expect(s.decision).toMatchObject({ explore: true, action: 2 });
  });

  it('I: trained agent reaches the goal with ε = 0', () => {
    const s = dqnNextEpisode(scenario('I'));
    expect(s.lastEpisode!.outcome).toBe('goal');
    expect(s.lastEpisode!.steps).toBeLessThanOrEqual(7);
  });

  it('every scenario builds and advances', () => {
    for (const sc of DQN_SCENARIOS) {
      const s = dqnNextAction(sc.build(createDqnState()));
      expect(s.pos).toBeDefined();
      expect(WORLD.inBounds(s.pos)).toBe(true);
    }
  });
});
