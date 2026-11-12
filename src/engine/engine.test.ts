import { describe, expect, it } from 'vitest';
import { bellmanUpdate, qValueIteration } from './agent';
import { DEFAULT_REWARDS, WORLD } from './environment';
import { SCENARIOS } from './scenarios';
import {
  SimState,
  advance,
  advanceAction,
  advanceEpisode,
  createInitialState,
  restartEpisode,
} from './simulation';
import { posKey } from './types';

const run = (s: SimState, n: number) => {
  for (let i = 0; i < n; i++) s = advance(s);
  return s;
};

describe('environment', () => {
  it('has 9 learnable states', () => {
    expect(WORLD.states()).toHaveLength(9);
  });
  it('bumps into the wall and edges with the bump penalty', () => {
    const wall = WORLD.step({ x: 2, y: 2 }, 0, DEFAULT_REWARDS);
    expect(wall).toMatchObject({ to: { x: 2, y: 2 }, reward: -1, outcome: 'bump', blockedBy: 'wall' });
    const edge = WORLD.step({ x: 0, y: 2 }, 2, DEFAULT_REWARDS);
    expect(edge).toMatchObject({ to: { x: 0, y: 2 }, reward: -1, blockedBy: 'edge' });
  });
  it('goal and trap are terminal', () => {
    expect(WORLD.step({ x: 3, y: 1 }, 0, DEFAULT_REWARDS)).toMatchObject({ reward: 10, terminal: true });
    expect(WORLD.step({ x: 1, y: 2 }, 0, DEFAULT_REWARDS)).toMatchObject({ reward: -10, terminal: true });
  });
});

describe('Bellman update', () => {
  it('matches the spec examples', () => {
    expect(bellmanUpdate(0, 10, 0, 0.5, 0.9, true).newQ).toBeCloseTo(5);
    expect(bellmanUpdate(0, -0.1, 5, 0.5, 0.9, false).newQ).toBeCloseTo(2.2);
  });
  it('ignores the next state value when terminal', () => {
    expect(bellmanUpdate(0, 10, 999, 0.5, 0.9, true).newQ).toBeCloseTo(5);
  });
});

describe('simulation state machine', () => {
  it('is deterministic for a given seed', () => {
    const a = advanceEpisode(createInitialState({ seed: 3 }));
    const b = advanceEpisode(createInitialState({ seed: 3 }));
    expect(a.q).toEqual(b.q);
    expect(a.lastEpisode).toEqual(b.lastEpisode);
  });

  it('runs the 10 microsteps in order and changes Q only at "update"', () => {
    let s = createInitialState();
    const phases: string[] = [];
    const q0 = s.q;
    for (let i = 0; i < 10; i++) {
      s = advance(s);
      phases.push(s.phase);
      if (s.phase !== 'update' && s.phase !== 'continue') expect(s.q).toBe(q0);
    }
    expect(phases).toEqual([
      'observe', 'inspect', 'decide', 'select', 'execute',
      'reward', 'inspectNext', 'bellman', 'update', 'continue',
    ]);
  });

  it('Next Action stops before the next action is chosen', () => {
    const s = advanceAction(createInitialState());
    expect(['observe', 'start']).toContain(s.phase);
    expect(s.step === 1 || s.episode === 2).toBe(true);
  });

  it('decays epsilon once per episode and keeps the Q-table', () => {
    let s = createInitialState();
    s = advanceEpisode(s);
    expect(s.phase).toBe('start');
    expect(s.episode).toBe(2);
    expect(s.params.epsilon).toBeCloseTo(0.95);
    expect(Object.values(s.q).some((row) => row.some((v) => v !== 0))).toBe(true);
    s = advanceEpisode(s);
    expect(s.params.epsilon).toBeCloseTo(0.9025);
  });

  it('respects the step limit and bootstraps from the (non-terminal) last state', () => {
    let s = createInitialState({ params: { maxSteps: 1 } });
    // force an edge bump: LEFT from start. ε=1 → explore; 0.6×4 → slot 2 = LEFT
    s = { ...s, script: [0.1, 0.6] };
    s = run(s, 7); // through inspectNext
    expect(s.timeout).toBe(true);
    expect(s.nextInfo!.terminal).toBe(false);
    s = run(s, 3); // bellman, update, continue
    expect(s.lastEpisode!.outcome).toBe('timeout');
  });

  it('does not let decay raise a manually zeroed epsilon', () => {
    let s = createInitialState({ params: { epsilon: 0 } });
    s = advanceEpisode(s);
    expect(s.params.epsilon).toBe(0);
  });

  it('restart keeps learning but resets the robot', () => {
    let s = run(createInitialState(), 30);
    const q = s.q;
    s = restartEpisode(s);
    expect(s.pos).toEqual(WORLD.start);
    expect(s.q).toBe(q);
  });

  it('learns to reach the goal with training', () => {
    let s = createInitialState();
    for (let i = 0; i < 300; i++) s = advanceEpisode(s);
    const last20 = s.history.slice(-20);
    expect(last20.filter((e) => e.outcome === 'goal').length).toBeGreaterThan(15);
  });
});

describe('guided scenarios', () => {
  const fresh = createInitialState();

  it('B: first reward gives Q((3,1),UP) = 5', () => {
    const s = run(SCENARIOS[1].build(fresh), 9);
    expect(s.phase).toBe('update');
    expect(s.transition!.outcome).toBe('goal');
    expect(s.q[posKey({ x: 3, y: 1 })][0]).toBeCloseTo(5);
  });

  it('C: propagation gives 2.2, then 7.5', () => {
    let s = run(SCENARIOS[2].build(fresh), 9);
    expect(s.bellman!.newQ).toBeCloseTo(2.2);
    expect(s.q[posKey({ x: 3, y: 2 })][0]).toBeCloseTo(2.2);
    s = run(s, 10);
    expect(s.decision!.explore).toBe(false);
    expect(s.q[posKey({ x: 3, y: 1 })][0]).toBeCloseTo(7.5);
  });

  it('D: exploit RIGHT, then explore LEFT', () => {
    let s = run(SCENARIOS[3].build(fresh), 4);
    expect(s.decision).toMatchObject({ explore: false, action: 3 });
    s = run(s, 10);
    expect(s.decision).toMatchObject({ explore: true, action: 2 });
  });

  it('E: trained agent reaches the goal in 5 moves with ε = 0', () => {
    const s = advanceEpisode(SCENARIOS[4].build(fresh));
    expect(s.lastEpisode).toMatchObject({ outcome: 'goal', steps: 5 });
    const qStar = qValueIteration(WORLD, DEFAULT_REWARDS, 0.9, 200);
    for (const k of Object.keys(qStar)) {
      qStar[k].forEach((v, a) => expect(s.q[k][a]).toBeCloseTo(v, 6));
    }
  });
});
