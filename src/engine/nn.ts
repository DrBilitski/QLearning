import { rngNext } from './rng';

/**
 * A tiny, dependency-free fully connected neural network with ReLU hidden
 * layers, a LINEAR output layer (Q-values are not probabilities, so no
 * softmax), backpropagation and the Adam optimizer.
 *
 * Everything is plain arrays and pure functions: a "training step" returns a
 * NEW network instead of mutating the old one. That lets the simulator keep
 * the before/after networks for visualisation and guarantees nothing changes
 * while the app is paused.
 */

export interface Layer {
  nIn: number;
  nOut: number;
  /** Row-major weights: W[o * nIn + i] connects input i to neuron o. */
  W: number[];
  b: number[];
}

export interface Net {
  layers: Layer[];
}

/** Same shape as a Net, used for gradients, Adam moments and weight deltas. */
export type Params = { W: number[]; b: number[] }[];

export const DQN_ARCHITECTURE = [2, 16, 16, 4];

/** He-uniform initialisation from the seeded RNG. Returns the net and the new RNG state. */
export function initNet(sizes: number[], rngState: number): [Net, number] {
  let st = rngState;
  const layers: Layer[] = [];
  for (let l = 0; l < sizes.length - 1; l++) {
    const nIn = sizes[l];
    const nOut = sizes[l + 1];
    const limit = Math.sqrt(6 / nIn);
    const W: number[] = [];
    for (let k = 0; k < nIn * nOut; k++) {
      const [u, next] = rngNext(st);
      st = next;
      W.push((u * 2 - 1) * limit);
    }
    layers.push({ nIn, nOut, W, b: new Array(nOut).fill(0) });
  }
  return [{ layers }, st];
}

export function cloneNet(net: Net): Net {
  return { layers: net.layers.map((L) => ({ nIn: L.nIn, nOut: L.nOut, W: [...L.W], b: [...L.b] })) };
}

export interface ForwardResult {
  /** acts[0] = input, acts[l+1] = output of layer l (after ReLU, except the last). */
  acts: number[][];
  /** pre[l] = weighted sum z of layer l (before activation). */
  pre: number[][];
  output: number[];
}

export function forward(net: Net, x: number[]): ForwardResult {
  const acts: number[][] = [x];
  const pre: number[][] = [];
  const last = net.layers.length - 1;
  net.layers.forEach((L, li) => {
    const inp = acts[li];
    const z = new Array(L.nOut);
    for (let o = 0; o < L.nOut; o++) {
      let sum = L.b[o];
      const row = o * L.nIn;
      for (let i = 0; i < L.nIn; i++) sum += L.W[row + i] * inp[i];
      z[o] = sum;
    }
    pre.push(z);
    acts.push(li === last ? z.slice() : z.map((v: number) => (v > 0 ? v : 0)));
  });
  return { acts, pre, output: acts[acts.length - 1] };
}

export const predict = (net: Net, x: number[]): number[] => forward(net, x).output;

export function zerosLike(net: Net): Params {
  return net.layers.map((L) => ({ W: new Array(L.W.length).fill(0), b: new Array(L.b.length).fill(0) }));
}

/**
 * Backpropagate dL/d(output) through the network and ADD the parameter
 * gradients into `grads`.
 */
export function backwardInto(net: Net, fr: ForwardResult, dOut: number[], grads: Params): void {
  let delta = dOut.slice(); // output layer is linear, so dL/dz = dL/dy
  for (let li = net.layers.length - 1; li >= 0; li--) {
    const L = net.layers[li];
    const inp = fr.acts[li];
    const g = grads[li];
    for (let o = 0; o < L.nOut; o++) {
      const d = delta[o];
      if (d === 0) continue;
      g.b[o] += d;
      const row = o * L.nIn;
      for (let i = 0; i < L.nIn; i++) g.W[row + i] += d * inp[i];
    }
    if (li > 0) {
      const prevPre = fr.pre[li - 1];
      const next = new Array(L.nIn).fill(0);
      for (let i = 0; i < L.nIn; i++) {
        if (prevPre[i] <= 0) continue; // ReLU derivative is 0 here
        let sum = 0;
        for (let o = 0; o < L.nOut; o++) sum += L.W[o * L.nIn + i] * delta[o];
        next[i] = sum;
      }
      delta = next;
    }
  }
}

export interface AdamState {
  t: number;
  m: Params;
  v: Params;
}

export function initAdam(net: Net): AdamState {
  return { t: 0, m: zerosLike(net), v: zerosLike(net) };
}

/** One Adam update. Returns the new network, the new optimizer state and the weight changes. */
export function adamStep(
  net: Net,
  grads: Params,
  adam: AdamState,
  lr: number,
  beta1 = 0.9,
  beta2 = 0.999,
  eps = 1e-8,
): { net: Net; adam: AdamState; delta: Params } {
  const t = adam.t + 1;
  const c1 = 1 - beta1 ** t;
  const c2 = 1 - beta2 ** t;
  const m: Params = [];
  const v: Params = [];
  const delta: Params = [];
  const layers: Layer[] = net.layers.map((L, li) => {
    const out = { W: [] as number[], b: [] as number[] };
    const mm = { W: [] as number[], b: [] as number[] };
    const vv = { W: [] as number[], b: [] as number[] };
    const dd = { W: [] as number[], b: [] as number[] };
    for (const key of ['W', 'b'] as const) {
      const p = L[key];
      const g = grads[li][key];
      const m0 = adam.m[li][key];
      const v0 = adam.v[li][key];
      for (let k = 0; k < p.length; k++) {
        const mk = beta1 * m0[k] + (1 - beta1) * g[k];
        const vk = beta2 * v0[k] + (1 - beta2) * g[k] * g[k];
        const step = (-lr * (mk / c1)) / (Math.sqrt(vk / c2) + eps);
        mm[key].push(mk);
        vv[key].push(vk);
        dd[key].push(step);
        out[key].push(p[k] + step);
      }
    }
    m.push(mm);
    v.push(vv);
    delta.push(dd);
    return { nIn: L.nIn, nOut: L.nOut, W: out.W, b: out.b };
  });
  return { net: { layers }, adam: { t, m, v }, delta };
}

/** Largest absolute difference between two networks' parameters (0 means identical). */
export function maxParamDiff(a: Net, b: Net): number {
  let max = 0;
  a.layers.forEach((L, li) => {
    const M = b.layers[li];
    L.W.forEach((w, k) => (max = Math.max(max, Math.abs(w - M.W[k]))));
    L.b.forEach((w, k) => (max = Math.max(max, Math.abs(w - M.b[k]))));
  });
  return max;
}

export function paramCount(net: Net): number {
  return net.layers.reduce((n, L) => n + L.W.length + L.b.length, 0);
}

/** Per-layer L2 norm of a gradient/delta. */
export function layerNorms(p: Params): number[] {
  return p.map((l) => Math.sqrt([...l.W, ...l.b].reduce((s, x) => s + x * x, 0)));
}
