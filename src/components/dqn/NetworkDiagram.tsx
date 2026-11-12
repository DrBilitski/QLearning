import { useMemo } from 'react';
import { fmtNum, fmtQ } from '../../engine/format';
import { Net, Params, forward } from '../../engine/nn';
import { ACTION_NAMES, Action } from '../../engine/types';

/**
 * A neural network drawn as connected nodes: 2 inputs → 16 → 16 → 4 Q-value
 * outputs. Connection thickness/opacity shows weight magnitude; hidden-node
 * fill shows activation. Optional `delta` highlights the weight changes made
 * by the last backpropagation step.
 */

export type NetTheme = 'main' | 'target';

const THEME = {
  main: { pos: '#7c3aed', neg: '#a5b4fc', node: '#8b5cf6', dark: '#4c1d95', light: '#ede9fe' },
  target: { pos: '#ca8a04', neg: '#cbd5e1', node: '#eab308', dark: '#713f12', light: '#fef9c3' },
};

export const NET_W = 600;
export const NET_H = 330;

interface Props {
  net: Net;
  input: number[] | null;
  inputLabels?: string[];
  inputCaption?: string;
  theme: NetTheme;
  title: string;
  badge: string;
  badgeTone?: 'learning' | 'frozen' | 'syncing';
  /** Output neuron(s) to emphasise (e.g. the action taken). */
  highlight?: Action[];
  /** Output neuron(s) marked as the maximum. */
  maxOut?: Action[];
  dim?: boolean;
  /** Weight changes from the last update (detailed view). */
  delta?: Params | null;
  /** Small text under the outputs, e.g. "only Q(s,RIGHT) is trained". */
  note?: string;
  /** Render as a nested <svg> at this position (inside a larger diagram). */
  at?: { x: number; y: number; w: number; h: number };
  pulseKey?: string | number;
}

export function NetworkDiagram(p: Props) {
  const th = THEME[p.theme];
  const fr = useMemo(() => (p.input ? forward(p.net, p.input) : null), [p.net, p.input]);
  const sizes = [p.net.layers[0].nIn, ...p.net.layers.map((l) => l.nOut)];
  const xs = [64, 215, 340, 440];
  const top = 46;
  const bottom = 28;
  const ys = sizes.map((n) => Array.from({ length: n }, (_, i) => top + ((i + 0.5) * (NET_H - top - bottom)) / n));
  const outputs = fr?.output ?? null;

  // biggest weight change, for scaling the detailed-view highlight
  const maxDelta = useMemo(() => {
    if (!p.delta) return 0;
    return Math.max(1e-12, ...p.delta.flatMap((l) => l.W.map(Math.abs)));
  }, [p.delta]);

  const lines: JSX.Element[] = [];
  p.net.layers.forEach((L, li) => {
    for (let o = 0; o < L.nOut; o++) {
      for (let i = 0; i < L.nIn; i++) {
        const w = L.W[o * L.nIn + i];
        let stroke = w >= 0 ? th.pos : th.neg;
        let opacity = Math.min(0.85, 0.08 + Math.abs(w) * 0.35);
        let width = Math.min(3.2, 0.5 + Math.abs(w) * 1.1);
        if (p.delta) {
          const d = Math.abs(p.delta[li].W[o * L.nIn + i]) / maxDelta;
          stroke = d > 0.02 ? '#dc2626' : '#cbd5e1';
          opacity = d > 0.02 ? 0.25 + 0.7 * d : 0.12;
          width = 0.6 + 2.6 * d;
        }
        lines.push(
          <line key={`${li}-${o}-${i}`} className="nn-edge" x1={xs[li]} y1={ys[li][i]} x2={xs[li + 1]} y2={ys[li + 1][o]} stroke={stroke} strokeOpacity={opacity} strokeWidth={width} />,
        );
      }
    }
  });

  const hiddenMax = fr ? Math.max(1e-9, ...fr.acts.slice(1, -1).flat()) : 1;

  const content = (
    <g className={`nn ${p.dim ? 'nn-dim' : ''}`}>
      <rect x={2} y={2} width={NET_W - 4} height={NET_H - 4} rx={18} fill={th.light} stroke={th.node} strokeWidth={3} />
      <text x={16} y={28} className="nn-title" fill={th.dark}>
        {p.title}
      </text>
      <g transform={`translate(${NET_W - 16}, 12)`}>
        <rect x={-260} y={0} width={260} height={26} rx={13} className={`nn-badge nn-badge-${p.badgeTone ?? 'learning'}`} />
        <text x={-130} y={18} textAnchor="middle" className="nn-badge-text">
          {p.badge}
        </text>
      </g>

      <g key={p.pulseKey} className={p.delta ? 'nn-edges nn-edges-update' : 'nn-edges'}>
        {lines}
      </g>

      {/* layer captions */}
      {['inputs', 'hidden 1 · ReLU', 'hidden 2 · ReLU', 'outputs · linear'].map((c, i) => (
        <text key={c} x={xs[i]} y={NET_H - 8} textAnchor="middle" className="nn-caption">
          {c}
        </text>
      ))}

      {/* input nodes */}
      {ys[0].map((y, i) => (
        <g key={`in${i}`}>
          <circle cx={xs[0]} cy={y} r={27} fill="white" stroke={th.dark} strokeWidth={3} />
          <text x={xs[0]} y={y - 4} textAnchor="middle" className="nn-in-label">
            {p.inputLabels?.[i] ?? `x${i}`}
          </text>
          <text x={xs[0]} y={y + 13} textAnchor="middle" className="nn-in-value">
            {p.input ? fmtNum(p.input[i]) : '?'}
          </text>
        </g>
      ))}
      {p.inputCaption && (
        <text x={xs[0]} y={top - 4} textAnchor="middle" className="nn-in-caption">
          {p.inputCaption}
        </text>
      )}

      {/* hidden nodes */}
      {[1, 2].map((l) =>
        ys[l].map((y, i) => {
          const a = fr ? fr.acts[l][i] / hiddenMax : 0;
          return <circle key={`h${l}-${i}`} cx={xs[l]} cy={y} r={7} fill={th.node} fillOpacity={fr ? 0.12 + 0.88 * a : 0.15} stroke={th.dark} strokeWidth={1.2} />;
        }),
      )}

      {/* output nodes */}
      {ys[3].map((y, a) => {
        const hi = p.highlight?.includes(a as Action);
        const mx = p.maxOut?.includes(a as Action);
        return (
          <g key={`out${a}`} className={hi ? 'nn-out-hi' : ''}>
            <circle cx={xs[3]} cy={y} r={hi ? 21 : 17} fill={hi ? (p.theme === 'main' ? '#c026d3' : '#eab308') : 'white'} stroke={mx ? '#ea580c' : th.dark} strokeWidth={mx ? 5 : 3} />
            <text x={xs[3] + 28} y={y - 4} className="nn-out-name">
              Q({ACTION_NAMES[a]})
            </text>
            <text x={xs[3] + 28} y={y + 16} className={`nn-out-value ${mx ? 'is-max' : ''}`}>
              {outputs ? fmtQ(outputs[a]) : '—'}
              {mx ? ' ★' : ''}
            </text>
          </g>
        );
      })}
      {p.note && (
        <text x={NET_W - 16} y={54} textAnchor="end" className="nn-note">
          {p.note}
        </text>
      )}
    </g>
  );

  if (p.at) {
    return (
      <svg x={p.at.x} y={p.at.y} width={p.at.w} height={p.at.h} viewBox={`0 0 ${NET_W} ${NET_H}`}>
        {content}
      </svg>
    );
  }
  return (
    <svg className="nn-svg" viewBox={`0 0 ${NET_W} ${NET_H}`} role="img" aria-label={p.title}>
      {content}
    </svg>
  );
}
