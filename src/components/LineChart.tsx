/** Minimal SVG line chart for classroom-readable training curves. */
export function LineChart({
  series,
  xLabel,
  emptyText,
  yMin,
  yMax,
  unit = '',
  maxPoints = 400,
}: {
  series: { values: number[]; className: string }[];
  xLabel: string;
  emptyText: string;
  yMin?: number;
  yMax?: number;
  unit?: string;
  maxPoints?: number;
}) {
  const W = 520;
  const H = 170;
  const pad = { l: 48, r: 12, t: 12, b: 28 };
  const n = Math.max(0, ...series.map((s) => s.values.length));
  if (n === 0) return <div className="chart-empty">{emptyText}</div>;

  // downsample long series so the SVG stays light
  const stride = Math.max(1, Math.ceil(n / maxPoints));
  const sampled = series.map((s) => ({ ...s, pts: s.values.map((v, i) => [i, v] as const).filter(([i]) => i % stride === 0 || i === s.values.length - 1) }));
  const all = series.flatMap((s) => s.values);
  const lo = yMin ?? Math.min(...all);
  let hi = yMax ?? Math.max(...all);
  if (hi - lo < 1e-9) hi = lo + 1;
  const sx = (i: number) => pad.l + (n === 1 ? 0.5 : i / (n - 1)) * (W - pad.l - pad.r);
  const sy = (v: number) => pad.t + ((hi - v) / (hi - lo)) * (H - pad.t - pad.b);
  const fmt = (v: number) => (Math.abs(hi - lo) >= 10 ? Math.round(v).toString() : v.toFixed(2)) + unit;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="chart-svg">
      {[lo, (lo + hi) / 2, hi].map((t) => (
        <g key={t}>
          <line x1={pad.l} x2={W - pad.r} y1={sy(t)} y2={sy(t)} className="grid" />
          <text x={pad.l - 6} y={sy(t) + 4} textAnchor="end" className="chart-tick">
            {fmt(t)}
          </text>
        </g>
      ))}
      {sampled.map((s, k) => (
        <path key={k} d={s.pts.map(([i, v], j) => `${j ? 'L' : 'M'}${sx(i)},${sy(v)}`).join(' ')} className={`ln ${s.className}`} />
      ))}
      <text x={pad.l} y={H - 8} className="chart-tick">
        {xLabel} 1
      </text>
      <text x={W - pad.r} y={H - 8} textAnchor="end" className="chart-tick">
        {xLabel} {n}
      </text>
    </svg>
  );
}
