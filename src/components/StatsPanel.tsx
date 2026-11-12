import { fmtNum, fmtProb } from '../engine/format';
import { EpisodeRecord, SimState } from '../engine/simulation';

export function StatsPanel({ state }: { state: SimState }) {
  const n = state.history.length;
  return (
    <div className="stats">
      <div className="stat-grid">
        <Stat label="Episode" value={state.episode} />
        <Stat label="Step" value={`${state.step} / ${state.params.maxSteps}`} />
        <Stat label="Reward this episode" value={fmtNum(state.episodeReward)} />
        <Stat label="Current ε" value={fmtProb(state.params.epsilon)} />
        <Stat label="Reached goal" value={state.successes} tone="good" />
        <Stat label="Fell in trap" value={state.traps} tone="bad" />
        <Stat label="Ran out of steps" value={state.timeouts} />
        <Stat label="Success rate (last 20)" value={n ? `${Math.round((state.history.slice(-20).filter((e) => e.outcome === 'goal').length / Math.min(20, n)) * 100)}%` : '—'} />
      </div>
      <RewardChart history={state.history} />
    </div>
  );
}

function Stat({ label, value, tone }: { label: string; value: string | number; tone?: 'good' | 'bad' }) {
  return (
    <div className={`stat ${tone ? `stat-${tone}` : ''}`}>
      <div className="stat-value">{value}</div>
      <div className="stat-label">{label}</div>
    </div>
  );
}

export function RewardChart({ history }: { history: EpisodeRecord[] }) {
  const W = 520;
  const H = 190;
  const pad = { l: 44, r: 12, t: 14, b: 30 };
  if (history.length === 0) {
    return <div className="chart-empty">Total reward per episode will be plotted here once the first episode ends.</div>;
  }
  const data = history.slice(-150);
  const ys = data.map((d) => d.totalReward);
  const yMin = Math.min(-12, ...ys);
  const yMax = Math.max(12, ...ys);
  const xMax = Math.max(data.length - 1, 1);
  const sx = (i: number) => pad.l + (i / xMax) * (W - pad.l - pad.r);
  const sy = (v: number) => pad.t + ((yMax - v) / (yMax - yMin)) * (H - pad.t - pad.b);
  const line = data.map((d, i) => `${i ? 'L' : 'M'}${sx(i)},${sy(d.totalReward)}`).join(' ');
  const avg = data.map((_, i) => {
    const w = data.slice(Math.max(0, i - 9), i + 1);
    return w.reduce((a, d) => a + d.totalReward, 0) / w.length;
  });
  const avgLine = avg.map((v, i) => `${i ? 'L' : 'M'}${sx(i)},${sy(v)}`).join(' ');
  const ticks = [yMin, 0, yMax].map((v) => Math.round(v));
  return (
    <div className="chart">
      <svg viewBox={`0 0 ${W} ${H}`} className="chart-svg" role="img" aria-label="Total reward per episode">
        {ticks.map((t) => (
          <g key={t}>
            <line x1={pad.l} x2={W - pad.r} y1={sy(t)} y2={sy(t)} className={t === 0 ? 'grid zero' : 'grid'} />
            <text x={pad.l - 8} y={sy(t) + 5} textAnchor="end" className="chart-tick">
              {t}
            </text>
          </g>
        ))}
        <path d={line} className="chart-line" />
        <path d={avgLine} className="chart-avg" />
        {data.map((d, i) => (
          <circle key={d.episode} cx={sx(i)} cy={sy(d.totalReward)} r={data.length > 60 ? 2.5 : 4.5} className={`dot dot-${d.outcome}`}>
            <title>
              Episode {d.episode}: {fmtNum(d.totalReward)} ({d.outcome}, {d.steps} moves)
            </title>
          </circle>
        ))}
        <text x={pad.l} y={H - 8} className="chart-tick">
          episode {data[0].episode}
        </text>
        <text x={W - pad.r} y={H - 8} textAnchor="end" className="chart-tick">
          episode {data[data.length - 1].episode}
        </text>
      </svg>
      <div className="chart-legend">
        <span className="lg lg-line">reward per episode</span>
        <span className="lg lg-avg">10-episode average</span>
        <span className="lg lg-goal">goal</span>
        <span className="lg lg-trap">trap</span>
        <span className="lg lg-timeout">step limit</span>
      </div>
    </div>
  );
}
