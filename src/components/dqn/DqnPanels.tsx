import { ReactNode } from 'react';
import { DQN_STAGES, DqnParams, DqnState, POSITION_ENCODER, dqnPhaseIndex } from '../../engine/dqn';
import { DQN_SCENARIOS } from '../../engine/dqnScenarios';
import { fmtNum, fmtProb, fmtQ } from '../../engine/format';
import { predict } from '../../engine/nn';
import { SimState } from '../../engine/simulation';
import { ACTIONS, ACTION_NAMES, fmtPos, posKey } from '../../engine/types';
import { DqnController } from '../../useDqn';
import { LineChart } from '../LineChart';
import { RewardChart } from '../StatsPanel';

// ---------------------------------------------------------------- explanation + stage stepper

export function DqnExplanation({ state: s }: { state: DqnState }) {
  const cur = dqnPhaseIndex(s.phase);
  const skipped = !!s.skipReason && s.phase === 'continue';
  const ep = (s.phase === 'start' || (s.phase === 'continue' && s.episodeDone)) && !s.scenario?.illustrative ? s.lastEpisode : null;
  return (
    <div className="explain">
      <ol className="stepper stepper-dqn">
        {DQN_STAGES.map((m, i) => {
          const idx = i + 1;
          let status = idx < cur ? 'done' : idx === cur ? 'now' : 'todo';
          if (m.training && skipped) status = 'skip';
          return (
            <li key={m.phase} className={`step-${status} ${m.training ? 'step-train' : ''}`} title={m.event}>
              <span className="step-num">{idx}</span>
              <span className="step-label">{m.label}</span>
            </li>
          );
        })}
      </ol>
      <div className="explain-text" key={s.tick}>
        {s.explanation}
      </div>
      {ep && (
        <div className={`episode-card ep-${ep.outcome}`}>
          <div className="ep-title">
            {ep.outcome === 'goal' ? '🏆' : ep.outcome === 'trap' ? '💥' : '⏱'} Episode {ep.episode}:{' '}
            {ep.outcome === 'goal' ? 'Reached the goal' : ep.outcome === 'trap' ? 'Fell into the trap' : 'Step limit reached'}
          </div>
          <div className="ep-stats">
            <span>
              Total reward <b>{fmtNum(ep.totalReward)}</b>
            </span>
            <span>
              Moves <b>{ep.steps}</b>
            </span>
            <span>
              ε <b>{fmtNum(ep.epsilonUsed)}</b> → <b>{fmtNum(ep.epsilonAfter)}</b>
            </span>
          </div>
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------- Q-table vs network (scenario A)

export function CompareCard({ dqn, tabular }: { dqn: DqnState; tabular: SimState }) {
  const p = dqn.current;
  const row = tabular.q[posKey(p)];
  const out = predict(dqn.main, POSITION_ENCODER.encode(p));
  const x = POSITION_ENCODER.encode(p);
  return (
    <div className="compare">
      <div className="cmp-col cmp-table">
        <div className="cmp-h">Q-learning: LOOK UP</div>
        <div className="cmp-sub">Q-table row {fmtPos(p)} (from your Q-learning mode)</div>
        <div className="cmp-vals">
          {ACTIONS.map((a) => (
            <span key={a}>
              {ACTION_NAMES[a]} <b className="mono">{row ? fmtQ(row[a]) : '—'}</b>
            </span>
          ))}
        </div>
      </div>
      <div className="cmp-vs">same 4 actions · same game · same rewards</div>
      <div className="cmp-col cmp-net">
        <div className="cmp-h">DQN: COMPUTE</div>
        <div className="cmp-sub">
          main network(X = {fmtNum(x[0])}, Y = {fmtNum(x[1])})
        </div>
        <div className="cmp-vals">
          {ACTIONS.map((a) => (
            <span key={a}>
              {ACTION_NAMES[a]} <b className="mono">{fmtQ(out[a])}</b>
            </span>
          ))}
        </div>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------- statistics & charts

export function DqnStats({ state: s }: { state: DqnState }) {
  const n = s.history.length;
  const recentLoss = s.lossHistory.slice(-50);
  const avgLoss = recentLoss.length ? recentLoss.reduce((a, l) => a + l.loss, 0) / recentLoss.length : null;
  const rate = s.history.map((_, i) => {
    const w = s.history.slice(Math.max(0, i - 19), i + 1);
    return (100 * w.filter((e) => e.outcome === 'goal').length) / w.length;
  });
  const lossSmoothed = s.lossHistory.map((_, i) => {
    const w = s.lossHistory.slice(Math.max(0, i - 19), i + 1);
    return w.reduce((a, l) => a + l.loss, 0) / w.length;
  });
  return (
    <div className="stats">
      <div className="stat-grid dqn-stat-grid">
        <Stat label="Episode" value={s.episode} />
        <Stat label="Step" value={`${s.step} / ${s.params.maxSteps}`} />
        <Stat label="Reward this episode" value={fmtNum(s.episodeReward)} />
        <Stat label="Current ε" value={fmtProb(s.params.epsilon)} />
        <Stat label="Reached goal" value={s.successes} tone="good" />
        <Stat label="Trap / step limit" value={`${s.traps} / ${s.timeouts}`} tone="bad" />
        <Stat label="Replay buffer" value={s.buffer.length} tone="blue" />
        <Stat label="Training updates" value={s.updates} />
        <Stat label="Target syncs" value={s.syncs} tone="yellow" />
        <Stat label="Avg loss (last 50)" value={avgLoss === null ? '—' : fmtNum(avgLoss)} tone="red" />
        <Stat label="Success (last 20)" value={n ? `${Math.round(rate[rate.length - 1])}%` : '—'} />
        <Stat label="Parameters / net" value={2 * 16 + 16 + 16 * 16 + 16 + 16 * 4 + 4} />
      </div>
      <div className="chart-grid">
        <ChartBox title="Episode reward">
          <RewardChart history={s.history} />
        </ChartBox>
        <ChartBox title="Training loss (20-update average)">
          <LineChart series={[{ values: lossSmoothed, className: 'ln-red' }]} xLabel="update" emptyText="Loss appears once training starts." yMin={0} />
        </ChartBox>
        <ChartBox title="Success rate (last 20 episodes)">
          <LineChart series={[{ values: rate, className: 'ln-green' }]} xLabel="episode" emptyText="Appears after the first episode." yMin={0} yMax={100} unit="%" />
        </ChartBox>
        <ChartBox title="Epsilon (exploration)">
          <LineChart series={[{ values: s.history.map((e) => e.epsilonUsed), className: 'ln-purple' }]} xLabel="episode" emptyText="Appears after the first episode." yMin={0} yMax={1} />
        </ChartBox>
      </div>
    </div>
  );
}

function ChartBox({ title, children }: { title: string; children: ReactNode }) {
  return (
    <div className="chart-box">
      <div className="chart-title">{title}</div>
      {children}
    </div>
  );
}

function Stat({ label, value, tone }: { label: string; value: string | number; tone?: 'good' | 'bad' | 'blue' | 'yellow' | 'red' }) {
  return (
    <div className={`stat ${tone ? `stat-${tone}` : ''}`}>
      <div className="stat-value">{value}</div>
      <div className="stat-label">{label}</div>
    </div>
  );
}

// ---------------------------------------------------------------- parameters

function Slider(props: { label: ReactNode; value: number; min: number; max: number; step: number; onChange: (v: number) => void; explain: string; display?: string }) {
  return (
    <div className="param">
      <div className="param-head">
        <span className="param-label">{props.label}</span>
        <b className="param-value mono">{props.display ?? props.value}</b>
      </div>
      <input type="range" min={props.min} max={props.max} step={props.step} value={props.value} onChange={(e) => props.onChange(Number(e.target.value))} />
      <div className="param-explain">{props.explain}</div>
    </div>
  );
}

export function DqnParamsPanel({ state: s, ctl }: { state: DqnState; ctl: DqnController }) {
  const p = s.params;
  const set = (x: Partial<DqnParams>) => ctl.setParams(x);
  return (
    <div className="params params-2col">
      <Slider label={<>Network learning rate</>} value={p.lr} min={0.001} max={0.05} step={0.001} onChange={(v) => set({ lr: v })} explain="Adam step size for the MAIN network’s weights. This is separate from tabular α." />
      <Slider label={<>Discount factor <i>γ</i></>} value={p.gamma} min={0} max={0.99} step={0.01} onChange={(v) => set({ gamma: v })} explain="How much future reward counts in the Bellman target." />
      <Slider label={<>Current epsilon <i>ε</i></>} value={p.epsilon} min={0} max={1} step={0.01} display={fmtProb(p.epsilon)} onChange={(v) => set({ epsilon: v })} explain="Probability of a random action right now." />
      <Slider label={<>Epsilon decay</>} value={p.epsilonDecay} min={0.8} max={1} step={0.005} onChange={(v) => set({ epsilonDecay: v })} explain="ε is multiplied by this at the end of each episode." />
      <Slider label={<>Minimum epsilon</>} value={p.minEpsilon} min={0} max={0.5} step={0.01} onChange={(v) => set({ minEpsilon: v })} explain="Decay stops at this floor." />
      <Slider label={<>Max steps per episode</>} value={p.maxSteps} min={5} max={100} step={1} onChange={(v) => set({ maxSteps: v })} explain="Episode ends if neither goal nor trap is reached." />
      <Slider label={<>Replay buffer capacity</>} value={p.bufferCapacity} min={50} max={5000} step={50} onChange={(v) => set({ bufferCapacity: v })} explain="When full, the oldest experience is dropped." />
      <Slider label={<>Mini-batch size</>} value={p.batchSize} min={1} max={32} step={1} onChange={(v) => set({ batchSize: v })} explain="Experiences sampled per training update." />
      <Slider label={<>Minimum experiences before training</>} value={p.minBuffer} min={1} max={200} step={1} onChange={(v) => set({ minBuffer: v })} explain="Collect this many before the first update." />
      <Slider label={<>Train every N steps</>} value={p.trainEvery} min={1} max={10} step={1} onChange={(v) => set({ trainEvery: v })} explain="How often a training update runs." />
      <Slider label={<>Target update interval</>} value={p.targetInterval} min={1} max={200} step={1} onChange={(v) => set({ targetInterval: v })} explain="Copy main → target after this many training updates." />
      <p className="param-foot">Network: 2 inputs (X, Y) → 16 ReLU → 16 ReLU → 4 linear outputs. Loss: mean squared error. Optimizer: Adam.</p>
    </div>
  );
}

// ---------------------------------------------------------------- scenarios

export function DqnScenariosPanel({ state: s, ctl }: { state: DqnState; ctl: DqnController }) {
  return (
    <div className="scenarios">
      <p className="sc-intro">
        Each demo loads a prepared starting point. Networks come from real training with a fixed seed, and buffers hold real experiences. Only F uses made-up (illustrative) numbers.
      </p>
      <div className="sc-list sc-list-dqn">
        {DQN_SCENARIOS.map((sc) => (
          <button key={sc.id} className={`sc-card ${s.scenario?.id === sc.id ? 'active' : ''}`} onClick={() => ctl.loadScenario(sc)}>
            <span className="sc-letter">{sc.letter}</span>
            <span className="sc-text">
              <b>
                {sc.title}
                {sc.illustrative && <span className="xp-illus-tag">illustrative</span>}
              </b>
              <span>{sc.summary}</span>
            </span>
          </button>
        ))}
      </div>
      <div className="sc-footer">
        <button className="btn btn-soft" onClick={ctl.exitScenario} disabled={!s.scenario}>
          Back to live DQN
        </button>
        <span className="seed">
          Seed <b className="mono">{s.seed}</b>
          <button className="link-btn" onClick={ctl.newSeed}>
            new seed
          </button>
        </span>
      </div>
    </div>
  );
}
