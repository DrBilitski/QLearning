import { ReactNode } from 'react';
import { RewardConfig } from '../engine/environment';
import { fmtProb } from '../engine/format';
import { SCENARIOS } from '../engine/scenarios';
import { Params, SimState } from '../engine/simulation';
import { SimController } from '../useSimulation';

export function RewardsPanel({ rewards, setRewards }: { rewards: RewardConfig; setRewards: (r: Partial<RewardConfig>) => void }) {
  const rows: { k: keyof RewardConfig; icon: string; label: string }[] = [
    { k: 'goal', icon: '🏆', label: 'Reach the goal' },
    { k: 'trap', icon: '💥', label: 'Enter the trap' },
    { k: 'move', icon: '👣', label: 'Normal successful move' },
    { k: 'bump', icon: '🧱', label: 'Hit a wall or board edge' },
  ];
  return (
    <div className="rewards">
      {rows.map((r) => {
        const v = rewards[r.k];
        return (
          <label key={r.k} className={`reward-row ${v > 0 ? 'pos' : v < 0 ? 'neg' : ''}`}>
            <span className="rw-icon">{r.icon}</span>
            <span className="rw-label">{r.label}</span>
            <input
              type="number"
              step={0.1}
              value={v}
              onChange={(e) => {
                const n = parseFloat(e.target.value);
                if (!Number.isNaN(n)) setRewards({ [r.k]: n });
              }}
            />
          </label>
        );
      })}
      <p className="teach-note">
        <b>The developer defines the rewards, not the correct actions.</b> No one tells the robot “go RIGHT here”. It has to discover good actions from these numbers alone.
      </p>
    </div>
  );
}

function Slider(props: {
  label: ReactNode;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (v: number) => void;
  explain: string;
  display?: string;
}) {
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

export function ParamsPanel({ state, ctl }: { state: SimState; ctl: SimController }) {
  const p = state.params;
  const set = (x: Partial<Params>) => ctl.setParams(x);
  return (
    <div className="params">
      <Slider label={<>Learning rate <i>α</i></>} value={p.alpha} min={0} max={1} step={0.05} onChange={(v) => set({ alpha: v })} explain="How strongly one new experience changes an existing Q-value." />
      <Slider label={<>Discount factor <i>γ</i></>} value={p.gamma} min={0} max={1} step={0.05} onChange={(v) => set({ gamma: v })} explain="How much the agent values future rewards compared with immediate ones." />
      <Slider
        label={<>Current epsilon <i>ε</i></>}
        value={p.epsilon}
        min={0}
        max={1}
        step={0.01}
        display={fmtProb(p.epsilon)}
        onChange={(v) => set({ epsilon: v })}
        explain="Probability of choosing a random action right now. Change it at any time. 0 means always exploit, but learning still happens."
      />
      <Slider label={<>Initial epsilon</>} value={p.initialEpsilon} min={0} max={1} step={0.05} onChange={(v) => set({ initialEpsilon: v })} explain="The ε used after Reset." />
      <Slider label={<>Epsilon decay</>} value={p.epsilonDecay} min={0.5} max={1} step={0.01} onChange={(v) => set({ epsilonDecay: v })} explain="At the end of each episode, ε is multiplied by this number." />
      <Slider label={<>Minimum epsilon</>} value={p.minEpsilon} min={0} max={0.5} step={0.01} onChange={(v) => set({ minEpsilon: v })} explain="Decay never pushes ε below this floor." />
      <Slider label={<>Max steps per episode</>} value={p.maxSteps} min={5} max={100} step={1} onChange={(v) => set({ maxSteps: v })} explain="The episode ends if the robot hasn’t reached the goal or trap by then." />
      <p className="param-foot">Reset restores all parameters and rewards to their defaults.</p>
    </div>
  );
}

export function ScenariosPanel({ state, ctl }: { state: SimState; ctl: SimController }) {
  return (
    <div className="scenarios">
      <p className="sc-intro">
        Guided demos load a prepared starting point (robot position, Q-table snapshot, ε, and sometimes predetermined random numbers). The math still runs live in the same engine.
      </p>
      <div className="sc-list">
        {SCENARIOS.map((sc) => (
          <button key={sc.id} className={`sc-card ${state.scenario?.id === sc.id ? 'active' : ''}`} onClick={() => ctl.loadScenario(sc)}>
            <span className="sc-letter">{sc.letter}</span>
            <span className="sc-text">
              <b>{sc.title}</b>
              <span>{sc.summary}</span>
            </span>
          </button>
        ))}
      </div>
      <div className="sc-footer">
        <button className="btn btn-soft" onClick={ctl.exitScenario} disabled={!state.scenario}>
          Back to live training
        </button>
        <span className="seed">
          Seed <b className="mono">{state.seed}</b>
          <button className="link-btn" onClick={ctl.newSeed} title="Reset with a different random seed">
            new seed
          </button>
        </span>
      </div>
    </div>
  );
}
