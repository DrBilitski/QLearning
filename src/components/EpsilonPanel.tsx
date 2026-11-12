import { fmtProb, fmtQ } from '../engine/format';
import { ACTIONS, ACTION_ARROWS, ACTION_NAMES } from '../engine/types';
import { EpsilonModel } from '../view';

/** Epsilon-greedy visualisation, shared by tabular Q-learning and DQN. */
export function EpsilonPanel({ model, valueSource = 'Q-value' }: { model: EpsilonModel; valueSource?: string }) {
  const { decision: d, epsilon: eps, values, selected } = model;
  const explorePct = Math.round(eps * 1000) / 10;
  const exploitPct = Math.round((100 - explorePct) * 10) / 10;

  return (
    <div className="eps">
      <div className="eps-top">
        <div className="eps-big">
          <span className="eps-sym">ε</span> = {fmtProb(eps)}
        </div>
        <div className="eps-split">
          <span className="chip chip-explore">{explorePct}% explore</span>
          <span className="chip chip-exploit">{exploitPct}% exploit</span>
        </div>
      </div>

      <div className="prob-bar">
        <div className="prob-explore" style={{ width: `${eps * 100}%` }}>
          {eps >= 0.14 && <span>EXPLORE</span>}
        </div>
        <div className="prob-exploit">{eps <= 0.86 && <span>EXPLOIT</span>}</div>
        {d && (
          <div key={model.animKey} className="prob-marker" style={{ left: `${d.random * 100}%` }}>
            <div className="prob-marker-label">{fmtProb(d.random, d.epsilon)}</div>
          </div>
        )}
      </div>
      <div className="prob-scale">
        <span>0</span>
        <span>ε</span>
        <span>1</span>
      </div>

      {d ? (
        <div className={`decision ${d.explore ? 'decision-explore' : 'decision-exploit'}`}>
          <div className="decision-math">
            <span className="mono">random = {fmtProb(d.random, d.epsilon)}</span>
            {d.randomScripted && <span className="scripted-tag">predetermined</span>}
            <span className="cmp mono">
              {fmtProb(d.random, d.epsilon)} {d.explore ? '<' : '≥'} {fmtProb(d.epsilon, d.random)}
            </span>
          </div>
          <div className="decision-word">{d.explore ? '🎲 EXPLORE' : '🎯 EXPLOIT'}</div>
        </div>
      ) : (
        <div className="decision decision-wait">
          Waiting for the epsilon decision. A random number in [0, 1) will be compared with ε.
        </div>
      )}

      <div className="action-chips">
        {ACTIONS.map((a) => {
          const isBest = d && !d.explore && d.best.includes(a);
          return (
            <div key={a} className={['action-chip', selected === a ? 'chosen' : '', isBest ? 'best' : '', d?.explore && selected === null ? 'candidate' : ''].join(' ')} title={valueSource}>
              <div className="ac-arrow">{ACTION_ARROWS[a]}</div>
              <div className="ac-name">{ACTION_NAMES[a]}</div>
              <div className="ac-q mono">{values ? fmtQ(values[a]) : '—'}</div>
              {d?.explore && <div className="ac-prob">25%</div>}
              {isBest && <div className="ac-star">★ max</div>}
            </div>
          );
        })}
      </div>

      {d && selected !== null && d.pickDetail && (
        <div className="pick-detail">
          <b>Selected: {ACTION_NAMES[selected]}</b>
          <span className="mono">{d.pickDetail}</span>
          {d.actionRandomScripted && <span className="scripted-tag">predetermined</span>}
        </div>
      )}
      <p className="eps-note">
        Exploring picks any of the 4 actions with equal chance, so it can still pick the best one. ε is the exploration probability. It is <b>not</b> the learning rate.
      </p>
    </div>
  );
}
