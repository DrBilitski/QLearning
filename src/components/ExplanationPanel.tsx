import { fmtNum } from '../engine/format';
import { MICROSTEPS, SimState, phaseIndex } from '../engine/simulation';

export function ExplanationPanel({ state }: { state: SimState }) {
  const cur = phaseIndex(state.phase);
  const ep = state.phase === 'start' ? state.lastEpisode : state.phase === 'continue' && state.episodeDone ? state.lastEpisode : null;
  return (
    <div className="explain">
      <ol className="stepper">
        {MICROSTEPS.map((m, i) => {
          const idx = i + 1;
          const status = idx < cur ? 'done' : idx === cur ? 'now' : 'todo';
          return (
            <li key={m.phase} className={`step-${status}`}>
              <span className="step-num">{idx}</span>
              <span className="step-label">{m.label}</span>
            </li>
          );
        })}
      </ol>
      <div className="explain-text" key={state.tick}>
        {state.explanation}
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
