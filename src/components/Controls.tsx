import { useEffect } from 'react';
import { SPEEDS } from '../useSimulation';

/** The controls both learning modes provide. */
export interface ControlActions {
  playing: boolean;
  speed: number;
  setSpeed: (n: number) => void;
  play: () => void;
  pause: () => void;
  nextStep: () => void;
  nextAction: () => void;
  nextEpisode: () => void;
  train: (episodes: number) => void;
  restartEpisode: () => void;
  reset: () => void;
  /** DQN only: run until one training update has finished. */
  nextUpdate?: () => void;
}

export function Controls({ ctl, resetLabel = 'Reset', trainCounts = [10] }: { ctl: ControlActions; resetLabel?: string; trainCounts?: number[] }) {
  // Keyboard shortcuts for the instructor (ignored while typing in an input).
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement)?.tagName;
      if (tag === 'INPUT' || tag === 'SELECT' || tag === 'TEXTAREA') return;
      if (e.metaKey || e.ctrlKey || e.altKey) return;
      const k = e.key.toLowerCase();
      if (k === ' ') {
        e.preventDefault();
        if (ctl.playing) ctl.pause();
        else ctl.play();
      } else if (k === 'arrowright' || k === 'n') {
        e.preventDefault();
        ctl.nextStep();
      } else if (k === 'a') ctl.nextAction();
      else if (k === 'e') ctl.nextEpisode();
      else if (k === 'u' && ctl.nextUpdate) ctl.nextUpdate();
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [ctl]);

  return (
    <div className="controls">
      <div className="btn-group">
        <button className={`btn btn-play ${ctl.playing ? 'active' : ''}`} onClick={ctl.play} disabled={ctl.playing} title="Play (Space)">
          ▶ Play
        </button>
        <button className={`btn btn-pause ${!ctl.playing ? 'active' : ''}`} onClick={ctl.pause} disabled={!ctl.playing} title="Pause (Space)">
          ❚❚ {ctl.playing ? 'Pause' : 'Paused'}
        </button>
      </div>
      <div className="btn-group">
        <button className="btn btn-step" onClick={ctl.nextStep} title="Advance exactly one microstep (→ or N)">
          ▸ Next Step
        </button>
        <button className="btn" onClick={ctl.nextAction} title="Finish this action and its Q update (A)">
          ⏩ Next Action
        </button>
        {ctl.nextUpdate && (
          <button className="btn" onClick={ctl.nextUpdate} title="Run until one training update (incl. target sync check) has finished (U)">
            🧠 Next Training Update
          </button>
        )}
        <button className="btn" onClick={ctl.nextEpisode} title="Finish this episode (E)">
          ⏭ Next Episode
        </button>
        {trainCounts.map((n) => (
          <button key={n} className="btn btn-soft" onClick={() => ctl.train(n)} title={`Run ${n} full episodes instantly`}>
            ⚡ Train ×{n}
          </button>
        ))}
      </div>
      <div className="btn-group">
        <button className="btn btn-soft" onClick={ctl.restartEpisode} title="Robot back to start; keep the Q-table">
          ↺ Restart Episode
        </button>
        <button className="btn btn-danger" onClick={ctl.reset} title="Clear everything back to the initial values">
          ⟲ {resetLabel}
        </button>
      </div>
      <label className="speed">
        <span>Speed</span>
        <input type="range" min={0} max={SPEEDS.length - 1} value={ctl.speed} onChange={(e) => ctl.setSpeed(Number(e.target.value))} />
        <b>{SPEEDS[ctl.speed].label}</b>
      </label>
      <div className="kbd-hint">
        <kbd>Space</kbd> play/pause · <kbd>→</kbd> step · <kbd>A</kbd> action · {ctl.nextUpdate && <><kbd>U</kbd> update · </>}<kbd>E</kbd> episode
      </div>
    </div>
  );
}
