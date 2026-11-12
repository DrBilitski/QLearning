import { useEffect } from 'react';
import { SPEEDS, SimController } from '../useSimulation';

export function Controls({ ctl }: { ctl: SimController }) {
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
        <button className="btn" onClick={ctl.nextEpisode} title="Finish this episode (E)">
          ⏭ Next Episode
        </button>
        <button className="btn btn-soft" onClick={() => ctl.train(10)} title="Run 10 full episodes instantly">
          ⚡ Train ×10
        </button>
      </div>
      <div className="btn-group">
        <button className="btn btn-soft" onClick={ctl.restartEpisode} title="Robot back to start; keep the Q-table">
          ↺ Restart Episode
        </button>
        <button className="btn btn-danger" onClick={ctl.reset} title="Clear everything back to the initial values">
          ⟲ Reset
        </button>
      </div>
      <label className="speed">
        <span>Speed</span>
        <input type="range" min={0} max={SPEEDS.length - 1} value={ctl.speed} onChange={(e) => ctl.setSpeed(Number(e.target.value))} />
        <b>{SPEEDS[ctl.speed].label}</b>
      </label>
      <div className="kbd-hint">
        <kbd>Space</kbd> play/pause · <kbd>→</kbd> step · <kbd>A</kbd> action · <kbd>E</kbd> episode
      </div>
    </div>
  );
}
