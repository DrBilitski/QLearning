import { ReactNode } from 'react';

export type Algorithm = 'q' | 'dqn';

export function TopBar(props: { subtitle: string; modeSelector: ReactNode; badge: string; guided: boolean; playing: boolean; children: ReactNode }) {
  return (
    <header className="topbar">
      <div className="brand">
        <span className="brand-logo">🤖</span>
        <div>
          <h1>Q-LEARNING EXPLORER</h1>
          <div className="subtitle">{props.subtitle}</div>
        </div>
      </div>
      {props.modeSelector}
      <div className="top-stats">{props.children}</div>
      <div className={`mode-badge ${props.guided ? 'mode-guided' : 'mode-live'}`}>
        {props.badge}
        {props.playing && <span className="playing-dot" />}
      </div>
    </header>
  );
}

export function TopStat({ label, value, accent, wide }: { label: string; value: string | number; accent?: boolean; wide?: boolean }) {
  return (
    <div className={`top-stat ${accent ? 'accent' : ''} ${wide ? 'wide' : ''}`}>
      <div className="ts-label">{label}</div>
      <div className="ts-value">{value}</div>
    </div>
  );
}

export function AlgorithmSelector({ value, onChange }: { value: Algorithm; onChange: (a: Algorithm) => void }) {
  return (
    <div className="algo" role="radiogroup" aria-label="Learning algorithm">
      <span className="algo-label">Learning algorithm</span>
      <div className="algo-buttons">
        <button role="radio" aria-checked={value === 'q'} className={value === 'q' ? 'on' : ''} onClick={() => onChange('q')}>
          📋 Q-Learning
        </button>
        <button role="radio" aria-checked={value === 'dqn'} className={value === 'dqn' ? 'on' : ''} onClick={() => onChange('dqn')}>
          🧠 Deep Q-Learning (DQN)
        </button>
      </div>
    </div>
  );
}
