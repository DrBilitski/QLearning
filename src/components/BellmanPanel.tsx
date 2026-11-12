import { ReactNode } from 'react';
import { fmtNum, fmtQ, fmtReward } from '../engine/format';
import { SimState, reached } from '../engine/simulation';
import { ACTION_NAMES, fmtPos, posKey } from '../engine/types';
import { AnimatedNumber } from './ui';

const paren = (n: number) => (n < 0 ? `(${fmtNum(n)})` : fmtNum(n));

function T({ k, children, delay }: { k: string; children: ReactNode; delay?: number }) {
  return (
    <span className={`term t-${k}`} style={delay !== undefined ? { animationDelay: `${delay}ms` } : undefined}>
      {children}
    </span>
  );
}

export function BellmanPanel({ state, animMs, stepMs }: { state: SimState; animMs: number; stepMs: number }) {
  const ph = state.phase;
  const live = ph !== 'start' && ph !== 'continue';
  const hasAction = live && reached(ph, 'select');
  const hasReward = live && reached(ph, 'reward');
  const hasNext = live && reached(ph, 'inspectNext');
  const b = live && reached(ph, 'bellman') ? state.bellman : null;
  const done = ph === 'update' && b;

  const a = state.decision?.action;
  const oldQ = b ? b.oldQ : hasAction && a !== undefined ? state.q[posKey(state.current)][a] : null;
  const tr = state.transition;
  const ni = state.nextInfo;
  const alpha = b ? b.alpha : state.params.alpha;
  const gamma = b ? b.gamma : state.params.gamma;
  // stagger the term-by-term highlight to fit inside one microstep when playing
  const stagger = Math.max(60, Math.min(420, stepMs / 9));

  const terms: { k: string; sym: string; name: string; value: string | null; note?: string }[] = [
    { k: 's', sym: 's', name: 'current state', value: hasAction ? fmtPos(state.current) : null },
    { k: 'a', sym: 'a', name: 'selected action', value: hasAction && a !== undefined ? ACTION_NAMES[a] : null },
    { k: 'q', sym: 'Q(s,a)', name: 'old value', value: oldQ !== null ? fmtQ(oldQ) : null },
    { k: 'r', sym: 'r', name: 'reward', value: hasReward && tr ? fmtReward(tr.reward) : null },
    { k: 'sn', sym: 's′', name: 'next state', value: hasReward && tr ? fmtPos(tr.to) : null },
    {
      k: 'max',
      sym: "max Q(s′,a′)",
      name: 'best next value',
      value: hasNext && ni ? (ni.terminal ? '0' : fmtQ(ni.maxNext)) : null,
      note: hasNext && ni?.terminal ? 'terminal' : undefined,
    },
    { k: 'alpha', sym: 'α', name: 'learning rate', value: fmtNum(alpha) },
    { k: 'gamma', sym: 'γ', name: 'discount', value: fmtNum(gamma) },
  ];

  return (
    <div className="bellman">
      <div className="equation">
        <T k="q">Q(s,a)</T> <span className="op">←</span> <T k="q">Q(s,a)</T> <span className="op">+</span> <T k="alpha">α</T>
        <span className="bracket">[</span>
        <T k="r">r</T> <span className="op">+</span> <T k="gamma">γ</T>
        <span className="op">·</span>
        <T k="max">max<sub>a′</sub> Q(s′,a′)</T> <span className="op">−</span> <T k="q">Q(s,a)</T>
        <span className="bracket">]</span>
      </div>

      <div className="term-grid">
        {terms.map((t) => (
          <div key={t.k} className={`term-card tc-${t.k} ${t.value === null ? 'pending' : 'known'}`}>
            <div className="tc-sym">{t.sym}</div>
            <div className="tc-val mono">{t.value ?? '?'}</div>
            <div className="tc-name">{t.note ? `${t.name} (${t.note})` : t.name}</div>
          </div>
        ))}
      </div>

      <div className="calc-area">
        {b ? (
          <div className="calc" key={state.tick - (ph === 'update' ? 1 : 0)}>
            <div className="calc-line">
              <span className="calc-lhs">New Q =</span>
              <T k="q" delay={0}>{fmtNum(b.oldQ)}</T> <span className="op">+</span>
              <T k="alpha" delay={stagger}>{fmtNum(b.alpha)}</T>
              <span className="bracket">[</span>
              <T k="r" delay={stagger * 2}>{fmtNum(b.r)}</T> <span className="op">+</span>
              {b.terminal ? (
                <T k="max" delay={stagger * 3}>0</T>
              ) : (
                <>
                  <T k="gamma" delay={stagger * 3}>{fmtNum(b.gamma)}</T>
                  <span className="op">×</span>
                  <T k="max" delay={stagger * 4}>{paren(b.maxNext)}</T>
                </>
              )}{' '}
              <span className="op">−</span> <T k="q" delay={stagger * 5}>{paren(b.oldQ)}</T>
              <span className="bracket">]</span>
              {b.terminal && <span className="terminal-note">s′ is terminal → no future term</span>}
            </div>
            <div className="calc-line">
              <span className="calc-lhs">New Q =</span>
              <span className="mono">{fmtNum(b.oldQ)}</span> <span className="op">+</span> <span className="mono">{fmtNum(b.alpha)}</span>
              <span className="op">×</span>
              <span className="mono">{paren(b.tdError)}</span>
              <span className="calc-hint">target {fmtNum(b.target)} − old {paren(b.oldQ)}</span>
            </div>
            <div className="calc-line result-line">
              <span className="calc-lhs">New Q =</span>
              <span className="mono new-q">{fmtNum(b.newQ)}</span>
            </div>
          </div>
        ) : (
          <div className="calc calc-wait">
            {live
              ? 'The cards above fill in as the robot acts. The full calculation appears at the Bellman step.'
              : 'Each action produces one experience (s, a, r, s′). The Bellman update turns that experience into a better Q-value.'}
          </div>
        )}

        <div className={`result-box ${done ? 'result-done' : ''}`}>
          <div className="rb-title">
            {b ? (
              <>
                Q({fmtPos(b.s)}, {ACTION_NAMES[b.a]})
              </>
            ) : (
              'Q(s, a)'
            )}
          </div>
          <div className="rb-row">
            <span>OLD</span>
            <b className="mono">{b ? fmtQ(b.oldQ) : '—'}</b>
          </div>
          <div className="rb-arrow">↓</div>
          <div className="rb-row rb-new">
            <span>NEW</span>
            <b className="mono">{done ? <AnimatedNumber value={b.newQ} initial={b.oldQ} duration={animMs} /> : b ? fmtQ(b.newQ) : '—'}</b>
          </div>
          <div className="rb-status">{done ? 'written to the Q-table' : b ? 'written at the next step' : ''}</div>
        </div>
      </div>
    </div>
  );
}
