import { DqnState, Experience, dqnReached } from '../../engine/dqn';
import { fmtReward } from '../../engine/format';
import { ACTION_ARROWS, ACTION_NAMES, fmtPos } from '../../engine/types';

const MAX_CARDS = 80;

export function ReplayBuffer({ state: s, focus, onFocus }: { state: DqnState; focus: number; onFocus: (i: number) => void }) {
  const training = !!s.train && dqnReached(s.phase, 'sample') && s.phase !== 'continue';
  const sampledIdx = new Map<number, number>(); // exp id → position in batch
  if (training) s.train!.samples.forEach((sm, k) => sampledIdx.set(sm.exp.id, k));
  const justStored = s.stored && dqnReached(s.phase, 'store') ? s.stored.id : null;

  // newest first; always include sampled experiences even if they are old
  const recent = s.buffer.slice(-MAX_CARDS).reverse();
  const extra = training ? s.train!.samples.map((sm) => sm.exp).filter((e) => !recent.some((r) => r.id === e.id)) : [];
  const cards = [...recent, ...extra];
  const hidden = s.buffer.length - cards.length;

  return (
    <div className="rb">
      <div className="rb-head">
        <div className="rb-meter">
          <div className="rb-fill" style={{ width: `${(s.buffer.length / s.params.bufferCapacity) * 100}%` }} />
          <div className="rb-min" style={{ left: `${(s.params.minBuffer / s.params.bufferCapacity) * 100}%` }} title="training starts here" />
        </div>
        <span className="rb-count">
          <b>{s.buffer.length}</b> / {s.params.bufferCapacity}
          {s.buffer.length < s.params.minBuffer && <em> (training starts at {s.params.minBuffer})</em>}
        </span>
      </div>
      <div className="rb-legend">
        <span className="rbl rbl-new">just stored</span>
        <span className="rbl rbl-sampled">sampled for training</span>
        <span className="rbl rbl-focus">inspected (click a sampled card)</span>
      </div>
      <div className="rb-list">
        {cards.length === 0 && <div className="rb-empty">Empty. Each step the robot takes will add one experience card here.</div>}
        {cards.map((e) => {
          const k = sampledIdx.get(e.id);
          const cls = ['xp', e.id === justStored ? 'xp-new' : '', k !== undefined ? 'xp-sampled' : '', k === focus ? 'xp-focus' : '', e.illustrative ? 'xp-illus' : ''].join(' ');
          return (
            <button key={e.id} className={cls} onClick={() => k !== undefined && onFocus(k)} disabled={k === undefined}>
              {k !== undefined && <span className="xp-k">{k + 1}</span>}
              <ExpBody e={e} />
            </button>
          );
        })}
        {hidden > 0 && <div className="rb-more">+ {hidden} older experiences not shown</div>}
      </div>
    </div>
  );
}

export function ExpBody({ e }: { e: Experience }) {
  return (
    <>
      <span className="xp-id">#{e.id}</span>
      <span className="xp-s mono">{fmtPos(e.s)}</span>
      <span className="xp-a" title={ACTION_NAMES[e.a]}>
        {ACTION_ARROWS[e.a]} {ACTION_NAMES[e.a]}
      </span>
      <span className={`xp-r mono ${e.r > 0 ? 'pos' : e.r <= -5 ? 'bad' : 'neg'}`}>{fmtReward(e.r)}</span>
      <span className="xp-sn mono">→ {fmtPos(e.sNext)}</span>
      <span className={`xp-done ${e.done ? 'yes' : ''}`}>{e.done ? 'done' : 'not done'}</span>
      {e.illustrative && <span className="xp-illus-tag">illustrative</span>}
    </>
  );
}
