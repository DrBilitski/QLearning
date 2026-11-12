import { DqnState, dqnReached } from '../../engine/dqn';
import { fmtNum, fmtQ, fmtReward } from '../../engine/format';
import { layerNorms, maxParamDiff, paramCount } from '../../engine/nn';
import { ACTIONS, ACTION_NAMES, fmtPos } from '../../engine/types';

/**
 * Prediction → target → loss → backprop → new prediction, for the inspected
 * sample. Basic view for beginners; detailed view adds the mini-batch table,
 * gradients and weight changes.
 */
export function TrainingPanel({ state: s, focus, detailed, onFocus }: { state: DqnState; focus: number; detailed: boolean; onFocus: (i: number) => void }) {
  const training = !!s.train && dqnReached(s.phase, 'sample') && s.phase !== 'continue';
  if (!training) {
    return (
      <div className="tp tp-wait">
        {s.skipReason ? (
          <p>
            <b>No training this step.</b> {s.skipReason}
          </p>
        ) : (
          <p>
            Training happens after an experience is stored. A random mini-batch is drawn from the replay buffer, and the main network learns from it. This panel will show the prediction, the Bellman target, the loss and the effect of backpropagation.
          </p>
        )}
        <div className="tp-counters">
          <span>
            updates <b>{s.updates}</b>
          </span>
          <span>
            target syncs <b>{s.syncs}</b>
          </span>
        </div>
      </div>
    );
  }

  const t = s.train!;
  const sm = t.samples[focus];
  const after = (p: Parameters<typeof dqnReached>[1]) => dqnReached(s.phase, p);
  const g = s.params.gamma;
  const step = (n: number, done: boolean) => `tp-step ${done ? 'known' : 'pending'} ${n}`;

  return (
    <div className="tp">
      <div className="tp-exp">
        <b>Sample {focus + 1}/{t.samples.length}</b>
        <span className="mono">
          s {fmtPos(sm.exp.s)} · a {ACTION_NAMES[sm.exp.a]} · r {fmtReward(sm.exp.r)} · s′ {fmtPos(sm.exp.sNext)} · {sm.exp.done ? 'done' : 'not done'}
        </span>
        {sm.exp.illustrative && <span className="xp-illus-tag">illustrative numbers</span>}
      </div>

      <div className="tp-ladder">
        <div className={step(1, after('mainPredict'))} data-tone="purple">
          <span className="tp-n">1</span>
          <span className="tp-label">Main network’s prediction Q_main(s, a)</span>
          <span className="tp-val mono">{after('mainPredict') ? fmtNum(sm.pred!) : '?'}</span>
        </div>
        <div className={step(2, after('targetPredict'))} data-tone="yellow">
          <span className="tp-n">2</span>
          <span className="tp-label">
            Target network’s best guess for s′
            {after('targetPredict') && sm.qTarget && (
              <span className="tp-qs">
                {ACTIONS.map((a) => (
                  <span key={a} className={sm.qTarget![a] === sm.maxNext ? 'mx' : ''}>
                    {ACTION_NAMES[a]} {fmtQ(sm.qTarget![a])}
                  </span>
                ))}
              </span>
            )}
            {after('targetPredict') && sm.qMainNext && (
              <span className="tp-ghost">main network would say max {fmtQ(Math.max(...sm.qMainNext))}. Not used!</span>
            )}
          </span>
          <span className="tp-val mono">{after('targetPredict') ? (sm.exp.done ? '0 (terminal)' : fmtQ(sm.maxNext!)) : '?'}</span>
        </div>
        <div className={step(3, after('bellman'))} data-tone="orange">
          <span className="tp-n">3</span>
          <span className="tp-label">
            Bellman target{' '}
            {after('bellman') && (
              <span className="mono tp-eq">
                {sm.exp.done ? `= r = ${fmtNum(sm.exp.r)}` : `= ${fmtNum(sm.exp.r)} + ${fmtNum(g)} × ${fmtNum(sm.maxNext!)}`}
              </span>
            )}
          </span>
          <span className="tp-val mono">{after('bellman') ? fmtNum(sm.target!) : '?'}</span>
        </div>
        <div className={step(4, after('loss'))} data-tone="red">
          <span className="tp-n">4</span>
          <span className="tp-label">
            Loss (target − prediction)²{' '}
            {after('loss') && <span className="mono tp-eq">= ({fmtNum(sm.target!)} − {fmtNum(sm.pred!)})²</span>}
          </span>
          <span className="tp-val mono">{after('loss') ? fmtNum(sm.loss!) : '?'}</span>
        </div>
        <div className={step(5, after('backprop'))} data-tone="purple">
          <span className="tp-n">5</span>
          <span className="tp-label">
            After backprop: new prediction{' '}
            {after('backprop') && (
              <span className="mono tp-eq">
                {fmtNum(sm.pred!)} → {fmtNum(sm.newPred!)} (loss {fmtNum(sm.loss!)} → {fmtNum(sm.newLoss!)})
              </span>
            )}
          </span>
          <span className="tp-val mono">{after('backprop') ? fmtNum(sm.newPred!) : '?'}</span>
        </div>
      </div>

      <div className="tp-batch-line">
        {after('loss') && (
          <span>
            Batch loss (mean of {t.samples.length}): <b className="mono">{fmtNum(t.batchLoss!)}</b>
            {after('backprop') && (
              <>
                {' '}
                → <b className="mono">{fmtNum(t.newBatchLoss!)}</b> after the update
              </>
            )}
          </span>
        )}
        {s.phase === 'backprop' && t.targetBefore && t.mainBefore && (
          <span className="tp-frozen">
            Weight change in this update: main max |Δ| = <b className="mono">{maxParamDiff(t.mainBefore, s.main).toExponential(2)}</b> · target max |Δ| ={' '}
            <b className="mono">{maxParamDiff(t.targetBefore, s.target)}</b> 🔒
          </span>
        )}
      </div>

      {detailed && (
        <div className="tp-detail">
          <table className="tp-table">
            <thead>
              <tr>
                <th>#</th>
                <th>s</th>
                <th>a</th>
                <th>r</th>
                <th>s′</th>
                <th>done</th>
                <th>Q_main(s,a)</th>
                <th>max Q_target(s′)</th>
                <th>target</th>
                <th>loss</th>
                <th>new Q</th>
              </tr>
            </thead>
            <tbody>
              {t.samples.map((x, k) => (
                <tr key={x.exp.id} className={k === focus ? 'focus' : ''} onClick={() => onFocus(k)}>
                  <td>{k + 1}</td>
                  <td>{fmtPos(x.exp.s)}</td>
                  <td>{ACTION_NAMES[x.exp.a]}</td>
                  <td>{fmtReward(x.exp.r)}</td>
                  <td>{fmtPos(x.exp.sNext)}</td>
                  <td>{x.exp.done ? '✓' : ''}</td>
                  <td>{x.pred !== undefined ? fmtNum(x.pred) : ''}</td>
                  <td>{x.maxNext !== undefined && after('targetPredict') ? (x.exp.done ? '0' : fmtNum(x.maxNext)) : ''}</td>
                  <td>{x.target !== undefined ? fmtNum(x.target) : ''}</td>
                  <td>{x.loss !== undefined ? fmtNum(x.loss) : ''}</td>
                  <td>{x.newPred !== undefined ? fmtNum(x.newPred) : ''}</td>
                </tr>
              ))}
            </tbody>
          </table>
          {after('backprop') && t.grads && t.delta && (
            <table className="tp-table tp-grad">
              <thead>
                <tr>
                  <th>layer</th>
                  <th>parameters</th>
                  <th>gradient size ‖∇‖</th>
                  <th>largest weight change</th>
                </tr>
              </thead>
              <tbody>
                {layerNorms(t.grads).map((n, li) => (
                  <tr key={li}>
                    <td>{['input → hidden 1', 'hidden 1 → hidden 2', 'hidden 2 → output'][li]}</td>
                    <td>{t.delta![li].W.length + t.delta![li].b.length}</td>
                    <td>{n.toExponential(2)}</td>
                    <td>{Math.max(...t.delta![li].W.map(Math.abs), ...t.delta![li].b.map(Math.abs)).toExponential(2)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
          <p className="tp-note">
            Loss function: mean squared error, (target − prediction)², averaged over the mini-batch. It is simple to explain, and its gradient, 2 × (prediction − target), points straight at the error. (Many DQN implementations use Huber loss, which limits the pull of very large errors.) The optimizer is Adam with learning rate {fmtNum(s.params.lr)}. All {paramCount(s.main)} main-network parameters can change on each update, and the target network’s parameters never do.
          </p>
        </div>
      )}
    </div>
  );
}
