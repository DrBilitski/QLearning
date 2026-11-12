import { DqnState, POSITION_ENCODER, TrainSample, dqnReached } from '../../engine/dqn';
import { fmtNum, fmtQ, fmtReward } from '../../engine/format';
import { maxParamDiff } from '../../engine/nn';
import { ACTION_NAMES, Action, fmtPos } from '../../engine/types';
import { NetworkDiagram } from './NetworkDiagram';

/**
 * The DQN Architecture View: both networks and every value that flows between
 * them during one training update. Arrows light up for the current stage.
 *
 *   replay ─ s ─▶ MAIN ─▶ Q(s,a) ───────────────┐
 *     │                     ▲ backprop          ▼
 *     │      copy every N   │            LOSS (target − Q)²
 *     │            ▼        │                   ▲
 *     ├─ s′ ─▶ TARGET ─▶ max Q(s′) ─▶ BELLMAN ──┘
 *     └─ r ────────────────────────────▲
 */

const W = 1600;
const H = 720;

type Flow =
  | 'buf-s'
  | 's-main'
  | 'main-q'
  | 'buf-sn'
  | 'sn-target'
  | 'target-max'
  | 'buf-r'
  | 'r-bell'
  | 'max-bell'
  | 'bell-loss'
  | 'q-loss'
  | 'backprop'
  | 'sync';

function activeFlows(s: DqnState): Set<Flow> {
  switch (s.phase) {
    case 'mainPredict':
      return new Set(['buf-s', 's-main', 'main-q']);
    case 'targetPredict':
      return new Set(['buf-sn', 'sn-target', 'target-max']);
    case 'bellman':
      return new Set(['buf-r', 'r-bell', 'max-bell']);
    case 'loss':
      return new Set(['bell-loss', 'q-loss']);
    case 'backprop':
      return new Set(['backprop']);
    case 'sync':
      return s.syncedNow ? new Set(['sync']) : new Set();
    default:
      return new Set();
  }
}

export function ArchitectureDiagram({ state: s, focus, detailed }: { state: DqnState; focus: number; detailed: boolean }) {
  const training = !!s.train && dqnReached(s.phase, 'sample') && s.phase !== 'continue';
  const acting = !training && s.phase !== 'start' && s.phase !== 'continue' && !(s.phase === 'sync');
  const sm: TrainSample | null = training ? s.train!.samples[focus] ?? null : null;
  const flows = activeFlows(s);
  const on = (f: Flow) => (flows.has(f) ? 'flow on' : 'flow');
  const after = (p: Parameters<typeof dqnReached>[1]) => training && dqnReached(s.phase, p);

  // ----- main network inputs/highlights
  const mainState = sm ? sm.exp.s : acting ? s.current : null;
  const mainInput = mainState ? POSITION_ENCODER.encode(mainState) : null;
  let mainHighlight: Action[] = [];
  let mainMax: Action[] = [];
  if (sm && after('mainPredict')) mainHighlight = [sm.exp.a];
  else if (acting && s.decision?.action !== undefined && dqnReached(s.phase, 'select')) mainHighlight = [s.decision.action];
  if (acting && s.decision) mainMax = s.decision.best;

  // ----- target network inputs/highlights
  const tState = sm && !sm.exp.done ? sm.exp.sNext : null;
  const targetInput = tState && after('targetPredict') ? POSITION_ENCODER.encode(tState) : null;
  const targetMax: Action[] = sm?.qTarget && after('targetPredict') ? sm.qTarget.map((v, i) => [v, i] as const).filter(([v]) => v === Math.max(...sm.qTarget!)).map(([, i]) => i as Action) : [];
  const terminalFocus = !!sm?.exp.done;

  const showDelta = detailed && s.phase === 'backprop' ? s.train?.delta ?? null : null;
  const diff = maxParamDiff(s.main, s.target);
  const syncing = s.phase === 'sync' && s.syncedNow;

  const g = s.params.gamma;

  return (
    <svg className="arch" viewBox={`0 0 ${W} ${H}`} role="img" aria-label="DQN architecture: main and target networks">
      <defs>
        {(['blue', 'purple', 'yellow', 'green', 'orange', 'red'] as const).map((c) => (
          <marker key={c} id={`ah-${c}`} viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse">
            <path d="M0 0 L10 5 L0 10 z" className={`ah ah-${c}`} />
          </marker>
        ))}
      </defs>

      {/* ---------------- replay buffer */}
      <g className={`box box-blue ${s.phase === 'sample' ? 'box-on' : ''}`}>
        <rect x={16} y={238} width={200} height={244} rx={18} />
        <text x={116} y={268} textAnchor="middle" className="box-title">
          REPLAY BUFFER
        </text>
        <text x={116} y={292} textAnchor="middle" className="box-sub">
          {s.buffer.length} / {s.params.bufferCapacity} experiences
        </text>
        {sm ? (
          <>
            <text x={116} y={322} textAnchor="middle" className="box-sub b">
              sample {focus + 1} of {s.train!.samples.length}
            </text>
            <text x={34} y={352} className="box-line">s = {fmtPos(sm.exp.s)}</text>
            <text x={34} y={378} className="box-line">a = {ACTION_NAMES[sm.exp.a]}</text>
            <text x={34} y={404} className="box-line">r = {fmtReward(sm.exp.r)}</text>
            <text x={34} y={430} className="box-line">s′ = {fmtPos(sm.exp.sNext)}</text>
            <text x={34} y={456} className="box-line">done = {sm.exp.done ? 'true' : 'false'}</text>
          </>
        ) : (
          <text x={116} y={360} textAnchor="middle" className="box-sub">
            {s.stored && dqnReached(s.phase, 'store') ? `stored #${s.stored.id}` : 'collecting…'}
          </text>
        )}
      </g>

      {/* ---------------- state chips */}
      <g className="chip-g chip-purple">
        <rect x={250} y={130} width={110} height={70} rx={14} />
        <text x={305} y={158} textAnchor="middle" className="chip-t">
          s
        </text>
        <text x={305} y={184} textAnchor="middle" className="chip-v">
          {mainState ? fmtPos(mainState) : '—'}
        </text>
      </g>
      <g className="chip-g chip-yellow">
        <rect x={250} y={520} width={110} height={70} rx={14} />
        <text x={305} y={548} textAnchor="middle" className="chip-t">
          s′
        </text>
        <text x={305} y={574} textAnchor="middle" className="chip-v">
          {sm ? fmtPos(sm.exp.sNext) : '—'}
        </text>
      </g>

      {/* ---------------- networks */}
      <NetworkDiagram
        at={{ x: 385, y: 6, w: 545, h: 300 }}
        net={s.main}
        input={mainInput}
        inputLabels={POSITION_ENCODER.labels}
        theme="main"
        title="MAIN NETWORK"
        badge="LEARNING · trained by backprop"
        badgeTone="learning"
        highlight={mainHighlight}
        maxOut={mainMax}
        delta={showDelta}
        pulseKey={s.updates}
        note={sm && after('mainPredict') ? `only Q(${ACTION_NAMES[sm.exp.a]}) is used for this sample` : acting ? 'choosing the action to take' : undefined}
      />
      <NetworkDiagram
        at={{ x: 385, y: 412, w: 545, h: 300 }}
        net={s.target}
        input={targetInput}
        inputLabels={POSITION_ENCODER.labels}
        theme="target"
        title="TARGET NETWORK"
        badge={syncing ? '⇣ RECEIVING WEIGHT COPY' : '🔒 FROZEN · not trained'}
        badgeTone={syncing ? 'syncing' : 'frozen'}
        maxOut={targetMax}
        dim={!training || terminalFocus}
        pulseKey={s.syncs}
        note={terminalFocus && after('targetPredict') ? 's′ is terminal → not consulted, future = 0' : 'estimates future value of s′'}
      />

      {/* ---------------- sync arrow (main → target) */}
      <g className={`sync ${syncing ? 'sync-on' : ''}`}>
        <path d="M657 312 L657 404" className={`${on('sync')} stroke-purple`} markerEnd="url(#ah-purple)" />
        <text x={676} y={330} className="sync-label">
          copy weights &amp; biases every {s.params.targetInterval} updates
        </text>
        <text x={676} y={354} className="sync-count">
          {syncing ? 'COPIED: networks now identical' : `${s.sinceSync} / ${s.params.targetInterval} updates since last copy`}
        </text>
        <text x={676} y={396} className="sync-diff">
          max |main − target| weight gap = {diff === 0 ? '0' : fmtNum(diff)}
        </text>
      </g>

      {/* ---------------- result boxes */}
      <g className={`box box-purple ${s.phase === 'mainPredict' ? 'box-on' : ''}`}>
        <rect x={966} y={110} width={210} height={100} rx={16} />
        <text x={1071} y={140} textAnchor="middle" className="box-title">
          PREDICTION
        </text>
        <text x={1071} y={166} textAnchor="middle" className="box-sub">
          Q_main(s, a)
        </text>
        <text x={1071} y={196} textAnchor="middle" className="box-num">
          {sm?.pred !== undefined && after('mainPredict') ? fmtNum(sm.pred) : '?'}
        </text>
      </g>

      <g className={`box box-green ${s.phase === 'bellman' ? 'box-on' : ''}`}>
        <rect x={966} y={330} width={210} height={84} rx={16} />
        <text x={1071} y={360} textAnchor="middle" className="box-title">
          REWARD (observed)
        </text>
        <text x={1071} y={396} textAnchor="middle" className="box-num">
          {sm ? `r = ${fmtReward(sm.exp.r)}` : '?'}
        </text>
      </g>

      <g className={`box box-yellow ${s.phase === 'targetPredict' ? 'box-on' : ''}`}>
        <rect x={966} y={512} width={210} height={100} rx={16} />
        <text x={1071} y={542} textAnchor="middle" className="box-title">
          FUTURE (estimate)
        </text>
        <text x={1071} y={568} textAnchor="middle" className="box-sub">
          max Q_target(s′, ·)
        </text>
        <text x={1071} y={598} textAnchor="middle" className="box-num">
          {sm && after('targetPredict') ? (terminalFocus ? '0 (terminal)' : fmtQ(sm.maxNext!)) : '?'}
        </text>
      </g>

      <g className={`box box-orange ${s.phase === 'bellman' ? 'box-on' : ''}`}>
        <rect x={1236} y={420} width={340} height={150} rx={16} />
        <text x={1406} y={452} textAnchor="middle" className="box-title">
          BELLMAN TARGET
        </text>
        <text x={1406} y={484} textAnchor="middle" className="box-sub">
          {terminalFocus ? 'target = r   (terminal)' : 'target = r + γ · max Q_target(s′)'}
        </text>
        {sm?.target !== undefined && after('bellman') && (
          <>
            <text x={1406} y={516} textAnchor="middle" className="box-calc">
              {terminalFocus ? `= ${fmtNum(sm.exp.r)}` : `= ${fmtNum(sm.exp.r)} + ${fmtNum(g)} × ${fmtNum(sm.maxNext!)}`}
            </text>
            <text x={1406} y={552} textAnchor="middle" className="box-num">
              = {fmtNum(sm.target)}
            </text>
          </>
        )}
      </g>

      <g className={`box box-red ${s.phase === 'loss' || s.phase === 'backprop' ? 'box-on' : ''}`}>
        <rect x={1236} y={110} width={340} height={170} rx={16} />
        <text x={1406} y={142} textAnchor="middle" className="box-title">
          LOSS
        </text>
        <text x={1406} y={172} textAnchor="middle" className="box-sub">
          (target − prediction)²
        </text>
        {sm?.loss !== undefined && after('loss') && (
          <>
            <text x={1406} y={206} textAnchor="middle" className="box-calc">
              = ({fmtNum(sm.target!)} − {fmtNum(sm.pred!)})²
            </text>
            <text x={1406} y={240} textAnchor="middle" className="box-num">
              = {fmtNum(sm.loss)}
            </text>
            <text x={1406} y={268} textAnchor="middle" className="box-sub">
              batch MSE = {fmtNum(s.train!.batchLoss!)}
            </text>
          </>
        )}
      </g>

      {/* ---------------- arrows */}
      <path d="M216 300 C 236 250, 236 190, 248 172" className={`${on('buf-s')} stroke-blue`} markerEnd="url(#ah-blue)" />
      <path d="M216 430 C 236 480, 236 530, 248 548" className={`${on('buf-sn')} stroke-blue`} markerEnd="url(#ah-blue)" />
      <path d="M362 166 L 386 166" className={`${on('s-main')} stroke-purple`} markerEnd="url(#ah-purple)" />
      <path d="M362 556 L 386 556" className={`${on('sn-target')} stroke-yellow`} markerEnd="url(#ah-yellow)" />
      <path d="M930 160 L 962 160" className={`${on('main-q')} stroke-purple`} markerEnd="url(#ah-purple)" />
      <path d="M930 562 L 962 562" className={`${on('target-max')} stroke-yellow`} markerEnd="url(#ah-yellow)" />
      <path d="M216 372 L 962 372" className={`${on('buf-r')} stroke-green`} markerEnd="url(#ah-green)" />
      <text x={240} y={364} className="arrow-label green">r (observed reward)</text>
      <path d="M1176 372 C 1210 372, 1220 440, 1234 452" className={`${on('r-bell')} stroke-green`} markerEnd="url(#ah-green)" />
      <path d="M1176 562 C 1206 562, 1216 530, 1234 520" className={`${on('max-bell')} stroke-yellow`} markerEnd="url(#ah-yellow)" />
      <path d="M1406 420 L 1406 284" className={`${on('bell-loss')} stroke-orange`} markerEnd="url(#ah-orange)" />
      <text x={1418} y={360} className="arrow-label orange">target (a number)</text>
      <path d="M1176 160 L 1232 160" className={`${on('q-loss')} stroke-purple`} markerEnd="url(#ah-purple)" />

      {/* backprop: loss → main network */}
      <path d="M1406 108 C 1406 40, 1200 20, 940 30" className={`${on('backprop')} stroke-red backprop`} markerEnd="url(#ah-red)" />
      <text x={1150} y={42} className="arrow-label red">
        backpropagation: gradients update MAIN weights only
      </text>
    </svg>
  );
}
