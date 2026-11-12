import { useId } from 'react';
import { bestActions } from '../engine/agent';
import { WORLD } from '../engine/environment';
import { fmtQ, fmtReward } from '../engine/format';
import { ACTION_DELTAS, Action, Pos, posKey, samePos } from '../engine/types';
import { BoardModel } from '../view';
import { valueColor } from './ui';

const C = 150; // tile size in SVG units
const M = 30; // margin for axis labels
const W = WORLD.cols * C + M + 8;
const H = WORLD.rows * C + M + 8;

const tx = (p: Pos) => M + p.x * C;
const ty = (p: Pos) => M + p.y * C;
const cx = (p: Pos) => tx(p) + C / 2;
const cy = (p: Pos) => ty(p) + C / 2;

interface Props {
  model: BoardModel;
  inspected: Pos | null;
  onTileClick: (p: Pos) => void;
  showValues: boolean;
  showPolicy: boolean;
}

export function Board({ model, inspected, onTileClick, showValues, showPolicy }: Props) {
  const uid = useId().replace(/:/g, '');
  const hl = model.hl;
  const tr = model.transition;
  const afterExec = tr && model.afterExecute;
  const mood = afterExec ? (tr!.outcome === 'goal' ? 'happy' : tr!.outcome === 'trap' ? 'sad' : 'normal') : 'normal';
  const look = model.look;
  const bumping = model.bumping;

  return (
    <svg className="board" viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Grid world">
      <defs>
        <pattern id={`bricks-${uid}`} width="50" height="30" patternUnits="userSpaceOnUse">
          <rect width="50" height="30" fill="#4c1d95" />
          <path d="M0 0.5 H50 M0 15.5 H50 M25 0 V15 M0 15 V30 M50 15 V30" stroke="#2e1065" strokeWidth="3" />
        </pattern>
        <linearGradient id={`gold-${uid}`} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0" stopColor="#fef08a" />
          <stop offset="1" stopColor="#facc15" />
        </linearGradient>
        <radialGradient id={`pit-${uid}`}>
          <stop offset="0" stopColor="#0f0518" />
          <stop offset="0.7" stopColor="#3b0764" />
          <stop offset="1" stopColor="#7f1d1d" />
        </radialGradient>
        <marker id={`arrow-${uid}`} viewBox="0 0 10 10" refX="5" refY="5" markerWidth="3.2" markerHeight="3.2" orient="auto-start-reverse">
          <path d="M0 0 L10 5 L0 10 z" fill="#c026d3" stroke="#fff" strokeWidth="1" />
        </marker>
      </defs>

      {/* axis labels */}
      {Array.from({ length: WORLD.cols }, (_, x) => (
        <text key={`cx${x}`} className="axis-label" x={M + x * C + C / 2} y={M - 9} textAnchor="middle">
          x = {x}
        </text>
      ))}
      {Array.from({ length: WORLD.rows }, (_, y) => (
        <text key={`ry${y}`} className="axis-label" x={M - 10} y={M + y * C + C / 2} textAnchor="middle" transform={`rotate(-90 ${M - 10} ${M + y * C + C / 2})`}>
          y = {y}
        </text>
      ))}

      {WORLD.allCells().map((p) => (
        <Tile
          key={posKey(p)}
          p={p}
          uid={uid}
          q={model.values(p)}
          showValues={showValues}
          showPolicy={showPolicy}
          onClick={() => onTileClick(p)}
          nextMax={samePos(hl.next, p) && hl.nextRow ? hl.nextMax : []}
          inspectingCurrent={hl.inspectingCurrent && samePos(hl.current, p)}
          selected={samePos(hl.current, p) ? hl.selectedAction : null}
          updated={hl.updated && samePos(hl.updated.s, p) ? hl.updated.a : null}
          updateKey={model.updateKey}
        />
      ))}

      {/* state highlights */}
      {inspected && <rect className="hl-inspect" x={tx(inspected) + 6} y={ty(inspected) + 6} width={C - 12} height={C - 12} rx={16} />}
      {hl.next && !samePos(hl.next, hl.current) && (
        <rect className="hl-next" x={tx(hl.next) + 5} y={ty(hl.next) + 5} width={C - 10} height={C - 10} rx={16} />
      )}
      {hl.current && <rect className="hl-current" x={tx(hl.current) + 3} y={ty(hl.current) + 3} width={C - 6} height={C - 6} rx={18} />}
      {hl.next && samePos(hl.next, hl.current) && (
        <rect className="hl-next" x={tx(hl.next) + 12} y={ty(hl.next) + 12} width={C - 24} height={C - 24} rx={12} />
      )}

      <Robot pos={model.pos} mood={mood} look={look} bump={bumping ? (tr!.action as Action) : null} bumpKey={model.tick} />

      {/* selected action arrow (drawn above the robot so it is never hidden) */}
      {hl.current && hl.selectedAction !== null && <ActionArrow from={hl.current} a={hl.selectedAction} uid={uid} key={model.arrowKey} />}

      {model.showReward && tr && <RewardPop key={model.tick} at={tr.to} reward={tr.reward} outcome={tr.outcome} />}
    </svg>
  );
}

function Tile(props: {
  p: Pos;
  uid: string;
  q: number[] | undefined;
  showValues: boolean;
  showPolicy: boolean;
  onClick: () => void;
  nextMax: Action[];
  inspectingCurrent: boolean;
  selected: Action | null;
  updated: Action | null;
  updateKey: number;
}) {
  const { p, uid, q } = props;
  const type = WORLD.cellType(p);
  const x = tx(p);
  const y = ty(p);
  const mx = cx(p);
  const my = cy(p);

  if (type === 'wall') {
    return (
      <g>
        <rect x={x + 3} y={y + 3} width={C - 6} height={C - 6} rx={14} fill={`url(#bricks-${uid})`} stroke="#2e1065" strokeWidth={3} />
        <text className="tile-label light" x={mx} y={my + 8} textAnchor="middle">
          WALL
        </text>
      </g>
    );
  }
  if (type === 'goal') return <GoalTile x={x} y={y} uid={uid} />;
  if (type === 'trap') return <TrapTile x={x} y={y} uid={uid} />;

  // triangles: up, down, left, right
  const tri = [
    `${x},${y} ${x + C},${y} ${mx},${my}`,
    `${x},${y + C} ${x + C},${y + C} ${mx},${my}`,
    `${x},${y} ${x},${y + C} ${mx},${my}`,
    `${x + C},${y} ${x + C},${y + C} ${mx},${my}`,
  ];
  const labelPos: [number, number][] = [
    [mx, y + 24],
    [mx, y + C - 12],
    [x + 23, my + 5],
    [x + C - 23, my + 5],
  ];
  const best = q ? bestActions(q) : [];
  const hasPolicy = q && best.length < 4;

  return (
    <g className="tile" onClick={props.onClick}>
      <clipPath id={`clip-${uid}-${p.x}${p.y}`}>
        <rect x={x + 3} y={y + 3} width={C - 6} height={C - 6} rx={14} />
      </clipPath>
      <g clipPath={`url(#clip-${uid}-${p.x}${p.y})`}>
        <rect x={x} y={y} width={C} height={C} fill={type === 'start' ? '#ede9fe' : '#f5f3ff'} />
        {props.showValues &&
          q &&
          tri.map((pts, a) => (
            <polygon
              key={a}
              points={pts}
              fill={valueColor(q[a], 0.9)}
              className={[
                'q-tri',
                props.selected === a ? 'q-tri-selected' : '',
                props.nextMax.includes(a as Action) ? 'q-tri-nextmax' : '',
                props.inspectingCurrent ? 'q-tri-inspect' : '',
              ].join(' ')}
            />
          ))}
        {props.showValues && <path d={`M${x} ${y} L${x + C} ${y + C} M${x + C} ${y} L${x} ${y + C}`} stroke="#c4b5fd" strokeWidth={1.5} />}
        {props.updated !== null && (
          <polygon key={props.updateKey} points={tri[props.updated]} className="q-tri-flash" />
        )}
      </g>
      <rect x={x + 3} y={y + 3} width={C - 6} height={C - 6} rx={14} fill="none" stroke="#7c3aed" strokeWidth={3} />
      {props.showValues &&
        q &&
        labelPos.map(([lx, ly], a) => (
          <text key={a} className={`tri-value ${best.includes(a as Action) && hasPolicy ? 'best' : ''}`} x={lx} y={ly} textAnchor="middle">
            {fmtQ(q[a])}
          </text>
        ))}
      {props.showPolicy && hasPolicy && <PolicyArrow p={p} actions={best} />}
      {type === 'start' && (
        <text className="start-label" x={x + 10} y={y + 17}>
          START
        </text>
      )}
    </g>
  );
}

function PolicyArrow({ p, actions }: { p: Pos; actions: Action[] }) {
  return (
    <g className="policy-arrow">
      {actions.map((a) => {
        const [dx, dy] = ACTION_DELTAS[a];
        const x1 = cx(p) + dx * 10;
        const y1 = cy(p) + dy * 10;
        const x2 = cx(p) + dx * 34;
        const y2 = cy(p) + dy * 34;
        return <line key={a} x1={x1} y1={y1} x2={x2} y2={y2} />;
      })}
    </g>
  );
}

function ActionArrow({ from, a, uid }: { from: Pos; a: Action; uid: string }) {
  const [dx, dy] = ACTION_DELTAS[a];
  const x1 = cx(from) + dx * 46;
  const y1 = cy(from) + dy * 46;
  const x2 = cx(from) + dx * 112;
  const y2 = cy(from) + dy * 112;
  return (
    <g className="action-arrow">
      <line x1={x1} y1={y1} x2={x2} y2={y2} className="action-arrow-outline" />
      <line x1={x1} y1={y1} x2={x2} y2={y2} className="action-arrow-line" markerEnd={`url(#arrow-${uid})`} />
    </g>
  );
}

function GoalTile({ x, y, uid }: { x: number; y: number; uid: string }) {
  const mx = x + C / 2;
  return (
    <g>
      <rect x={x + 3} y={y + 3} width={C - 6} height={C - 6} rx={14} fill="#fef9c3" stroke="#eab308" strokeWidth={4} />
      <g transform={`translate(${mx}, ${y + 78})`}>
        {/* chest */}
        <rect x={-38} y={-6} width={76} height={40} rx={6} fill="#92400e" stroke="#451a03" strokeWidth={3} />
        <path d="M-38 -6 Q-38 -34 0 -34 Q38 -34 38 -6 Z" fill="#b45309" stroke="#451a03" strokeWidth={3} />
        <rect x={-38} y={-8} width={76} height={8} fill={`url(#gold-${uid})`} stroke="#451a03" strokeWidth={2} />
        <rect x={-6} y={-4} width={12} height={16} rx={2} fill={`url(#gold-${uid})`} stroke="#451a03" strokeWidth={2} />
        <rect x={-30} y={6} width={6} height={26} fill="#fcd34d" opacity={0.7} />
        <rect x={24} y={6} width={6} height={26} fill="#fcd34d" opacity={0.7} />
        {/* coins */}
        <circle cx={-20} cy={-36} r={7} fill={`url(#gold-${uid})`} stroke="#a16207" strokeWidth={2} />
        <circle cx={-4} cy={-42} r={7} fill={`url(#gold-${uid})`} stroke="#a16207" strokeWidth={2} />
        <circle cx={14} cy={-37} r={7} fill={`url(#gold-${uid})`} stroke="#a16207" strokeWidth={2} />
        <path className="sparkle s1" d="M-44 -50 l3 8 l8 3 l-8 3 l-3 8 l-3 -8 l-8 -3 l8 -3 z" fill="#facc15" />
        <path className="sparkle s2" d="M40 -56 l2.5 6 l6 2.5 l-6 2.5 l-2.5 6 l-2.5 -6 l-6 -2.5 l6 -2.5 z" fill="#a855f7" />
      </g>
      <text className="tile-label gold" x={mx} y={y + C - 14} textAnchor="middle">
        GOAL
      </text>
    </g>
  );
}

function TrapTile({ x, y, uid }: { x: number; y: number; uid: string }) {
  const mx = x + C / 2;
  const my = y + C / 2;
  const spikes = [-30, -15, 0, 15, 30];
  return (
    <g>
      <rect x={x + 3} y={y + 3} width={C - 6} height={C - 6} rx={14} fill="#fee2e2" stroke="#dc2626" strokeWidth={4} strokeDasharray="14 8" />
      <ellipse cx={mx} cy={my + 4} rx={50} ry={34} fill={`url(#pit-${uid})`} className="trap-pit" />
      {spikes.map((sx) => (
        <path key={sx} d={`M${mx + sx - 7} ${my + 20} L${mx + sx} ${my - 8 - (sx === 0 ? 10 : 0)} L${mx + sx + 7} ${my + 20} Z`} fill="#e5e7eb" stroke="#7f1d1d" strokeWidth={2} />
      ))}
      <text className="tile-label danger" x={mx} y={y + 26} textAnchor="middle">
        ⚠ TRAP
      </text>
    </g>
  );
}

function Robot({ pos, mood, look, bump, bumpKey }: { pos: Pos; mood: 'normal' | 'happy' | 'sad'; look: Action | null; bump: Action | null; bumpKey: number }) {
  const [lx, ly] = look !== null ? ACTION_DELTAS[look] : [0, 0];
  return (
    <g className="robot" style={{ transform: `translate(${cx(pos)}px, ${cy(pos) + 2}px) scale(0.8)` }}>
      <g className={`robot-mood robot-${mood}`}>
        <g key={bump !== null ? bumpKey : 'still'} className={bump !== null ? `robot-bump bump-${bump}` : ''}>
          <ellipse cx={0} cy={44} rx={30} ry={7} fill="rgba(46,16,101,0.22)" />
          <g className="robot-bob">
            <line x1={0} y1={-34} x2={0} y2={-46} stroke="#4c1d95" strokeWidth={4} strokeLinecap="round" />
            <circle className="antenna" cx={0} cy={-50} r={6} fill="#facc15" stroke="#4c1d95" strokeWidth={2.5} />
            <rect x={-28} y={8} width={56} height={32} rx={11} fill="#a78bfa" stroke="#4c1d95" strokeWidth={3} />
            <circle cx={0} cy={23} r={6} fill="#facc15" stroke="#4c1d95" strokeWidth={2} />
            <rect x={-40} y={12} width={12} height={20} rx={6} fill="#8b5cf6" stroke="#4c1d95" strokeWidth={2.5} />
            <rect x={28} y={12} width={12} height={20} rx={6} fill="#8b5cf6" stroke="#4c1d95" strokeWidth={2.5} />
            <rect x={-34} y={-36} width={68} height={46} rx={16} fill="#8b5cf6" stroke="#4c1d95" strokeWidth={3} />
            <rect x={-25} y={-28} width={50} height={30} rx={10} fill="#1e1b4b" />
            {mood === 'happy' ? (
              <g stroke="#facc15" strokeWidth={3.5} fill="none" strokeLinecap="round">
                <path d="M-17 -12 q6 -8 12 0" />
                <path d="M5 -12 q6 -8 12 0" />
                <path d="M-8 -4 q8 6 16 0" />
              </g>
            ) : mood === 'sad' ? (
              <g stroke="#f87171" strokeWidth={3.5} strokeLinecap="round">
                <path d="M-16 -19 l8 8 M-8 -19 l-8 8" />
                <path d="M8 -19 l8 8 M16 -19 l-8 8" />
                <path d="M-7 -2 q7 -5 14 0" fill="none" />
              </g>
            ) : (
              <g>
                <circle cx={-11 + lx * 4} cy={-14 + ly * 4} r={5.5} fill="#facc15" className="eye" />
                <circle cx={11 + lx * 4} cy={-14 + ly * 4} r={5.5} fill="#facc15" className="eye" />
                <path d="M-7 -4 q7 4 14 0" stroke="#facc15" strokeWidth={2.5} fill="none" strokeLinecap="round" />
              </g>
            )}
          </g>
        </g>
      </g>
    </g>
  );
}

function RewardPop({ at, reward, outcome }: { at: Pos; reward: number; outcome: string }) {
  return (
    <g className={`reward-pop reward-${outcome}`} style={{ transformOrigin: `${cx(at)}px ${cy(at)}px` }}>
      {(outcome === 'goal' || outcome === 'trap') && <circle className="burst" cx={cx(at)} cy={cy(at)} r={40} />}
      <text x={cx(at)} y={ty(at) + 34} textAnchor="middle">
        {fmtReward(reward)}
      </text>
    </g>
  );
}
