import { TabularQ } from '../engine/agent';
import { WORLD } from '../engine/environment';
import { SimState } from '../engine/simulation';
import { ACTIONS, ACTION_ARROWS, ACTION_NAMES, Pos, fmtPos, posKey, samePos } from '../engine/types';
import { highlights } from '../view';
import { AnimatedNumber, valueColor, valueTextColor } from './ui';

interface Props {
  state: SimState;
  inspected: Pos | null;
  onRowClick: (p: Pos) => void;
  animMs: number;
}

export function QTable({ state, inspected, onRowClick, animMs }: Props) {
  const hl = highlights(state);
  // The table reads values through the ActionValueProvider interface, so a
  // future DQN can supply the same numbers from a network.
  const provider = new TabularQ(state.q);

  return (
    <div className="qtable-wrap">
      <table className="qtable">
        <thead>
          <tr>
            <th>State</th>
            {ACTIONS.map((a) => (
              <th key={a} className={hl.selectedAction === a ? 'th-selected' : ''}>
                <span className="th-arrow">{ACTION_ARROWS[a]}</span> {ACTION_NAMES[a]}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {WORLD.states().map((s) => {
            const values = provider.getQValues(s);
            const isCurrent = samePos(hl.current, s);
            const isNext = hl.nextRow && samePos(hl.next, s);
            const isInspected = samePos(inspected, s);
            const cls = ['qrow', isCurrent ? 'row-current' : '', isNext ? 'row-next' : '', isInspected ? 'row-inspected' : ''].join(' ');
            return (
              <tr key={posKey(s)} className={cls} onClick={() => onRowClick(s)} title="Click to highlight this tile on the board">
                <td className="state-cell">
                  <span className="state-name">{fmtPos(s)}</span>
                  {samePos(s, WORLD.start) && <span className="state-tag">start</span>}
                  {isCurrent && <span className="state-tag tag-s">s</span>}
                  {isNext && <span className="state-tag tag-snext">s′</span>}
                </td>
                {ACTIONS.map((a) => {
                  const v = values[a];
                  const selected = isCurrent && hl.selectedAction === a;
                  const inspecting = isCurrent && hl.inspectingCurrent;
                  const nextMax = isNext && hl.nextMax.includes(a);
                  const updated = hl.updated && samePos(hl.updated.s, s) && hl.updated.a === a;
                  return (
                    <td
                      key={a}
                      className={[
                        'qcell',
                        selected ? 'cell-selected' : '',
                        inspecting ? 'cell-inspect' : '',
                        nextMax ? 'cell-nextmax' : '',
                        updated ? 'cell-updated' : '',
                      ].join(' ')}
                      style={{ background: valueColor(v), color: valueTextColor(v) }}
                    >
                      <AnimatedNumber value={v} duration={animMs} />
                      {nextMax && <span className="cell-badge">max</span>}
                    </td>
                  );
                })}
              </tr>
            );
          })}
        </tbody>
      </table>
      <p className="qtable-note">
        Rows are the 9 states the robot can act from. The goal and trap end the episode, so no action is ever chosen there and they have no rows. Their future value is always 0.
      </p>
    </div>
  );
}
