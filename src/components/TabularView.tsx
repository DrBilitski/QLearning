import { CSSProperties, ReactNode, useState } from 'react';
import { WORLD } from '../engine/environment';
import { fmtNum, fmtProb, fmtReward } from '../engine/format';
import { MICROSTEPS, phaseIndex } from '../engine/simulation';
import { Pos, samePos } from '../engine/types';
import { SimController } from '../useSimulation';
import { boardModelFromSim, epsilonModelFromSim } from '../view';
import { BellmanPanel } from './BellmanPanel';
import { Board } from './Board';
import { Controls } from './Controls';
import { EpsilonPanel } from './EpsilonPanel';
import { ExplanationPanel } from './ExplanationPanel';
import { ParamsPanel, RewardsPanel, ScenariosPanel } from './InstructorPanels';
import { QTable } from './QTable';
import { StatsPanel } from './StatsPanel';
import { TopBar, TopStat } from './TopBar';
import { Panel } from './ui';

export function TabularView({ ctl, modeSelector }: { ctl: SimController; modeSelector: ReactNode }) {
  const { state } = ctl;
  const [showValues, setShowValues] = useState(true);
  const [showPolicy, setShowPolicy] = useState(true);

  // Animation lengths follow the play speed so animations never lag behind the logic.
  const stepMs = ctl.playing ? ctl.delay : 900;
  const moveMs = Math.max(40, Math.min(650, stepMs * 0.6));
  const animMs = Math.max(0, Math.min(900, stepMs * 0.7));

  const toggleInspect = (p: Pos) => {
    if (WORLD.isWall(p) || WORLD.isTerminal(p)) return;
    ctl.setInspected(samePos(ctl.inspected, p) ? null : p);
  };

  const micro = phaseIndex(state.phase);

  return (
    <div className="app" style={{ '--move-ms': `${moveMs}ms`, '--anim-ms': `${animMs}ms` } as CSSProperties}>
      <TopBar
        subtitle="Tabular Q-learning, one step at a time"
        modeSelector={modeSelector}
        badge={state.scenario ? `GUIDED SCENARIO ${state.scenario.letter}` : 'LIVE TRAINING'}
        guided={!!state.scenario}
        playing={ctl.playing}
      >
        <TopStat label="Episode" value={state.episode} />
        <TopStat label="Step" value={`${state.step}/${state.params.maxSteps}`} />
        <TopStat label="Episode reward" value={fmtNum(state.episodeReward)} />
        <TopStat label="Epsilon" value={fmtProb(state.params.epsilon)} accent />
        <TopStat label="Microstep" value={micro === 0 ? 'ready' : `${micro}/10 · ${MICROSTEPS[micro - 1].label}`} wide />
      </TopBar>

      {state.scenario && (
        <div className="scenario-banner">
          <b>
            Scenario {state.scenario.letter}: {state.scenario.title}
          </b>
          <span>{state.scenario.description}</span>
          <span className="sb-honest">
            {state.script.length > 0
              ? `${state.script.length} predetermined random number${state.script.length > 1 ? 's' : ''} remaining, then the seeded RNG takes over.`
              : 'This is a prepared demonstration, not the result of a live training run.'}
          </span>
          <button className="btn btn-soft btn-small" onClick={ctl.exitScenario}>
            Exit scenario
          </button>
        </div>
      )}

      <main className="dashboard">
        <Panel
          className="area-board"
          title="Game World"
          icon="🗺️"
          right={
            <span className="toggles">
              <label>
                <input type="checkbox" checked={showValues} onChange={(e) => setShowValues(e.target.checked)} /> Q on tiles
              </label>
              <label>
                <input type="checkbox" checked={showPolicy} onChange={(e) => setShowPolicy(e.target.checked)} /> best-action arrows
              </label>
            </span>
          }
        >
          <Board model={boardModelFromSim(state)} inspected={ctl.inspected} onTileClick={toggleInspect} showValues={showValues} showPolicy={showPolicy} />
          <div className="reward-legend">
            <span className="rl rl-goal">🏆 goal {fmtReward(state.rewards.goal)}</span>
            <span className="rl rl-trap">💥 trap {fmtReward(state.rewards.trap)}</span>
            <span className="rl rl-move">👣 move {fmtReward(state.rewards.move)}</span>
            <span className="rl rl-bump">🧱 bump {fmtReward(state.rewards.bump)}</span>
          </div>
        </Panel>

        <Panel
          className="area-qtable"
          title="Q-Table"
          icon="📋"
          right={
            <span className="legend-mini">
              <span className="lm lm-s">s</span> current <span className="lm lm-sn">s′</span> next <span className="lm lm-a">a</span> chosen
            </span>
          }
        >
          <QTable state={state} inspected={ctl.inspected} onRowClick={toggleInspect} animMs={animMs} />
        </Panel>

        <Panel className="area-eps" title="Exploration vs. Exploitation" icon="🎲">
          <EpsilonPanel model={epsilonModelFromSim(state)} />
        </Panel>

        <Panel className="area-bellman" title="Bellman Update" icon="🧮">
          <BellmanPanel state={state} animMs={animMs} stepMs={stepMs} />
        </Panel>

        <Panel className="area-explain" title="What is happening?" icon="💬">
          <ExplanationPanel state={state} />
        </Panel>
      </main>

      <Controls ctl={ctl} />

      <section className="instructor">
        <Panel className="area-rewards" title="Reward System" icon="🏆">
          <RewardsPanel rewards={state.rewards} setRewards={ctl.setRewards} />
        </Panel>
        <Panel className="area-params" title="Learning Parameters" icon="🎛️">
          <ParamsPanel state={state} ctl={ctl} />
        </Panel>
        <Panel className="area-stats" title="Episode History" icon="📈">
          <StatsPanel state={state} />
        </Panel>
        <Panel className="area-scenarios" title="Guided Demonstrations" icon="🎓">
          <ScenariosPanel state={state} ctl={ctl} />
        </Panel>
      </section>

      <footer className="footer">
        Tabular Q-learning mode. Switch to Deep Q-Learning (DQN) at the top: same game, same rewards, but a neural network replaces the Q-table.
      </footer>
    </div>
  );
}
