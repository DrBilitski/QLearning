import { CSSProperties, ReactNode, useState } from 'react';
import { DQN_STAGES, POSITION_ENCODER, dqnPhaseIndex, dqnReached } from '../../engine/dqn';
import { WORLD } from '../../engine/environment';
import { fmtProb } from '../../engine/format';
import { SimState } from '../../engine/simulation';
import { Action, Pos, samePos } from '../../engine/types';
import { DqnController } from '../../useDqn';
import { boardModelFromDqn, epsilonModelFromDqn } from '../../view';
import { Board } from '../Board';
import { Controls } from '../Controls';
import { EpsilonPanel } from '../EpsilonPanel';
import { RewardsPanel } from '../InstructorPanels';
import { TopBar, TopStat } from '../TopBar';
import { Panel } from '../ui';
import { ArchitectureDiagram } from './ArchitectureDiagram';
import { CompareCard, DqnExplanation, DqnParamsPanel, DqnScenariosPanel, DqnStats } from './DqnPanels';
import { NetworkDiagram } from './NetworkDiagram';
import { ReplayBuffer } from './ReplayBuffer';
import { TrainingPanel } from './TrainingPanel';

export function DqnView({ ctl, tabular, modeSelector }: { ctl: DqnController; tabular: SimState; modeSelector: ReactNode }) {
  const s = ctl.state;
  const [showValues, setShowValues] = useState(true);
  const [showPolicy, setShowPolicy] = useState(true);
  const [inspected, setInspected] = useState<Pos | null>(null);

  const stepMs = ctl.playing ? ctl.delay : 900;
  const moveMs = Math.max(40, Math.min(650, stepMs * 0.6));
  const animMs = Math.max(0, Math.min(900, stepMs * 0.7));
  const idx = dqnPhaseIndex(s.phase);
  const stage = idx === 0 ? 'ready' : `${idx}/15 · ${DQN_STAGES[idx - 1].label}`;

  const toggleInspect = (p: Pos) => {
    if (WORLD.isWall(p) || WORLD.isTerminal(p)) return;
    setInspected(samePos(inspected, p) ? null : p);
  };

  // Game-view network: shows the state being acted on (or an inspected tile).
  const acting = dqnReached(s.phase, 'observe') && s.phase !== 'continue';
  const netState = inspected ?? (acting || s.phase === 'start' ? s.current : s.pos);
  const sel = s.decision?.action !== undefined && dqnReached(s.phase, 'select') && !inspected ? [s.decision.action as Action] : [];
  const best = s.decision && !inspected && dqnReached(s.phase, 'decide') ? s.decision.best : [];

  return (
    <div className="app" style={{ '--move-ms': `${moveMs}ms`, '--anim-ms': `${animMs}ms` } as CSSProperties}>
      <TopBar
        subtitle="Deep Q-Network: a neural network replaces the Q-table"
        modeSelector={modeSelector}
        badge={s.scenario ? `GUIDED SCENARIO ${s.scenario.letter}` : 'LIVE DQN TRAINING'}
        guided={!!s.scenario}
        playing={ctl.playing}
      >
        <TopStat label="Episode" value={s.episode} />
        <TopStat label="Step" value={`${s.step}/${s.params.maxSteps}`} />
        <TopStat label="Epsilon" value={fmtProb(s.params.epsilon)} accent />
        <TopStat label="Buffer" value={s.buffer.length} />
        <TopStat label="Updates" value={s.updates} />
        <TopStat label="Target sync" value={`${s.sinceSync}/${s.params.targetInterval}`} />
        <TopStat label="Stage" value={stage} wide />
      </TopBar>

      <div className="view-tabs">
        <button className={ctl.view === 'game' ? 'on' : ''} onClick={() => ctl.setView('game')}>
          🎮 Game View
        </button>
        <button className={ctl.view === 'training' ? 'on' : ''} onClick={() => ctl.setView('training')}>
          🧠 Training View: both networks
        </button>
        <span className="view-hint">Switching views never changes the simulation. Pause, then switch to explain what is going on inside.</span>
        <label className="detail-toggle">
          <input type="checkbox" checked={ctl.detailed} onChange={(e) => ctl.setDetailed(e.target.checked)} /> Detailed view (weights &amp; gradients)
        </label>
      </div>

      {s.scenario && (
        <div className={`scenario-banner ${s.scenario.illustrative ? 'illustrative' : ''}`}>
          <b>
            Scenario {s.scenario.letter}: {s.scenario.title}
          </b>
          <span>{s.scenario.description}</span>
          <span className="sb-honest">
            {s.scenario.illustrative
              ? 'ILLUSTRATIVE NUMBERS: a worked example, not the result of game play.'
              : s.script.length > 0
                ? `${s.script.length} predetermined random number${s.script.length > 1 ? 's' : ''} remaining.`
                : 'Prepared demonstration, built from real training with a fixed seed.'}
          </span>
          <button className="btn btn-soft btn-small" onClick={ctl.exitScenario}>
            Exit scenario
          </button>
        </div>
      )}

      {ctl.view === 'game' ? (
        <main className="dashboard dqn-game">
          <Panel
            className="area-board"
            title="Game World"
            icon="🗺️"
            right={
              <span className="toggles">
                <label>
                  <input type="checkbox" checked={showValues} onChange={(e) => setShowValues(e.target.checked)} /> network Q on tiles
                </label>
                <label>
                  <input type="checkbox" checked={showPolicy} onChange={(e) => setShowPolicy(e.target.checked)} /> best-action arrows
                </label>
              </span>
            }
          >
            <Board model={boardModelFromDqn(s)} inspected={inspected} onTileClick={toggleInspect} showValues={showValues} showPolicy={showPolicy} />
            <div className="reward-legend">
              <span className="rl rl-goal">🏆 goal {s.rewards.goal}</span>
              <span className="rl rl-trap">💥 trap {s.rewards.trap}</span>
              <span className="rl rl-move">👣 move {s.rewards.move}</span>
              <span className="rl rl-bump">🧱 bump {s.rewards.bump}</span>
            </div>
          </Panel>

          <Panel className="area-net" title="Main Network: replaces the Q-table" icon="🧠">
            <NetworkDiagram
              net={s.main}
              input={POSITION_ENCODER.encode(netState)}
              inputLabels={POSITION_ENCODER.labels}
              inputCaption={`state ${`(${netState.x},${netState.y})`}`}
              theme="main"
              title="MAIN NETWORK"
              badge="LEARNING"
              highlight={sel}
              maxOut={best}
              pulseKey={s.updates}
            />
            {s.scenario?.compareWithTable ? (
              <CompareCard dqn={s} tabular={tabular} />
            ) : (
              <p className="net-note">
                {inspected
                  ? `Inspecting ${`(${inspected.x},${inspected.y})`}: click the tile again to return. `
                  : ''}
                The 4 outputs are <b>Q-values</b>: estimates of future reward, not probabilities. There is no softmax. Tile colors on the board show this network’s prediction for every state.
              </p>
            )}
          </Panel>

          <Panel className="area-eps" title="Exploration vs. Exploitation" icon="🎲">
            <EpsilonPanel model={epsilonModelFromDqn(s)} valueSource="predicted by the main network" />
          </Panel>

          <Panel className="area-buffer" title="Experience Replay Buffer" icon="🗃️">
            <ReplayBuffer state={s} focus={ctl.focus} onFocus={ctl.setFocus} />
          </Panel>

          <Panel className="area-explain" title="What is happening?" icon="💬">
            <DqnExplanation state={s} />
          </Panel>
        </main>
      ) : (
        <main className="dashboard dqn-train">
          <Panel className="area-arch" title="DQN Architecture: how the two networks work together" icon="🔗">
            <ArchitectureDiagram state={s} focus={ctl.focus} detailed={ctl.detailed} />
            <TwoNetworkKey />
          </Panel>
          <Panel className="area-buffer" title="Experience Replay Buffer" icon="🗃️">
            <ReplayBuffer state={s} focus={ctl.focus} onFocus={ctl.setFocus} />
          </Panel>
          <Panel className="area-train" title={`Training Calculation ${ctl.detailed ? '(detailed)' : '(basic)'}`} icon="🧮">
            <TrainingPanel state={s} focus={ctl.focus} detailed={ctl.detailed} onFocus={ctl.setFocus} />
          </Panel>
          <Panel className="area-explain" title="What is happening?" icon="💬">
            <DqnExplanation state={s} />
          </Panel>
        </main>
      )}

      <Controls ctl={ctl} resetLabel="Reset DQN" trainCounts={[10, 50]} />

      <section className="instructor">
        <Panel className="area-rewards" title="Reward System" icon="🏆">
          <RewardsPanel rewards={s.rewards} setRewards={ctl.setRewards} />
        </Panel>
        <Panel className="area-params" title="DQN Parameters" icon="🎛️">
          <DqnParamsPanel state={s} ctl={ctl} />
        </Panel>
        <Panel className="area-scenarios" title="Guided DQN Demonstrations" icon="🎓">
          <DqnScenariosPanel state={s} ctl={ctl} />
        </Panel>
      </section>
      <section className="instructor-wide">
        <Panel title="Training Progress" icon="📈">
          <DqnStats state={s} />
        </Panel>
      </section>

      <footer className="footer">
        DQN mode keeps its own networks and statistics. Switching to Q-Learning and back loses nothing. Rewards and parameters are set separately for each mode.
      </footer>
    </div>
  );
}

function TwoNetworkKey() {
  return (
    <div className="two-net-key">
      <div className="tnk tnk-main">
        <b>MAIN network (purple)</b> chooses actions and is the one that LEARNS. Every training update changes its weights by backpropagation.
      </div>
      <div className="tnk tnk-target">
        <b>TARGET network (yellow)</b> only answers one question: “how good is the next state s′?” It is FROZEN, and it changes only when the main network’s weights are copied into it.
      </div>
      <div className="tnk tnk-why">
        <b>Why two?</b> If the same network produced both the prediction and the target, every update would also move the target, like chasing your own shadow. A frozen copy gives a steadier target to learn toward.
      </div>
    </div>
  );
}

