# Q-Learning Explorer

An interactive classroom visualizer for **tabular Q-learning** and **Deep Q-Networks (DQN)** on a 4×3 grid world.
Use the *Learning algorithm* switch at the top to change modes. Each mode keeps its own training state.
Built with React + TypeScript + Vite. It runs entirely in the browser with no backend.

## Run it

```bash
npm install
npm run dev        # open the printed localhost URL
npm test           # engine tests (Bellman math, decay, scenarios, determinism)
npm run build      # static build in dist/ — serve it (npm run preview) or host on any static host
```

## Instructor controls

| Control | Key | What it does |
|---|---|---|
| Play / Pause | Space | Auto-advance one microstep per tick. Pause freezes everything. |
| Next Step | → or N | Exactly one of the 10 microsteps |
| Next Action | A | Finish the current action, including its Q update |
| Next Episode | E | Finish the episode and stop at the start of the next one |
| Train ×10 | | Run 10 full episodes instantly |
| Restart Episode | | Robot back to start, Q-table kept |
| Reset | | Everything back to defaults (same seed, so it is reproducible) |

Click any Q-table row or board tile to highlight it without changing the simulation.

## DQN mode

- **Main network** (purple), 2 → 16 ReLU → 16 ReLU → 4 linear outputs. It predicts Q(UP/DOWN/LEFT/RIGHT) from (x/3, y/2),
  chooses actions, and is trained by backpropagation (MSE loss, Adam).
- **Target network** (yellow): the same architecture. It is frozen and only supplies max Q(s′) for the Bellman target.
  Every *N* training updates it receives a copy of the main network's weights.
- **Replay buffer** (blue): stores (s, a, r, s′, done). Training samples random mini-batches from it.
- **Game View** shows the board, the main network, and epsilon-greedy selection. **Training View** shows the full
  two-network data flow: prediction → target → loss → backprop → sync. A detailed view adds gradients and weight changes.
- Extra controls: **Next Training Update** (key U) and **Train ×50**.
- Scenarios A–I. F is the worked example (3 → target 11 → loss 64). It uses illustrative numbers and is labeled as such.
  The other scenarios load networks produced by real training with a fixed seed.
- The network is implemented from scratch in `src/engine/nn.ts`, with no ML library. It is checked against a numerical
  gradient in the tests.

## Design notes

- `src/engine/` holds the simulation. It is pure TypeScript with no React:
  - `environment.ts`: the grid world, transitions, rewards. It knows nothing about Q-values.
  - `agent.ts`: the Q-table, the Bellman update, and the `ActionValueProvider` interface.
  - `simulation.ts`: a pure state machine. `advance(state)` performs exactly one microstep.
    All randomness comes from a seeded RNG stored inside the state, so nothing can change while paused.
  - `nn.ts`, `dqn.ts`, `dqnScenarios.ts`: the neural network, the DQN state machine, and the DQN demos.
  - `policy.ts`: epsilon-greedy selection, shared by both modes.
  - `scenarios.ts`: guided demos A–E. They are prepared starting points run by the same engine,
    with optional predetermined random numbers. The UI labels them as scripted.
- `src/components/` only *displays* engine results. It never computes Q-values itself.
- Episodes that hit the step limit still use the last state's best Q-value, because the step limit is not a terminal state.
  Only the goal and trap use a future value of 0.
- **DQN later:** the UI reads values through `ActionValueProvider.getQValues(state)`, and
  `GridWorld.encodeState()` already gives normalized (x, y) inputs for a network.
