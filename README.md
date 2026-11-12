# Q-Learning Explorer

An interactive classroom visualizer for **tabular Q-learning** on a 4×3 grid world.
Built with React + TypeScript + Vite. It runs entirely in the browser with no backend.

## Run it

```bash
npm install
npm run dev        # open the printed localhost URL
npm test           # engine tests (Bellman math, decay, scenarios, determinism)
npm run build      # static build in dist/ (open dist/index.html or host anywhere)
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

## Design notes

- `src/engine/` holds the simulation. It is pure TypeScript with no React:
  - `environment.ts`: the grid world, transitions, rewards. It knows nothing about Q-values.
  - `agent.ts`: the Q-table, the Bellman update, and the `ActionValueProvider` interface.
  - `simulation.ts`: a pure state machine. `advance(state)` performs exactly one microstep.
    All randomness comes from a seeded RNG stored inside the state, so nothing can change while paused.
  - `scenarios.ts`: guided demos A–E. They are prepared starting points run by the same engine,
    with optional predetermined random numbers. The UI labels them as scripted.
- `src/components/` only *displays* engine results. It never computes Q-values itself.
- Episodes that hit the step limit still use the last state's best Q-value, because the step limit is not a terminal state.
  Only the goal and trap use a future value of 0.
- **DQN later:** the UI reads values through `ActionValueProvider.getQValues(state)`, and
  `GridWorld.encodeState()` already gives normalized (x, y) inputs for a network.
