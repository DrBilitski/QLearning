# Project Specification: Interactive Q-Learning Visualizer

## 1. Project Overview

Build a colorful, highly graphical, interactive application that teaches how Q-learning works.

The primary audience is university students learning reinforcement learning for the first time.

**The main goal is to visualize the algorithm, not simply execute it.**

Students should be able to watch an agent make decisions, inspect Q-values, see epsilon-greedy exploration, examine Bellman calculations, and observe the Q-table changing over time.

The instructor must be able to pause the simulation, advance through individual steps, change parameters, and explain what is happening.

The application should feel like an interactive educational animation rather than a programming demonstration.

### Important Scope Restriction

Implement traditional tabular Q-learning first.

We will eventually add Deep Q-Networks (DQN), but do not implement neural networks in this version.

Design the environment and UI so the same game can later be reused with DQN.

---

## 2. Technology

Choose the technology that produces the best interactive graphical experience.

A browser-based application is preferred because it will be easy to run in a classroom, distribute to students, and project during lectures.

Recommended stack:

- React and TypeScript
- Vite
- SVG or Canvas for game visualization
- A modern component library if useful
- Client-side execution with no backend required

The implementation should be easy to run locally.

Prioritize graphical quality, clarity, responsiveness, and maintainability over framework complexity.

---

## 3. Game Environment

### Simple Grid World

Create a small, visually attractive game board.

Use a 4-column by 3-row grid.

The board contains:

- One robot agent
- One starting position
- One goal represented by a treasure chest or flag
- One dangerous tile representing a trap
- One wall that cannot be entered
- Normal movement tiles

Example layout:

```text
      0        1        2        3

0   [   ]    [   ]    [   ]   [GOAL]

1   [   ]    [TRAP]   [WALL]  [   ]

2   [ROBOT]  [   ]    [   ]   [   ]
```

The robot starts at position (0,2).

The goal is at (3,0).

The trap is at (1,1).

The wall is at (2,1).

The robot can choose four actions:

- Up
- Down
- Left
- Right

Each action attempts to move the robot one cell in the corresponding direction.

If the robot moves into a wall or outside the board, it stays in its current position and receives a penalty.

Reaching the goal or entering the trap ends the episode.

The game should be simple and deterministic so students can easily understand how each action changes the environment.

### Visual Design

The board should resemble a polished, colorful educational game.

Use:

- Purple and yellow as primary colors
- Bright accent colors for important events
- A friendly animated robot
- A treasure chest or flag for the goal
- A clearly visible hazard for the trap
- Distinct tile borders
- Smooth movement animations
- Highlighting for current and next states
- Colored directional arrows

Use large readable text appropriate for classroom projection.

Avoid a plain spreadsheet or console appearance.

---

## 4. Reward System

The game must have a clearly defined reward system.

Use the following default rewards:

| Event | Reward |
|---|---:|
| Reach the goal | +10 |
| Enter the trap | -10 |
| Normal successful movement | -0.1 |
| Hit a wall or board boundary | -1 |

The rewards should be adjustable through the interface.

Display the reward system prominently in a dedicated panel.

When the agent performs an action, visually display the reward received.

Examples:

- Reaching the goal produces a bright +10 animation.
- Entering the trap produces a red -10 animation.
- A normal move displays -0.1.
- Attempting to move into a wall displays -1.

**Teaching objective:** Students should understand that the developer defines rewards, but does not tell the robot which actions are correct.

There are no manually provided action labels.

---

## 5. Q-Table Visualization

The Q-table is one of the most important components of the application.

Display the complete Q-table alongside the game.

Rows represent nonterminal, non-wall states.

Columns represent the four possible actions.

Example:

| State | Up | Down | Left | Right |
|---|---:|---:|---:|---:|
| (0,2) | 0.00 | 0.00 | 0.00 | 0.00 |
| (0,1) | 0.00 | 0.00 | 0.00 | 0.00 |
| (0,0) | 0.00 | 0.00 | 0.00 | 0.00 |
| ... | ... | ... | ... | ... |

Initially, all Q-values should be zero.

### Q-Table Interaction

As the robot moves:

1. Highlight the current state's row.
2. Highlight the four available Q-values.
3. Highlight the action selected.
4. After the action, show which Q-value will be updated.
5. Animate the numerical change.
6. Keep the updated value visible.

Use color to distinguish positive and negative Q-values.

For example:

- Green or yellow for favorable values
- Red for unfavorable values
- Purple for the currently selected value

Also highlight the current state and selected action directly on the game board.

Students should be able to click any Q-table row to highlight the corresponding game tile.

Do not change the actual simulation state when merely inspecting a row.

---

## 6. Epsilon-Greedy Action Selection

Students must be able to see exactly how the agent chooses an action.

Provide an **Exploration vs. Exploitation** panel.

Epsilon controls the probability of exploration.

For example:

```text
Epsilon = 0.80

80% probability of exploration
20% probability of exploitation
```

Represent this visually using a colorful probability bar.

### Show the Actual Decision

Every time the agent chooses an action:

1. Display the current epsilon.
2. Generate and display a random number between 0 and 1.
3. Compare the random number to epsilon.
4. Clearly indicate whether the agent will explore or exploit.
5. Highlight the action it selects.

Example:

```text
Epsilon:       0.80
Random number: 0.43

0.43 < 0.80

Decision: EXPLORE

Selected Action: LEFT
```

If exploiting, select an action with the highest Q-value.

If exploring, choose randomly among the available actions, including actions that may have lower Q-values.

Explain that exploration can still randomly select the action with the highest Q-value.

When multiple actions share the highest Q-value, break the tie randomly.

### Epsilon Decay

Start with epsilon = 1.0 by default.

Provide an adjustable decay rate.

Example:

```text
Episode 1:   epsilon = 1.00
Episode 2:   epsilon = 0.95
Episode 3:   epsilon = 0.90
...
```

Use multiplicative decay with configurable minimum epsilon.

The decay should occur at the end of each episode, not after each movement.

Students must see exploration gradually decrease as training progresses.

Allow the instructor to change epsilon manually at any time.

---

## 7. Bellman Equation Visualization

This is another critical feature.

Display the Q-learning update equation prominently:

Q(s,a) ← Q(s,a) + α [r + γ max Q(s',a') - Q(s,a)]

Do not simply display the equation.

**Show how actual numbers are substituted into the equation after every move.**

### Example

Suppose:

```text
Current state:          (0,1)
Selected action:        UP
Old Q-value:            0.00
Reward:                 -0.1
Next state's best Q:    5.00

Alpha:                  0.50
Gamma:                  0.90
```

Display the calculation:

```text
New Q = 0 + 0.5[-0.1 + 0.9(5) - 0]

New Q = 0 + 0.5(4.4)

New Q = 2.2
```

Then show:

```text
Q((0,1), UP)

OLD VALUE: 0.00
NEW VALUE: 2.20
```

Animate the corresponding Q-table cell changing from 0.00 to 2.20.

### Additional Requirements

Highlight each term in the equation as it is explained.

The interface should identify:

- Current state (s)
- Selected action (a)
- Reward (r)
- Next state (s')
- Highest Q-value in the next state
- Learning rate (alpha)
- Discount factor (gamma)

For terminal states, the future Q-value must be zero. Do not bootstrap from the next state after reaching the goal or trap.

When displaying a calculation, make the correspondence between the numerical values and Q-table cells obvious.

For example, highlight the highest Q-value in the next state's row while displaying the Bellman calculation.

---

## 8. Step-by-Step Simulation Controls

This is the application's highest-priority feature.

The instructor must have precise control over execution.

Provide the following buttons:

**Play**  
Run the simulation automatically with animations.

**Pause**  
Freeze the simulation at its current stage without advancing or changing values.

**Next Step**  
Advance exactly one logical stage of the algorithm.

**Next Action**  
Complete the current action, including its Q-value update, and stop before selecting another action.

**Next Episode**  
Complete the current episode and pause at the beginning of the following episode.

**Reset**  
Reset the robot, Q-table, episode count, parameters, and visualization to initial values.

**Restart Episode**  
Restart the current episode without clearing learned Q-values.

**Speed Control**  
Change animation speed from very slow to fast.

### Required Microsteps

The instructor should be able to advance through the following sequence individually:

1. **Observe State:** Highlight the robot's position and the corresponding Q-table row.
2. **Inspect Q-values:** Show the Q-values of available actions.
3. **Epsilon Decision:** Show epsilon, generate the random value, and decide explore or exploit.
4. **Select Action:** Highlight the selected action and explain why it was chosen.
5. **Execute Action:** Animate the robot moving.
6. **Receive Reward:** Display the observed reward and new state.
7. **Inspect Next State:** Highlight the next state's Q-values and identify their maximum.
8. **Calculate Bellman Update:** Display all terms and the completed calculation.
9. **Update Q-Table:** Animate the modified Q-value.
10. **Continue:** Begin the next action or end the episode.

If a transition ends the episode, skip future-value bootstrapping and explain why.

Each microstep should include a short plain-English explanation.

Example:

"The robot chose LEFT because epsilon caused it to explore. The robot received a reward of -0.1. We will now update its Q-value using the Bellman equation."

**Important:** No game state, random decision, or Q-value should change while the application is paused. Advancing a stage must happen only once per user command.

---

## 9. Episodes and Training

Explain and visualize the concept of an episode.

An episode starts when the robot is placed at its initial position.

An episode ends when:

- The robot reaches the goal.
- The robot enters the trap.
- The maximum number of steps is reached.

Use a default maximum of 30 actions per episode.

After an episode:

1. Display the result.
2. Show the total accumulated reward.
3. Show the number of moves.
4. Update epsilon according to the decay schedule.
5. Reset the robot to the starting position.
6. Preserve the Q-table.

The robot should gradually learn which actions lead toward the goal.

### Episode History

Display:

- Current episode number
- Current step number
- Total reward this episode
- Number of successful episodes
- Number of trap episodes
- Current epsilon

Include a small line chart showing total episode reward over time.

Students should be able to observe whether performance improves with training.

---

## 10. Visualizing How Learning Spreads Backward

This deserves special attention.

In Q-learning, a robot may initially wander because it has no information about the environment.

Eventually, it discovers the goal and receives +10.

The action leading directly to the goal receives an increased Q-value.

During later updates, previous states can gain value because they lead toward that high-value state.

Demonstrate this progression graphically.

For example, with alpha = 0.5, gamma = 0.9, and initial Q-values of zero:

```text
State immediately before goal:

Old Q = 0
Reward = +10
Next state is terminal

New Q = 0 + 0.5(10) = 5
```

Then, when learning from a prior transition into that state:

```text
Old Q = 0
Reward = -0.1
Best next Q = 5

New Q = 0 + 0.5[-0.1 + 0.9(5)]

New Q = 2.2
```

The student should see the value of reaching the goal propagate backward through earlier state-action pairs.

Use tile colors, directional arrows, or animated numerical labels to illustrate the process.

Do not imply that all previous states update automatically when a goal is reached. They update when their state-action experiences are revisited and learned from.

---

## 11. Guided Demonstration Mode

In addition to a functioning Q-learning simulator, provide a guided demonstration mode.

This is very important for classroom use.

Sometimes an instructor wants to demonstrate a particular concept without waiting through hundreds of random actions.

Provide predefined scenarios such as:

**Scenario A: Initial Exploration**

- Q-values are all zero.
- Epsilon is high.
- The agent makes random decisions.
- Show why exploration is necessary.

**Scenario B: First Reward**

- Start from a known state shortly before the goal.
- Show the robot receiving +10.
- Show its Q-value being updated.

**Scenario C: Reward Propagation**

- Use a prepared Q-table containing a valuable action near the goal.
- Show how an earlier action gets updated using the next state's Q-values.

**Scenario D: Exploration vs. Exploitation**

- Use a state with different Q-values.
- Show the epsilon-greedy decision.
- Demonstrate the difference between selecting the highest Q-value and selecting randomly.

**Scenario E: Trained Agent**

- Load a Q-table representing a successfully trained policy.
- Set epsilon to zero.
- Let the robot follow its learned policy to the goal.

For each scenario, define valid game states, rewards, Q-values, and transitions.

All displayed calculations must be mathematically correct.

Guided scenarios may use predetermined random values and saved Q-table snapshots to make demonstrations reproducible.

Clearly distinguish guided scenarios from live training. Do not falsely present a scripted trajectory as the result of an actual training run.

---

## 12. Learning Parameters

Create an instructor control panel.

Provide adjustable values for:

| Parameter | Default |
|---|---:|
| Learning rate (alpha) | 0.5 |
| Discount factor (gamma) | 0.9 |
| Initial epsilon | 1.0 |
| Epsilon decay | 0.95 |
| Minimum epsilon | 0.05 |
| Maximum episode steps | 30 |

Changing parameters must update the simulation behavior.

Include a short explanation next to each parameter.

For example:

**Alpha:** How strongly a new experience changes an existing Q-value.

**Gamma:** How much the agent values future rewards.

**Epsilon:** Probability of choosing a random action.

Avoid overwhelming students with advanced configuration.

---

## 13. User Interface Layout

Create a dashboard optimized for a 16:9 presentation screen.

Suggested layout:

```text
+----------------------------------------------------------+
|                  Q-LEARNING EXPLORER                     |
|          Episode 4    Step 7    Epsilon 0.73             |
+--------------------------+-------------------------------+
|                          |                               |
|       GAME WORLD         |           Q-TABLE             |
|                          |                               |
|       Robot              |    State  Up Down Left Right  |
|       Walls              |                               |
|       Goal               |    Highlight current state    |
|       Trap               |    Highlight chosen action    |
|                          |                               |
+--------------------------+-------------------------------+
|                                                          |
|               ACTION SELECTION                           |
|        Epsilon = 0.73     Random = 0.45                   |
|        EXPLORE -> Selected action: UP                     |
|                                                          |
+----------------------------------------------------------+
|                                                          |
|                  BELLMAN CALCULATION                     |
|                                                          |
|       Old Q + alpha[reward + gamma*maxNext - Old Q]        |
|                                                          |
|                  Result: 0.00 -> 2.20                    |
|                                                          |
+----------------------------------------------------------+
|                                                          |
| PLAY | PAUSE | NEXT STEP | NEXT ACTION | NEXT EPISODE     |
|                                                          |
+----------------------------------------------------------+
```

Use panels, subtle shadows, rounded corners, crisp typography, and purposeful animation.

The interface should remain readable when projected onto a large classroom screen.

Use consistent colors to connect concepts across the UI.

For example, the current state could be outlined in purple on both the board and Q-table, while the reward and Bellman target are highlighted in yellow.

Include an optional explanation panel showing what the current step means in plain English.

---

## 14. Implementation Architecture

Separate the simulation logic from the visualization.

Recommended components:

**Environment**

Handles:

- Game board
- Current state
- Available actions
- State transitions
- Rewards
- Episode termination

**QLearningAgent**

Handles:

- Q-table
- Epsilon-greedy action selection
- Bellman updates
- Learning parameters
- Q-value lookup

**SimulationController**

Handles:

- Play
- Pause
- Individual microsteps
- Next action
- Next episode
- Reset
- Animation speed
- Guided demonstration scenarios

**Visualization**

Handles:

- Game board rendering
- Q-table rendering
- Bellman calculation
- Epsilon-greedy selection visualization
- Episode statistics
- Animation

The Q-learning algorithm must be implemented correctly and independently of the graphical interface.

Use a deterministic, seedable random number generator for reproducible demonstrations.

Keep calculations and state changes in the simulation engine; the UI should display those results rather than independently calculating them.

---

## 15. Design for Future DQN Support

We will eventually implement Deep Q-Networks.

Do not build DQN yet.

However, the architecture should make it possible to reuse the game environment and most of the user interface.

In the future:

**Tabular Q-Learning**

```text
Current State
      |
      v
    Q-Table
      |
      v
Q-values for:
UP, DOWN, LEFT, RIGHT
```

**Deep Q-Learning**

```text
Current State
      |
      v
  Main Neural Network
      |
      v
Q-values for:
UP, DOWN, LEFT, RIGHT
```

For DQN, the state coordinates can be numerical neural network inputs, with four linear output neurons representing the four actions.

The eventual DQN implementation will add:

- Main Q-network
- Target network
- Replay buffer
- Bellman training targets
- Neural network loss
- Backpropagation
- Periodic target-network weight updates

Consider defining an interface for predicting action values so the visualization does not depend directly on a Q-table implementation.

Do not build unnecessary neural network infrastructure now.

---

## 16. Educational Accuracy

The simulator must correctly demonstrate:

- Q-values estimate expected discounted future rewards.
- Epsilon controls exploration probability.
- Exploration can select any available action.
- Exploitation chooses an action with the highest Q-value.
- Epsilon is not the learning rate.
- Alpha controls the size of Q-value updates.
- Gamma discounts future rewards.
- Q-values are updated after observing transitions.
- Terminal states do not contribute future Q-values.
- Updates do not automatically propagate through the entire trajectory.
- Q-values persist between episodes.
- Setting epsilon to zero stops random exploration but does not automatically disable learning.

Use correct Q-learning equations and numerically accurate demonstrations.

---

## 17. Priorities

Implement the project in this order:

1. Create a visually attractive game environment.
2. Implement the Q-table and display it clearly.
3. Implement correct Q-learning state transitions and updates.
4. Implement granular pause and step-through controls.
5. Implement visible epsilon-greedy decision-making.
6. Implement the animated Bellman equation.
7. Implement episode tracking and epsilon decay.
8. Implement guided teaching scenarios.
9. Add charts and visualization improvements.
10. Refine the appearance and usability.

### Acceptance Criteria

The application is successful when an instructor can:

- Start with a robot that knows nothing.
- Point to the Q-table and explain how an action is selected.
- Demonstrate epsilon-greedy exploration, including the actual random-number comparison.
- Step through an action one stage at a time.
- Pause before an action is performed.
- Watch the robot move and receive a reward.
- Point to the next state's Q-values.
- Show every numerical term in the Bellman equation.
- Advance one step and watch the correct Q-table cell change.
- Show a complete episode.
- Demonstrate how a goal reward eventually influences earlier actions.
- Set epsilon to zero and demonstrate a trained policy.
- Reset and repeat the demonstration reliably.

### Final Instruction

The visual educational experience is more important than implementing a sophisticated reinforcement learning environment.

Keep the game simple.

Make the algorithm extremely transparent.

Avoid hiding important computations behind animations.

The student should always be able to answer four questions:

1. **Why did the agent choose that action?**
2. **What happened when it took the action?**
3. **How was the new Q-value calculated?**
4. **What did the agent learn from the experience?**

Build a functional, attractive Q-learning teaching application, not just a static mockup.
