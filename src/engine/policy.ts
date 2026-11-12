import { bestActions, maxValue } from './agent';
import { fmtNum, fmtProb, fmtQ } from './format';
import { ACTION_NAMES, Action } from './types';

/**
 * Epsilon-greedy action selection, shared by tabular Q-learning and DQN.
 * It only needs four Q-values, whether they come from a table or a network.
 * Random numbers are passed in, so the caller controls (and records) them.
 */

export interface Decision {
  epsilon: number;
  random: number;
  randomScripted: boolean;
  explore: boolean;
  qValues: number[];
  best: Action[];
  /** Filled in at the "select" stage. */
  action?: Action;
  /** Second random number: which action to explore, or which tied action to exploit. */
  actionRandom?: number;
  actionRandomScripted?: boolean;
  /** How the action was picked, e.g. "random 0.612 × 4 = 2.45 → LEFT". */
  pickDetail?: string;
}

export interface Draw {
  value: number;
  scripted: boolean;
}

const names = (acts: Action[]) => acts.map((a) => ACTION_NAMES[a]).join(', ');

/** Stage 1 of epsilon-greedy: compare a random number with ε. */
export function epsilonDecision(qValues: number[], epsilon: number, d: Draw): { decision: Decision; explanation: string } {
  const explore = d.value < epsilon;
  const decision: Decision = {
    epsilon,
    random: d.value,
    randomScripted: d.scripted,
    explore,
    qValues: [...qValues],
    best: bestActions(qValues),
  };
  const cmp = explore ? '<' : '≥';
  const explanation =
    `EPSILON DECISION: The random number is ${fmtProb(d.value, epsilon)}${d.scripted ? ' (predetermined for this scenario)' : ''}, and ε = ${fmtProb(epsilon, d.value)}. ` +
    `${fmtProb(d.value, epsilon)} ${cmp} ${fmtProb(epsilon, d.value)}, so the robot will ${explore ? 'EXPLORE: it picks an action at random' : 'EXPLOIT: it picks the action with the highest Q-value'}.`;
  return { decision, explanation };
}

/** True when choosing the action needs a second random number. */
export function needsActionDraw(dec: Decision): boolean {
  return dec.explore || dec.best.length > 1;
}

/** Stage 2 of epsilon-greedy: pick the action (draw is required when needsActionDraw). */
export function selectAction(dec: Decision, d: Draw | null): { decision: Decision; explanation: string } {
  let action: Action;
  let pickDetail: string;
  let explanation: string;

  if (dec.explore) {
    const idx = Math.min(3, Math.floor(d!.value * 4)) as Action;
    action = idx;
    pickDetail = `random ${fmtProb(d!.value)} × 4 = ${fmtNum(d!.value * 4)} → slot ${idx} → ${ACTION_NAMES[idx]}`;
    const isBest = dec.best.includes(action) && dec.best.length < 4;
    explanation =
      `SELECT ACTION: The robot is exploring, so every one of the four actions had an equal 25% chance. A second random number picked ${ACTION_NAMES[action]}. ` +
      (isBest
        ? 'This happens to also be the highest-valued action. Exploration can still land on the best action by chance.'
        : 'Its Q-value did not matter. Exploring is how the robot tries things it does not yet know about.');
  } else if (dec.best.length > 1) {
    const idx = Math.min(dec.best.length - 1, Math.floor(d!.value * dec.best.length));
    action = dec.best[idx];
    pickDetail = `tie between ${names(dec.best)}; random ${fmtProb(d!.value)} × ${dec.best.length} = ${fmtNum(d!.value * dec.best.length)} → ${ACTION_NAMES[action]}`;
    explanation = `SELECT ACTION: The robot is exploiting, but ${names(dec.best)} are tied for the highest Q-value (${fmtQ(maxValue(dec.qValues))}). The tie is broken randomly, which gives ${ACTION_NAMES[action]}.`;
  } else {
    action = dec.best[0];
    pickDetail = `highest Q-value: ${ACTION_NAMES[action]} = ${fmtQ(dec.qValues[action])}`;
    explanation = `SELECT ACTION: The robot is exploiting, so it chooses ${ACTION_NAMES[action]}. That action has the highest Q-value in this state (${fmtQ(dec.qValues[action])}).`;
  }

  return {
    decision: { ...dec, action, actionRandom: d?.value, actionRandomScripted: d?.scripted, pickDetail },
    explanation,
  };
}
