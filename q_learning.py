import random
import numpy as np
from config import N_ACTIONS

# ── Q-table ───────────────────────────────────────────────────────
Q: dict[tuple, np.ndarray] = {}

# ── Hyperparameters ───────────────────────────────────────────────
ALPHA         = 0.12
GAMMA         = 0.95
EPSILON_MIN   = 0.05
EPSILON_DECAY = 0.994


def get_q(state: tuple) -> np.ndarray:
    """Return Q-values for a state, initialising to zeros if unseen."""
    if state not in Q:
        Q[state] = np.zeros(N_ACTIONS)
    return Q[state]


def choose_action(state: tuple, valid: list[int], epsilon: float) -> int:
    """ε-greedy action selection restricted to valid actions."""
    if random.random() < epsilon:
        return random.choice(valid)

    q_arr = get_q(state).copy()
    mask  = np.full(N_ACTIONS, -1e9)
    for a in valid:
        mask[a] = q_arr[a]
    return int(np.argmax(mask))


def update_q(state: tuple, action: int, reward: float,
             next_state: tuple, next_valid: list[int]) -> None:
    """Standard Q-learning update (Bellman equation)."""
    q_vals    = get_q(state)
    next_q    = get_q(next_state).copy()
    mask_next = np.full(N_ACTIONS, -1e9)
    for a in next_valid:
        mask_next[a] = next_q[a]
    best_next = float(np.max(mask_next))
    q_vals[action] += ALPHA * (reward + GAMMA * best_next - q_vals[action])


def decay_epsilon(epsilon: float) -> float:
    """Apply one decay step, respecting the minimum floor."""
    return max(EPSILON_MIN, epsilon * EPSILON_DECAY)