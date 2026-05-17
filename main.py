"""
Hospital Bed Allocation — Pure RL (Refactored)
Entry point: runs the Q-learning training loop and shows post-training charts.
"""

import pygame
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from config import MAX_EPISODES, STEPS_PER_EP, DECISIONS_PER_STEP, FPS
import environment as env_module
from environment import HospitalEnv
from logger import EpisodeLogger, CSVWriter
from q_learning import Q, choose_action, update_q, decay_epsilon
from renderer import draw, init_fonts

# ── Initialise pygame ────────────────────────────────────────────
pygame.init()
init_fonts()

from config import WIDTH, HEIGHT
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Hospital Triage RL — Pure Q-Learning")
clock = pygame.time.Clock()

# ── Wire up the episode logger into the environment module ───────
ep_logger = EpisodeLogger()
env_module.ep_logger = ep_logger          # injected dependency

# ── CSV writer ───────────────────────────────────────────────────
CSV_PATH   = f"hospital_rl_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
csv_writer = CSVWriter(CSV_PATH)

# ── Training state ───────────────────────────────────────────────
reward_history: list[float] = []
avg_rewards:    list[float] = []
epsilon = 1.0

env     = HospitalEnv()
episode = 1
running = True

# =========================================
# Training loop
# =========================================

while running and episode <= MAX_EPISODES:
    state = env.reset()
    ep_logger.new_episode()
    done = False

    while not done and running:
        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

        for i in range(DECISIONS_PER_STEP):
            clock.tick(FPS)
            valid  = env.valid_actions()
            action = choose_action(state, valid, epsilon)

            next_state, reward, done = env.step(action, i)

            next_valid = env.valid_actions()
            update_q(state, action, reward, next_state, next_valid)

            state = next_state

            # Snapshot occupancy on the last decision of each step
            if i == DECISIONS_PER_STEP - 1:
                ep_logger.log_step_snapshot(env)

            draw(screen, env, episode, avg_rewards, epsilon, Q, CSV_PATH)

    reward_history.append(env.total_reward)
    if len(reward_history) >= 20:
        avg_rewards.append(float(np.mean(reward_history[-20:])))

    epsilon = decay_epsilon(epsilon)

    row = ep_logger.flush(episode, env, epsilon, Q, avg_rewards, reward_history)
    csv_writer.write_row(row)

    episode += 1

csv_writer.close()
pygame.quit()

# =========================================
# Post-training charts
# =========================================

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.patch.set_facecolor("#0D0F14")
for ax in axes:
    ax.set_facecolor("#161921")
    for sp in ax.spines.values():
        sp.set_color("#2A2D38")
    ax.tick_params(colors="#555")

ax1 = axes[0]
if avg_rewards:
    ax1.plot(avg_rewards, color="#26B294", linewidth=2)
ax1.axhline(0, color="#333", linewidth=0.8, linestyle="--")
ax1.set_title("Learning curve (20-ep avg)", color="white")
ax1.set_xlabel("Episode", color="#666")
ax1.set_ylabel("Avg reward", color="#666")

ax2 = axes[1]
ax2.plot(reward_history, color="#6666CC", linewidth=0.8, alpha=0.5, label="raw")
if len(reward_history) >= 20:
    sm = np.convolve(reward_history, np.ones(20) / 20, mode="valid")
    ax2.plot(range(19, len(reward_history)), sm, color="#FFAA44", linewidth=2, label="smooth")
ax2.legend(facecolor="#1A1D28", labelcolor="white", framealpha=0.6)
ax2.set_title("Episode rewards", color="white")
ax2.set_xlabel("Episode", color="#666")

ax3 = axes[2]
ax3.bar(["Q-states\nexplored"], [len(Q)], color="#4870C0")
ax3.set_title("Q-table size", color="white")

plt.suptitle("Hospital Bed Allocation — Pure Q-Learning Results", color="white", fontsize=14)
plt.tight_layout()
plt.show()