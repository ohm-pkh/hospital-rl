import csv
import os
from datetime import datetime

import numpy as np

from config import N_ACTIONS, STEPS_PER_EP, Action
from patient import Patient

# =========================================
# CSV column schema
# =========================================

CSV_COLUMNS = [
    # ── Episode metadata ──────────────────────────────────────────
    "episode",
    "timestamp",

    # ── RL training state ─────────────────────────────────────────
    "epsilon",
    "q_states_explored",
    "avg_q_value_icu",
    "avg_q_value_hdu",
    "avg_q_value_ward",
    "avg_q_value_transfer",
    "avg_q_value_defer",

    # ── Reward signals ────────────────────────────────────────────
    "total_reward",
    "reward_per_step",
    "rolling_avg_20ep",

    # ── Outcome: mortality ────────────────────────────────────────
    "deaths_total",
    "deaths_in_queue",
    "deaths_in_ward",
    "deaths_on_transfer",
    "death_rate",
    "critical_death_rate",

    # ── Outcome: triage quality ───────────────────────────────────
    "saves",
    "correct_placements",
    "placement_accuracy",
    "transfers_total",
    "deferrals_total",

    # ── Action distribution ───────────────────────────────────────
    "pct_action_icu",
    "pct_action_hdu",
    "pct_action_ward",
    "pct_action_transfer",
    "pct_action_defer",

    # ── Bed utilisation ───────────────────────────────────────────
    "admitted_total",
    "avg_icu_occupancy",
    "avg_hdu_occupancy",
    "avg_ward_occupancy",
    "peak_icu_occupancy",
    "peak_hdu_occupancy",
    "peak_ward_occupancy",
    "icu_overflow_attempts",
    "hdu_overflow_attempts",
    "ward_overflow_attempts",

    # ── Queue dynamics ────────────────────────────────────────────
    "avg_queue_length",
    "peak_queue_length",
    "avg_wait_steps_at_decision",
    "avg_wait_steps_at_death",

    # ── Patient mix ───────────────────────────────────────────────
    "patients_seen_total",
    "pct_severity_3",
    "pct_severity_2",
    "pct_severity_1",
    "pct_from_queue",
    "night_shift_steps",
    "mass_casualty_steps",

    # ── Financial ─────────────────────────────────────────────────
    "total_cost_thb",
    "cost_per_save",
    "cost_per_admitted",
]


# =========================================
# EpisodeLogger
# =========================================

class EpisodeLogger:
    """Accumulates per-step stats during an episode, flushes to CSV at end."""

    def __init__(self):
        self._reset()

    def _reset(self):
        self.icu_occ_snapshots  = []
        self.hdu_occ_snapshots  = []
        self.ward_occ_snapshots = []
        self.queue_len_snapshots = []

        self.action_counts = {a: 0 for a in range(N_ACTIONS)}
        self.overflow = {"icu": 0, "hdu": 0, "ward": 0}

        self.wait_at_decision    = []
        self.wait_at_queue_death = []

        self.sev_counts      = {1: 0, 2: 0, 3: 0}
        self.from_queue_count = 0
        self.patients_seen    = 0

        self.deaths_ward     = 0
        self.deaths_transfer = 0
        self.critical_presented = 0
        self.critical_died      = 0

        self.night_steps    = 0
        self.mass_cas_steps = 0
        self.deferrals      = 0

    def new_episode(self):
        self._reset()

    def log_step_snapshot(self, env) -> None:
        """Call once per env step (after tick) to record occupancy/queue."""
        self.icu_occ_snapshots.append(len(env.icu))
        self.hdu_occ_snapshots.append(len(env.hdu))
        self.ward_occ_snapshots.append(len(env.ward))
        self.queue_len_snapshots.append(len(env.queue))
        if env.is_night:
            self.night_steps += 1
        if env.is_mass_cas:
            self.mass_cas_steps += 1

    def log_decision(self, patient: Patient, action: int,
                     outcome_death: bool = False) -> None:
        """Call each time the agent makes a decision."""
        self.action_counts[action] += 1
        self.wait_at_decision.append(patient.wait_steps)
        self.patients_seen += 1
        self.sev_counts[patient.severity] += 1
        if patient.from_queue:
            self.from_queue_count += 1
        if patient.severity == 3:
            self.critical_presented += 1
        if action == Action.DEFER:
            self.deferrals += 1
        if action == Action.WARD and patient.severity == 3 and outcome_death:
            self.deaths_ward += 1
            self.critical_died += 1
        if action == Action.TRANSFER and patient.severity == 3 and outcome_death:
            self.deaths_transfer += 1
            self.critical_died += 1

    def log_overflow(self, unit: str) -> None:
        self.overflow[unit] += 1

    def log_queue_death(self, patient: Patient) -> None:
        self.wait_at_queue_death.append(patient.wait_steps)
        if patient.severity == 3:
            self.critical_died += 1

    def flush(self, episode: int, env,
              epsilon: float, Q: dict,
              avg_rewards: list, reward_history: list) -> dict:
        """Build and return the row dict for this episode."""
        total_decisions = max(sum(self.action_counts.values()), 1)

        def pct_action(a):
            return round(self.action_counts[a] / total_decisions * 100, 2)

        avg_q = {a: 0.0 for a in range(N_ACTIONS)}
        if Q:
            for q_arr in Q.values():
                for a in range(N_ACTIONS):
                    avg_q[a] += q_arr[a]
            n = len(Q)
            for a in avg_q:
                avg_q[a] = round(avg_q[a] / n, 4)

        total_patients     = env.admitted + env.transfers
        death_rate         = round(env.deaths / max(total_patients, 1), 4)
        critical_death_rate = round(
            self.critical_died / max(self.critical_presented, 1), 4
        )

        return {
            "episode":                    episode,
            "timestamp":                  datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "epsilon":                    round(epsilon, 5),
            "q_states_explored":          len(Q),
            "avg_q_value_icu":            avg_q[Action.ICU],
            "avg_q_value_hdu":            avg_q[Action.HDU],
            "avg_q_value_ward":           avg_q[Action.WARD],
            "avg_q_value_transfer":       avg_q[Action.TRANSFER],
            "avg_q_value_defer":          avg_q[Action.DEFER],
            "total_reward":               round(env.total_reward, 2),
            "reward_per_step":            round(env.total_reward / STEPS_PER_EP, 4),
            "rolling_avg_20ep":           round(avg_rewards[-1], 2) if avg_rewards else "",
            "deaths_total":               env.deaths,
            "deaths_in_queue":            env.queue_deaths,
            "deaths_in_ward":             self.deaths_ward,
            "deaths_on_transfer":         self.deaths_transfer,
            "death_rate":                 death_rate,
            "critical_death_rate":        critical_death_rate,
            "saves":                      env.saves,
            "correct_placements":         env.correct_place,
            "placement_accuracy":         round(env.accuracy, 2),
            "transfers_total":            env.transfers,
            "deferrals_total":            self.deferrals,
            "pct_action_icu":             pct_action(Action.ICU),
            "pct_action_hdu":             pct_action(Action.HDU),
            "pct_action_ward":            pct_action(Action.WARD),
            "pct_action_transfer":        pct_action(Action.TRANSFER),
            "pct_action_defer":           pct_action(Action.DEFER),
            "admitted_total":             env.admitted,
            "avg_icu_occupancy":          round(np.mean(self.icu_occ_snapshots),  2) if self.icu_occ_snapshots  else 0,
            "avg_hdu_occupancy":          round(np.mean(self.hdu_occ_snapshots),  2) if self.hdu_occ_snapshots  else 0,
            "avg_ward_occupancy":         round(np.mean(self.ward_occ_snapshots), 2) if self.ward_occ_snapshots else 0,
            "peak_icu_occupancy":         max(self.icu_occ_snapshots,  default=0),
            "peak_hdu_occupancy":         max(self.hdu_occ_snapshots,  default=0),
            "peak_ward_occupancy":        max(self.ward_occ_snapshots, default=0),
            "icu_overflow_attempts":      self.overflow["icu"],
            "hdu_overflow_attempts":      self.overflow["hdu"],
            "ward_overflow_attempts":     self.overflow["ward"],
            "avg_queue_length":           round(np.mean(self.queue_len_snapshots), 2) if self.queue_len_snapshots else 0,
            "peak_queue_length":          max(self.queue_len_snapshots, default=0),
            "avg_wait_steps_at_decision": round(np.mean(self.wait_at_decision),   2) if self.wait_at_decision    else 0,
            "avg_wait_steps_at_death":    round(np.mean(self.wait_at_queue_death), 2) if self.wait_at_queue_death else 0,
            "patients_seen_total":        self.patients_seen,
            "pct_severity_3":             round(self.sev_counts[3] / max(self.patients_seen, 1) * 100, 2),
            "pct_severity_2":             round(self.sev_counts[2] / max(self.patients_seen, 1) * 100, 2),
            "pct_severity_1":             round(self.sev_counts[1] / max(self.patients_seen, 1) * 100, 2),
            "pct_from_queue":             round(self.from_queue_count / max(self.patients_seen, 1) * 100, 2),
            "night_shift_steps":          self.night_steps,
            "mass_casualty_steps":        self.mass_cas_steps,
            "total_cost_thb":             round(env.total_cost, 0),
            "cost_per_save":              round(env.total_cost / max(env.saves, 1), 0),
            "cost_per_admitted":          round(env.total_cost / max(env.admitted, 1), 0),
        }


# =========================================
# CSVWriter
# =========================================

class CSVWriter:
    """Opens the CSV once, writes header, then appends a row per episode."""

    def __init__(self, path: str):
        self.path   = path
        self._file  = open(path, "w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=CSV_COLUMNS)
        self._writer.writeheader()
        self._file.flush()
        print(f"[CSV] Logging to: {os.path.abspath(path)}")

    def write_row(self, row: dict) -> None:
        self._writer.writerow(row)
        self._file.flush()

    def close(self) -> None:
        self._file.close()
        print(f"[CSV] Saved: {self.path}")