# Hospital Bed Allocation — Reinforcement Learning

*Artificial Intelligence: Mini-Project*

---

## Introduction

Hospitals face three core challenges simultaneously: **limited beds**, **unpredictable patient flow**, and **complex triage decisions**. Emergency beds are scarce — wrong allocation blocks critical patients from life-saving care. Patient arrivals vary drastically across normal shifts, night shifts, and mass-casualty events. And every triage decision must balance severity, mortality risk, comorbidities, wait time, and bed availability all at once.

This project applies **tabular Q-learning** to train an autonomous agent that learns to assign incoming patients to the correct care unit — with no hard-coded rules — purely through trial and error across 600 episodes.

### Project Objectives

- **Minimize patient mortality** — reduce deaths in queue, in ward (critical misplacement), and during transfers by learning optimal triage priority.
- **Correct placement accuracy** — match patient severity (Critical → ICU, Moderate → HDU, Mild → Ward) with >75% accuracy.
- **Maximize bed utilization** — allocate ICU, HDU, and Ward beds efficiently, preventing overflow while avoiding unnecessary high-acuity placement.
- **Improve over time** — learn from 600 episodes using tabular Q-learning with ε-greedy exploration, maximizing long-run cumulative reward.

## Project Structure

```
hospital_rl/
├── main.py          # Entry point — training loop and post-training charts
├── config.py        # All constants: colors, bed sizes, costs, hyperparams
├── patient.py       # Patient dataclass and make_patient() factory
├── q_learning.py    # Q-table, choose_action(), update_q(), decay_epsilon()
├── environment.py   # HospitalEnv — state, actions, rewards, world tick
├── logger.py        # EpisodeLogger, CSVWriter, CSV column schema
├── renderer.py      # All pygame drawing functions and the main draw() frame
└── requirements.txt # Python dependencies
```

---

## Setup

### Requirements

- Python **3.10 or later**
- A monitor — pygame opens a 1260 × 800 window

### 1. Clone or download

```bash
git clone https://github.com/ohm-pkh/hospital-rl.git
cd hospital-rl
```

Or place all the `.py` files and `requirements.txt` in the same folder.

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it:

```bash
# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** `requirements.txt` uses `pygame-ce` (Community Edition). If you have the original `pygame` installed, remove it first:
> ```bash
> pip uninstall pygame
> pip install -r requirements.txt
> ```

### 4. Run

```bash
python main.py
```

Press **ESC** or close the window to stop early — the CSV log up to that point is still saved.

---

## Controls

| Key / Action | Effect |
|---|---|
| `ESC` | Stop training and save CSV |
| Close window | Same as ESC |

---

## Output

- **Live pygame window** — real-time bed occupancy, presenting patient details, queue state, decision log, and rolling reward curve.
- **CSV log** — `hospital_rl_YYYYMMDD_HHMMSS.csv` written to the working directory, with 50+ metrics per episode (mortality, accuracy, action distribution, financials, queue dynamics).
- **Post-training charts** — learning curve, episode rewards, and Q-table size shown automatically after training ends.