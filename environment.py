import random
from patient import Patient, make_patient
from config import (
    ICU_BEDS, HDU_BEDS, WARD_BEDS, MAX_QUEUE,
    ICU_COST, HDU_COST, WARD_COST,
    STEPS_PER_EP, DECISIONS_PER_STEP,
    QUEUE_DEATH, Action,
)

# ep_logger is injected at runtime by main.py to avoid a circular import.
# Call  environment.ep_logger = <EpisodeLogger instance>  before creating HospitalEnv.
ep_logger = None  # type: ignore


class HospitalEnv:

    def __init__(self):
        self.reset()

    def reset(self):
        self.icu:  list[Patient] = []
        self.hdu:  list[Patient] = []
        self.ward: list[Patient] = []
        self.queue: list[Patient] = []

        self.step_num      = 0
        self.total_reward  = 0.0
        self.deaths        = 0
        self.queue_deaths  = 0
        self.saves         = 0
        self.transfers     = 0
        self.correct_place = 0
        self.admitted      = 0
        self.total_cost    = 0.0

        self.is_night      = False
        self.is_mass_cas   = False
        self.event_left    = 0

        self.log: list[str] = []

        self._arrive_new_patient()
        self._select_current()
        return self.get_state()

    # ── Internal helpers ──────────────────────────────────────────

    def _arrive_new_patient(self):
        p = make_patient(self.is_night, self.is_mass_cas)
        self.queue.append(p)
        self._sort_queue()

    def _sort_queue(self):
        self.queue.sort(key=lambda p: (p.severity, p.wait_steps), reverse=True)

    def _select_current(self):
        if self.queue:
            self.current = self.queue[0]
        else:
            p = make_patient(self.is_night, self.is_mass_cas)
            self.queue.append(p)
            self.current = self.queue[0]

    # ── Public API ────────────────────────────────────────────────

    def get_state(self) -> tuple:
        p    = self.current
        rest = self.queue[1:]
        max_rest_sev = max((r.severity for r in rest), default=0)
        return (
            len(self.icu),
            len(self.hdu),
            min(len(self.ward) // 2, 5),
            p.severity,
            min(p.wait_steps, 8),
            min(len(rest), 8),
            max_rest_sev,
            ICU_BEDS  - len(self.icu),
            HDU_BEDS  - len(self.hdu),
            min((WARD_BEDS - len(self.ward)) // 2, 5),
            int(self.is_night),
            int(self.is_mass_cas),
            int(p.from_queue),
        )

    def valid_actions(self) -> list[int]:
        valid = []
        p = self.current

        if len(self.icu)  < ICU_BEDS:  valid.append(Action.ICU)
        if len(self.hdu)  < HDU_BEDS:  valid.append(Action.HDU)
        if len(self.ward) < WARD_BEDS: valid.append(Action.WARD)

        valid.append(Action.TRANSFER)

        queue_rest = len(self.queue) - 1
        can_defer  = (
            queue_rest < MAX_QUEUE - 1 and
            not (p.severity == 3 and p.wait_steps >= 3)
        )
        if can_defer:
            valid.append(Action.DEFER)

        return valid

    def step(self, action: int, round_num: int) -> tuple[tuple, float, bool]:
        p = self.current

        reward  = self._apply_action(action, p)
        reward += self._tick()

        if round_num == DECISIONS_PER_STEP - 1:
            self._update_shift_and_arrivals()
            self.step_num += 1

        self.total_reward += reward

        self._sort_queue()
        self._select_current()

        done = self.step_num >= STEPS_PER_EP
        return self.get_state(), reward, done

    # ── Action handler ────────────────────────────────────────────

    def _apply_action(self, action: int, p: Patient) -> float:
        sev  = p.severity
        risk = p.mortality_risk
        wait = p.wait_steps

        if action != Action.DEFER:
            if self.queue and self.queue[0] is p:
                self.queue.pop(0)

        wait_penalty = wait * {3: 8, 2: 4, 1: 2}[sev]

        if action == Action.ICU:
            if len(self.icu) >= ICU_BEDS:
                self.queue.insert(0, p)
                self.log.append(f"[!] P#{p.pid} ICU FULL — re-queued")
                ep_logger.log_overflow("icu")
                return -80.0
            self.icu.append(p)
            self.admitted   += 1
            self.total_cost += ICU_COST * p.days_needed
            if sev == 3:
                r = 110 + risk * 70 - wait_penalty
                self.saves += 1
                self.correct_place += 1
                self.log.append(f"✓ P#{p.pid} {p.diagnosis} → ICU  (critical ✓)")
            elif sev == 2:
                r = 10 - wait_penalty
                self.log.append(f"  P#{p.pid} → ICU (moderate, slight overkill)")
            else:
                r = -55 - wait_penalty
                self.log.append(f"✗ P#{p.pid} → ICU wasted (mild patient)")
            ep_logger.log_decision(p, action)
            return r

        elif action == Action.HDU:
            if len(self.hdu) >= HDU_BEDS:
                self.queue.insert(0, p)
                self.log.append(f"[!] P#{p.pid} HDU FULL — re-queued")
                ep_logger.log_overflow("hdu")
                return -55.0
            self.hdu.append(p)
            self.admitted   += 1
            self.total_cost += HDU_COST * p.days_needed
            if sev == 3:
                r = 50 + risk * 25 - wait_penalty
                self.saves += 1
                self.log.append(f"  P#{p.pid} → HDU (critical, sub-optimal)")
            elif sev == 2:
                r = 80 - wait_penalty
                self.saves += 1
                self.correct_place += 1
                self.log.append(f"✓ P#{p.pid} → HDU  (moderate ✓)")
            else:
                r = -12 - wait_penalty
                self.log.append(f"  P#{p.pid} → HDU (mild, minor waste)")
            ep_logger.log_decision(p, action)
            return r

        elif action == Action.WARD:
            if len(self.ward) >= WARD_BEDS:
                self.queue.insert(0, p)
                self.log.append(f"[!] P#{p.pid} Ward FULL — re-queued")
                ep_logger.log_overflow("ward")
                return -45.0
            self.ward.append(p)
            self.admitted   += 1
            self.total_cost += WARD_COST * p.days_needed
            if sev == 3:
                r    = -110 - wait_penalty
                died = random.random() < risk * 0.85
                if died:
                    self.deaths += 1
                    self.ward.remove(p)
                    self.log.append(f"✗ P#{p.pid} {p.diagnosis} → Ward DIED (critical)")
                else:
                    self.log.append(f"✗ P#{p.pid} → Ward (critical, survived by luck)")
                ep_logger.log_decision(p, action, outcome_death=died)
            elif sev == 2:
                r = 38 - wait_penalty
                self.log.append(f"  P#{p.pid} → Ward (moderate)")
                ep_logger.log_decision(p, action)
            else:
                r = 70 - wait_penalty
                self.correct_place += 1
                self.log.append(f"✓ P#{p.pid} → Ward  (mild ✓)")
                ep_logger.log_decision(p, action)
            return r

        elif action == Action.TRANSFER:
            self.transfers += 1
            if sev == 3:
                r    = -110 - wait_penalty
                died = random.random() < risk * 0.55
                if died:
                    self.deaths += 1
                self.log.append(f"✗ P#{p.pid} transferred out (critical!)")
                ep_logger.log_decision(p, action, outcome_death=died)
            elif sev == 2:
                r = -30
                self.log.append(f"  P#{p.pid} transferred out (moderate)")
                ep_logger.log_decision(p, action)
            else:
                r = 6
                self.log.append(f"  P#{p.pid} transferred out (mild, ok)")
                ep_logger.log_decision(p, action)
            return r

        elif action == Action.DEFER:
            ep_logger.log_decision(p, action)
            if sev == 3:
                r = -90
                self.log.append(f"  P#{p.pid} DEFERRED (critical — risky!)")
            elif sev == 2:
                r = -30
                self.log.append(f"  P#{p.pid} deferred (moderate)")
            else:
                r = -10
                self.log.append(f"  P#{p.pid} deferred (mild)")
            return r

        return 0.0

    # ── Tick (per-decision world update) ─────────────────────────

    def _tick(self) -> float:
        for unit in (self.icu, self.hdu, self.ward):
            for p in unit:
                p.days_left = max(0, p.days_left - 1)
                if p.days_left <= p.days_needed // 3 and random.random() < 0.12:
                    p.discharge_ready = True
            unit[:] = [p for p in unit if p.days_left > 0]

        reward_delta = 0.0
        survived     = []
        for qp in self.queue:
            qp.wait_steps += 1
            qp.from_queue  = True

            d          = QUEUE_DEATH[qp.severity]
            death_prob = min(d["base"] + d["rate"] * qp.wait_steps, 0.90)

            if random.random() < death_prob:
                self.deaths       += 1
                self.queue_deaths += 1
                penalty            = {3: -150, 2: -100, 1: -70}[qp.severity]
                reward_delta      += penalty
                self.log.append(
                    f"✗ P#{qp.pid} DIED IN QUEUE  sev={qp.severity} waited={qp.wait_steps}s"
                )
                ep_logger.log_queue_death(qp)
            else:
                survived.append(qp)

        self.queue = survived
        return reward_delta

    # ── Shift / arrival update (once per step) ───────────────────

    def _update_shift_and_arrivals(self):
        hour          = self.step_num % 24
        self.is_night = (hour >= 18 or hour < 6)

        if self.event_left > 0:
            self.event_left  -= 1
            self.is_mass_cas  = True
        elif random.random() < 0.025:
            self.is_mass_cas = True
            self.event_left  = random.randint(3, 6)
        else:
            self.is_mass_cas = False

        n_arrivals = 5
        if self.is_mass_cas:
            n_arrivals = random.randint(6, 12)
        elif self.is_night:
            n_arrivals = random.choices([1, 2, 3, 4], weights=[20, 40, 25, 15])[0]

        for _ in range(n_arrivals):
            if len(self.queue) < MAX_QUEUE + 2:
                self._arrive_new_patient()

    # ── Properties ───────────────────────────────────────────────

    @property
    def occupancy(self):
        return {
            "ICU":  (len(self.icu),  ICU_BEDS),
            "HDU":  (len(self.hdu),  HDU_BEDS),
            "Ward": (len(self.ward), WARD_BEDS),
        }

    @property
    def accuracy(self) -> float:
        return (self.correct_place / self.admitted * 100) if self.admitted else 0.0