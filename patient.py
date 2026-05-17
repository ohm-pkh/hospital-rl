import random
from dataclasses import dataclass
from config import DIAGNOSES, COMORBIDITIES

_pid_counter = [1000]


@dataclass
class Patient:
    pid:          int
    severity:     int
    age:          int
    diagnosis:    str
    comorbidity:  str
    days_needed:  int
    days_left:    int   = 0
    wait_steps:   int   = 0
    from_queue:   bool  = False
    discharge_ready: bool = False

    bp_sys: int   = 120
    hr:     int   = 80
    spo2:   int   = 98
    mortality_risk: float = 0.0

    def __post_init__(self):
        self.days_left = self.days_needed
        if self.severity == 3:
            self.bp_sys = random.randint(60, 90)
            self.hr     = random.randint(110, 145)
            self.spo2   = random.randint(80, 92)
            self.mortality_risk = round(0.28 + random.random() * 0.42, 2)
        elif self.severity == 2:
            self.bp_sys = random.randint(90, 112)
            self.hr     = random.randint(95, 115)
            self.spo2   = random.randint(90, 96)
            self.mortality_risk = round(0.05 + random.random() * 0.18, 2)
        else:
            self.bp_sys = random.randint(110, 145)
            self.hr     = random.randint(68, 95)
            self.spo2   = random.randint(96, 100)
            self.mortality_risk = round(0.01 + random.random() * 0.06, 2)

        if self.comorbidity != "None":
            self.mortality_risk = min(0.99, self.mortality_risk + 0.08)


def make_patient(night: bool = False, mass_cas: bool = False) -> Patient:
    _pid_counter[0] += 1
    if mass_cas:
        sev = random.choices([1, 2, 3], weights=[10, 38, 62])[0]
    elif night:
        sev = random.choices([1, 2, 3], weights=[25, 40, 35])[0]
    else:
        sev = random.choices([1, 2, 3], weights=[45, 35, 20])[0]

    los = {3: random.randint(5, 14), 2: random.randint(3, 8), 1: random.randint(1, 4)}[sev]

    return Patient(
        pid=_pid_counter[0], severity=sev,
        age=random.randint(18, 92),
        diagnosis=random.choice(DIAGNOSES[sev]),
        comorbidity=random.choice(COMORBIDITIES),
        days_needed=los,
    )