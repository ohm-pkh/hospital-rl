from enum import IntEnum

# ── Window ────────────────────────────────────────────────────────
WIDTH, HEIGHT = 1260, 800

# ── Colors ────────────────────────────────────────────────────────
BG         = (13,  15,  20)
PANEL      = (22,  25,  33)
PANEL2     = (30,  34,  44)
BORDER     = (45,  51,  65)

WHITE      = (228, 230, 236)
MUTED      = (105, 114, 135)
DIM        = (58,  65,  82)

RED        = (218,  68,  68)
RED_DIM    = ( 72,  24,  24)

ORANGE     = (218, 128,  48)
ORANGE_DIM = ( 75,  46,  12)

YELLOW     = (208, 188,  58)
YELLOW_DIM = ( 70,  64,  12)

BLUE       = ( 58, 128, 218)
BLUE_DIM   = ( 16,  42,  88)

TEAL       = ( 38, 178, 148)
TEAL_DIM   = ( 10,  62,  52)

GREEN      = ( 68, 188,  88)
GREEN_DIM  = ( 18,  62,  22)

PURPLE     = (138,  88, 218)
PINK       = (218,  88, 158)

# ── Bed config ────────────────────────────────────────────────────
ICU_BEDS   = 4
HDU_BEDS   = 4
WARD_BEDS  = 10
MAX_QUEUE  = 40

# ── Cost per admission (THB) ──────────────────────────────────────
ICU_COST   = 8000
HDU_COST   = 3500
WARD_COST  = 1200

# ── Training hyperparams ──────────────────────────────────────────
MAX_EPISODES       = 600
STEPS_PER_EP       = 100
DECISIONS_PER_STEP = 6
FPS                = 20

# ── Queue mortality model ─────────────────────────────────────────
QUEUE_DEATH = {
    3: dict(base=0.06,  rate=0.035),
    2: dict(base=0.008, rate=0.004),
    1: dict(base=0.001, rate=0.0005),
}

# ── Action space ──────────────────────────────────────────────────
class Action(IntEnum):
    ICU      = 0
    HDU      = 1
    WARD     = 2
    TRANSFER = 3
    DEFER    = 4

N_ACTIONS = 5

# ── Patient data ──────────────────────────────────────────────────
DIAGNOSES = {
    3: ["Septic shock", "Acute MI", "Resp. failure",
        "Multi-organ failure", "Haemorrhagic stroke", "Polytrauma"],
    2: ["Pneumonia", "Post-op monitoring", "Acute pancreatitis",
        "DVT w/ PE", "Cellulitis+sepsis", "Subdural haematoma"],
    1: ["UTI", "Elective post-op", "Dehydration",
        "Controlled asthma", "Hypertensive episode", "Mild chest pain"],
}

COMORBIDITIES = [
    "None", "Diabetes", "COPD", "Heart disease",
    "Renal failure", "Immunocompromised", "Obesity",
]