import pygame

from config import (
    WIDTH, HEIGHT,
    BG, PANEL, PANEL2, BORDER,
    WHITE, MUTED, DIM,
    RED, RED_DIM, ORANGE, ORANGE_DIM,
    YELLOW, BLUE, BLUE_DIM,
    TEAL, GREEN, GREEN_DIM,
    PURPLE,
    ICU_BEDS, HDU_BEDS, WARD_BEDS,
    MAX_EPISODES, STEPS_PER_EP, N_ACTIONS,
    Action,
)
from patient import Patient

# ── Fonts (initialised lazily after pygame.init()) ───────────────
font_lg = font_md = font_sm = font_xs = None


def init_fonts():
    """Call once after pygame.init() to create font objects."""
    global font_lg, font_md, font_sm, font_xs
    font_lg = pygame.font.SysFont("Segoe UI", 22, bold=True)
    font_md = pygame.font.SysFont("Segoe UI", 17)
    font_sm = pygame.font.SysFont("Segoe UI", 14)
    font_xs = pygame.font.SysFont("Segoe UI", 12)


# ── Low-level draw helpers ────────────────────────────────────────

def draw_rect(s, rect, color=PANEL, r=8):
    pygame.draw.rect(s, color, rect, border_radius=r)


def draw_border(s, rect, color=BORDER, r=8, w=1):
    pygame.draw.rect(s, color, rect, width=w, border_radius=r)


def label(s, text, pos, fnt=None, color=WHITE, align="left"):
    if fnt is None:
        fnt = font_sm
    img = fnt.render(text, True, color)
    if   align == "right":  pos = (pos[0] - img.get_width(), pos[1])
    elif align == "center": pos = (pos[0] - img.get_width() // 2, pos[1])
    s.blit(img, pos)


# ── Colour helpers ────────────────────────────────────────────────

def sev_col(sev): return {3: RED,      2: ORANGE, 1: GREEN}.get(sev, MUTED)
def sev_bg(sev):  return {3: RED_DIM,  2: ORANGE_DIM, 1: GREEN_DIM}.get(sev, DIM)
def unit_col(u):  return {"ICU": RED,  "HDU": ORANGE, "Ward": BLUE}.get(u, MUTED)
def unit_bg(u):   return {"ICU": RED_DIM, "HDU": ORANGE_DIM, "Ward": BLUE_DIM}.get(u, DIM)


# ── Compound widgets ──────────────────────────────────────────────

def draw_beds(surface, patients, capacity, x, y, bw=64, bh=52):
    cols = min(capacity, 10)
    for i in range(capacity):
        bx   = x + (i % cols) * (bw + 5)
        by   = y + (i // cols) * (bh + 5)
        rect = pygame.Rect(bx, by, bw, bh)
        if i < len(patients):
            p  = patients[i]
            sc = sev_col(p.severity)
            pygame.draw.rect(surface, sev_bg(p.severity), rect, border_radius=6)
            pygame.draw.rect(surface, sc, rect, width=1, border_radius=6)
            surface.blit(font_xs.render(f"S{p.severity}", True, sc), (bx+4, by+4))
            surface.blit(font_xs.render(f"{p.age}y", True, MUTED), (bx+4, by+bh-22))
            if p.days_needed > 0:
                fw = int(p.days_left / p.days_needed * (bw - 8))
                pygame.draw.rect(surface, sc, (bx+4, by+bh-10, fw, 5), border_radius=2)
            if p.discharge_ready:
                pygame.draw.circle(surface, YELLOW, (bx+bw-6, by+6), 4)
        else:
            pygame.draw.rect(surface, PANEL2, rect, border_radius=6)
            pygame.draw.rect(surface, BORDER, rect, width=1, border_radius=6)
            surface.blit(font_xs.render("—", True, DIM), (bx+bw//2-4, by+bh//2-7))


def draw_patient_card(surface, p: Patient, rx, ry, rw=305, rh=200):
    sc = sev_col(p.severity)
    draw_rect(surface, (rx, ry, rw, rh), sev_bg(p.severity))
    draw_border(surface, (rx, ry, rw, rh), sc)
    sev_lbl = {1: "MILD", 2: "MODERATE", 3: "CRITICAL"}[p.severity]
    label(surface, f"P#{p.pid}", (rx+12, ry+10), font_md, WHITE)
    label(surface, sev_lbl, (rx+rw-12, ry+12), font_sm, sc, align="right")
    if p.from_queue:
        label(surface, "from queue", (rx+rw//2, ry+12), font_xs, YELLOW, align="center")
    rows = [
        ("Diagnosis",      p.diagnosis),
        ("Age",            f"{p.age} yrs"),
        ("Comorbidity",    p.comorbidity),
        ("Est. LOS",       f"{p.days_needed} days"),
        ("Mortality risk", f"{p.mortality_risk*100:.0f}%"),
        ("Waited",         f"{p.wait_steps} steps"),
    ]
    for i, (k, v) in enumerate(rows):
        label(surface, k, (rx+12,    ry+38+i*24), font_xs, MUTED)
        label(surface, v, (rx+rw-12, ry+38+i*24), font_xs, WHITE, align="right")
    vitals = f"BP {p.bp_sys}/80  HR {p.hr}  SpO₂ {p.spo2}%"
    label(surface, vitals, (rx+12, ry+rh-18), font_xs, MUTED)


def draw_queue_panel(surface, queue, rx, ry, rw=310, rh=215):
    from config import QUEUE_DEATH
    draw_rect(surface, (rx, ry, rw, rh), PANEL)
    draw_border(surface, (rx, ry, rw, rh), BORDER)
    label(surface, f"Queue  ({len(queue)} patients)", (rx+10, ry+8), font_sm, YELLOW)
    if not queue:
        label(surface, "Empty", (rx+10, ry+34), font_xs, MUTED)
        return
    for i, qp in enumerate(queue[:8]):
        cy  = ry + 30 + i * 22
        col = sev_col(qp.severity)
        pygame.draw.circle(surface, col, (rx+11, cy+7), 5)
        d   = QUEUE_DEATH[qp.severity]
        dp  = min((d["base"] + d["rate"] * qp.wait_steps) * 100, 90)
        tag = "◄ NEXT" if i == 0 else ""
        info = f"#{qp.pid} S{qp.severity}  {qp.diagnosis[:14]}  w={qp.wait_steps}  ☠{dp:.0f}%  {tag}"
        label(surface, info, (rx+22, cy), font_xs, WHITE if i == 0 else MUTED)
    if len(queue) > 8:
        label(surface, f"  …+{len(queue)-8} more", (rx+10, ry+rh-18), font_xs, MUTED)


def draw_log(surface, log, rx, ry, rw, rh):
    draw_rect(surface, (rx, ry, rw, rh), PANEL)
    draw_border(surface, (rx, ry, rw, rh), BORDER)
    label(surface, "Decision log", (rx+10, ry+8), font_xs, MUTED)
    for i, line in enumerate(log[-10:]):
        col = GREEN if line.startswith("✓") else RED if line.startswith("✗") else MUTED
        label(surface, line[:60], (rx+10, ry+26+i*16), font_xs, col)


def draw_curve(surface, data, rx, ry, rw, rh):
    draw_rect(surface, (rx, ry, rw, rh), PANEL)
    draw_border(surface, (rx, ry, rw, rh), BORDER)
    label(surface, "20-episode rolling reward", (rx+10, ry+8), font_xs, MUTED)
    if len(data) < 2:
        return
    mn, mx = min(data), max(data)
    rng = max(mx - mn, 1)
    pts = [
        (rx+10 + int(i / max(len(data)-1, 1) * (rw-20)),
         ry+rh-18 - int((v - mn) / rng * (rh-32)))
        for i, v in enumerate(data[-rw:])
    ]
    if len(pts) > 1:
        pygame.draw.lines(surface, TEAL, False, pts, 2)
    if mn < 0 < mx:
        zy = ry+rh-18 - int((0 - mn) / rng * (rh-32))
        pygame.draw.line(surface, DIM, (rx+10, zy), (rx+rw-10, zy), 1)
    label(surface, f"{data[-1]:+.0f}", (rx+rw-10, ry+8), font_xs, TEAL, align="right")


# ── Main frame render ─────────────────────────────────────────────

def draw(screen, env, episode: int, avg_rewards: list,
         epsilon: float, Q: dict, csv_path: str):
    screen.fill(BG)
    p   = env.current
    occ = env.occupancy

    # Header
    label(screen, "Hospital Triage — Pure Q-Learning", (20, 13), font_lg, WHITE)
    shift_txt = "NIGHT SHIFT" if env.is_night else "Day Shift"
    label(screen, shift_txt, (WIDTH-175, 13), font_md, PURPLE if env.is_night else YELLOW)
    if env.is_mass_cas:
        label(screen, "⚠ MASS CASUALTY", (WIDTH//2, 13), font_md, RED, align="center")
    pygame.draw.line(screen, BORDER, (0, 44), (WIDTH, 44), 1)

    # Stats bar
    stats = [
        ("Episode",      f"{episode}/{MAX_EPISODES}"),
        ("Step",         f"{env.step_num}/{STEPS_PER_EP}"),
        ("Ep. reward",   f"{env.total_reward:+.0f}"),
        ("Saves",        f"{env.saves}"),
        ("Deaths total", f"{env.deaths}"),
        ("Queue deaths", f"{env.queue_deaths}"),
        ("Transfers",    f"{env.transfers}"),
        ("Accuracy",     f"{env.accuracy:.0f}%"),
        ("ε",            f"{epsilon:.3f}"),
        ("Q-states",     f"{len(Q)}"),
    ]
    sw = (WIDTH - 40) // len(stats)
    for i, (k, v) in enumerate(stats):
        sx  = 20 + i * sw
        col = (GREEN if "reward" in k and env.total_reward > 0
               else RED if "death" in k.lower() and v != "0"
               else WHITE)
        label(screen, k, (sx, 52), font_xs, MUTED)
        label(screen, v, (sx, 68), font_md, col)

    pygame.draw.line(screen, BORDER, (0, 96), (WIDTH, 96), 1)

    # Bed units
    uy = 104
    for name, patients, cap in [("ICU", env.icu, ICU_BEDS),
                                  ("HDU", env.hdu, HDU_BEDS),
                                  ("Ward", env.ward, WARD_BEDS)]:
        oc, tot = occ[name]
        label(screen, f"{name}  {oc}/{tot}", (20, uy), font_sm, unit_col(name))
        draw_beds(screen, patients, cap, 20, uy+18)
        uy += 80

    # Queue panel
    draw_queue_panel(screen, env.queue, rx=20, ry=365, rw=320, rh=220)

    # Presenting patient
    label(screen, "PRESENTING PATIENT  (◄ queue priority #1)", (350, 92), font_xs, MUTED)
    draw_patient_card(screen, p, rx=800, ry=108, rw=310, rh=200)

    # Valid actions
    valid         = env.valid_actions()
    ax, ay        = 810, 320
    action_names  = ["ICU", "HDU", "Ward", "Transfer", "Defer"]
    action_colors = [RED, ORANGE, BLUE, MUTED, YELLOW]
    label(screen, "Valid actions this step:", (ax, ay), font_xs, MUTED)
    for i, (name, col) in enumerate(zip(action_names, action_colors)):
        a        = Action(i)
        is_valid = a in valid
        bg       = unit_bg(name) if name in ("ICU", "HDU", "Ward") and is_valid else PANEL2
        c        = col if is_valid else DIM
        rect     = pygame.Rect(ax + i*60, ay+16, 52, 22)
        pygame.draw.rect(screen, bg, rect, border_radius=4)
        pygame.draw.rect(screen, c, rect, width=1, border_radius=4)
        label(screen, f"{i} {name}", (ax+i*60+4, ay+20), font_xs, c)

    # Occupancy bars
    bx, by = 810, 360
    label(screen, "Occupancy", (bx, by), font_xs, MUTED)
    for i, (name, (oc, tot)) in enumerate(occ.items()):
        col  = unit_col(name)
        fill = int(oc / tot * 150)
        label(screen, name, (bx, by+20+i*26), font_xs, col)
        pygame.draw.rect(screen, PANEL2, (bx+55, by+20+i*26, 150, 14), border_radius=4)
        pygame.draw.rect(screen, col,    (bx+55, by+20+i*26, fill, 14), border_radius=4)
        if oc / tot > 0.75:
            label(screen, f"{oc}/{tot} !", (bx+215, by+20+i*26), font_xs, RED)
        else:
            label(screen, f"{oc}/{tot}", (bx+215, by+20+i*26), font_xs, MUTED)

    # Log + curve
    draw_log(screen, env.log, rx=360, ry=365, rw=320, rh=220)
    draw_curve(screen, avg_rewards, rx=20, ry=600, rw=660, rh=130)

    # Legend
    lx, ly = 20, 740
    label(screen, "Severity:", (lx, ly), font_xs, MUTED)
    for j, (txt, col) in enumerate([("Critical", RED), ("Moderate", ORANGE), ("Mild", GREEN)]):
        pygame.draw.circle(screen, col, (lx+72+j*105, ly+6), 5)
        label(screen, txt, (lx+82+j*105, ly), font_xs, col)
    label(screen,
          "Yellow dot = discharge soon  |  ◄ NEXT = agent will decide this patient",
          (lx, ly+18), font_xs, DIM)

    # Footer
    label(screen, f"[ESC] quit  |  CSV → {csv_path}", (20, HEIGHT-20), font_xs, DIM)

    pygame.display.flip()