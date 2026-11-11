"""The repressilator with three transcriptional reporters: paper Eqs. (1)-(4).

    reporter  promoter  repressor  protein      maturation  tag (half-life)
    lac       P_Llac    LacI       mJuniper     1.7 min     LVA (40 min)
    tet       P_Ltet    TetR       GFPmut3      4.1 min     AAV (60 min)
    cI        P_R       CI         mScarlet-I3  2.0 min     LVA (40 min)

Units: time in mRNA lifetimes (2.885 min), protein in units of K_M.
State (15): m_lacI, m_tetR, m_cI | p_LacI, p_TetR, p_CI | (r, D, F) of the lac, tet and cI reporters.
"""

import math

import torch

TAU_M_MIN = 2.0 / math.log(2.0)  # model time unit (mRNA lifetime) in minutes


def rate(half_time_min: float) -> float:
    """First-order rate in model time units from a half-time in minutes."""
    return math.log(2.0) * TAU_M_MIN / half_time_min


REPORTERS = ("lac", "tet", "cI")
FP_BRIGHTNESS = {"lac": 13.76, "tet": 34.87, "cI": 68.25}  # FPbase
K_MAT = torch.tensor([rate(1.7), rate(4.1), rate(2.0)], dtype=torch.float64)  # maturation (FPbase)
DELTA = torch.tensor([rate(40.0), rate(60.0), rate(40.0)], dtype=torch.float64)  # tag decay (Andersen 1998)

# Which protein represses which promoter, as indices 0..2 into (LacI, TetR, CI).
GENE_REPRESSOR = [2, 0, 1]  # lacI <- CI, tetR <- LacI, cI <- TetR
REPORTER_REPRESSOR = [0, 1, 2]  # lac <- LacI, tet <- TetR, cI <- CI

N_STATE = 15
M_IDX = [0, 1, 2]
P_IDX = [3, 4, 5]
R_IDX = [6, 9, 12]
D_IDX = [7, 10, 13]
F_IDX = [8, 11, 14]


def repression(y: torch.Tensor, p: dict, present: torch.Tensor) -> torch.Tensor:
    """Hill term 1 / (1 + (p_j / K_j)^n) of Eqs. (1) and (3), per repressor (LacI, TetR, CI)."""
    K = 1.0 + p["tau"] * present  # titration: K_j = 1 + tau_j if reporter j is built
    protein = y[..., P_IDX].clamp_min(0.0)
    return 1.0 / (1.0 + (protein / K) ** p["n"].unsqueeze(-1))


def rhs(y: torch.Tensor, p: dict, present: torch.Tensor) -> torch.Tensor:
    """d(state)/dt ``(..., 15)``; ``p`` holds alpha, alpha_0, n, beta and per-reporter a, a_0, tau."""
    h = repression(y, p, present)
    alpha, alpha_0 = p["alpha"].unsqueeze(-1), p["alpha_0"].unsqueeze(-1)
    beta = p["beta"].unsqueeze(-1)
    k, delta = K_MAT.to(y), DELTA.to(y)

    m, prot = y[..., M_IDX], y[..., P_IDX]
    dm = -m + alpha * h[..., GENE_REPRESSOR] + alpha_0  # Eq. (1)
    dp = -beta * (prot - m)  # Eq. (2)

    r, D, F = y[..., R_IDX], y[..., D_IDX], y[..., F_IDX]
    dr = -r + p["a"] * h[..., REPORTER_REPRESSOR] + p["a_0"]  # Eq. (3)
    dD = delta * r - (k + delta) * D  # Eq. (4), dark protein
    dF = k * D - delta * F  # Eq. (4), mature protein: what the camera sees

    return torch.cat([dm, dp, torch.stack([dr, dD, dF], -1).flatten(-2)], -1)  # the state layout above


# The same model as a reaction network in molecule counts (for Gillespie). omega = K_M in molecules.
# Counts per model unit: proteins omega, circuit mRNA omega beta / EFF, reporter mRNA omega delta / EFF;
# with these scales the network's drift equals rhs() (tested).

EFF = 20.0  # proteins per transcript


def count_scales(p: dict, omega: float) -> torch.Tensor:
    """Molecules per model unit, for each of the 15 species."""
    s = torch.full((N_STATE,), float(omega), dtype=torch.float64)
    s[M_IDX] = omega * float(p["beta"]) / EFF
    s[R_IDX] = omega * DELTA / EFF
    return s


def stoichiometry() -> torch.Tensor:
    """Change in each species per reaction, ``(30, 15)``, in :func:`propensities` order."""
    changes = []
    for i in range(3):  # circuit gene i: transcription, mRNA decay, translation, protein decay
        m, pr = M_IDX[i], P_IDX[i]
        changes += [{m: 1}, {m: -1}, {pr: 1}, {pr: -1}]
    for j in range(3):  # reporter j: transcription, mRNA decay, translation, maturation, 2 decays
        r, D, F = R_IDX[j], D_IDX[j], F_IDX[j]
        changes += [{r: 1}, {r: -1}, {D: 1}, {D: -1, F: 1}, {D: -1}, {F: -1}]
    S = torch.zeros(len(changes), N_STATE, dtype=torch.float64)
    for row, change in enumerate(changes):
        for species, delta_count in change.items():
            S[row, species] = delta_count
    return S


def propensities(x: torch.Tensor, p: dict, present: torch.Tensor, omega: float) -> torch.Tensor:
    """Reaction rates (events per model time unit) for counts ``x`` ``(B, 15)`` -> ``(B, 30)``."""
    x = x.clamp_min(0.0)
    h = repression(x / count_scales(p, omega), p, present)  # (B, 3)
    beta = p["beta"]
    rates = []
    for i in range(3):
        s_m = omega * beta / EFF
        m, pr = x[:, M_IDX[i]], x[:, P_IDX[i]]
        rates += [s_m * (p["alpha"] * h[:, GENE_REPRESSOR[i]] + p["alpha_0"]),  # transcription
                  m,  # mRNA decay
                  EFF * m,  # translation
                  beta * pr]  # protein decay
    for j in range(3):
        k, delta = float(K_MAT[j]), float(DELTA[j])
        s_r = omega * delta / EFF
        r, D, F = x[:, R_IDX[j]], x[:, D_IDX[j]], x[:, F_IDX[j]]
        rates += [s_r * (p["a"][j] * h[:, REPORTER_REPRESSOR[j]] + p["a_0"][j]),  # transcription
                  r,  # mRNA decay
                  EFF * r,  # translation (dark)
                  k * D,  # maturation
                  delta * D,  # dark decay
                  delta * F]  # fluorescent decay
    return torch.stack(torch.broadcast_tensors(*rates), dim=-1)
