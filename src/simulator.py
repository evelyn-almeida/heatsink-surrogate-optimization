"""Physics-based thermal model of a ducted, fan-cooled plate-fin heat sink.

Stands in for an expensive CFD run. Correlations:
  - channel heat transfer: Teertstra, Yovanovich & Culham (1999)
  - developing-flow friction: Muzychka & Yovanovich (1998)
  - spreading resistance: Lee, Song, Au & Moran (1995)
"""
import numpy as np

VARIABLES = {
    "n_fins": {"bounds": (8, 60), "label": "Number of fins", "unit": "", "scale": 1},
    "fill_ratio": {"bounds": (0.15, 0.60), "label": "Fin thickness / pitch", "unit": "", "scale": 1},
    "fin_height": {"bounds": (10, 50), "label": "Fin height", "unit": "mm", "scale": 1e-3},
    "base_thickness": {"bounds": (2, 10), "label": "Base thickness", "unit": "mm", "scale": 1e-3},
    "length": {"bounds": (40, 100), "label": "Length (flow direction)", "unit": "mm", "scale": 1e-3},
}
NAMES = list(VARIABLES)
LOWER = np.array([v["bounds"][0] for v in VARIABLES.values()], dtype=float)
UPPER = np.array([v["bounds"][1] for v in VARIABLES.values()], dtype=float)

WIDTH = 0.060                 # m, fixed by the product enclosure
SOURCE_SIDE = 0.020           # m, square heat source (chip)
POWER = 60.0                  # W
T_AMBIENT = 25.0              # degC
K_AL, RHO_AL = 200.0, 2700.0  # aluminum 6063
RHO, MU, K_AIR, CP, PR = 1.184, 1.849e-5, 0.0261, 1007.0, 0.71
NU = MU / RHO
FAN_DP_MAX, FAN_FLOW_MAX = 80.0, 0.010  # Pa, m^3/s


def to_unit(x):
    return (np.asarray(x, dtype=float) - LOWER) / (UPPER - LOWER)


def from_unit(u):
    return LOWER + np.asarray(u, dtype=float) * (UPPER - LOWER)


def geometry(x):
    x = np.atleast_2d(x)
    n = np.round(x[:, 0])
    pitch = WIDTH / n
    t = x[:, 1] * pitch
    gap = (WIDTH - n * t) / (n - 1)
    H, tb, L = (x[:, i] * VARIABLES[NAMES[i]]["scale"] for i in (2, 3, 4))
    return n, t, gap, H, tb, L


def mass_and_gap(x):
    """Cheap geometric quantities (known from CAD, no simulation needed)."""
    n, t, gap, H, tb, L = geometry(x)
    return RHO_AL * (WIDTH * L * tb + n * t * H * L) * 1e3, gap * 1e3


def pressure_drop(flow, n, t, gap, H, L):
    u = flow / ((n - 1) * gap * H)
    dh = 2 * gap * H / (gap + H)
    re = np.maximum(u * dh / NU, 1e-9)
    eps = gap / H
    f_fd = 24 * (1 - 1.3553 * eps + 1.9467 * eps**2 - 1.7012 * eps**3
                 + 0.9564 * eps**4 - 0.2537 * eps**5)
    l_plus = L / (dh * re)
    f_app = np.sqrt((3.44 / np.sqrt(l_plus)) ** 2 + f_fd**2) / re
    sigma = 1 - n * t / WIDTH
    k_contraction, k_expansion = 0.42 * (1 - sigma**2), (1 - sigma**2) ** 2
    return (k_contraction + k_expansion + 4 * f_app * L / dh) * 0.5 * RHO * u**2


def fan_pressure(flow):
    return FAN_DP_MAX * (1 - (flow / FAN_FLOW_MAX) ** 2)


def operating_point(n, t, gap, H, L, iterations=60):
    """Bisection on the fan curve: fan pressure = heat sink pressure drop."""
    lo, hi = np.zeros_like(H), np.full_like(H, FAN_FLOW_MAX)
    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        above = fan_pressure(mid) > pressure_drop(mid, n, t, gap, H, L)
        lo, hi = np.where(above, mid, lo), np.where(above, hi, mid)
    return 0.5 * (lo + hi)


def spreading_resistance(h_eff, tb, L):
    a = SOURCE_SIDE / np.sqrt(np.pi)
    b = np.sqrt(WIDTH * L / np.pi)
    eps, tau, bi = a / b, tb / b, h_eff * b / K_AL
    lam = np.pi + 1 / (np.sqrt(np.pi) * eps)
    phi = (np.tanh(lam * tau) + lam / bi) / (1 + lam / bi * np.tanh(lam * tau))
    psi = eps * tau / np.sqrt(np.pi) + (1 - eps) * phi / np.sqrt(np.pi)
    return psi / (K_AL * np.pi * a)


def simulate(x):
    """x: (n, 5) array in physical units (see VARIABLES). Returns a dict of arrays."""
    n, t, gap, H, tb, L = geometry(x)
    flow = operating_point(n, t, gap, H, L)
    u = flow / ((n - 1) * gap * H)

    re_star = (u * gap / NU) * gap / L
    nu_b = ((re_star * PR / 2) ** -3
            + (0.664 * np.sqrt(re_star) * PR ** (1 / 3) * np.sqrt(1 + 3.65 / np.sqrt(re_star))) ** -3) ** (-1 / 3)
    h = nu_b * K_AIR / gap

    m = np.sqrt(2 * h / (K_AL * t))
    fin_eff = np.tanh(m * H) / (m * H)
    r_conv = 1 / (h * (n * fin_eff * 2 * H * L + (n - 1) * gap * L))
    r_air = 1 / (2 * RHO * CP * flow)
    r_base = tb / (K_AL * WIDTH * L)
    r_spread = spreading_resistance(1 / (r_conv * WIDTH * L), tb, L)

    resistance = r_conv + r_air + r_base + r_spread
    return {
        "resistance": resistance,
        "t_max": T_AMBIENT + POWER * resistance,
        "mass": mass_and_gap(x)[0],
        "flow": flow,
        "pressure_drop": pressure_drop(flow, n, t, gap, H, L),
        "gap_mm": gap * 1e3,
        "r_conv": r_conv, "r_air": r_air, "r_base": r_base, "r_spread": r_spread,
    }
