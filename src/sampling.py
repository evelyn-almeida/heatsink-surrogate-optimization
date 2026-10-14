import numpy as np
from scipy.stats import qmc

from config import MASS_LIMIT, MIN_GAP
from src.simulator import NAMES, from_unit, mass_and_gap

DIM = len(NAMES)


def lhs(n, seed):
    return qmc.LatinHypercube(d=DIM, seed=seed).random(n)


def is_feasible(u):
    mass, gap = mass_and_gap(from_unit(np.atleast_2d(u)))
    return (mass <= MASS_LIMIT) & (gap >= MIN_GAP)


def feasible_lhs(n, seed):
    """Latin hypercube candidates filtered by the (cheap) design constraints."""
    u = lhs(20 * n, seed)
    return u[is_feasible(u)][:n]


def random_feasible(n, rng):
    out = np.empty((0, DIM))
    while len(out) < n:
        u = rng.random((4 * n, DIM))
        out = np.vstack([out, u[is_feasible(u)]])
    return out[:n]
