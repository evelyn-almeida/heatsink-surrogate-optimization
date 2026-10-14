import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.optimization import expected_improvement, pareto_mask
from src.sensitivity import sobol_indices
from src.simulator import LOWER, UPPER, from_unit, simulate, to_unit


def test_unit_scaling_roundtrip():
    x = np.array([[20, 0.3, 30, 5, 60]], dtype=float)
    assert np.allclose(from_unit(to_unit(x)), x)
    assert np.allclose(to_unit(LOWER), 0) and np.allclose(to_unit(UPPER), 1)


def test_simulator_physics():
    base = np.array([[20, 0.3, 30, 5, 60]], dtype=float)
    taller = base.copy()
    taller[0, 2] = 45
    r, r_tall = simulate(base), simulate(taller)
    assert 0.2 < r["resistance"][0] < 2.0
    assert r_tall["mass"][0] > r["mass"][0]
    assert r_tall["resistance"][0] < r["resistance"][0]   # more fin area cools better
    assert 0 < r["flow"][0] < 0.010                     # operating point lies on the fan curve


def test_expected_improvement():
    ei = expected_improvement(np.array([0.0, 1.0, 1.0]), np.array([1.0, 1.0, 1e-9]), best=0.5)
    assert ei[0] > ei[1] > 0
    assert ei[2] < 1e-6


def test_pareto_mask():
    a = np.array([1, 2, 3, 2.5])
    b = np.array([3, 2, 1, 2.5])
    assert pareto_mask(a, b).tolist() == [True, True, True, False]


def test_sobol_on_ishigami():
    def ishigami(u):
        x = -np.pi + 2 * np.pi * u
        return np.sin(x[:, 0]) + 7 * np.sin(x[:, 1]) ** 2 + 0.1 * x[:, 2] ** 4 * np.sin(x[:, 0])

    first, total = sobol_indices(ishigami, 3, 50_000, seed=0)
    assert np.allclose(first, [0.314, 0.442, 0.0], atol=0.03)
    assert np.allclose(total, [0.558, 0.442, 0.244], atol=0.03)
