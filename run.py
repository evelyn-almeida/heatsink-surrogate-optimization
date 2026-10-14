"""End-to-end study: python run.py [--quick]"""
import argparse
import json
import time
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.metrics import r2_score

from config import BASELINE, MASS_LIMIT, MIN_GAP, MODES, N_INIT, RESULTS_DIR, SEED
from src import plots
from src.optimization import bayesian_optimization, expected_improvement, pareto_mask, random_search
from src.sampling import DIM, lhs
from src.sensitivity import sobol_indices
from src.simulator import VARIABLES, from_unit, mass_and_gap, simulate, to_unit
from src.surrogates import SURROGATES, gaussian_process

warnings.filterwarnings("ignore")
to_temp = plots.to_temp


def objective(u):
    """One 'expensive simulation' per row of u (unit-cube coordinates)."""
    return simulate(from_unit(np.atleast_2d(u)))["resistance"]


def true_optimum():
    """Brute-force reference: differential evolution directly on the simulator."""
    history = []

    def penalized(u):
        r = simulate(from_unit(u[None]))
        mass, gap, res = r["mass"][0], r["gap_mm"][0], r["resistance"][0]
        history.append((res if mass <= MASS_LIMIT and gap >= MIN_GAP else np.inf, u.copy()))
        return res + 0.01 * max(0, mass - MASS_LIMIT) + max(0, MIN_GAP - gap)

    differential_evolution(penalized, [(0, 1)] * DIM, seed=SEED, popsize=30, tol=1e-8, maxiter=300)
    values = np.array([h[0] for h in history])
    best = int(np.argmin(values))
    target = values[best] + 0.1 / plots.POWER
    evals_to_target = int(np.argmax(np.minimum.accumulate(values) <= target)) + 1
    return history[best][1], values[best], evals_to_target


def learning_curves(sizes, n_seeds, u_test, y_test):
    rows = []
    for n in sizes:
        for seed in range(n_seeds):
            u = lhs(n, seed)
            y = objective(u)
            for name, make in SURROGATES.items():
                pred = np.exp(make().fit(u, np.log(y)).predict(u_test))
                rows.append({"model": name, "n": n, "seed": seed,
                             "mape": 100 * np.mean(np.abs(pred - y_test) / y_test),
                             "r2": r2_score(y_test, pred)})
    return pd.DataFrame(rows)


def bo_1d_states(u_star, iterations=6):
    grid = np.linspace(0, 1, 300)

    def slice_temp(g):
        u = np.tile(u_star, (len(g), 1))
        u[:, 0] = g
        return to_temp(objective(u))

    x, y = np.array([0.05, 0.5, 0.95]), slice_temp(np.array([0.05, 0.5, 0.95]))
    kernel = ConstantKernel() * Matern(0.3, (0.1, 10), nu=2.5) + WhiteKernel(1e-6, (1e-10, 1e-2))
    states = []
    for _ in range(iterations):
        gp = GaussianProcessRegressor(kernel, normalize_y=True, n_restarts_optimizer=3, random_state=SEED)
        mu, sd = gp.fit(x[:, None], y).predict(grid[:, None], return_std=True)
        ei = expected_improvement(mu, sd, y.min())
        nxt = grid[np.argmax(ei)]
        states.append({"x": from_unit_1d(x), "y": y.copy(), "mu": mu, "sd": sd, "ei": ei, "next": from_unit_1d(nxt)})
        x, y = np.append(x, nxt), np.append(y, slice_temp(np.array([nxt])))
    return from_unit_1d(grid), slice_temp(grid), states


def from_unit_1d(g):
    lo, hi = VARIABLES["n_fins"]["bounds"]
    return lo + np.asarray(g) * (hi - lo)


def describe(x):
    r = simulate(np.asarray(x, dtype=float)[None])
    return {"x": list(map(float, x)), "t_max": float(r["t_max"][0]), **{k: float(v[0]) for k, v in r.items()}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="fewer runs, for fast iteration")
    mode = MODES["quick" if parser.parse_args().quick else "full"]
    RESULTS_DIR.mkdir(exist_ok=True)
    clock = time.time()

    u_star, r_star, de_evals = true_optimum()
    x_star = from_unit(u_star)
    print(f"[{time.time() - clock:5.0f}s] true optimum: {to_temp(r_star):.2f} degC, DE needed {de_evals} runs")

    u_test = lhs(2000, SEED + 1)
    y_test = objective(u_test)
    lc = learning_curves(mode["train_sizes"], mode["lc_seeds"], u_test, y_test)
    lc.to_csv(RESULTS_DIR / "learning_curves.csv", index=False, float_format="%.4f")
    print(f"[{time.time() - clock:5.0f}s] learning curves")

    u80 = lhs(80, 0)
    gp80 = gaussian_process().fit(u80, np.log(objective(u80)))
    rsm80 = SURROGATES["Quadratic response surface"]().fit(u80, np.log(objective(u80)))
    mu, sd = gp80.predict(u_test, return_std=True)
    gp_coverage = float(np.mean((y_test >= np.exp(mu - 1.645 * sd)) & (y_test <= np.exp(mu + 1.645 * sd))))

    n_grid = 80
    nn, hh = np.meshgrid(np.linspace(0, 1, n_grid), np.linspace(0, 1, n_grid))
    u_grid = np.tile(u_star, (n_grid**2, 1))
    u_grid[:, 0], u_grid[:, 2] = nn.ravel(), hh.ravel()
    g_mu, g_sd = gp80.predict(u_grid, return_std=True)
    mass_grid, gap_grid = mass_and_gap(from_unit(u_grid))
    shape = (n_grid, n_grid)

    sobol_true = sobol_indices(objective, DIM, mode["sobol_n"], SEED)
    sobol_gp = sobol_indices(lambda u: np.exp(gp80.predict(u)), DIM, mode["sobol_n"], SEED)
    pd.DataFrame({"variable": list(VARIABLES), "first_true": sobol_true[0], "total_true": sobol_true[1],
                  "first_gp": sobol_gp[0], "total_gp": sobol_gp[1]}).to_csv(
        RESULTS_DIR / "sobol_indices.csv", index=False, float_format="%.4f")
    print(f"[{time.time() - clock:5.0f}s] surrogates and sensitivity")

    rows, bo_best = [], None
    for seed in range(mode["seeds"]):
        for method, run in [("Bayesian optimization", bayesian_optimization), ("Random search", random_search)]:
            u, y = run(objective, mode["budget"], seed)
            for i, best in enumerate(np.minimum.accumulate(to_temp(y)), start=1):
                rows.append({"method": method, "seed": seed, "evaluation": i, "best_t": best})
            if method == "Bayesian optimization" and seed == 0:
                bo_best = from_unit(u[np.argmin(y)])
    bo = pd.DataFrame(rows)
    bo.to_csv(RESULTS_DIR / "optimization_runs.csv", index=False, float_format="%.4f")
    print(f"[{time.time() - clock:5.0f}s] optimization runs")

    rng = np.random.default_rng(SEED)
    u_cloud = rng.random((100_000, DIM))
    u_cloud = u_cloud[mass_and_gap(from_unit(u_cloud))[1] >= MIN_GAP]
    sim = simulate(from_unit(u_cloud))
    cloud_t, cloud_mass = to_temp(sim["resistance"]), sim["mass"]
    front = pareto_mask(cloud_mass, cloud_t)
    sur_front = pareto_mask(cloud_mass, np.exp(gp80.predict(u_cloud)))

    base = describe(BASELINE)
    same_t = front & (cloud_t <= base["t_max"])
    lightest = from_unit(u_cloud[same_t][np.argmin(cloud_mass[same_t])])
    designs = {"Hand-made baseline": base,
               f"Bayesian optimization ({mode['budget']} runs)": describe(bo_best),
               "Lightest design, same temperature": describe(lightest)}
    best_same_mass = cloud_t[front & (cloud_mass <= base["mass"])].min()

    plots.setup_style()
    plots.fan_operating_point(x_star)
    plots.learning_curves(lc)
    idx = np.random.default_rng(0).choice(len(u_test), 400, replace=False)
    plots.parity(y_test[idx], np.exp(mu[idx]), np.exp(mu[idx] - 1.645 * sd[idx]),
                 np.exp(mu[idx] + 1.645 * sd[idx]), np.exp(rsm80.predict(u_test[idx])), 80)
    x80 = from_unit(u80)
    plots.response_surface(from_unit_1d(nn), from_unit(u_grid)[:, 2].reshape(shape),
                           to_temp(objective(u_grid)).reshape(shape), to_temp(np.exp(g_mu)).reshape(shape),
                           ((to_temp(np.exp(g_mu)) - plots.T_AMBIENT) * g_sd).reshape(shape),
                           mass_grid.reshape(shape), gap_grid.reshape(shape), x80[:, [0, 2]],
                           (x_star[0], x_star[2]))
    labels = [v["label"] for v in VARIABLES.values()]
    plots.sensitivity(labels, sobol_true, sobol_gp, mode["sobol_n"] * (DIM + 2), 80)
    plots.bo_in_action(*bo_1d_states(u_star))
    plots.bo_convergence(bo, to_temp(r_star), base["t_max"], N_INIT, de_evals)
    plots.pareto(cloud_mass, cloud_t, front, cloud_mass[sur_front], cloud_t[sur_front], designs, 80)
    plots.design_comparison(designs)
    plots.resistance_breakdown(designs)

    final = bo[bo["evaluation"] == mode["budget"]].groupby("method")["best_t"].median()
    summary = {
        "true_optimum_t": float(to_temp(r_star)), "true_optimum_x": dict(zip(VARIABLES, map(float, x_star))),
        "de_evaluations_within_0.1C": de_evals, "budget": mode["budget"],
        "median_final_t": final.to_dict(), "baseline": base, "designs": designs,
        "best_t_at_baseline_mass": float(best_same_mass), "lightest_mass_at_baseline_t": designs[
            "Lightest design, same temperature"]["mass"],
        "gp80_coverage_90": gp_coverage,
        "mape_by_n": {str(n): g.groupby("model")["mape"].median().to_dict() for n, g in lc.groupby("n")},
    }
    (RESULTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    table = pd.DataFrame({name: {"Fins": int(round(d["x"][0])), "Fin thickness / pitch": round(d["x"][1], 2),
                                 "Fin height (mm)": round(d["x"][2], 1), "Base (mm)": round(d["x"][3], 1),
                                 "Length (mm)": round(d["x"][4], 1), "Fin gap (mm)": round(d["gap_mm"], 2),
                                 "Air flow (L/s)": round(1e3 * d["flow"], 2), "Mass (g)": round(d["mass"], 1),
                                 "T max (°C)": round(d["t_max"], 1)} for name, d in designs.items()})
    (RESULTS_DIR / "designs.md").write_text(table.to_markdown())
    print(f"[{time.time() - clock:5.0f}s] done\n")
    print(json.dumps({k: v for k, v in summary.items() if k not in ("designs", "baseline")}, indent=2))
    print(table.to_string())


if __name__ == "__main__":
    main()
