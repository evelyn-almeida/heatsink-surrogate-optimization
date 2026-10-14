import numpy as np
from scipy.stats import norm

from config import N_INIT
from src.sampling import feasible_lhs, is_feasible, random_feasible
from src.surrogates import gaussian_process


def expected_improvement(mu, sigma, best):
    """EI for minimization."""
    sigma = np.maximum(sigma, 1e-12)
    z = (best - mu) / sigma
    return (best - mu) * norm.cdf(z) + sigma * norm.pdf(z)


def _candidates(rng, best_u, n_global=4000, n_local=2000):
    """Global random feasible points plus a local cloud around the incumbent."""
    local = np.clip(best_u + rng.normal(0, 0.05, (3 * n_local, best_u.size)), 0, 1)
    local = local[is_feasible(local)][:n_local]
    return np.vstack([random_feasible(n_global, rng), local])


def bayesian_optimization(objective, budget, seed, n_init=N_INIT):
    rng = np.random.default_rng(seed)
    u = feasible_lhs(n_init, seed)
    y = objective(u)
    while len(y) < budget:
        gp = gaussian_process(u.shape[1], seed).fit(u, np.log(y))
        candidates = _candidates(rng, u[np.argmin(y)])
        mu, sd = gp.predict(candidates, return_std=True)
        u_next = candidates[np.argmax(expected_improvement(mu, sd, np.log(y).min()))]
        u = np.vstack([u, u_next])
        y = np.append(y, objective(u_next[None]))
    return u, y


def random_search(objective, budget, seed, n_init=N_INIT):
    """Same initial design as Bayesian optimization, then uniform random proposals."""
    u = np.vstack([feasible_lhs(n_init, seed), random_feasible(budget - n_init, np.random.default_rng(seed))])
    return u, objective(u)


def pareto_mask(a, b):
    """Non-dominated points when minimizing both a and b."""
    mask = np.zeros(len(a), dtype=bool)
    best = np.inf
    for i in np.argsort(a, kind="stable"):
        if b[i] < best:
            mask[i], best = True, b[i]
    return mask
