import numpy as np


def sobol_indices(f, dim, n, seed):
    """First-order (Saltelli 2010) and total (Jansen 1999) Sobol indices on [0, 1]^dim."""
    rng = np.random.default_rng(seed)
    a, b = rng.random((n, dim)), rng.random((n, dim))
    fa, fb = f(a), f(b)
    var = np.var(np.concatenate([fa, fb]))
    first, total = np.empty(dim), np.empty(dim)
    for i in range(dim):
        ab = a.copy()
        ab[:, i] = b[:, i]
        fab = f(ab)
        first[i] = np.mean(fb * (fab - fa)) / var
        total[i] = 0.5 * np.mean((fa - fab) ** 2) / var
    return first, total
