from pathlib import Path

ROOT = Path(__file__).resolve().parent
FIG_DIR = ROOT / "figures"
RESULTS_DIR = ROOT / "results"

SEED = 42
MASS_LIMIT = 150.0   # g, product weight budget
MIN_GAP = 1.0        # mm, minimum fin gap for aluminum extrusion
N_INIT = 10          # initial space-filling design for Bayesian optimization

# A reasonable hand-made design, used as the reference to beat
BASELINE = [20, 0.30, 30.0, 5.0, 60.0]

MODES = {
    "quick": {"seeds": 3, "budget": 30, "train_sizes": [10, 20, 40, 80], "lc_seeds": 2, "sobol_n": 2048},
    "full": {"seeds": 10, "budget": 50, "train_sizes": [10, 20, 40, 80, 160], "lc_seeds": 5, "sobol_n": 8192},
}
