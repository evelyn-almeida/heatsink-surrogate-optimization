"""Surrogate models. All of them are trained on log(thermal resistance) in the unit cube."""
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from config import SEED


def gaussian_process(dim=5, seed=SEED):
    kernel = (ConstantKernel(1.0, (1e-3, 1e3))
              * Matern(length_scale=np.full(dim, 0.5), length_scale_bounds=(1e-2, 1e2), nu=2.5)
              + WhiteKernel(1e-6, (1e-10, 1e-2)))
    return GaussianProcessRegressor(kernel, normalize_y=True, n_restarts_optimizer=3, random_state=seed)


def response_surface():
    """Classical second-order response surface (RSM), the DoE workhorse."""
    return make_pipeline(PolynomialFeatures(2), StandardScaler(), RidgeCV(alphas=np.logspace(-6, 2, 20)))


def random_forest(seed=SEED):
    return RandomForestRegressor(n_estimators=300, min_samples_leaf=2, random_state=seed)


SURROGATES = {
    "Gaussian process": gaussian_process,
    "Quadratic response surface": response_surface,
    "Random forest": random_forest,
}
