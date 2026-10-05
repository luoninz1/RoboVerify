"""Cached demonstration densities with reproducible, local Monte Carlo KL."""

from typing import Protocol

import numpy as np
from scipy.special import logsumexp


class TrajectoryDistance(Protocol):
    name: str

    def __call__(self, states, *, samples=256): ...


class KDEDistance:
    name = "kl_kde"

    def __init__(self, demos, *, bandwidth=0.02, seed=0):
        self.demos = self._states(demos)
        if bandwidth <= 0:
            raise ValueError("KDE bandwidth must be positive")
        self.bandwidth = bandwidth
        self.seed = seed
        # Cache demo-side scaling/density components across every CEM evaluation.
        self.scale = np.maximum(self.demos.std(axis=0), bandwidth)
        self.reference = self.demos / self.scale

    @staticmethod
    def _states(values):
        array = np.asarray(values, dtype=float)
        if array.ndim != 2 or len(array) == 0 or not np.isfinite(array).all():
            raise ValueError("Distance requires finite nonempty N x D states")
        return array

    def _log_density(self, points, centers):
        # Isotropic Gaussian KDE in demo-normalized coordinates, chunked by samples.
        result = []
        for chunk in np.array_split(points, max(1, (len(points) + 63) // 64)):
            square = ((chunk[:, None, :] - centers[None, :, :]) / self.bandwidth) ** 2
            result.extend(
                logsumexp(-0.5 * square.sum(axis=2), axis=1) - np.log(len(centers))
            )
        return np.asarray(result)

    def __call__(self, states, *, samples=256):
        policy = self._states(states) / self.scale
        if policy.shape[1] != self.reference.shape[1] or samples < 1:
            raise ValueError("Mismatched dimensions or sample budget")
        rng = np.random.default_rng(self.seed)
        points = policy[rng.integers(len(policy), size=samples)] + rng.normal(
            0, self.bandwidth, (samples, policy.shape[1])
        )
        # Monte Carlo can be slightly negative. Keep that estimator value; do not
        # silently rescale MMD or manufacture a KL threshold from it.
        return float(
            np.mean(
                self._log_density(points, policy)
                - self._log_density(points, self.reference)
            )
        )


class GMMDistance:
    name = "kl_gmm"

    def __init__(self, demos, *, components=3, seed=0):
        from sklearn.mixture import GaussianMixture

        self.seed, self.components = seed, components
        self.factory = GaussianMixture
        self.reference = self._fit(KDEDistance._states(demos))

    def _fit(self, states):
        if len(states) == 1:
            states = np.concatenate([states, states], axis=0)
        return self.factory(
            n_components=min(self.components, len(states)),
            covariance_type="diag",
            reg_covar=1e-5,
            random_state=self.seed,
        ).fit(states)

    def __call__(self, states, *, samples=256):
        policy = self._fit(KDEDistance._states(states))
        points, _ = policy.sample(samples)
        return float(
            np.mean(policy.score_samples(points) - self.reference.score_samples(points))
        )


class MMDDistance:
    name = "mmd"

    def __init__(self, demos):
        self.demos = KDEDistance._states(demos)

    def __call__(self, states, *, samples=256):
        from synthesis.mcmc.cost_func import maximum_mean_discrepancy_rbf

        return float(
            maximum_mean_discrepancy_rbf(KDEDistance._states(states), self.demos)
        )


def make_distance(name, demos, **kwargs):
    return {"kl_kde": KDEDistance, "kl_gmm": GMMDistance, "mmd": MMDDistance}[name](
        demos, **kwargs
    )
