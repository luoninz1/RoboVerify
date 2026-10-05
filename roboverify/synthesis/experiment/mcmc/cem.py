"""Cross-entropy-method optimizer with structured output instead of prints.

This is a deliberate copy of :func:`synthesis.mcmc.cem.cem_optimize`. That version
prints the full sorted score list plus two ndarrays on every iteration and dumps the
complete score and mean history at the end, which for a 2000-iteration MCMC search
is the single largest source of log volume. It is also the most informative signal
about whether parameter optimization is doing anything, so it is worth turning into
metrics rather than discarding.

The sampling order, the ``f`` call order and therefore the consumption of
``numpy.random`` are identical to the original, which is what lets
``test_mcmc_parity`` hold. Note that the objective reseeds the global RNG per
rollout (``set_np_seed`` inside ``rollout_demos``), so the order of ``f`` calls
relative to ``np.random.randn`` is load-bearing -- do not reorder it.
"""

import multiprocessing as mp
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np


@dataclass
class CEMStats:
    """Diagnostics for one CEM optimization of one candidate program.

    ``delta`` is the headline number: if parameter optimization is working it is
    positive, and if it is persistently zero the search is spending its whole
    budget on an inner loop that never improves anything.
    """

    evals: int = 0
    iterations: int = 0
    dim: int = 0
    initial_score: Optional[float] = None
    best_score: Optional[float] = None
    sigma_norm_final: Optional[float] = None
    per_iteration: list = field(default_factory=list)

    @property
    def delta(self) -> Optional[float]:
        if self.initial_score is None or self.best_score is None:
            return None
        return self.best_score - self.initial_score

    def summary(self) -> dict:
        """Flat, JSON-safe fields for ``metrics.jsonl``."""

        def rounded(value):
            return None if value is None else round(float(value), 6)

        return {
            "cem_evals": self.evals,
            "cem_iterations": self.iterations,
            "cem_dim": self.dim,
            "cem_initial": rounded(self.initial_score),
            "cem_best": rounded(self.best_score),
            "cem_delta": rounded(self.delta),
            "cem_sigma_norm": rounded(self.sigma_norm_final),
        }


def cem_optimize(
    f: Callable,
    dim: int,
    iterations: int = 50,
    N: int = 100,
    K: int = 10,
    init_mu=None,
    init_std: float = 0.1,
    num_workers: Optional[int] = None,
    on_iteration: Optional[Callable[[int, dict], None]] = None,
) -> tuple:
    """Maximize ``f`` by the cross-entropy method.

    Returns ``(best_score, best_mu, stats)``. The first two match the original
    function's return value exactly.
    """
    mu = np.zeros(dim) if init_mu is None else np.array(init_mu, dtype=float)
    sigma = np.ones(dim) * init_std
    stats = CEMStats(iterations=iterations, dim=dim)

    initial_score = f(mu)
    stats.evals += 1
    stats.initial_score = float(initial_score)

    mu_list, score_list = [mu], [initial_score]

    if iterations > 0:
        # The pool is only created when there is work for it; the original opened
        # one unconditionally, which forks workers even for a parameterless program.
        with (
            nullcontext(None) if num_workers == 0 else mp.Pool(processes=num_workers)
        ) as pool:
            for iteration in range(iterations):
                samples = np.random.randn(N, dim) * sigma + mu
                scores = np.array(
                    list(map(f, samples)) if pool is None else pool.map(f, samples)
                )
                stats.evals += len(samples)

                order = np.argsort(scores)
                elites = samples[order[-K:]]
                mu = elites.mean(axis=0)
                sigma = elites.std(axis=0)

                mu_list.append(mu)
                iterate_score = f(mu)
                stats.evals += 1
                score_list.append(iterate_score)

                record = {
                    "iteration": iteration,
                    "sample_best": float(scores.max()),
                    "sample_mean": float(scores.mean()),
                    "elite_mean": float(scores[order[-K:]].mean()),
                    "mu_score": float(iterate_score),
                    "sigma_norm": float(np.linalg.norm(sigma)),
                }
                stats.per_iteration.append(record)
                if on_iteration is not None:
                    on_iteration(iteration, record)

    max_idx = int(np.argmax(score_list))
    stats.best_score = float(score_list[max_idx])
    stats.sigma_norm_final = float(np.linalg.norm(sigma))
    return score_list[max_idx], mu_list[max_idx], stats
