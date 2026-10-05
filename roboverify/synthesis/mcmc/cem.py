import multiprocessing as mp
from contextlib import nullcontext

import numpy as np


def cem_optimize(
    f, dim, iterations=50, N=100, K=10, init_mu=None, init_std=0.1, num_workers=None
) -> tuple[float, np.ndarray]:
    """
    Parallelized Cross-Entropy Method optimizer using the multiprocess library.

    Args:
        f: Objective function to maximize (takes ndarray, returns scalar)
        dim: Dimension of the search space
        iterations: Number of optimization iterations
        N: Number of samples per iteration
        K: Number of elite samples to keep
        init_mu: Initial mean vector
        init_std: Initial standard deviation
        num_workers: Number of parallel workers (defaults to os.cpu_count())

    Returns:
        (best_score, best_mu)
    """
    mu = np.zeros(dim) if init_mu is None else np.array(init_mu, dtype=float)
    sigma = np.ones(dim) * init_std

    mu_list, score_list = [mu], [f(mu)]

    # Create pool outside loop for efficiency
    with (
        nullcontext(None) if num_workers == 0 else mp.Pool(processes=num_workers)
    ) as pool:
        for _ in range(iterations):
            # Sample candidate solutions
            samples = np.random.randn(N, dim) * sigma + mu

            # Parallel evaluation of f(x)
            scores = list(map(f, samples)) if pool is None else pool.map(f, samples)

            scores = np.array(scores)
            print("scores", sorted(scores))
            print("average of best k", np.mean(sorted(scores)[-K:]))
            print("best mu", samples[scores.argsort()[-1]])
            elites = samples[np.argsort(scores)[-K:]]  # top-K elites

            # Update mean and std
            mu = elites.mean(axis=0)
            print("next mu:", mu)
            sigma = elites.std(axis=0)

            mu_list.append(mu)
            score_list.append(f(mu))

    # Pick the best mu seen
    max_idx = np.argmax(score_list)
    print(max_idx, score_list[max_idx], mu_list[max_idx])
    print(score_list)
    print(mu_list)
    return score_list[max_idx], mu_list[max_idx]
