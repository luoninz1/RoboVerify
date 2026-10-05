import numpy as np
import torch
from scipy.stats import gaussian_kde
from sklearn.mixture import GaussianMixture


class TorchGMM:
    def __init__(self, gmm: GaussianMixture):
        self.gmm = gmm
        # Convert parameters to torch tensors
        self.weights = torch.tensor(gmm.weights_, dtype=torch.float32)
        self.means = torch.tensor(gmm.means_, dtype=torch.float32)
        self.covariances = torch.tensor(gmm.covariances_, dtype=torch.float32)
        self.n_components = gmm.n_components
        self.covariance_type = gmm.covariance_type

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """Compute log-likelihood of each sample in x."""
        x_np = x.detach().cpu().numpy()
        log_probs = self.gmm.score_samples(x_np)
        return torch.tensor(log_probs, dtype=torch.float32, device=x.device)

    def sample(self, n_samples: int) -> torch.Tensor:
        """Generate samples from the GMM."""
        samples, _ = self.gmm.sample(n_samples)
        return torch.tensor(samples, dtype=torch.float32)


def estimate_gmm_from_states_torch(
    states: torch.Tensor,
    n_components: int = 5,
    covariance_type: str = "diag",
    random_state: int = 42,
) -> TorchGMM:
    """
    Fit a GMM to the given states (PyTorch tensor) and return a Torch-compatible wrapper.

    Args:
        states (torch.Tensor): Tensor of shape [num_samples, state_dim].
        n_components (int): Number of Gaussian components.
        covariance_type (str): Covariance type: 'full', 'tied', 'diag', or 'spherical'.
        random_state (int): Random seed.

    Returns:
        TorchGMM: A PyTorch-compatible wrapper for the fitted GMM.
    """
    states_np = states.detach().cpu().numpy()
    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type=covariance_type,
        max_iter=200,
        random_state=random_state,
    )
    gmm.fit(states_np)
    return TorchGMM(gmm)


def kl_divergence_gmm(gmm_p, gmm_q, n_samples=10000):
    """
    Estimate KL divergence KL(P || Q) between two GMMs using Monte Carlo sampling.

    Args:
        gmm_p (TorchGMM): First GMM (from which we sample).
        gmm_q (TorchGMM): Second GMM.
        n_samples (int): Number of samples to use for approximation.

    Returns:
        float: Estimated KL divergence.
    """
    # Sample from P
    samples = gmm_p.sample(n_samples)  # shape: [n_samples, dim]

    # Evaluate log probabilities
    log_p = gmm_p.log_prob(samples)
    log_q = gmm_q.log_prob(samples)

    # KL(P || Q) = E_P[log P - log Q]
    kl_estimate = torch.mean(log_p - log_q)
    return kl_estimate.item()


def kl_divergence_kde(states_P, states_Q, n_samples=10000):
    """
    Estimate KL(P || Q) where states_P and states_Q are arrays of shape (N, d)
    representing states from two different trajectory sets.
    KDE is used to estimate occupancy distributions.
    """
    # import pdb; pdb.set_trace()
    states_P_noisy = states_P + 1e-6 * np.random.randn(*states_P.shape)
    # states_Q_noisy = states_Q + 1e-6 * np.random.randn(*states_Q.shape)
    kde_P = gaussian_kde(states_P_noisy.T)
    kde_Q = gaussian_kde(states_Q.T)
    # Sample from P (or use its raw samples)
    samples = states_P[np.random.choice(len(states_P), size=n_samples, replace=True)]
    # Evaluate densities
    p_vals = kde_P(samples.T)
    q_vals = kde_Q(samples.T)
    # Avoid log(0); use small value for numerical stability
    epsilon = 1e-10
    p_vals = np.clip(p_vals, epsilon, None)
    q_vals = np.clip(q_vals, epsilon, None)
    kl = np.mean(np.log(p_vals / q_vals))
    return kl


def _flatten_states(states) -> np.ndarray:
    """Convert a batch of states to a 2D float array."""
    arr = np.asarray(states, dtype=np.float64)
    if arr.ndim == 1:
        return arr[:, None]
    return arr.reshape(arr.shape[0], -1)


def _sample_rows(states: np.ndarray, max_samples: int, rng: np.random.Generator) -> np.ndarray:
    """Subsample rows without replacement to bound pairwise MMD cost."""
    if len(states) <= max_samples:
        return states
    indices = rng.choice(len(states), size=max_samples, replace=False)
    return states[indices]


def _median_heuristic_bandwidth(X: np.ndarray, Y: np.ndarray) -> float:
    """Choose an RBF bandwidth from pairwise distances over combined samples."""
    combined = np.vstack([X, Y])
    if len(combined) < 2:
        return 1.0
    sq_norms = np.sum(combined * combined, axis=1, keepdims=True)
    sq_dists = sq_norms + sq_norms.T - 2.0 * combined @ combined.T
    sq_dists = np.maximum(sq_dists, 0.0)
    upper = sq_dists[np.triu_indices_from(sq_dists, k=1)]
    positive = upper[upper > 0.0]
    if len(positive) == 0:
        return 1.0
    return float(np.sqrt(np.median(positive)))


def _rbf_kernel(X: np.ndarray, Y: np.ndarray, bandwidth: float) -> np.ndarray:
    """Compute an RBF kernel matrix."""
    bandwidth = max(float(bandwidth), 1e-6)
    gamma = 1.0 / (2.0 * bandwidth * bandwidth)
    X_sq = np.sum(X * X, axis=1, keepdims=True)
    Y_sq = np.sum(Y * Y, axis=1, keepdims=True).T
    sq_dists = X_sq + Y_sq - 2.0 * X @ Y.T
    sq_dists = np.maximum(sq_dists, 0.0)
    return np.exp(-gamma * sq_dists)


def maximum_mean_discrepancy_rbf(
    states_P,
    states_Q,
    bandwidth: float | None = None,
    max_samples: int = 512,
    random_state: int = 42,
) -> float:
    """
    Estimate squared MMD between two state distributions with an RBF kernel.

    Uses the unbiased estimator and standardizes the joint sample first so the
    kernel bandwidth is not dominated by raw feature scale.
    """
    X = _flatten_states(states_P)
    Y = _flatten_states(states_Q)
    if len(X) == 0 or len(Y) == 0:
        raise ValueError("MMD requires non-empty state sets")

    rng = np.random.default_rng(random_state)
    X = _sample_rows(X, max_samples=max_samples, rng=rng)
    Y = _sample_rows(Y, max_samples=max_samples, rng=rng)

    combined = np.vstack([X, Y])
    mean = combined.mean(axis=0, keepdims=True)
    std = combined.std(axis=0, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    X = (X - mean) / std
    Y = (Y - mean) / std

    if bandwidth is None:
        bandwidth = _median_heuristic_bandwidth(X, Y)

    K_xx = _rbf_kernel(X, X, bandwidth)
    K_yy = _rbf_kernel(Y, Y, bandwidth)
    K_xy = _rbf_kernel(X, Y, bandwidth)

    n = len(X)
    m = len(Y)
    if n < 2 or m < 2:
        return 0.0

    np.fill_diagonal(K_xx, 0.0)
    np.fill_diagonal(K_yy, 0.0)
    mmd2 = (
        K_xx.sum() / (n * (n - 1))
        + K_yy.sum() / (m * (m - 1))
        - 2.0 * K_xy.mean()
    )
    return float(max(mmd2, 0.0))
