"""
Pure-NumPy Gaussian HMM — drop-in replacement for hmmlearn.GaussianHMM.
No compiled C extensions required. Works on any Python version.
"""
import numpy as np
from sklearn.cluster import KMeans


class GaussianHMM:
    """
    Gaussian Hidden Markov Model implemented in pure NumPy.
    Interface mirrors hmmlearn.hmm.GaussianHMM for drop-in compatibility.
    """

    def __init__(
        self,
        n_components: int = 3,
        covariance_type: str = "full",
        n_iter: int = 100,
        tol: float = 1e-4,
        random_state: int = 42,
        **kwargs,
    ):
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.n_iter = n_iter
        self.tol = tol
        self.random_state = random_state
        self.monitor_ = _Monitor()

    # ── Public API ────────────────────────────────────────────

    def fit(self, X: np.ndarray, lengths=None) -> "GaussianHMM":
        X = _ensure_2d(X)
        n, d = X.shape
        k = self.n_components

        # Init via K-Means
        km = KMeans(n_clusters=k, random_state=self.random_state, n_init=10).fit(X)
        labels = km.labels_

        self.startprob_ = np.bincount(labels, minlength=k).astype(float) + 1.0
        self.startprob_ /= self.startprob_.sum()

        self.transmat_ = np.ones((k, k)) + 0.1
        self.transmat_ /= self.transmat_.sum(axis=1, keepdims=True)

        self.means_ = km.cluster_centers_.copy()
        self.covars_ = np.array([
            np.cov(X[labels == i].T) if (labels == i).sum() > 1 else np.eye(d)
            for i in range(k)
        ])
        # Regularize
        self.covars_ += np.eye(d)[None] * 1e-3

        log_transmat = np.log(self.transmat_ + 1e-300)
        prev_ll = -np.inf

        for iteration in range(self.n_iter):
            # E-step
            log_obs = self._log_obs(X)
            log_alpha = self._forward(log_obs, log_transmat)
            log_beta = self._backward(log_obs, log_transmat)

            ll = _logsumexp(log_alpha[-1])
            if abs(ll - prev_ll) < self.tol:
                self.monitor_.converged = True
                break
            prev_ll = ll

            log_gamma = log_alpha + log_beta
            log_gamma -= _logsumexp(log_gamma, axis=1, keepdims=True)
            gamma = np.exp(log_gamma)

            # xi[t, i, j] = P(q_t=i, q_{t+1}=j | X)
            log_xi = (
                log_alpha[:-1, :, None]
                + log_transmat[None, :, :]
                + log_obs[1:, None, :]
                + log_beta[1:, None, :]
            )
            log_xi -= _logsumexp(log_xi.reshape(n - 1, -1), axis=1)[:, None, None]
            xi_sum = np.exp(log_xi).sum(axis=0)  # (k, k)

            # M-step
            self.startprob_ = gamma[0] + 1e-10
            self.startprob_ /= self.startprob_.sum()

            self.transmat_ = xi_sum + 1e-10
            self.transmat_ /= self.transmat_.sum(axis=1, keepdims=True)
            log_transmat = np.log(self.transmat_)

            gamma_sum = gamma.sum(axis=0) + 1e-10
            self.means_ = (gamma.T @ X) / gamma_sum[:, None]

            for i in range(k):
                diff = X - self.means_[i]
                self.covars_[i] = (gamma[:, i:i+1] * diff).T @ diff / gamma_sum[i]
                self.covars_[i] += np.eye(diff.shape[1]) * 1e-3

        return self

    def predict(self, X: np.ndarray, lengths=None) -> np.ndarray:
        """Viterbi decoding."""
        X = _ensure_2d(X)
        log_obs = self._log_obs(X)
        log_transmat = np.log(self.transmat_ + 1e-300)
        n, k = log_obs.shape

        viterbi = np.full((n, k), -np.inf)
        psi = np.zeros((n, k), dtype=int)
        viterbi[0] = np.log(self.startprob_ + 1e-300) + log_obs[0]

        for t in range(1, n):
            trans = viterbi[t - 1, :, None] + log_transmat  # (k, k)
            psi[t] = trans.argmax(axis=0)
            viterbi[t] = trans.max(axis=0) + log_obs[t]

        states = np.zeros(n, dtype=int)
        states[-1] = viterbi[-1].argmax()
        for t in range(n - 2, -1, -1):
            states[t] = psi[t + 1, states[t + 1]]
        return states

    def predict_proba(self, X: np.ndarray, lengths=None) -> np.ndarray:
        """Posterior state probabilities via forward-backward."""
        X = _ensure_2d(X)
        log_obs = self._log_obs(X)
        log_transmat = np.log(self.transmat_ + 1e-300)
        log_alpha = self._forward(log_obs, log_transmat)
        log_beta = self._backward(log_obs, log_transmat)
        log_gamma = log_alpha + log_beta
        log_gamma -= _logsumexp(log_gamma, axis=1, keepdims=True)
        return np.exp(log_gamma)

    # ── Internal helpers ──────────────────────────────────────

    def _log_obs(self, X: np.ndarray) -> np.ndarray:
        """Log observation probabilities: shape (n, k)."""
        n, d = X.shape
        log_p = np.zeros((n, self.n_components))
        for i in range(self.n_components):
            log_p[:, i] = _log_mvn(X, self.means_[i], self.covars_[i])
        return log_p

    def _forward(self, log_obs: np.ndarray, log_transmat: np.ndarray) -> np.ndarray:
        n, k = log_obs.shape
        log_alpha = np.empty((n, k))
        log_alpha[0] = np.log(self.startprob_ + 1e-300) + log_obs[0]
        for t in range(1, n):
            # log_alpha[t, j] = logsumexp_i(log_alpha[t-1, i] + log_transmat[i, j]) + log_obs[t, j]
            log_alpha[t] = _logsumexp(log_alpha[t - 1, :, None] + log_transmat, axis=0) + log_obs[t]
        return log_alpha

    def _backward(self, log_obs: np.ndarray, log_transmat: np.ndarray) -> np.ndarray:
        n, k = log_obs.shape
        log_beta = np.zeros((n, k))
        for t in range(n - 2, -1, -1):
            # log_beta[t, i] = logsumexp_j(log_transmat[i, j] + log_obs[t+1, j] + log_beta[t+1, j])
            log_beta[t] = _logsumexp(log_transmat + log_obs[t + 1] + log_beta[t + 1], axis=1)
        return log_beta


# ── Utilities ─────────────────────────────────────────────────

class _Monitor:
    converged: bool = False


def _ensure_2d(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    return np.ascontiguousarray(X if X.ndim == 2 else X.reshape(-1, 1))


def _logsumexp(a: np.ndarray, axis=None, keepdims=False) -> np.ndarray:
    a_max = np.max(a, axis=axis, keepdims=True)
    a_max = np.where(np.isfinite(a_max), a_max, 0.0)
    out = np.log(np.sum(np.exp(a - a_max), axis=axis, keepdims=keepdims))
    if not keepdims:
        a_max = np.squeeze(a_max, axis=axis)
    return out + a_max


def _log_mvn(X: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """Log-density of multivariate normal N(mean, cov) evaluated at each row of X."""
    d = mean.shape[0]
    cov_reg = cov + np.eye(d) * 1e-6
    try:
        L = np.linalg.cholesky(cov_reg)
    except np.linalg.LinAlgError:
        return np.full(len(X), -1e10)
    log_det = 2.0 * np.sum(np.log(np.diag(L)))
    diff = X - mean
    alpha = np.linalg.solve(L, diff.T).T
    maha = np.sum(alpha ** 2, axis=1)
    return -0.5 * (d * np.log(2 * np.pi) + log_det + maha)
