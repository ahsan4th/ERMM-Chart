"""Phase I estimators of the in-control mean vector and covariance matrix.

The reweighted Fast-MCD estimator follows robustbase::covMcd (0.99-x) step by
step, so that limits calibrated here and limits calibrated in R refer to the
same estimator:

1. subset size h = h.alpha.n(alpha, N, p);
2. Fast-MCD with `n_starts` random (p+1)-subsets, two C-steps each, the ten
   best subsets iterated to convergence (Rousseeuw and Van Driessen, 1999);
3. raw covariance of the h-subset (divisor h-1) multiplied by the consistency
   factor .MCDcons(p, h/N) and the small-sample factor .MCDcnp2(p, N, alpha);
4. weights w_i = 1{d_i^2 < chi2_{p;q}} from the raw estimates (q = 0.975);
5. weighted mean and covariance (divisor sum(w)-1), multiplied by
   .MCDcons(p, q) and .MCDcnp2.rew(p, N, alpha) when sum(w) < N.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import special, stats


@dataclass(frozen=True)
class PhaseIEstimate:
    method: str                    # "classical" or "fastmcd"
    center: np.ndarray             # (p,) in-control mean vector estimate
    cov: np.ndarray                # (p, p) in-control covariance estimate
    weights: np.ndarray            # (N,) 1 = observation used in the estimate
    d2: np.ndarray                 # (N,) squared distance used for weighting
    h: int                         # MCD subset size (N for classical)
    raw_center: np.ndarray
    raw_cov: np.ndarray


# ---------------------------------------------------------------------------
# robustbase helper functions, translated literally
# ---------------------------------------------------------------------------
def h_alpha_n(alpha: float, n: int, p: int) -> int:
    """robustbase:::h.alpha.n"""
    if not (0.5 <= alpha <= 1):
        raise ValueError(f"alpha must lie in [0.5, 1], got {alpha}")
    n2 = (n + p + 1) // 2
    return int(np.floor(2 * n2 - n + 2 * (n - n2) * alpha))


def mcd_consistency(p: int, alpha: float) -> float:
    """robustbase:::.MCDcons: consistency factor for a trimming fraction alpha."""
    q = stats.chi2.ppf(alpha, p)
    return alpha / special.gammainc(p / 2 + 1, q / 2)


def _cnp2(p, n, alpha, c500, c875, p2, p1):
    if p > 2:
        # R: matrix(c(a1, b1, k1, a2, b2, k2), ncol = 2) -> rows (a, b, k)
        c500 = np.asarray(c500, dtype=float).reshape(2, 3).T
        c875 = np.asarray(c875, dtype=float).reshape(2, 3).T

        def coeffic(c):
            y = np.log(-c[0, :] / p ** c[1, :])
            A = np.column_stack([np.ones(2), -np.log(c[2, :] * p ** 2)])
            return np.linalg.solve(A, y)

        k500, k875 = coeffic(c500), coeffic(c875)
        fp500 = 1 - np.exp(k500[0]) / n ** k500[1]
        fp875 = 1 - np.exp(k875[0]) / n ** k875[1]
    elif p == 2:
        fp500 = 1 - np.exp(p2[0]) / n ** p2[1]
        fp875 = 1 - np.exp(p2[2]) / n ** p2[3]
    elif p == 1:
        fp500 = 1 - np.exp(p1[0]) / n ** p1[1]
        fp875 = 1 - np.exp(p1[2]) / n ** p1[3]
    else:
        raise ValueError("p must be a positive integer")
    if alpha == 0.5:
        raise NotImplementedError("alpha = 0.5 needs robustbase's simulated "
                                  "correction table; use 0.5 < alpha <= 1")
    if alpha <= 0.875:
        fp = fp500 + (fp875 - fp500) / 0.375 * (alpha - 0.5)
    else:
        fp = fp875 + (1 - fp875) / 0.125 * (alpha - 0.875)
    return 1.0 / fp


def mcd_small_sample_raw(p: int, n: int, alpha: float) -> float:
    """robustbase:::.MCDcnp2 (Pison, Van Aelst and Willems, 2002)."""
    return _cnp2(p, n, alpha,
                 c500=[-1.42764571687802, 1.26263336932151, 2,
                       -1.06141115981725, 1.28907991440387, 3],
                 c875=[-0.455179464070565, 1.11192541278794, 2,
                       -0.294241208320834, 1.09649329149811, 3],
                 p2=[0.673292623522027, 0.691365864961895,
                     0.446537815635445, 1.06690782995919],
                 p1=[0.262024211897096, 0.604756680630497,
                     -0.351584646688712, 1.01646567502486])


def mcd_small_sample_rew(p: int, n: int, alpha: float) -> float:
    """robustbase:::.MCDcnp2.rew"""
    return _cnp2(p, n, alpha,
                 c500=[-1.02842572724793, 1.67659883081926, 2,
                       -0.26800273450853, 1.35968562893582, 3],
                 c875=[-0.544482443573914, 1.25994483222292, 2,
                       -0.343791072183285, 1.25159004257133, 3],
                 p2=[3.11101712909049, 1.91401056721863,
                     0.79473550581058, 1.10081930350091],
                 p1=[1.11098143415027, 1.5182890270453,
                     -0.66046776772861, 0.88939595831888])


# ---------------------------------------------------------------------------
# Fast-MCD, vectorized over the random starts
# ---------------------------------------------------------------------------
def _subset_mean_cov(X, idx):
    sub = X[idx]                                   # (S, m, p)
    m = idx.shape[1]
    mu = sub.mean(axis=1)                          # (S, p)
    d = sub - mu[:, None, :]
    cov = np.matmul(d.transpose(0, 2, 1), d) / (m - 1)
    return mu, cov


def _mahalanobis_batch(X, mu, cov):
    """Squared distances of all rows of X to every (mu_s, cov_s); shape (S, N)."""
    diff = X[None, :, :] - mu[:, None, :]          # (S, N, p)
    return (np.matmul(diff, np.linalg.inv(cov)) * diff).sum(axis=2)


def _c_step(X, mu, cov, h):
    d2 = _mahalanobis_batch(X, mu, cov)
    idx = np.argpartition(d2, h - 1, axis=1)[:, :h]
    idx.sort(axis=1)
    mu_new, cov_new = _subset_mean_cov(X, idx)
    sign, logdet = np.linalg.slogdet(cov_new)
    if np.any(sign <= 0):
        raise np.linalg.LinAlgError(
            "singular h-subset covariance: h observations lie on a hyperplane "
            "(exact fit); check for constant or duplicated characteristics")
    return idx, mu_new, cov_new, logdet


def _initial_fits(X, n_starts, rng):
    N, p = X.shape
    keys = rng.random((n_starts, N))
    order = np.argsort(keys, axis=1)
    idx = order[:, : p + 1]
    mu, cov = _subset_mean_cov(X, idx)
    sign, logdet = np.linalg.slogdet(cov)
    # as in robustbase, a singular (p+1)-subset is enlarged with further points
    for s in np.where((sign <= 0) | (logdet < -700))[0]:
        for m in range(p + 2, N + 1):
            sub = order[s, :m]
            c = np.cov(X[sub].T)
            sg, ld = np.linalg.slogdet(c)
            if sg > 0 and ld > -700:
                mu[s], cov[s] = X[sub].mean(axis=0), c
                break
        else:
            raise np.linalg.LinAlgError("all observations lie on a hyperplane")
    return mu, cov


def fast_mcd_raw(X: np.ndarray, h: int, n_starts: int, rng: np.random.Generator,
                 n_keep: int = 10, max_csteps: int = 200):
    """Raw Fast-MCD. Returns (best subset, mean, covariance with divisor h-1, logdet)."""
    X = np.asarray(X, dtype=float)
    N, p = X.shape
    if not (p < h <= N):
        raise ValueError(f"need p < h <= N, got p={p}, h={h}, N={N}")
    mu, cov = _initial_fits(X, n_starts, rng)
    for _ in range(2):
        idx, mu, cov, logdet = _c_step(X, mu, cov, h)
    best = np.argsort(logdet)[:n_keep]
    idx, mu, cov, logdet = idx[best], mu[best], cov[best], logdet[best]
    for _ in range(max_csteps):
        idx_new, mu, cov, logdet = _c_step(X, mu, cov, h)
        converged = np.array_equal(idx_new, idx)
        idx = idx_new
        if converged:
            break
    k = int(np.argmin(logdet))
    return idx[k], mu[k], cov[k], float(logdet[k])


def reweighted_fast_mcd(X: np.ndarray, alpha: float, n_starts: int,
                        reweight_quantile: float, rng: np.random.Generator
                        ) -> PhaseIEstimate:
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be an N x p matrix")
    if not np.all(np.isfinite(X)):
        raise ValueError("X contains missing or non-finite values")
    N, p = X.shape
    h = h_alpha_n(alpha, N, p)
    _, raw_mu, raw_cov, _ = fast_mcd_raw(X, h, n_starts, rng)
    raw_cov = raw_cov * mcd_consistency(p, h / N) * mcd_small_sample_raw(p, N, alpha)
    diff = X - raw_mu
    d2 = np.einsum("ij,ij->i", diff @ np.linalg.inv(raw_cov), diff)
    w = (d2 < stats.chi2.ppf(reweight_quantile, p)).astype(float)
    m = w.sum()
    if m <= p:
        raise np.linalg.LinAlgError("too few observations retained by reweighting")
    center = (w[:, None] * X).sum(axis=0) / m
    dc = X - center
    cov = (w[:, None] * dc).T @ dc / (m - 1)
    if m != N:
        cov = cov * mcd_consistency(p, reweight_quantile) * mcd_small_sample_rew(p, N, alpha)
    return PhaseIEstimate("fastmcd", center, cov, w, d2, h, raw_mu, raw_cov)


def classical_estimate(X: np.ndarray) -> PhaseIEstimate:
    X = np.asarray(X, dtype=float)
    if not np.all(np.isfinite(X)):
        raise ValueError("X contains missing or non-finite values")
    N, p = X.shape
    center = X.mean(axis=0)
    cov = np.atleast_2d(np.cov(X.T, ddof=1))
    diff = X - center
    d2 = np.einsum("ij,ij->i", diff @ np.linalg.inv(cov), diff)
    return PhaseIEstimate("classical", center, cov, np.ones(N), d2, N, center, cov)


def estimate(X: np.ndarray, method: str, alpha: float, n_starts: int,
             reweight_quantile: float, rng: np.random.Generator) -> PhaseIEstimate:
    if method == "classical":
        return classical_estimate(X)
    if method == "fastmcd":
        return reweighted_fast_mcd(X, alpha, n_starts, reweight_quantile, rng)
    raise ValueError(f"unknown method '{method}'")
