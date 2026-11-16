"""Chart statistics: Max-M components, the (E)RMM EWMA recursion and MEWMA.

Notation follows the manuscript:
  Q_U,t = (x_t - mu)' S^{-1} (x_t - mu)                         Eq. (1)
  Q_V,t = 0.5 (x_t - x_{t-1})' S^{-1} (x_t - x_{t-1})
  U_t = Phi^{-1}{H_p(Q_U,t)},  V_t = Phi^{-1}{H_p(Q_V,t)}       Eq. (2)
  M_t = max(|U_t|, |V_t|)                                       Eq. (3)
  ERMM_t = lam M_t + (1 - lam) ERMM_{t-1},  ERMM_0 = mu_M        Eq. (4)
  UCL = mu_M + L sigma_M sqrt(lam / (2 - lam))                   Eq. (5)
"""
from __future__ import annotations

import numpy as np
from scipy import integrate, signal, stats


def _max_abs_moments():
    """Mean and SD of max(|Z1|, |Z2|) for independent standard normals.

    The CDF is F(m) = (2 Phi(m) - 1)^2, m >= 0.
    """
    sf = lambda m: 1.0 - (2.0 * stats.norm.cdf(m) - 1.0) ** 2
    mean, _ = integrate.quad(sf, 0, np.inf, epsabs=1e-13)
    second, _ = integrate.quad(lambda m: 2.0 * m * sf(m), 0, np.inf, epsabs=1e-13)
    return mean, np.sqrt(second - mean ** 2)


MU_M, SIGMA_M = _max_abs_moments()   # 2/sqrt(pi) = 1.128379..., 0.6028...
HALF_NORMAL_MEAN = float(np.sqrt(2.0 / np.pi))   # in-control mean of |U_t| and |V_t|


def chi2_to_normal(q: np.ndarray, p: int) -> np.ndarray:
    """Phi^{-1}(H_p(q)) evaluated without loss of precision in either tail."""
    q = np.asarray(q, dtype=float)
    cdf = stats.chi2.cdf(q, p)
    sf = stats.chi2.sf(q, p)
    return np.where(cdf < 0.5, stats.norm.ppf(cdf), stats.norm.isf(sf))


def max_m_components(X: np.ndarray, x_prev: np.ndarray, center: np.ndarray,
                     cov: np.ndarray) -> dict:
    """Location and successive-difference components for every row of X.

    X      : (T, p) monitored observations, or (R, T, p) for R replications
    x_prev : (p,) or (R, p) observation preceding X[0] (for Q_V of the first row)
    """
    X = np.asarray(X, dtype=float)
    p = X.shape[-1]
    if center.shape != (p,) or cov.shape != (p, p):
        raise ValueError("center/cov do not match the number of characteristics")
    S_inv = np.linalg.inv(cov)
    prev = np.concatenate([np.asarray(x_prev, dtype=float)[..., None, :],
                           X[..., :-1, :]], axis=-2)
    du = X - center
    dv = X - prev
    q_u = np.einsum("...i,ij,...j->...", du, S_inv, du)
    q_v = 0.5 * np.einsum("...i,ij,...j->...", dv, S_inv, dv)
    u = chi2_to_normal(q_u, p)
    v = chi2_to_normal(q_v, p)
    return {"Q_U": q_u, "Q_V": q_v, "U": u, "V": v,
            "M": np.maximum(np.abs(u), np.abs(v))}


def ewma(z: np.ndarray, lam: float, start: float) -> np.ndarray:
    """EWMA along the last axis: y_t = lam z_t + (1 - lam) y_{t-1}, y_0 = start."""
    if not (0 < lam <= 1):
        raise ValueError(f"lambda must lie in (0, 1], got {lam}")
    z = np.asarray(z, dtype=float)
    zi = np.full(z.shape[:-1] + (1,), (1 - lam) * start)
    y, _ = signal.lfilter([lam], [1.0, -(1.0 - lam)], z, axis=-1, zi=zi)
    return y


def ucl_from_L(L: float, lam: float) -> float:
    return MU_M + L * SIGMA_M * np.sqrt(lam / (2 - lam))


def L_from_ucl(ucl: float, lam: float) -> float:
    return (ucl - MU_M) / (SIGMA_M * np.sqrt(lam / (2 - lam)))


def ermm_path(X: np.ndarray, x_prev: np.ndarray, center: np.ndarray,
              cov: np.ndarray, lam: float) -> dict:
    """ERMM (or EMM, depending on the estimates passed) statistic for X."""
    comp = max_m_components(X, x_prev, center, cov)
    comp["stat"] = ewma(comp["M"], lam, MU_M)
    # EWMA of |U| and |V| for diagnosis plots; in control E|U| = E|V| = sqrt(2/pi)
    comp["U_ewma"] = ewma(np.abs(comp["U"]), lam, HALF_NORMAL_MEAN)
    comp["V_ewma"] = ewma(np.abs(comp["V"]), lam, HALF_NORMAL_MEAN)
    return comp


def mewma_path(X: np.ndarray, center: np.ndarray, cov: np.ndarray,
               lam: float) -> np.ndarray:
    """MEWMA statistic of Lowry et al. (1992) with the asymptotic covariance."""
    X = np.asarray(X, dtype=float)
    Z = ewma(np.moveaxis(X - center, -2, -1), lam, 0.0)      # (..., p, T)
    Z = np.moveaxis(Z, -1, -2)                               # (..., T, p)
    S_inv = np.linalg.inv(cov)
    return (2 - lam) / lam * np.einsum("...i,ij,...j->...", Z, S_inv, Z)
