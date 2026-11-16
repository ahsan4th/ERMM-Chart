"""Design of the chart by Monte Carlo simulation.

calibrate()      Algorithm 1 of the manuscript: the constant L (or the MEWMA
                 limit h) that gives the target in-control ARL when the
                 parameters are estimated from a Phase I sample of size N.
evaluate_arl()   Algorithm 2: ARL, SDRL and MRL for a list of shifts with
                 common random numbers.
spec_driven_shifts()  translates design specifications of the product into
                 the mean and variance shifts that the chart must detect.

Every function takes its experimental parameters explicitly; default values
for the user interface live in config.py.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import signal, stats

from .charts import MU_M, SIGMA_M, chi2_to_normal, ucl_from_L
from .estimators import estimate


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _rep_rng(seed: int, rep: int) -> np.random.Generator:
    """Independent, reproducible stream for replication `rep`."""
    return np.random.default_rng(np.random.SeedSequence([seed, rep]))


def _estimate_rep(Xi, method, alpha, n_starts, reweight_quantile, rng):
    est = estimate(Xi, method, alpha, n_starts, reweight_quantile, rng)
    return est.center, np.linalg.inv(est.cov)


def _stat_segment(chart, Xseg, x_prev, center, S_inv, lam, state):
    """Chart statistic for one segment of one replication; returns (stat, state)."""
    p = Xseg.shape[1]
    a, b = [lam], [1.0, -(1.0 - lam)]
    if chart == "maxm":
        prev = np.vstack([x_prev[None, :], Xseg[:-1]])
        du, dv = Xseg - center, Xseg - prev
        u = chi2_to_normal(np.einsum("ti,ij,tj->t", du, S_inv, du), p)
        v = chi2_to_normal(0.5 * np.einsum("ti,ij,tj->t", dv, S_inv, dv), p)
        m = np.maximum(np.abs(u), np.abs(v))
        out, _ = signal.lfilter(a, b, m, zi=[(1 - lam) * state])
        return out, out[-1]
    if chart == "mewma":
        z, _ = signal.lfilter(a, b, Xseg - center, axis=0,
                              zi=((1 - lam) * state)[None, :])
        out = (2 - lam) / lam * np.einsum("ti,ij,tj->t", z, S_inv, z)
        return out, z[-1]
    raise ValueError(f"unknown chart '{chart}'")


def _initial_state(chart, p):
    return MU_M if chart == "maxm" else np.zeros(p)


# ---------------------------------------------------------------------------
# Algorithm 1: calibration of the control limit
# ---------------------------------------------------------------------------
@dataclass
class Calibration:
    chart: str            # "maxm" (EMM/ERMM) or "mewma"
    method: str           # "classical" or "fastmcd"
    p: int
    N: int
    lam: float
    arl0_target: float
    n_rep: int
    horizon: int
    seed: int
    limit: float          # UCL (Max-M) or h (MEWMA)
    L: float | None       # Max-M constant, None for MEWMA
    grid: np.ndarray = field(repr=False)
    arl_grid: np.ndarray = field(repr=False)
    sdrl_grid: np.ndarray = field(repr=False)
    truncated_share: float = 0.0   # share of runs that reached the horizon at the limit


def _limit_grid(chart, p, lam, centre, half_width, n_points):
    return np.linspace(centre - half_width, centre + half_width, n_points)


def _run_lengths_for_grid(chart, method, p, N, lam, grid, n_rep, horizon, seed,
                          alpha, n_starts, reweight_quantile, segment, progress):
    """Run lengths (n_rep, len(grid)) of in-control charts for every candidate limit."""
    grid = np.sort(np.asarray(grid, dtype=float))
    rl = np.full((n_rep, grid.size), horizon, dtype=np.int64)
    top = grid[-1]
    for r in range(n_rep):
        rng = _rep_rng(seed, r)
        X1 = rng.standard_normal((N, p))      # affine invariance: Sigma_0 = I
        center, S_inv = _estimate_rep(X1, method, alpha, n_starts,
                                      reweight_quantile, rng)
        state, x_prev, run_max, t0 = _initial_state(chart, p), X1[-1], -np.inf, 0
        while t0 < horizon:
            n = min(segment, horizon - t0)
            Xs = rng.standard_normal((n, p))
            s, state = _stat_segment(chart, Xs, x_prev, center, S_inv, lam, state)
            x_prev = Xs[-1]
            cm = np.maximum.accumulate(np.maximum(s, run_max))
            # first index with running maximum above each candidate limit
            hit = np.searchsorted(cm, grid, side="right")
            new = (hit < n) & (rl[r] == horizon)
            rl[r, new] = t0 + hit[new] + 1
            run_max = cm[-1]
            t0 += n
            if run_max > top:
                break
        if progress is not None and (r + 1) % max(1, n_rep // 100) == 0:
            progress((r + 1) / n_rep)
    return grid, rl


def calibrate(chart: str, method: str, p: int, N: int, lam: float,
              arl0_target: float, n_rep: int, horizon: int, seed: int,
              alpha: float, n_starts: int, reweight_quantile: float,
              segment: int = 500, pilot_fraction: float = 0.1,
              progress=None) -> Calibration:
    """Calibrate the control limit so that the in-control ARL equals arl0_target.

    A pilot run on a wide grid brackets the limit; the main run uses a narrow
    grid around it and interpolates log(ARL) linearly in the limit.
    """
    if chart == "maxm":
        centre, half = ucl_from_L(3.0, lam), 1.6 * SIGMA_M * np.sqrt(lam / (2 - lam))
    elif chart == "mewma":
        centre = stats.chi2.ppf(1 - 1 / arl0_target, p) + 2.0
        half = 0.6 * centre
    else:
        raise ValueError(f"unknown chart '{chart}'")
    n_pilot = max(200, int(pilot_fraction * n_rep))
    wrap = (lambda f: progress(0.25 * f)) if progress else None
    g, rl = _run_lengths_for_grid(chart, method, p, N, lam,
                                  _limit_grid(chart, p, lam, centre, half, 41),
                                  n_pilot, horizon, seed + 1_000_003, alpha,
                                  n_starts, reweight_quantile, segment, wrap)
    arl = rl.mean(axis=0)
    if not (arl[0] < arl0_target < arl[-1]):
        raise RuntimeError(f"pilot grid does not bracket ARL0={arl0_target}: "
                           f"ARL range {arl[0]:.1f}-{arl[-1]:.1f}")
    pilot_limit = np.interp(np.log(arl0_target), np.log(arl), g)
    step = g[1] - g[0]
    wrap = (lambda f: progress(0.25 + 0.75 * f)) if progress else None
    g, rl = _run_lengths_for_grid(chart, method, p, N, lam,
                                  _limit_grid(chart, p, lam, pilot_limit, 3 * step, 25),
                                  n_rep, horizon, seed, alpha, n_starts,
                                  reweight_quantile, segment, wrap)
    arl, sdrl = rl.mean(axis=0), rl.std(axis=0, ddof=1)
    if not (arl[0] < arl0_target < arl[-1]):
        raise RuntimeError("main grid does not bracket the target ARL0; "
                           "increase n_rep or check the design")
    if np.any(np.diff(arl) < 0):
        # Monte Carlo noise cannot make ARL decrease with the limit when the
        # same replications are used for every candidate (running maximum).
        raise AssertionError("ARL is not monotone in the limit")
    limit = float(np.interp(np.log(arl0_target), np.log(arl), g))
    k = int(np.clip(np.searchsorted(g, limit), 0, g.size - 1))
    trunc = float(np.mean(rl[:, k] >= horizon))
    L = None
    if chart == "maxm":
        L = float((limit - MU_M) / (SIGMA_M * np.sqrt(lam / (2 - lam))))
    return Calibration(chart, method, p, N, lam, arl0_target, n_rep, horizon,
                       seed, limit, L, g, arl, sdrl, trunc)


# ---------------------------------------------------------------------------
# Algorithm 2: out-of-control run length with common random numbers
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Shift:
    label: str
    mean_sd: np.ndarray        # (p,) mean shift in units of the SD of each characteristic
    var_factor: np.ndarray     # (p,) factor multiplying each variance


def uniform_shift(p: int, a: float, b: float, label: str | None = None) -> Shift:
    """The shifts of the manuscript: every mean moves by a SD, every variance x b."""
    return Shift(label or f"a={a:g}, b={b:g}", np.full(p, float(a)), np.full(p, float(b)))


def evaluate_arl(charts: list[dict], shifts: list[Shift], corr: np.ndarray, N: int,
                 n_rep: int, max_rl: int, seed: int, alpha: float, n_starts: int,
                 reweight_quantile: float, segment: int = 200, progress=None) -> list[dict]:
    """ARL, SDRL and MRL of several charts for several shifts.

    charts: list of {"name", "chart" ("maxm"/"mewma"), "method", "lam", "limit"}.
    corr  : (p, p) in-control correlation matrix (unit variances).
    The same Phase I sample and Phase II noise are used for all charts and
    shifts of a replication (common random numbers).
    """
    corr = np.asarray(corr, dtype=float)
    p = corr.shape[0]
    chol = np.linalg.cholesky(corr)
    rl = np.full((n_rep, len(charts), len(shifts)), -1, dtype=np.int64)
    methods = sorted({c["method"] for c in charts})
    for r in range(n_rep):
        rng = _rep_rng(seed, r)
        X1 = rng.standard_normal((N, p)) @ chol.T
        est = {m: _estimate_rep(X1, m, alpha, n_starts, reweight_quantile,
                                _rep_rng(seed + 7_919, r)) for m in methods}
        noise_rng = _rep_rng(seed + 104_729, r)
        segments: list[np.ndarray] = []        # cached Phase II noise of this replication

        def noise(k):
            while len(segments) <= k:
                segments.append(noise_rng.standard_normal((segment, p)) @ chol.T)
            return segments[k]

        for j, sh in enumerate(shifts):
            if sh.mean_sd.shape != (p,) or sh.var_factor.shape != (p,):
                raise ValueError(f"shift '{sh.label}' does not have p={p} components")
            scale = np.sqrt(sh.var_factor)
            for c_i, c in enumerate(charts):
                center, S_inv = est[c["method"]]
                state, x_prev, t0, k = _initial_state(c["chart"], p), X1[-1], 0, 0
                while t0 < max_rl:
                    Xs = sh.mean_sd + noise(k) * scale
                    s, state = _stat_segment(c["chart"], Xs, x_prev, center,
                                             S_inv, c["lam"], state)
                    above = np.nonzero(s > c["limit"])[0]
                    if above.size:
                        rl[r, c_i, j] = min(t0 + above[0] + 1, max_rl)
                        break
                    x_prev, t0, k = Xs[-1], t0 + segment, k + 1
                if rl[r, c_i, j] < 0:
                    rl[r, c_i, j] = max_rl
        if progress is not None and (r + 1) % max(1, n_rep // 100) == 0:
            progress((r + 1) / n_rep)
    out = []
    for c_i, c in enumerate(charts):
        for j, sh in enumerate(shifts):
            x = rl[:, c_i, j]
            out.append({"chart": c["name"], "shift": sh.label,
                        "ARL": x.mean(), "SE": x.std(ddof=1) / np.sqrt(n_rep),
                        "SDRL": x.std(ddof=1), "MRL": float(np.median(x)),
                        "truncated": float(np.mean(x >= max_rl))})
    return out


# ---------------------------------------------------------------------------
# From product specifications to the shifts the chart has to detect
# ---------------------------------------------------------------------------
def spec_driven_shifts(names, center, sd, lsl, usl, target_nc):
    """Critical shifts implied by the specification limits of each characteristic.

    For characteristic j with robust centre m_j and robust SD s_j, d_j is the
    distance to the nearer specification limit in SD units. Under normality the
    nonconforming fraction reaches `target_nc` when
      - the mean moves towards that limit by  a_j = d_j - z_{1 - target_nc}, or
      - the variance grows by the factor      b_j = (d_j / z_{1 - target_nc})^2.
    Characteristics without limits are skipped.
    """
    if not (0 < target_nc < 0.5):
        raise ValueError("target_nc must lie in (0, 0.5)")
    z = stats.norm.isf(target_nc)
    rows = []
    for j, name in enumerate(names):
        lo, hi = lsl[j], usl[j]
        if np.isnan(lo) and np.isnan(hi):
            continue
        d_hi = (hi - center[j]) / sd[j] if not np.isnan(hi) else np.inf
        d_lo = (center[j] - lo) / sd[j] if not np.isnan(lo) else np.inf
        d = min(d_hi, d_lo)
        direction = 1.0 if d_hi <= d_lo else -1.0
        nc_now = (stats.norm.sf(d_hi) if np.isfinite(d_hi) else 0.0) + \
                 (stats.norm.sf(d_lo) if np.isfinite(d_lo) else 0.0)
        rows.append({"characteristic": name, "nearer_limit": "USL" if direction > 0 else "LSL",
                     "distance_sd": d, "nonconforming_now": nc_now,
                     "critical_mean_shift_sd": d - z, "direction": direction,
                     "critical_var_factor": (d / z) ** 2 if d > 0 else np.nan})
    return rows
