"""Interpretation of signals: which component, which characteristic, which unit."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def signal_table(dates: pd.Series, stat: np.ndarray, limit: float,
                 gap_days: int) -> pd.DataFrame:
    """One row per monitored day with the statistic, the limit and flags."""
    d = pd.to_datetime(pd.Series(dates).reset_index(drop=True))
    gap = d.diff().dt.days
    return pd.DataFrame({"date": d, "statistic": stat, "limit": limit,
                         "signal": stat > limit,
                         "after_stop": (gap > gap_days).fillna(False)})


def first_signal(tbl: pd.DataFrame):
    hits = tbl.index[tbl["signal"]]
    return None if len(hits) == 0 else int(hits[0])


def component_summary(U: np.ndarray, V: np.ndarray, start: int) -> dict:
    """Share of days dominated by each component and the sign of V after `start`."""
    u, v = np.abs(U[start:]), np.abs(V[start:])
    return {"days": int(u.size),
            "mean_abs_U": float(u.mean()), "mean_abs_V": float(v.mean()),
            "in_control_mean_abs": float(np.sqrt(2 / np.pi)),
            "share_location_dominant": float(np.mean(u >= v)),
            "share_V_negative": float(np.mean(V[start:] < 0))}


def characteristic_deviations(X: np.ndarray, names: list[str], center: np.ndarray,
                              cov: np.ndarray, start: int) -> pd.DataFrame:
    """Mean shift (in robust SD) and SD ratio of each characteristic after `start`."""
    sd = np.sqrt(np.diag(cov))
    after = X[start:]
    if after.shape[0] < 2:
        raise ValueError("need at least two monitored days after the signal")
    mean_after = after.mean(axis=0)
    sd_after = after.std(axis=0, ddof=1)
    df = pd.DataFrame({"characteristic": names, "reference_centre": center,
                       "mean_after": mean_after, "shift_robust_sd": (mean_after - center) / sd,
                       "reference_sd": sd, "sd_after": sd_after,
                       "sd_ratio": sd_after / sd})
    df["sd_change_pct"] = 100 * (df["sd_ratio"] - 1)
    return df


def rank_actions(dev: pd.DataFrame, ocap: pd.DataFrame,
                 mean_threshold: float, sd_threshold_pct: float) -> pd.DataFrame:
    """Join deviations with the out-of-control action plan and rank them."""
    rows = []
    for _, r in dev.iterrows():
        kind = []
        if abs(r["shift_robust_sd"]) >= mean_threshold:
            kind.append("level " + ("up" if r["shift_robust_sd"] > 0 else "down"))
        if abs(r["sd_change_pct"]) >= sd_threshold_pct:
            kind.append("spread " + ("larger" if r["sd_change_pct"] > 0 else "smaller"))
        if not kind:
            continue
        m = ocap[ocap["characteristic"] == r["characteristic"]]
        unit = m["process_unit"].iloc[0] if len(m) else "(not mapped)"
        check = m["first_check"].iloc[0] if len(m) else "(add to the action plan)"
        rows.append({"characteristic": r["characteristic"], "change": ", ".join(kind),
                     "shift_robust_sd": r["shift_robust_sd"],
                     "sd_change_pct": r["sd_change_pct"],
                     "process_unit": unit, "first_check": check,
                     "_score": max(abs(r["shift_robust_sd"]),
                                   abs(np.log(r["sd_ratio"])) / np.log(2))})
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values("_score", ascending=False).drop(columns="_score")


def property_comparison(P: pd.DataFrame, ref_mask: np.ndarray,
                        after_mask: np.ndarray) -> pd.DataFrame:
    """Welch tests of product properties between the reference and the signal period."""
    rows = []
    for col in P.columns:
        a = P.loc[ref_mask, col].dropna().to_numpy(float)
        b = P.loc[after_mask, col].dropna().to_numpy(float)
        if a.size < 2 or b.size < 2:
            continue
        t = stats.ttest_ind(b, a, equal_var=False)
        rows.append({"property": col, "n_ref": a.size, "mean_ref": a.mean(),
                     "sd_ref": a.std(ddof=1), "n_after": b.size,
                     "mean_after": b.mean(), "sd_after": b.std(ddof=1),
                     "welch_p": t.pvalue})
    return pd.DataFrame(rows)
