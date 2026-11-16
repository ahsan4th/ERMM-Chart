"""Run (or load) the 10,000-replication calibrations of the case design.

Used by validation.ipynb. Each design is stored with its full ARL grid so that
the Monte Carlo error of the calibrated limit can be computed.
"""
import json, sys, time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ermm.design import calibrate  # noqa: E402

CACHE = Path(__file__).parent / "results" / "calibration_case_design.json"
SETTINGS = dict(p=5, N=100, arl0=370.0, n_rep=10000, horizon=10000, seed=2026,
                alpha=0.75, n_starts=500, reweight_quantile=0.975)
DESIGNS = [("maxm", m, lam) for lam in (0.1, 0.3, 0.5, 0.7, 1.0)
           for m in ("classical", "fastmcd")] + [("mewma", "classical", 0.1),
                                                 ("mewma", "fastmcd", 0.1)]


def key(chart, method, lam):
    return f"{chart}|{method}|{lam}"


def load():
    return json.loads(CACHE.read_text()) if CACHE.exists() else {}


def run_or_load(chart, method, lam, recompute=False):
    cache = load()
    k = key(chart, method, lam)
    if k in cache and not recompute:
        return cache[k]
    s = SETTINGS
    t0 = time.perf_counter()
    c = calibrate(chart, method, s["p"], s["N"], lam, s["arl0"], s["n_rep"], s["horizon"],
                  s["seed"], s["alpha"], s["n_starts"], s["reweight_quantile"])
    rec = dict(chart=chart, method=method, lam=lam, limit=c.limit, L=c.L,
               grid=c.grid.tolist(), arl_grid=c.arl_grid.tolist(),
               sdrl_grid=c.sdrl_grid.tolist(), truncated_share=c.truncated_share,
               seconds=time.perf_counter() - t0, **s)
    if not recompute:
        cache = load()
        cache[k] = rec
        CACHE.write_text(json.dumps(cache, indent=1))
    return rec


if __name__ == "__main__":
    for d in sys.argv[1:]:
        chart, method, lam = d.split(",")
        r = run_or_load(chart, method, float(lam))
        print(d, round(r["limit"], 4), round(r["seconds"]))
