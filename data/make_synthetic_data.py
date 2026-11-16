"""Generate the synthetic OPC finish-mill data set shipped with the app.

The plant records of the manuscript are confidential. This script creates a
data set with the same structure (160 production days, 5 quality
characteristics, 4 product properties) whose parameters are the published
summary statistics of the case study:

* regular Phase I days: multivariate normal with the Fast-MCD centre and SDs of
  Table 7 and the correlations of Table 5;
* eight unusual Phase I days with the values reported in Section 4.4;
* Phase II: 15 days with the 'before' deviations of Table 15, then, from
  27 September 2022, 45 days with the 'after' deviations and SDs of Table 15;
* product properties: linear in the standardized characteristics with the
  correlations of Table 6 and the Phase I means and SDs of Table 16.

The values are synthetic. They illustrate the tool and must not be read as
plant data.
"""
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 20221
HERE = Path(__file__).parent
NAMES = ["Blaine", "Residue45", "FreeLime", "LOI", "SO3"]
CENTRE = np.array([356.815, 12.492, 0.942, 4.259, 1.780])         # Table 7, Fast-MCD
SD = np.array([17.872, 1.800, 0.108, 0.378, 0.126])               # Table 7, Fast-MCD
CORR = np.array([[1.00, -0.23, -0.08, 0.29, 0.23],                # Table 5
                 [-0.23, 1.00, 0.41, 0.08, -0.42],
                 [-0.08, 0.41, 1.00, 0.13, -0.36],
                 [0.29, 0.08, 0.13, 1.00, -0.17],
                 [0.23, -0.42, -0.36, -0.17, 1.00]])
DEV_BEFORE = np.array([0.28, -1.01, -0.96, 0.48, 1.34])           # Table 15
DEV_AFTER = np.array([0.46, -1.10, -0.66, 0.23, 0.87])            # Table 15
SD_AFTER = np.array([13.913, 0.922, 0.076, 0.596, 0.088])         # Table 15
UNUSUAL = {"2022-01-03": ("LOI", 6.78), "2022-01-25": ("SO3", 1.39),  # Section 4.4
           "2022-01-26": ("SO3", 1.41), "2022-02-02": ("LOI", 5.34),
           "2022-02-25": ("LOI", 6.00), "2022-02-28": ("FreeLime", 0.59),
           "2022-04-28": ("LOI", 5.93), "2022-05-13": ("LOI", 7.19)}
PROPS = ["InitialSetting_min", "FinalSetting_min", "Strength3d_kgcm2", "Strength7d_kgcm2"]
PROP_MEAN = np.array([169.7, 274.2, 214.1, 310.8])                # Table 16, Phase I
PROP_SD = np.array([19.5, 18.3, 16.0, 25.4])
PROP_CORR = np.array([[0.20, -0.03, 0.02, -0.21, 0.23],           # Table 6
                      [0.27, -0.04, 0.01, -0.10, 0.24],
                      [0.39, -0.47, -0.16, 0.05, 0.24],
                      [0.33, -0.45, -0.07, -0.10, 0.18]])


def production_days(rng):
    """160 production days of 2022 with stops, 100 in Phase I and 60 in Phase II."""
    def pick(start, end, n, keep, exclude=()):
        cal = pd.date_range(start, end, freq="D")
        cal = cal[~cal.isin(pd.DatetimeIndex(exclude))]
        keep = pd.DatetimeIndex(keep).unique()
        free = cal[~cal.isin(keep)]
        extra = rng.choice(len(free), n - len(keep), replace=False)
        return keep.union(free[np.sort(extra)])
    stop_1 = pd.date_range("2022-03-14", "2022-04-20")           # 38-day stop
    stop_2 = pd.date_range("2022-10-11", "2022-10-15")
    stop_3 = pd.date_range("2022-11-05", "2022-11-09")
    p1 = pick("2022-01-03", "2022-09-07", 100,
              ["2022-01-03", "2022-03-13", "2022-04-21", "2022-09-07", *UNUSUAL], stop_1)
    p2a = pick("2022-09-08", "2022-09-26", 15, ["2022-09-08", "2022-09-26"])
    p2b = pick("2022-09-27", "2022-11-30", 45, ["2022-09-27", "2022-11-30"],
               stop_2.union(stop_3))
    return p1.union(p2a).union(p2b)


def main():
    rng = np.random.default_rng(SEED)
    dates = production_days(rng)
    assert len(dates) == 160
    L = np.linalg.cholesky(CORR)
    Z = rng.standard_normal((160, 5)) @ L.T                      # standardized, Phase I scale
    dev = np.zeros((160, 5))
    scale = np.ones((160, 5))
    dev[100:115] = DEV_BEFORE
    dev[115:] = DEV_AFTER
    scale[115:] = SD_AFTER / SD
    X = CENTRE + SD * (dev + scale * Z)
    df = pd.DataFrame(X, columns=NAMES)
    df.insert(0, "Date", dates.strftime("%Y-%m-%d"))
    for day, (col, val) in UNUSUAL.items():
        df.loc[df["Date"] == day, col] = val
    # product properties from the standardized characteristics (Phase I scale)
    zc = (df[NAMES].to_numpy() - CENTRE) / SD
    Rinv = np.linalg.inv(CORR)
    for k, prop in enumerate(PROPS):
        beta = Rinv @ PROP_CORR[k]
        r2 = PROP_CORR[k] @ beta
        assert 0 < r2 < 1
        e = rng.standard_normal(160)
        df[prop] = PROP_MEAN[k] + PROP_SD[k] * (zc @ beta + np.sqrt(1 - r2) * e)
    df["Blaine"] = df["Blaine"].round(0)
    for col in ["Residue45", "FreeLime", "LOI", "SO3"]:
        df[col] = df[col].round(3)
    for col in PROPS:
        df[col] = df[col].round(1)
    df["Phase"] = ["I"] * 100 + ["II"] * 60
    df.to_csv(HERE / "opc_synthetic.csv", index=False)
    print(df.groupby("Phase")[NAMES].agg(["mean", "std"]).round(3).T)


if __name__ == "__main__":
    main()
