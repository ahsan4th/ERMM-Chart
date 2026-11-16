"""Default settings of the user interface.

The scientific functions take every parameter explicitly; the values below are
the defaults offered by the app and used in the manuscript.
"""

ARL0 = 370.0                 # target in-control average run length
LAMBDA = 0.1                 # design weighting constant of the case study
LAMBDA_CANDIDATES = (0.1, 0.3, 0.5, 0.7)

MCD_ALPHA = 0.75             # robustbase alpha: h = h.alpha.n(0.75, N, p)
MCD_N_STARTS = 500           # random (p+1)-subsets of Fast-MCD (robustbase nsamp)
REWEIGHT_QUANTILE = 0.975    # reweighting cut-off chi2_{p;0.975}

CAL_N_REP_QUICK = 2000       # replications for an interactive calibration
CAL_N_REP_FINAL = 10000      # replications used in the manuscript
HORIZON = 10000              # Phase II horizon of the calibration runs
ARL_N_REP = 500              # replications for interactive ARL evaluation
SEED = 2026

GAP_DAYS = 3                 # a production stop longer than this is flagged
TARGET_NONCONFORMING = 0.01  # nonconforming fraction defining a critical shift
