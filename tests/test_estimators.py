"""The Python estimator must reproduce robustbase::covMcd (alpha = 0.75)."""
from pathlib import Path

import numpy as np
import pytest

from ermm.estimators import (h_alpha_n, mcd_consistency, mcd_small_sample_raw,
                             mcd_small_sample_rew, reweighted_fast_mcd)

DATA = Path(__file__).parent / "data"


@pytest.mark.parametrize("k", [0, 1, 3, 4])
def test_agrees_with_robustbase(k):
    X = np.loadtxt(DATA / f"x{k}.csv", delimiter=",")
    p = X.shape[1]
    est = reweighted_fast_mcd(X, 0.75, 500, 0.975, np.random.default_rng(7))
    np.testing.assert_allclose(est.center, np.loadtxt(DATA / f"r_center{k}.txt"), atol=1e-12)
    np.testing.assert_allclose(est.cov, np.loadtxt(DATA / f"r_cov{k}.txt").reshape(p, p), atol=1e-12)
    np.testing.assert_array_equal(est.weights, np.loadtxt(DATA / f"r_weights{k}.txt"))


def test_robustbase_factors():
    # values printed by robustbase 0.99-2 for N = 100, p = 5, alpha = 0.75
    assert h_alpha_n(0.75, 100, 5) == 76
    assert mcd_consistency(5, 76 / 100) == pytest.approx(1.395754, abs=1e-6)
    assert mcd_small_sample_raw(5, 100, 0.75) == pytest.approx(1.063854, abs=1e-6)
    assert mcd_small_sample_rew(5, 100, 0.75) == pytest.approx(1.028611, abs=1e-6)


def test_exact_fit_fails_loudly():
    X = np.random.default_rng(0).standard_normal((50, 3))
    X[:, 2] = X[:, 0] + X[:, 1]          # all points on a hyperplane
    with pytest.raises(np.linalg.LinAlgError):
        reweighted_fast_mcd(X, 0.75, 100, 0.975, np.random.default_rng(1))
