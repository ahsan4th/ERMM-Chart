import numpy as np
import pytest

from ermm.charts import MU_M, SIGMA_M, chi2_to_normal, ermm_path, ewma, max_m_components, mewma_path
from ermm.design import spec_driven_shifts
from ermm.estimators import classical_estimate


def test_max_m_moments():
    assert MU_M == pytest.approx(2 / np.sqrt(np.pi), abs=1e-12)
    z = np.random.default_rng(0).standard_normal((2, 2_000_000))
    m = np.abs(z).max(axis=0)
    assert m.mean() == pytest.approx(MU_M, abs=3e-3)
    assert m.std() == pytest.approx(SIGMA_M, abs=3e-3)


def test_chi2_to_normal_tails():
    q = np.array([1e-12, 1.0, 5.0, 200.0])
    u = chi2_to_normal(q, 5)
    assert np.all(np.isfinite(u)) and np.all(np.diff(u) > 0)


def test_ewma_recursion():
    z = np.random.default_rng(1).random(50)
    ref, e = [], 0.7
    for v in z:
        e = 0.2 * v + 0.8 * e
        ref.append(e)
    np.testing.assert_allclose(ewma(z, 0.2, 0.7), ref, atol=1e-14)


def test_affine_invariance():
    rng = np.random.default_rng(2)
    X1, X2 = rng.standard_normal((80, 3)), rng.standard_normal((40, 3))
    A, b = np.array([[2.0, 0.3, 0], [0, 1.5, -0.4], [0.1, 0, 0.7]]), np.array([5.0, -2.0, 1.0])
    e1, e2 = classical_estimate(X1), classical_estimate(X1 @ A.T + b)
    s1 = ermm_path(X2, X1[-1], e1.center, e1.cov, 0.1)["stat"]
    s2 = ermm_path(X2 @ A.T + b, X1[-1] @ A.T + b, e2.center, e2.cov, 0.1)["stat"]
    np.testing.assert_allclose(s1, s2, atol=1e-10)
    np.testing.assert_allclose(mewma_path(X2, e1.center, e1.cov, 0.1),
                               mewma_path(X2 @ A.T + b, e2.center, e2.cov, 0.1), atol=1e-9)


def test_successive_difference_ignores_mean_shift():
    rng = np.random.default_rng(3)
    X = rng.standard_normal((30, 2))
    c = max_m_components(X, X[0], np.zeros(2), np.eye(2))
    d = max_m_components(X + 3.0, X[0] + 3.0, np.zeros(2), np.eye(2))
    np.testing.assert_allclose(c["Q_V"], d["Q_V"], atol=1e-12)


def test_spec_driven_shift_loi_example():
    # LOI: robust centre 4.259, robust SD 0.378 (Table 7), USL 5.0 (SNI 2049)
    r = spec_driven_shifts(["LOI"], np.array([4.259]), np.array([0.378]),
                           np.array([np.nan]), np.array([5.0]), 0.05)[0]
    assert r["distance_sd"] == pytest.approx(1.960, abs=1e-3)
    assert r["critical_mean_shift_sd"] == pytest.approx(1.960 - 1.645, abs=2e-3)
