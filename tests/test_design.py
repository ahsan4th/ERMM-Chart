import numpy as np

from ermm.design import calibrate, evaluate_arl, uniform_shift


def test_calibration_brackets_and_is_reproducible():
    a = calibrate("maxm", "classical", 2, 60, 0.2, 100.0, 300, 3000, 11, 0.75, 100, 0.975)
    b = calibrate("maxm", "classical", 2, 60, 0.2, 100.0, 300, 3000, 11, 0.75, 100, 0.975)
    assert a.limit == b.limit
    assert np.all(np.diff(a.arl_grid) >= 0)


def test_larger_shift_is_detected_faster():
    charts = [dict(name="EMM", chart="maxm", method="classical", lam=0.2, limit=2.2)]
    out = evaluate_arl(charts, [uniform_shift(2, 0.5, 1.0), uniform_shift(2, 2.0, 1.0)],
                       np.eye(2), 60, 200, 2000, 5, 0.75, 100, 0.975)
    assert out[1]["ARL"] < out[0]["ARL"]
