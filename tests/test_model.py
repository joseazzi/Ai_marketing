import numpy as np

from budget_agent.data import quality_report, simulate
from budget_agent.mmm import MMM, adstock, hill


def test_adstock_steady_state_and_carryover():
    x = np.r_[np.full(200, 100.0)]
    a = adstock(x, 0.6)
    assert abs(a[-1] - 100) < 1e-6          # normalised: constant spend -> same level
    pulse = adstock(np.r_[100.0, np.zeros(5)], 0.5)
    assert pulse[1] > 0 and pulse[2] < pulse[1]  # effect fades gradually


def test_hill_saturates():
    assert hill(1000, 1000, 2) == 0.5
    assert hill(1e9, 1000, 2) > 0.999
    assert hill(2000, 1000, 1.5) < 2 * hill(1000, 1000, 1.5)  # double spend != double effect


def test_quality_report_flags_short_data():
    sim = simulate(weeks=40)
    q = quality_report(sim.data)
    assert not q["ok_to_model"] and q["issues"]


def test_model_fits_and_roundtrips(tmp_path):
    sim = simulate(weeks=130, seed=3)
    m = MMM(maxiter=25).fit(sim.data)
    assert m.fit_metrics["r2"] > 0.8
    assert m.fit_metrics["mape"] < 0.05
    m.save(tmp_path / "m.json")
    m2 = MMM.load(tmp_path / "m.json")
    assert np.allclose(m.predict(sim.data), m2.predict(sim.data))
    s = m.channel_summary(sim.data)
    assert set(s.channel) == set(sim.truth)
