import numpy as np
from nominal_price_invariance.analysis import (
    sign_flip_p,
    tost_p,
    utility,
    wilcoxon_p,
)
from nominal_price_invariance.scenarios import optimal_weight_a


def test_optimum_beats_nearby_weights():
    mu_a, mu_b = 0.11, 0.09
    sa, sb, rho, gamma = 0.22, 0.18, 0.25, 3.0
    w = optimal_weight_a(mu_a, mu_b, sa, sb, rho, gamma)
    u0 = utility(
        np.array([w]), np.array([mu_a]), np.array([mu_b]), np.array([sa]),
        np.array([sb]), np.array([rho]), np.array([gamma])
    )[0]
    for delta in [-0.05, 0.05]:
        w2 = min(1, max(0, w + delta))
        u2 = utility(
            np.array([w2]), np.array([mu_a]), np.array([mu_b]), np.array([sa]),
            np.array([sb]), np.array([rho]), np.array([gamma])
        )[0]
        assert u0 >= u2 - 1e-12


def test_equivalence_accepts_tight_zero_effects():
    x = np.array([0.1, -0.1, 0.0, 0.2, -0.2] * 20)
    assert tost_p(x, margin=2.0) < 0.05


def test_equivalence_rejects_large_effect():
    x = np.array([5.0, 4.5, 5.5, 5.2, 4.8] * 20)
    assert tost_p(x, margin=2.0) > 0.05


def test_sign_flip_detects_consistent_effect():
    x = np.linspace(3.0, 6.0, 40)
    assert sign_flip_p(x, draws=4000, seed=123) < 0.01


def test_wilcoxon_all_zero_is_one():
    assert wilcoxon_p(np.zeros(20)) == 1.0
