import numpy as np
from nominal_price_invariance.analysis import utility
from nominal_price_invariance.scenarios import optimal_weight_a


def test_optimum_beats_nearby_weights():
    mu_a, mu_b = 0.11, 0.09
    sa, sb, rho, gamma = 0.22, 0.18, 0.25, 3.0
    w = optimal_weight_a(mu_a, mu_b, sa, sb, rho, gamma)
    u0 = utility(np.array([w]), np.array([mu_a]), np.array([mu_b]), np.array([sa]), np.array([sb]), np.array([rho]), np.array([gamma]))[0]
    for delta in [-0.05, 0.05]:
        w2 = min(1, max(0, w + delta))
        u2 = utility(np.array([w2]), np.array([mu_a]), np.array([mu_b]), np.array([sa]), np.array([sb]), np.array([rho]), np.array([gamma]))[0]
        assert u0 >= u2 - 1e-12
