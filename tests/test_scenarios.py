import math
from nominal_price_invariance.scenarios import generate_scenarios, optimal_weight_a


def test_design_counts_and_determinism():
    a = generate_scenarios()
    b = generate_scenarios()
    assert a == b
    assert len(a) == 300
    assert sum(s.scenario_type == "symmetric" for s in a) == 100
    assert sum(s.scenario_type == "asymmetric" for s in a) == 200


def test_symmetric_optimum_is_half():
    for s in generate_scenarios():
        if s.scenario_type == "symmetric":
            w = optimal_weight_a(s.mu_a, s.mu_b, s.sigma_a, s.sigma_b, s.rho, s.gamma)
            assert math.isclose(w, 0.5, abs_tol=1e-10)


def test_asymmetric_optima_are_interior():
    for s in generate_scenarios():
        if s.scenario_type == "asymmetric":
            assert 0.15 <= s.optimal_weight_a <= 0.85
