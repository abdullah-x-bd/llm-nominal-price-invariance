from nominal_price_invariance.prompts import CONDITIONS, build_prompt, prices_for
from nominal_price_invariance.scenarios import generate_scenarios


def test_exact_neutral_replicates():
    s = generate_scenarios()[0]
    assert build_prompt(s, CONDITIONS[0]) == build_prompt(s, CONDITIONS[1])


def test_treatment_pair_only_changes_price_assignment():
    s = generate_scenarios()[0]
    for ai, bi in [(2, 3), (4, 5), (6, 7)]:
        pa1, pb1 = prices_for(s, CONDITIONS[ai])
        pa2, pb2 = prices_for(s, CONDITIONS[bi])
        assert pa1 == pb2
        assert pb1 == pa2
        assert pa1 != pb1
