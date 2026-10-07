import pytest
from nominal_price_invariance.openrouter import parse_allocation


def test_parse_clean_json():
    x = parse_allocation('{"A":61,"B":39}')
    assert (x.weight_a, x.weight_b) == (61, 39)


def test_parse_embedded_json():
    x = parse_allocation('Answer: {"A": 50, "B": 50}')
    assert x.weight_a == 50


def test_reject_bad_sum():
    with pytest.raises(ValueError):
        parse_allocation('{"A":60,"B":50}')
