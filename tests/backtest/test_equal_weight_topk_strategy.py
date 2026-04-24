# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import pandas as pd

from qlib.contrib.strategy.signal_strategy import EqualWeightTopKStrategy


def test_equal_weight_topk_target_weights_basic():
    strat = EqualWeightTopKStrategy.__new__(EqualWeightTopKStrategy)
    strat.topk = 3

    scores = pd.Series({"A": 0.1, "B": 0.9, "C": 0.5, "D": 0.8, "E": float("nan")})
    current = None  # unused by this method

    out = strat.generate_target_weight_position(scores, current, None, None)

    assert out is not None
    assert set(out) == {"B", "D", "C"}
    for v in out.values():
        assert abs(v - 1.0 / 3.0) < 1e-9


def test_equal_weight_topk_target_weights_fewer_than_topk():
    strat = EqualWeightTopKStrategy.__new__(EqualWeightTopKStrategy)
    strat.topk = 5

    scores = pd.Series({"X": 1.0, "Y": 2.0})
    out = strat.generate_target_weight_position(scores, None, None, None)

    assert out is not None
    assert set(out) == {"Y", "X"}
    assert abs(out["Y"] - 0.5) < 1e-9
    assert abs(out["X"] - 0.5) < 1e-9


def test_equal_weight_topk_target_weights_empty():
    strat = EqualWeightTopKStrategy.__new__(EqualWeightTopKStrategy)
    strat.topk = 3

    scores = pd.Series({"A": float("nan")})
    assert strat.generate_target_weight_position(scores, None, None, None) is None


def test_equal_weight_topk_dataframe_uses_first_column():
    strat = EqualWeightTopKStrategy.__new__(EqualWeightTopKStrategy)
    strat.topk = 2

    scores = pd.DataFrame({"s1": [0.0, 1.0], "s2": [1.0, 0.0]}, index=["a", "b"])
    out = strat.generate_target_weight_position(scores, None, None, None)

    assert out is not None
    assert set(out) == {"b", "a"}
    assert abs(out["b"] - 0.5) < 1e-9


if __name__ == "__main__":
    import pytest

    pytest.main([__file__])
