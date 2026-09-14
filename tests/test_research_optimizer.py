import numpy as np
import pandas as pd

from ashare_picker.research_optimizer import (
    _stock_features,
    metrics,
    parameter_grid,
    select_signals,
)


def test_target_enters_next_open_and_charges_costs():
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=5),
            "open": [10.0, 10.0, 10.2, 10.3, 10.4],
            "high": [10.1, 10.2, 10.3, 10.4, 10.5],
            "low": [9.9, 9.9, 10.1, 10.2, 10.3],
            "close": [10.0, 10.1, 10.2, 10.3, 10.4],
            "amount": [1e8] * 5,
            "main_net": [1e7] * 5,
        }
    )
    out = _stock_features(frame, costs=0.0016, stop_loss=0.03, take_profit=0.06)
    expected = 10.1 / 10.0 - 1.0 - 0.0016
    assert np.isclose(out.loc[0, "net_return_1d"], expected)


def test_select_signals_caps_day_and_industry():
    panel = pd.DataFrame(
        {
            "date": [pd.Timestamp("2026-07-01")] * 6,
            "industry": ["A", "A", "A", "B", "B", "C"],
            "net_return_2d": [0.01] * 6,
        }
    )
    mask = pd.Series([True] * 6)
    score = pd.Series([6, 5, 4, 3, 2, 1], dtype=float)
    selected = select_signals(
        panel, mask, score, pd.Timestamp("2026-07-01"), pd.Timestamp("2026-07-01"),
        horizon=2, max_per_day=5, max_per_industry=2
    )
    assert len(selected) == 5
    assert selected["industry"].value_counts().max() <= 2


def test_optimizer_grid_and_metrics_are_nonempty():
    assert len(parameter_grid()) > 100
    signals = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-07-01", "2026-07-01", "2026-07-02"]),
            "net_return_1d": [0.01, -0.005, 0.02],
        }
    )
    result = metrics(signals, 1)
    assert result["trades"] == 3
    assert result["signal_days"] == 2
    assert result["mean_return"] > 0
