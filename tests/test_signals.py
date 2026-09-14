import numpy as np
import pandas as pd

from ashare_picker.signals import (
    accumulation_features,
    add_growth_features,
    rounded_bottom_features,
    summarize_backtest,
)


def _price_frame(close: np.ndarray, volume: np.ndarray | None = None) -> pd.DataFrame:
    if volume is None:
        volume = np.full(len(close), 1_000_000.0)
    return pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=len(close), freq="B"),
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": volume,
            "amount": close * volume,
        }
    )


def test_completed_rounded_bottom_is_detected():
    x = np.linspace(-1.0, 1.0, 60)
    close = 10.0 * np.exp(0.12 * np.square(x + 0.20))
    result = rounded_bottom_features(
        _price_frame(close),
        {
            "window": 60,
            "min_r2": 0.42,
            "min_curvature": 0.035,
            "min_recovery": 0.08,
            "min_last_5d_return": 0.0,
        },
    )
    assert result["rounded_bottom"] is True
    assert result["rounded_r2"] > 0.95
    assert result["rounded_recovery"] > 0.08


def test_flat_market_is_not_rounded_bottom():
    result = rounded_bottom_features(
        _price_frame(np.full(60, 10.0)),
        {"window": 60, "min_r2": 0.42, "min_curvature": 0.035, "min_recovery": 0.08},
    )
    assert result["rounded_bottom"] is False


def test_accumulation_requires_flow_and_volume_confirmation():
    rng = np.random.default_rng(7)
    dates = pd.date_range("2026-01-01", periods=60, freq="B")
    main_net = rng.normal(0.0, 800_000.0, 60)
    main_net[-10:] = np.linspace(2_000_000.0, 5_000_000.0, 10)
    flow = pd.DataFrame({"date": dates, "main_net": main_net})
    close = np.linspace(10.0, 10.8, 60)
    volume = np.r_[np.full(50, 1_000_000.0), np.full(10, 1_500_000.0)]
    result = accumulation_features(
        flow,
        _price_frame(close, volume),
        {
            "recent_days": 10,
            "baseline_days": 60,
            "min_positive_ratio": 0.6,
            "min_zscore": 0.45,
            "min_volume_ratio": 1.05,
            "max_abs_10d_price_return": 0.18,
        },
    )
    assert result["accumulation_anomaly"] is True
    assert result["main_net_10d"] > 0
    assert result["volume_ratio"] > 1


def test_growth_uses_financials_and_compares_inside_industry():
    frame = pd.DataFrame(
        {
            "industry": ["软件", "软件", "软件"],
            "revenue_yoy": [10.0, 20.0, 30.0],
            "profit_yoy": [5.0, 15.0, 40.0],
            "momentum_60d": [0.01, 0.02, 0.03],
        }
    )
    out = add_growth_features(frame)
    assert bool(out.iloc[-1]["growth_above_avg"])
    assert out.iloc[-1]["growth_source"] == "财报同比"
    assert out.iloc[-1]["growth_percentile"] == 1.0


def test_backtest_summary_is_grouped_by_horizon():
    trades = pd.DataFrame(
        {
            "horizon": [1, 1, 2],
            "net_return": [0.01, -0.02, 0.03],
        }
    )
    result = summarize_backtest(trades)
    one_day = result.loc[result["horizon"] == 1].iloc[0]
    assert one_day["trades"] == 2
    assert one_day["win_rate"] == 0.5
