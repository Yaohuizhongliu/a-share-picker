import numpy as np

import pandas as pd

from ashare_picker.adaptive_walkforward import (
    global_walkforward_gate,
    monthly_breakdown,
    rolling_gate,
)


def test_rolling_gate_requires_return_profit_factor_and_drawdown():
    good = {
        "trades": 10,
        "signal_days": 5,
        "mean_return": 0.002,
        "profit_factor": 1.2,
        "max_drawdown": -0.05,
    }
    assert rolling_gate(good)
    assert not rolling_gate({**good, "trades": 7})
    assert not rolling_gate({**good, "mean_return": 0.0005})
    assert not rolling_gate({**good, "profit_factor": 1.0})
    assert not rolling_gate({**good, "max_drawdown": -0.11})
    assert not rolling_gate({**good, "mean_return": np.nan})



def test_global_walkforward_gate_requires_stable_full_period_edge():
    monthly = pd.DataFrame(
        [
            {"month": "2026-07", "mean_return": 0.001},
            {"month": "2026-08", "mean_return": -0.0002},
            {"month": "2026-09", "mean_return": 0.0015},
        ]
    )
    good = {
        "trades": 40,
        "signal_days": 15,
        "mean_return": 0.001,
        "profit_factor": 1.2,
        "max_drawdown": -0.05,
    }
    assert global_walkforward_gate(good, monthly)
    assert not global_walkforward_gate({**good, "mean_return": 0.0004}, monthly)
    assert not global_walkforward_gate({**good, "profit_factor": 1.0}, monthly)
    losing_months = monthly.assign(mean_return=[-0.001, -0.002, 0.001])
    assert not global_walkforward_gate(good, losing_months)


def test_monthly_breakdown_splits_realized_returns():
    trades = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-07-01", "2026-07-02", "2026-08-03"]),
            "net_return": [0.01, -0.005, 0.02],
        }
    )
    result = monthly_breakdown(trades)
    assert result["month"].tolist() == ["2026-07", "2026-08"]
    assert result["trades"].tolist() == [2, 1]
    assert result.loc[0, "mean_return"] == 0.0025
