import numpy as np

from ashare_picker.adaptive_walkforward import rolling_gate


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
