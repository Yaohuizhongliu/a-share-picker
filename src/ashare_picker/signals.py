from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
        return result if np.isfinite(result) else default
    except (TypeError, ValueError):
        return default


def rounded_bottom_features(prices: pd.DataFrame, cfg: dict[str, Any]) -> dict[str, Any]:
    window = int(cfg.get("window", 60))
    result = {
        "rounded_bottom": False,
        "rounded_score": 0.0,
        "rounded_r2": 0.0,
        "rounded_curvature": 0.0,
        "rounded_recovery": 0.0,
        "last_5d_return": 0.0,
    }
    clean = prices.dropna(subset=["close"]).sort_values("date").tail(window)
    if len(clean) < window:
        return result

    close = clean["close"].to_numpy(dtype=float)
    if np.any(close <= 0):
        return result
    x = np.linspace(-1.0, 1.0, window)
    y = np.log(close)
    a, b, c = np.polyfit(x, y, 2)
    fitted = a * x * x + b * x + c
    residual = float(np.square(y - fitted).sum())
    total = float(np.square(y - y.mean()).sum())
    r2 = max(0.0, 1.0 - residual / total) if total > 1e-12 else 0.0
    vertex = -b / (2 * a) if a > 1e-12 else 99.0
    bottom = float(close.min())
    recovery = float(close[-1] / bottom - 1.0)
    last_5d = float(close[-1] / close[-6] - 1.0)
    min_r2 = float(cfg.get("min_r2", 0.42))
    min_curvature = float(cfg.get("min_curvature", 0.035))
    min_recovery = float(cfg.get("min_recovery", 0.08))
    min_last_5d = float(cfg.get("min_last_5d_return", 0.0))

    # The vertex must be inside the middle portion of the observation window.
    shape_ok = a >= min_curvature and -0.60 <= vertex <= 0.35
    recovery_ok = recovery >= min_recovery and last_5d >= min_last_5d
    # Do not require a perfect neckline breakout; allow a 2% tolerance around
    # the highest price in the first quarter of the bowl.
    neckline = float(np.max(close[: max(10, window // 4)]))
    completion_ok = close[-1] >= neckline * 0.98

    score = np.clip(
        0.35 * min(r2 / max(min_r2, 1e-9), 1.5)
        + 0.30 * min(max(a, 0.0) / max(min_curvature, 1e-9), 1.5)
        + 0.20 * min(recovery / max(min_recovery, 1e-9), 1.5)
        + 0.15 * min(max(last_5d, 0.0) / 0.05, 1.5),
        0.0,
        1.0,
    )
    result.update(
        {
            "rounded_bottom": bool(shape_ok and recovery_ok and completion_ok and r2 >= min_r2),
            "rounded_score": float(score),
            "rounded_r2": float(r2),
            "rounded_curvature": float(a),
            "rounded_recovery": recovery,
            "last_5d_return": last_5d,
        }
    )
    return result


def accumulation_features(
    flow: pd.DataFrame,
    prices: pd.DataFrame,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    result = {
        "accumulation_anomaly": False,
        "accumulation_score": 0.0,
        "main_net_10d": 0.0,
        "main_positive_ratio": 0.0,
        "main_flow_zscore": 0.0,
        "volume_ratio": 0.0,
        "price_return_10d": 0.0,
    }
    recent_days = int(cfg.get("recent_days", 10))
    baseline_days = int(cfg.get("baseline_days", 60))
    clean_flow = flow.dropna(subset=["main_net"]).sort_values("date").tail(baseline_days)
    clean_prices = prices.dropna(subset=["close", "volume"]).sort_values("date")
    if len(clean_flow) < max(recent_days + 10, baseline_days // 2) or len(clean_prices) < recent_days + 20:
        return result

    recent = clean_flow.tail(recent_days)["main_net"].astype(float)
    prior = clean_flow.iloc[:-recent_days]["main_net"].astype(float)
    prior_std = float(prior.std(ddof=1))
    zscore = (
        float((recent.mean() - prior.mean()) / prior_std)
        if prior_std > 1e-12
        else (3.0 if recent.mean() > prior.mean() else 0.0)
    )
    main_sum = float(recent.sum())
    positive_ratio = float((recent > 0).mean())

    price_tail = clean_prices.tail(max(30, recent_days + 20))
    price_return = float(price_tail["close"].iloc[-1] / price_tail["close"].iloc[-recent_days] - 1.0)
    recent_volume = float(price_tail["volume"].tail(recent_days).mean())
    prior_volume = float(price_tail["volume"].iloc[:-recent_days].tail(20).mean())
    volume_ratio = recent_volume / prior_volume if prior_volume > 0 else 0.0

    min_positive_ratio = float(cfg.get("min_positive_ratio", 0.60))
    min_zscore = float(cfg.get("min_zscore", 0.45))
    min_volume_ratio = float(cfg.get("min_volume_ratio", 1.05))
    max_abs_return = float(cfg.get("max_abs_10d_price_return", 0.18))
    anomaly = (
        main_sum > 0
        and positive_ratio >= min_positive_ratio
        and zscore >= min_zscore
        and volume_ratio >= min_volume_ratio
        and abs(price_return) <= max_abs_return
    )
    score = np.clip(
        0.30 * max(0.0, min(positive_ratio, 1.0))
        + 0.30 * max(0.0, min(zscore / 2.5, 1.0))
        + 0.20 * max(0.0, min(volume_ratio / 2.0, 1.0))
        + 0.20 * (1.0 if main_sum > 0 else 0.0),
        0.0,
        1.0,
    )
    result.update(
        {
            "accumulation_anomaly": bool(anomaly),
            "accumulation_score": float(score),
            "main_net_10d": main_sum,
            "main_positive_ratio": positive_ratio,
            "main_flow_zscore": zscore,
            "volume_ratio": volume_ratio,
            "price_return_10d": price_return,
        }
    )
    return result


def add_growth_features(candidates: pd.DataFrame) -> pd.DataFrame:
    out = candidates.copy()
    for col in ("revenue_yoy", "profit_yoy", "momentum_60d"):
        if col not in out:
            out[col] = np.nan
        out[col] = pd.to_numeric(out[col], errors="coerce")

    def winsorized(series: pd.Series) -> pd.Series:
        valid = series.dropna()
        if len(valid) < 5:
            return series
        return series.clip(valid.quantile(0.05), valid.quantile(0.95))

    out["revenue_yoy_w"] = out.groupby("industry", group_keys=False)["revenue_yoy"].transform(winsorized)
    out["profit_yoy_w"] = out.groupby("industry", group_keys=False)["profit_yoy"].transform(winsorized)
    out["financial_growth"] = out[["revenue_yoy_w", "profit_yoy_w"]].mean(axis=1, skipna=True)
    out.loc[out[["revenue_yoy_w", "profit_yoy_w"]].isna().all(axis=1), "financial_growth"] = np.nan
    out["industry_financial_growth_mean"] = out.groupby("industry")["financial_growth"].transform("mean")
    out["industry_momentum_mean"] = out.groupby("industry")["momentum_60d"].transform("mean")
    has_financial = out["financial_growth"].notna() & out["industry_financial_growth_mean"].notna()
    out["growth_metric"] = out["financial_growth"].where(has_financial, out["momentum_60d"])
    out["industry_growth_mean"] = out["industry_financial_growth_mean"].where(
        has_financial, out["industry_momentum_mean"]
    )
    out["growth_above_avg"] = out["growth_metric"] > out["industry_growth_mean"]
    out["growth_source"] = np.where(has_financial, "财报同比", "60日价格动量(财报缺失后备)")
    out["growth_percentile"] = out.groupby("industry")["growth_metric"].rank(pct=True).fillna(0.0)
    return out


@dataclass(frozen=True)
class TradeRules:
    stop_loss: float = 0.03
    take_profit: float = 0.06
    max_holding_days: int = 3
    commission_bps_each_side: float = 3.0
    slippage_bps_each_side: float = 5.0


def _simulate_trade(
    prices: pd.DataFrame,
    signal_position: int,
    horizon: int,
    rules: TradeRules,
) -> dict[str, Any] | None:
    entry_position = signal_position + 1
    if entry_position >= len(prices):
        return None
    entry_row = prices.iloc[entry_position]
    entry = _finite(entry_row["open"])
    if entry <= 0:
        return None
    stop = entry * (1.0 - rules.stop_loss)
    target = entry * (1.0 + rules.take_profit)
    last_position = min(entry_position + horizon - 1, len(prices) - 1)
    exit_price = _finite(prices.iloc[last_position]["close"])
    exit_reason = "time"

    for position in range(entry_position, last_position + 1):
        row = prices.iloc[position]
        hit_stop = _finite(row["low"], entry) <= stop
        hit_target = _finite(row["high"], entry) >= target
        if hit_stop:
            exit_price = stop
            exit_reason = "stop"
            last_position = position
            break
        if hit_target:
            exit_price = target
            exit_reason = "target"
            last_position = position
            break

    one_way_cost = (rules.commission_bps_each_side + rules.slippage_bps_each_side) / 10000.0
    net_return = exit_price / entry - 1.0 - 2.0 * one_way_cost
    return {
        "signal_date": pd.Timestamp(prices.iloc[signal_position]["date"]),
        "entry_date": pd.Timestamp(entry_row["date"]),
        "exit_date": pd.Timestamp(prices.iloc[last_position]["date"]),
        "entry": entry,
        "exit": exit_price,
        "horizon": horizon,
        "actual_holding_days": last_position - entry_position + 1,
        "exit_reason": exit_reason,
        "net_return": float(net_return),
    }


def walk_forward_backtest(
    code: str,
    prices: pd.DataFrame,
    flow: pd.DataFrame,
    rounded_cfg: dict[str, Any],
    accumulation_cfg: dict[str, Any],
    rules: TradeRules,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    prices = prices.sort_values("date").reset_index(drop=True)
    flow = flow.sort_values("date").reset_index(drop=True)
    rows: list[dict[str, Any]] = []
    last_signal_position = -10

    for position in range(60, len(prices) - 1):
        date = pd.Timestamp(prices.iloc[position]["date"])
        if date < start or date > end or position - last_signal_position < 4:
            continue
        price_window = prices.iloc[: position + 1]
        flow_window = flow.loc[flow["date"] <= date]
        rounded = rounded_bottom_features(price_window, rounded_cfg)
        accumulation = accumulation_features(flow_window, price_window, accumulation_cfg)
        if not (rounded["rounded_bottom"] or accumulation["accumulation_anomaly"]):
            continue
        last_signal_position = position
        for horizon in range(1, rules.max_holding_days + 1):
            trade = _simulate_trade(prices, position, horizon, rules)
            if trade is not None:
                trade.update(
                    {
                        "code": code,
                        "rounded_bottom": rounded["rounded_bottom"],
                        "accumulation_anomaly": accumulation["accumulation_anomaly"],
                    }
                )
                rows.append(trade)
    return pd.DataFrame(rows)


def summarize_backtest(trades: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "horizon",
        "trades",
        "win_rate",
        "average_return",
        "median_return",
        "worst_trade",
        "best_trade",
    ]
    if trades.empty:
        return pd.DataFrame(columns=columns)
    grouped = trades.groupby("horizon")["net_return"]
    out = grouped.agg(trades="size", average_return="mean", median_return="median", worst_trade="min", best_trade="max")
    out["win_rate"] = grouped.apply(lambda series: float((series > 0).mean()))
    return out.reset_index()[columns]
