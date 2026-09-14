from __future__ import annotations

import argparse
import itertools
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

LOG = logging.getLogger("ashare_picker.optimizer")


def load_inputs(report_dir: Path) -> pd.DataFrame:
    prices = pd.read_csv(report_dir / "daily_prices.csv", dtype={"code": str})
    industries = pd.read_csv(report_dir / "universe_industries.csv", dtype={"code": str})
    prices["code"] = prices["code"].str.zfill(6)
    industries["code"] = industries["code"].str.zfill(6)
    prices["date"] = pd.to_datetime(prices["date"], errors="coerce")
    keep = ["code", "industry", "name"]
    frame = prices.merge(industries[keep].drop_duplicates("code"), on="code", how="left")
    numeric = ["open", "high", "low", "close", "volume", "amount", "main_net"]
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["date", "open", "high", "low", "close", "amount"])
    return frame.sort_values(["code", "date"]).reset_index(drop=True)


def _stock_features(group: pd.DataFrame, costs: float, stop_loss: float, take_profit: float) -> pd.DataFrame:
    out = group.copy().sort_values("date")
    close = out["close"]
    amount = out["amount"]
    net = out["main_net"]
    returns = close.pct_change()
    out["ret5"] = close.pct_change(5)
    out["ret10"] = close.pct_change(10)
    out["ret20"] = close.pct_change(20)
    out["ret60"] = close.pct_change(60)
    out["ma20_gap"] = close / close.rolling(20).mean() - 1.0
    out["breakout20"] = close / out["high"].rolling(20).max().shift(1) - 1.0
    out["vol20"] = returns.rolling(20).std()
    out["amount_ratio5_20"] = amount.rolling(5).mean() / amount.rolling(20).mean()
    out["flow5"] = net.rolling(5).sum() / amount.rolling(5).sum()
    out["flow10"] = net.rolling(10).sum() / amount.rolling(10).sum()
    out["flow_positive_ratio10"] = net.gt(0).rolling(10).mean()

    daily_flow = np.where(amount.gt(0), net / amount, 0.0)
    daily_flow = pd.Series(daily_flow, index=out.index)
    prior_mean = daily_flow.shift(10).rolling(40).mean()
    prior_std = daily_flow.shift(10).rolling(40).std()
    out["flow_z"] = (daily_flow.rolling(10).mean() - prior_mean) / prior_std.replace(0, np.nan)

    n = len(out)
    for horizon in (1, 2, 3):
        values = np.full(n, np.nan)
        for signal_pos in range(n - 1):
            entry_pos = signal_pos + 1
            entry = float(out.iloc[entry_pos]["open"])
            if not np.isfinite(entry) or entry <= 0:
                continue
            stop = entry * (1.0 - stop_loss)
            target = entry * (1.0 + take_profit)
            last_pos = min(entry_pos + horizon - 1, n - 1)
            exit_price = float(out.iloc[last_pos]["close"])
            for pos in range(entry_pos, last_pos + 1):
                row = out.iloc[pos]
                if float(row["low"]) <= stop:
                    exit_price = stop
                    break
                if float(row["high"]) >= target:
                    exit_price = target
                    break
            values[signal_pos] = exit_price / entry - 1.0 - costs
        out[f"net_return_{horizon}d"] = values
    return out


def build_feature_panel(frame: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    strategy = cfg["strategy"]
    costs = 2.0 * (
        float(strategy["commission_bps_each_side"]) + float(strategy["slippage_bps_each_side"])
    ) / 10000.0
    pieces = [
        _stock_features(group, costs, float(strategy["stop_loss"]), float(strategy["take_profit"]))
        for _, group in frame.groupby("code", sort=False)
    ]
    panel = pd.concat(pieces, ignore_index=True)

    daily_market = (
        panel.groupby("date", as_index=False)
        .agg(
            market_ret20=("ret20", "median"),
            market_breadth=("ma20_gap", lambda x: float((x > 0).mean())),
            median_vol20=("vol20", "median"),
        )
    )
    sector_day = (
        panel.groupby(["industry", "date"], as_index=False)
        .agg(sector_net=("main_net", "sum"), sector_amount=("amount", "sum"))
        .sort_values(["industry", "date"])
    )
    sector_day["sector_flow5"] = (
        sector_day.groupby("industry", group_keys=False)["sector_net"]
        .transform(lambda x: x.rolling(5).sum())
        / sector_day.groupby("industry", group_keys=False)["sector_amount"]
        .transform(lambda x: x.rolling(5).sum())
    )

    panel = panel.merge(daily_market, on="date", how="left")
    panel = panel.merge(sector_day[["industry", "date", "sector_flow5"]], on=["industry", "date"], how="left")
    panel["mom20_rank"] = panel.groupby("date")["ret20"].rank(pct=True)
    panel["mom60_rank"] = panel.groupby("date")["ret60"].rank(pct=True)
    panel["flow10_rank"] = panel.groupby("date")["flow10"].rank(pct=True)
    panel["vol20_rank"] = panel.groupby("date")["vol20"].rank(pct=True)
    panel["ret5_industry_rank"] = panel.groupby(["date", "industry"])["ret5"].rank(pct=True)
    panel["sector_flow_rank"] = panel.groupby("date")["sector_flow5"].rank(pct=True)
    return panel.replace([np.inf, -np.inf], np.nan)


def strategy_mask_score(panel: pd.DataFrame, params: dict[str, Any]) -> tuple[pd.Series, pd.Series]:
    template = params["template"]
    common = (
        panel["market_breadth"].ge(params["breadth"])
        & panel["sector_flow5"].ge(params["sector_flow"])
        & panel["flow10"].ge(params["stock_flow"])
        & panel["vol20_rank"].le(params["max_vol_rank"])
        & panel["amount"].ge(3e7)
    )
    if template == "flow_trend":
        mask = common & panel["mom20_rank"].ge(params["momentum"]) & panel["ret5"].ge(-0.03)
        score = (
            0.30 * panel["mom20_rank"]
            + 0.30 * panel["flow10_rank"]
            + 0.25 * panel["sector_flow_rank"]
            + 0.15 * (1.0 - panel["vol20_rank"])
        )
    elif template == "trend_pullback":
        mask = (
            common
            & panel["ret20"].gt(0)
            & panel["mom60_rank"].ge(params["momentum"])
            & panel["ret5_industry_rank"].le(params["pullback_rank"])
            & panel["ma20_gap"].between(-0.06, 0.08)
        )
        score = (
            0.25 * panel["mom60_rank"]
            + 0.30 * panel["flow10_rank"]
            + 0.25 * panel["sector_flow_rank"]
            + 0.20 * (1.0 - panel["ret5_industry_rank"])
        )
    elif template == "volume_breakout":
        mask = (
            common
            & panel["mom20_rank"].ge(params["momentum"])
            & panel["breakout20"].ge(params["breakout"])
            & panel["amount_ratio5_20"].ge(params["volume_ratio"])
        )
        score = (
            0.30 * panel["mom20_rank"]
            + 0.25 * panel["flow10_rank"]
            + 0.25 * panel["sector_flow_rank"]
            + 0.20 * panel["amount_ratio5_20"].clip(0, 2) / 2
        )
    else:
        raise ValueError(template)
    return mask.fillna(False), score.fillna(-999.0)


def baseline_mask_score(panel: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    mask = (
        panel["flow10"].gt(0)
        & panel["flow_positive_ratio10"].ge(0.60)
        & panel["flow_z"].ge(0.45)
        & panel["amount_ratio5_20"].ge(1.05)
        & panel["ret10"].abs().le(0.18)
    )
    flow_z_rank = panel.groupby("date")["flow_z"].rank(pct=True)
    score = 0.45 * panel["flow10_rank"] + 0.30 * flow_z_rank + 0.25 * panel["amount_ratio5_20"].clip(0, 2) / 2
    return mask.fillna(False), score.fillna(-999.0)


def select_signals(
    panel: pd.DataFrame,
    mask: pd.Series,
    score: pd.Series,
    start: pd.Timestamp,
    end: pd.Timestamp,
    horizon: int,
    max_per_day: int = 5,
    max_per_industry: int = 2,
) -> pd.DataFrame:
    candidates = panel.loc[
        mask & panel["date"].between(start, end) & panel[f"net_return_{horizon}d"].notna()
    ].copy()
    candidates["signal_score"] = score.loc[candidates.index]
    candidates = candidates.sort_values(["date", "signal_score"], ascending=[True, False])
    selected_rows = []
    for _, day in candidates.groupby("date", sort=True):
        industry_counts: dict[str, int] = {}
        for idx, row in day.iterrows():
            industry = str(row["industry"])
            if industry_counts.get(industry, 0) >= max_per_industry:
                continue
            selected_rows.append(idx)
            industry_counts[industry] = industry_counts.get(industry, 0) + 1
            if sum(industry_counts.values()) >= max_per_day:
                break
    return candidates.loc[selected_rows].copy() if selected_rows else candidates.head(0)


def select_live_signals(
    panel: pd.DataFrame,
    mask: pd.Series,
    score: pd.Series,
    signal_date: pd.Timestamp,
    max_per_day: int = 5,
    max_per_industry: int = 2,
) -> pd.DataFrame:
    candidates = panel.loc[mask & panel["date"].eq(signal_date)].copy()
    candidates["signal_score"] = score.loc[candidates.index]
    candidates = candidates.sort_values("signal_score", ascending=False)
    selected_rows = []
    industry_counts: dict[str, int] = {}
    for idx, row in candidates.iterrows():
        industry = str(row["industry"])
        if industry_counts.get(industry, 0) >= max_per_industry:
            continue
        selected_rows.append(idx)
        industry_counts[industry] = industry_counts.get(industry, 0) + 1
        if len(selected_rows) >= max_per_day:
            break
    return candidates.loc[selected_rows].copy() if selected_rows else candidates.head(0)


def metrics(signals: pd.DataFrame, horizon: int) -> dict[str, float]:
    if signals.empty:
        return {
            "trades": 0, "signal_days": 0, "win_rate": np.nan, "mean_return": np.nan,
            "median_return": np.nan, "daily_sharpe": np.nan, "max_drawdown": np.nan,
            "profit_factor": np.nan,
        }
    column = f"net_return_{horizon}d"
    values = signals[column].astype(float)
    daily = signals.groupby("date")[column].mean().sort_index()
    equity = (1.0 + daily).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    std = float(daily.std(ddof=1))
    gains = float(values.loc[values > 0].sum())
    losses = abs(float(values.loc[values < 0].sum()))
    return {
        "trades": int(len(signals)),
        "signal_days": int(signals["date"].nunique()),
        "win_rate": float((values > 0).mean()),
        "mean_return": float(values.mean()),
        "median_return": float(values.median()),
        "daily_sharpe": float(daily.mean() / std * np.sqrt(252)) if std > 1e-12 else np.nan,
        "max_drawdown": float(drawdown.min()),
        "profit_factor": gains / losses if losses > 1e-12 else np.nan,
    }


def passes_validation_gate(result: dict[str, Any]) -> bool:
    """Require positive validation economics before any live/paper signal is enabled."""
    required = ("mean_return", "profit_factor", "max_drawdown")
    if any(not np.isfinite(float(result.get(key, np.nan))) for key in required):
        return False
    return bool(
        int(result.get("trades", 0)) >= 15
        and int(result.get("signal_days", 0)) >= 5
        and float(result["mean_return"]) > 0
        and float(result["profit_factor"]) > 1.0
        and float(result["max_drawdown"]) > -0.12
    )


def parameter_grid() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    common = itertools.product(
        [0.30, 0.40, 0.50],
        [-0.03, 0.0, 0.03],
        [-0.03, 0.0, 0.03],
        [0.75, 1.0],
    )
    common_values = list(common)
    for template in ("flow_trend", "trend_pullback", "volume_breakout"):
        for breadth, sector_flow, stock_flow, max_vol_rank in common_values:
            for momentum in (0.35, 0.50, 0.65):
                base = {
                    "template": template,
                    "breadth": breadth,
                    "sector_flow": sector_flow,
                    "stock_flow": stock_flow,
                    "max_vol_rank": max_vol_rank,
                    "momentum": momentum,
                }
                if template == "trend_pullback":
                    for pullback_rank in (0.45, 0.70):
                        rows.append({**base, "pullback_rank": pullback_rank})
                elif template == "volume_breakout":
                    for breakout, volume_ratio in itertools.product((-0.05, 0.0), (0.8, 1.1)):
                        rows.append({**base, "breakout": breakout, "volume_ratio": volume_ratio})
                else:
                    rows.append(base)
    return rows


def evaluate_period(
    panel: pd.DataFrame,
    mask: pd.Series,
    score: pd.Series,
    start: pd.Timestamp,
    end: pd.Timestamp,
    horizon: int,
) -> tuple[pd.DataFrame, dict[str, float]]:
    signals = select_signals(panel, mask, score, start, end, horizon)
    return signals, metrics(signals, horizon)


def optimize(panel: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    train_start, train_end = pd.Timestamp("2026-03-15"), pd.Timestamp("2026-04-30")
    validation_start, validation_end = pd.Timestamp("2026-05-01"), pd.Timestamp("2026-06-30")
    test_start, test_end = pd.Timestamp("2026-07-01"), panel["date"].max()

    rows = []
    best: dict[str, Any] | None = None
    best_score = -np.inf
    fallback_best: dict[str, Any] | None = None
    fallback_score = -np.inf
    for params in parameter_grid():
        mask, score = strategy_mask_score(panel, params)
        for horizon in (1, 2, 3):
            _, train = evaluate_period(panel, mask, score, train_start, train_end, horizon)
            _, validation = evaluate_period(panel, mask, score, validation_start, validation_end, horizon)
            eligible = validation["trades"] >= 15 and validation["signal_days"] >= 5
            if eligible and np.isfinite(validation["mean_return"]):
                # Conservative validation objective: reward return and profit factor,
                # penalize drawdown; the July-September holdout is never used here.
                objective = (
                    validation["mean_return"]
                    + 0.00025 * min(validation["profit_factor"], 3.0)
                    + 0.03 * validation["max_drawdown"]
                )
            else:
                objective = -999.0
            row = {
                **params,
                "horizon": horizon,
                "train_trades": train["trades"],
                "train_mean_return": train["mean_return"],
                "train_win_rate": train["win_rate"],
                "validation_trades": validation["trades"],
                "validation_days": validation["signal_days"],
                "validation_mean_return": validation["mean_return"],
                "validation_win_rate": validation["win_rate"],
                "validation_profit_factor": validation["profit_factor"],
                "validation_max_drawdown": validation["max_drawdown"],
                "selection_objective": objective,
            }
            rows.append(row)
            if objective > best_score:
                best_score = objective
                best = {
                    **params,
                    "horizon": horizon,
                    "validation_sample_sufficient": True,
                    "validation": validation,
                }
            if validation["trades"] >= 5 and validation["signal_days"] >= 3 and np.isfinite(validation["mean_return"]):
                fallback_objective = (
                    validation["mean_return"]
                    + 0.00025 * min(validation["profit_factor"], 3.0)
                    + 0.03 * validation["max_drawdown"]
                )
                if fallback_objective > fallback_score:
                    fallback_score = fallback_objective
                    fallback_best = {
                        **params,
                        "horizon": horizon,
                        "validation_sample_sufficient": False,
                        "validation": validation,
                    }

    if best is None:
        if fallback_best is None:
            raise RuntimeError("no strategy produced even five validation trades")
        best = fallback_best
        best_score = fallback_score

    best_mask, best_rank = strategy_mask_score(panel, best)
    test_signals, test = evaluate_period(
        panel, best_mask, best_rank, test_start, test_end, int(best["horizon"])
    )
    best.update({"selection_objective": best_score, "test": test})

    baseline_mask, baseline_score = baseline_mask_score(panel)
    baseline_validation = []
    for horizon in (1, 2, 3):
        _, result = evaluate_period(
            panel, baseline_mask, baseline_score, validation_start, validation_end, horizon
        )
        baseline_validation.append({"horizon": horizon, **result})
    eligible_baseline = [
        row for row in baseline_validation
        if row["trades"] >= 15 and row["signal_days"] >= 5 and np.isfinite(row["mean_return"])
    ]
    if eligible_baseline:
        selected_baseline = max(eligible_baseline, key=lambda row: row["mean_return"])
    else:
        selected_baseline = max(
            baseline_validation,
            key=lambda row: row["mean_return"] if np.isfinite(row["mean_return"]) else -999.0,
        )
    best["baseline_selected_horizon"] = int(selected_baseline["horizon"])
    best["baseline_validation"] = selected_baseline

    comparisons = []
    for horizon in (1, 2, 3):
        _, result = evaluate_period(panel, baseline_mask, baseline_score, test_start, test_end, horizon)
        comparisons.append({
            "model": "baseline_accumulation",
            "horizon": horizon,
            "selected_on_validation": horizon == best["baseline_selected_horizon"],
            **result,
        })
    comparisons.append({
        "model": f"candidate_{best['template']}",
        "horizon": best["horizon"],
        "selected_on_validation": True,
        **test,
    })
    comparison = pd.DataFrame(comparisons)
    return pd.DataFrame(rows).sort_values("selection_objective", ascending=False), best, comparison


def latest_recommendations(
    panel: pd.DataFrame,
    best: dict[str, Any],
    comparison: pd.DataFrame,
) -> pd.DataFrame:
    candidate = comparison.loc[comparison["model"].str.startswith("candidate_")].iloc[0]
    baseline_selected = comparison.loc[
        comparison["model"].eq("baseline_accumulation")
        & comparison["horizon"].eq(best["baseline_selected_horizon"])
    ].iloc[0]
    promote_candidate = bool(
        best.get("validation_sample_sufficient", False)
        and passes_validation_gate(best.get("validation", {}))
        and candidate["trades"] >= 20
        and candidate["signal_days"] >= 8
        and candidate["mean_return"] > 0
        and candidate["mean_return"] > baseline_selected["mean_return"]
    )
    baseline_qualified = passes_validation_gate(best["baseline_validation"])
    if promote_candidate:
        mask, score = strategy_mask_score(panel, best)
        horizon = int(best["horizon"])
        selected_model = f"candidate_{best['template']}"
        trade_enabled = True
    elif baseline_qualified:
        mask, score = baseline_mask_score(panel)
        horizon = int(best["baseline_selected_horizon"])
        selected_model = "baseline_accumulation"
        trade_enabled = True
    else:
        mask, score = baseline_mask_score(panel)
        horizon = int(best["baseline_selected_horizon"])
        selected_model = "cash_no_trade"
        trade_enabled = False

    latest_date = panel["date"].max()
    signals = (
        select_live_signals(panel, mask, score, latest_date)
        if trade_enabled else panel.loc[panel["date"].lt(latest_date)].head(0).copy()
    )
    if signals.empty:
        relaxed = panel.loc[panel["date"].eq(latest_date)].copy()
        relaxed["signal_score"] = score.loc[relaxed.index]
        relaxed = relaxed.sort_values("signal_score", ascending=False).head(5)
        relaxed["model_signal"] = False
        signals = relaxed
    else:
        signals["model_signal"] = True
    signals["selected_model"] = selected_model
    signals["trade_enabled"] = trade_enabled
    signals["planned_holding_days"] = horizon
    signals["entry_rule"] = "下一交易日开盘；高开超过3%跳过"
    signals["exit_rule"] = f"止损3% / 止盈6% / 最长{horizon}日"
    columns = [
        "date", "code", "name", "industry", "selected_model", "trade_enabled", "model_signal",
        "signal_score", "close", "ret5", "ret20", "ret60", "flow10",
        "sector_flow5", "market_breadth", "vol20_rank",
        "planned_holding_days", "entry_rule", "exit_rule",
    ]
    return signals[[column for column in columns if column in signals]]


def format_pct(value: Any) -> str:
    return "—" if pd.isna(value) else f"{float(value):.2%}"


def write_report(
    output_dir: Path,
    grid: pd.DataFrame,
    best: dict[str, Any],
    comparison: pd.DataFrame,
    recommendations: pd.DataFrame,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    grid.to_csv(output_dir / "strategy_grid.csv", index=False, encoding="utf-8-sig")
    comparison.to_csv(output_dir / "strategy_comparison.csv", index=False, encoding="utf-8-sig")
    recommendations.to_csv(output_dir / "recommendations_v2.csv", index=False, encoding="utf-8-sig")
    (output_dir / "selected_strategy.json").write_text(
        json.dumps(best, ensure_ascii=False, indent=2, default=float), encoding="utf-8"
    )

    candidate = comparison.loc[comparison["model"].str.startswith("candidate_")].iloc[0]
    baseline = comparison.loc[
        comparison["model"].eq("baseline_accumulation")
        & comparison["horizon"].eq(best["baseline_selected_horizon"])
    ].iloc[0]
    promote = bool(
        bool(best.get("validation_sample_sufficient", False))
        and passes_validation_gate(best.get("validation", {}))
        and candidate["trades"] >= 20
        and candidate["signal_days"] >= 8
        and candidate["mean_return"] > 0
        and candidate["mean_return"] > baseline["mean_return"]
    )
    baseline_qualified = passes_validation_gate(best["baseline_validation"])

    compare_rows = []
    for _, row in comparison.iterrows():
        compare_rows.append(
            f"| {row['model']} | {int(row['horizon'])} | {int(row['trades'])} | "
            f"{int(row['signal_days'])} | {format_pct(row['win_rate'])} | "
            f"{format_pct(row['mean_return'])} | {format_pct(row['max_drawdown'])} | "
            f"{row['profit_factor']:.2f} |"
        )
    rec_rows = []
    for _, row in recommendations.iterrows():
        rec_rows.append(
            f"| {row['code']} | {row['name']} | {row['industry']} | "
            f"{'触发' if row['model_signal'] else '观察'} | {row['signal_score']:.3f} | "
            f"{format_pct(row['ret20'])} | {format_pct(row['flow10'])} | "
            f"{int(row['planned_holding_days'])} |"
        )

    selected_model = recommendations["selected_model"].iloc[0]
    selected_horizon = int(recommendations["planned_holding_days"].iloc[0])
    conclusion = (
        "新候选通过门槛，进入纸面跟踪"
        if promote else
        "新动量模型未通过；吸筹基线通过验证，继续纸面跟踪"
        if baseline_qualified else
        "候选与基线均未通过验证；启动空仓保护，不发布买入信号"
    )
    report = f"""# A股短线策略增益研究

研究日期：{recommendations['date'].max().date()}  
最终结论：**{conclusion}**。  
当前状态：`{selected_model}`，研究持有期 {selected_horizon} 日。

## 验证设计

- 训练观察：2026-03-15 至 2026-04-30。
- 参数选择：2026-05-01 至 2026-06-30，正式门槛为至少15笔交易且覆盖5个信号日。
- 完全留出测试：2026-07-01 至最新交易日；这一段不参与参数选择。
- 每天最多5只、每个细分行业最多2只，信号在收盘后计算，下一交易日开盘成交。
- 已计入止损3%、止盈6%、双边佣金和滑点合计16bp。
- 共比较三种逻辑与多个阈值：资金趋势、上涨趋势中的短期回撤、放量突破。

## 入选参数

- 模板：`{best['template']}`
- 持有期：{best['horizon']}日
- 市场宽度下限：{best['breadth']:.0%}
- 行业5日资金压力下限：{best['sector_flow']:.0%}
- 个股10日资金压力下限：{best['stock_flow']:.0%}
- 最大波动率分位：{best['max_vol_rank']:.0%}
- 动量分位下限：{best['momentum']:.0%}
- 验证期样本门槛：{'达到' if best.get('validation_sample_sufficient') else '未达到（仅作探索，不可晋级）'}
- 交易闸门：验证期必须平均收益 > 0、盈亏比 > 1、最大回撤优于 -12%，否则空仓。
- 吸筹基线验证：平均每笔 {format_pct(best['baseline_validation']['mean_return'])}，盈亏比 {best['baseline_validation']['profit_factor']:.2f}，最大回撤 {format_pct(best['baseline_validation']['max_drawdown'])}，{'通过' if baseline_qualified else '未通过'}。

## 7–9月完全留出测试

| 模型 | 持有日 | 交易数 | 信号日 | 胜率 | 平均每笔 | 最大回撤 | 盈亏比 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(compare_rows)}

只有新候选策略在验证期先通过交易闸门、且留出期平均收益为正、优于验证期选定的吸筹基线并满足样本数时才允许晋级。本轮新动量模型没有晋级；吸筹基线的验证期也未通过，因此系统保持空仓。每天最多5只、每行业最多2只的约束仍保留。没有通过时，正确动作是继续收集数据，而不是根据7–9月结果再次调参。

## 最新候选

| 代码 | 名称 | 细分行业 | 状态 | 分数 | 20日收益 | 10日资金代理 | 持有日 |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
{chr(10).join(rec_rows) if rec_rows else '| — | — | — | 无信号 | — | — | — | — |'}

“观察”只展示最接近条件的股票；当状态为 `cash_no_trade` 或交易闸门关闭时，即使技术条件接近也不可当作买入信号。

## 为什么这比旧回测更可靠

旧版回测只验证圆弧底/吸筹技术触发，未完整复现每日推荐中的行业资金与成长过滤。新版对信号日逐日计算市场、行业和个股特征，并把7–9月锁成一次性留出集。参数选择不看留出结果，避免用目标月份反复拟合。

## 下一步

即使本轮通过，也只进入至少20个交易日的纸面跟踪，不直接实盘。若未通过，应扩展历史到至少3年，并加入涨跌停不可成交、停牌、动态指数成分和公告财报时点，之后采用滚动年度验证。
"""
    (output_dir / "strategy_research.md").write_text(report, encoding="utf-8")


def run(config_path: str, report_dir_text: str) -> Path:
    with Path(config_path).open(encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    report_dir = Path(report_dir_text)
    panel = build_feature_panel(load_inputs(report_dir), cfg)
    grid, best, comparison = optimize(panel)
    recommendations = latest_recommendations(panel, best, comparison)
    output_dir = report_dir / "research"
    write_report(output_dir, grid, best, comparison, recommendations)
    LOG.info("optimizer outputs written to %s", output_dir)
    return output_dir


def main() -> int:
    parser = argparse.ArgumentParser(description="Walk-forward optimizer with untouched July-September holdout")
    parser.add_argument("--config", default="config/default.yml")
    parser.add_argument("--report-dir", default="reports/latest")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
    try:
        print(f"完成：{run(args.config, args.report_dir)}")
        return 0
    except Exception:
        LOG.exception("optimizer failed")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
