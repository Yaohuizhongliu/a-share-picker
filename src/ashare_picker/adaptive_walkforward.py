from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from ashare_picker.research_optimizer import (
    build_feature_panel,
    load_inputs,
    metrics,
    select_live_signals,
    select_signals,
)

LOG = logging.getLogger("ashare_picker.adaptive")


def candidate_library(panel: pd.DataFrame) -> dict[str, tuple[pd.Series, pd.Series]]:
    """Small, economically distinct strategy library; no parameter grid mining."""
    ret5_rank = panel.groupby("date")["ret5"].rank(pct=True)
    liquid = panel["amount"].ge(3e7)
    flow_rank = panel["flow10_rank"]
    sector_rank = panel["sector_flow_rank"]
    low_vol = 1.0 - panel["vol20_rank"]

    libraries: dict[str, tuple[pd.Series, pd.Series]] = {}
    libraries["broad_flow"] = (
        liquid
        & panel["flow10"].gt(0)
        & panel["flow_positive_ratio10"].ge(0.50)
        & flow_rank.ge(0.70)
        & sector_rank.ge(0.60)
        & panel["ret20"].between(-0.10, 0.25)
        & panel["vol20_rank"].le(0.75),
        0.40 * flow_rank + 0.30 * sector_rank + 0.20 * low_vol + 0.10 * panel["mom20_rank"],
    )
    libraries["bull_momentum"] = (
        liquid
        & panel["market_breadth"].ge(0.45)
        & panel["mom20_rank"].ge(0.65)
        & panel["ret5"].ge(-0.03)
        & flow_rank.ge(0.55)
        & sector_rank.ge(0.50)
        & panel["vol20_rank"].le(0.80),
        0.35 * panel["mom20_rank"] + 0.30 * flow_rank + 0.25 * sector_rank + 0.10 * low_vol,
    )
    libraries["bull_pullback"] = (
        liquid
        & panel["market_breadth"].ge(0.40)
        & panel["mom60_rank"].ge(0.60)
        & ret5_rank.le(0.35)
        & panel["ma20_gap"].between(-0.06, 0.08)
        & panel["flow10"].ge(-0.02)
        & panel["vol20_rank"].le(0.80),
        0.30 * panel["mom60_rank"] + 0.30 * flow_rank + 0.20 * sector_rank + 0.20 * (1.0 - ret5_rank),
    )
    libraries["weak_reversal"] = (
        liquid
        & panel["market_breadth"].lt(0.45)
        & ret5_rank.le(0.20)
        & panel["ret5"].lt(-0.02)
        & panel["ret20"].gt(-0.20)
        & panel["flow5"].gt(0)
        & panel["sector_flow5"].gt(-0.02)
        & panel["vol20_rank"].le(0.75),
        0.35 * (1.0 - ret5_rank) + 0.30 * flow_rank + 0.20 * sector_rank + 0.15 * low_vol,
    )
    libraries["weak_defensive_flow"] = (
        liquid
        & panel["market_breadth"].lt(0.45)
        & flow_rank.ge(0.75)
        & sector_rank.ge(0.65)
        & panel["vol20_rank"].le(0.45)
        & panel["ret20"].gt(-0.10),
        0.40 * flow_rank + 0.30 * sector_rank + 0.30 * low_vol,
    )
    libraries["cross_sectional_reversal"] = (
        liquid
        & ret5_rank.le(0.15)
        & flow_rank.ge(0.60)
        & panel["amount_ratio5_20"].ge(0.80)
        & panel["vol20_rank"].le(0.70),
        0.40 * (1.0 - ret5_rank) + 0.35 * flow_rank + 0.15 * sector_rank + 0.10 * low_vol,
    )
    return {name: (mask.fillna(False), score.fillna(-999.0)) for name, (mask, score) in libraries.items()}


def rolling_gate(result: dict[str, Any]) -> bool:
    required = ("mean_return", "profit_factor", "max_drawdown")
    if any(not np.isfinite(float(result.get(key, np.nan))) for key in required):
        return False
    return bool(
        int(result.get("trades", 0)) >= 8
        and int(result.get("signal_days", 0)) >= 4
        and float(result["mean_return"]) > 0.0010
        and float(result["profit_factor"]) > 1.10
        and float(result["max_drawdown"]) > -0.10
    )


def choose_for_date(
    panel: pd.DataFrame,
    library: dict[str, tuple[pd.Series, pd.Series]],
    dates: list[pd.Timestamp],
    date_pos: int,
    lookback: int = 40,
) -> dict[str, Any]:
    best: dict[str, Any] = {"strategy": "cash", "horizon": 1, "objective": -999.0}
    for name, (mask, score) in library.items():
        for horizon in (1, 2, 3):
            cutoff_pos = date_pos - horizon
            if cutoff_pos < 0:
                continue
            start_pos = max(0, cutoff_pos - lookback + 1)
            signals = select_signals(
                panel,
                mask,
                score,
                dates[start_pos],
                dates[cutoff_pos],
                horizon,
                max_per_day=3,
                max_per_industry=1,
            )
            result = metrics(signals, horizon)
            if not rolling_gate(result):
                continue
            shrink = result["trades"] / (result["trades"] + 20.0)
            objective = (
                shrink * result["mean_return"]
                + 0.00015 * min(result["profit_factor"], 2.5)
                + 0.02 * result["max_drawdown"]
                - 0.00005 * (horizon - 1)
            )
            if objective > best["objective"]:
                best = {
                    "strategy": name,
                    "horizon": horizon,
                    "objective": objective,
                    "lookback_metrics": result,
                }
    return best


def walk_forward(
    panel: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    lookback: int = 40,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    dates = sorted(pd.Timestamp(value) for value in panel["date"].dropna().unique())
    position = {date: idx for idx, date in enumerate(dates)}
    library = candidate_library(panel)
    selected_parts: list[pd.DataFrame] = []
    decisions: list[dict[str, Any]] = []
    for signal_date in [date for date in dates if start <= date <= end]:
        decision = choose_for_date(panel, library, dates, position[signal_date], lookback)
        decisions.append({"date": signal_date, **decision})
        if decision["strategy"] == "cash":
            continue
        mask, score = library[decision["strategy"]]
        chosen = select_live_signals(
            panel, mask, score, signal_date, max_per_day=3, max_per_industry=1
        )
        if chosen.empty:
            continue
        horizon = int(decision["horizon"])
        chosen = chosen.loc[chosen[f"net_return_{horizon}d"].notna()].copy()
        chosen["strategy"] = decision["strategy"]
        chosen["horizon"] = horizon
        chosen["net_return"] = chosen[f"net_return_{horizon}d"]
        selected_parts.append(chosen)
    trades = pd.concat(selected_parts, ignore_index=True) if selected_parts else pd.DataFrame()
    decisions_frame = pd.DataFrame(decisions)
    if trades.empty:
        summary = {
            "trades": 0, "signal_days": 0, "win_rate": np.nan, "mean_return": np.nan,
            "daily_sharpe": np.nan, "max_drawdown": np.nan, "profit_factor": np.nan,
            "cash_days": int((decisions_frame["strategy"] == "cash").sum()),
        }
    else:
        generic = trades.rename(columns={"net_return": "net_return_1d"})
        summary = metrics(generic, 1)
        summary["cash_days"] = int((decisions_frame["strategy"] == "cash").sum())
    return trades, decisions_frame, summary


def latest_state(panel: pd.DataFrame, lookback: int = 40) -> tuple[dict[str, Any], pd.DataFrame]:
    dates = sorted(pd.Timestamp(value) for value in panel["date"].dropna().unique())
    library = candidate_library(panel)
    decision = choose_for_date(panel, library, dates, len(dates) - 1, lookback)
    if decision["strategy"] == "cash":
        return decision, panel.loc[panel["date"].lt(dates[-1])].head(0).copy()
    mask, score = library[decision["strategy"]]
    signals = select_live_signals(panel, mask, score, dates[-1], max_per_day=3, max_per_industry=1)
    return decision, signals


def write_outputs(
    output_dir: Path,
    trades: pd.DataFrame,
    decisions: pd.DataFrame,
    summary: dict[str, Any],
    live_decision: dict[str, Any],
    live_signals: pd.DataFrame,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    trades.to_csv(output_dir / "adaptive_trades.csv", index=False, encoding="utf-8-sig")
    decisions.to_csv(output_dir / "adaptive_daily_decisions.csv", index=False, encoding="utf-8-sig")
    live_signals.to_csv(output_dir / "adaptive_live_signals.csv", index=False, encoding="utf-8-sig")
    payload = {"july_september": summary, "latest_decision": live_decision}
    (output_dir / "adaptive_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=float), encoding="utf-8"
    )
    mean_text = "—" if pd.isna(summary["mean_return"]) else f"{summary['mean_return']:.2%}"
    dd_text = "—" if pd.isna(summary["max_drawdown"]) else f"{summary['max_drawdown']:.2%}"
    pf_text = "—" if pd.isna(summary["profit_factor"]) else f"{summary['profit_factor']:.2f}"
    live_names = "、".join(live_signals.get("name", pd.Series(dtype=str)).astype(str).tolist()) or "无"
    report = f"""# 7–9月在线自适应策略研究

本报告每天只使用当时已经完成的交易结果，在过去40个交易日内选择策略；未来收益不参与当日决策。

## 7–9月走步结果

- 交易数：{summary['trades']}
- 信号日：{summary['signal_days']}
- 空仓日：{summary['cash_days']}
- 胜率：{'—' if pd.isna(summary['win_rate']) else f"{summary['win_rate']:.2%}"}
- 平均每笔：{mean_text}
- 盈亏比：{pf_text}
- 最大回撤：{dd_text}

## 最新状态

- 策略：`{live_decision['strategy']}`
- 持有期：{live_decision['horizon']}日
- 当前信号：{live_names}

策略库只包含六种预先定义的经济逻辑：广义资金趋势、牛市动量、牛市回撤、弱市反转、弱市防御资金、横截面反转。滚动窗口未达到正收益、盈亏比1.10和回撤门槛时自动空仓。
"""
    (output_dir / "adaptive_research.md").write_text(report, encoding="utf-8")


def run(config_path: str, report_dir_text: str) -> Path:
    with Path(config_path).open(encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    report_dir = Path(report_dir_text)
    panel = build_feature_panel(load_inputs(report_dir), cfg)
    trades, decisions, summary = walk_forward(
        panel, pd.Timestamp("2026-07-01"), panel["date"].max(), lookback=40
    )
    live_decision, live_signals = latest_state(panel, lookback=40)
    output_dir = report_dir / "research"
    write_outputs(output_dir, trades, decisions, summary, live_decision, live_signals)
    LOG.info("adaptive outputs written to %s", output_dir)
    return output_dir


def main() -> int:
    parser = argparse.ArgumentParser(description="Daily online walk-forward A-share strategy selection")
    parser.add_argument("--config", default="config/default.yml")
    parser.add_argument("--report-dir", default="reports/latest")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
    try:
        print(f"完成：{run(args.config, args.report_dir)}")
        return 0
    except Exception:
        LOG.exception("adaptive walk-forward failed")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
