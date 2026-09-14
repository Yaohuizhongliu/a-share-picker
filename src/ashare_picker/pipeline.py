from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from .data import AkshareClient, DataSourceError
from .signals import (
    TradeRules,
    accumulation_features,
    add_growth_features,
    rounded_bottom_features,
    summarize_backtest,
    walk_forward_backtest,
)

LOG = logging.getLogger(__name__)


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("配置文件必须是 YAML 映射")
    return config


def _week_end(series: pd.Series) -> pd.Series:
    return series.dt.to_period("W-FRI").dt.end_time.dt.normalize()


def _fmt(value: Any, kind: str = "plain") -> str:
    if pd.isna(value):
        return "-"
    if kind == "money":
        return f"{float(value) / 100_000_000:.2f}亿"
    if kind == "pct":
        return f"{float(value) * 100:.2f}%"
    if kind == "score":
        return f"{float(value):.3f}"
    return str(value)


def _markdown_table(frame: pd.DataFrame, specs: list[tuple[str, str, str]], limit: int = 15) -> str:
    if frame.empty:
        return "_无符合条件的数据_"
    header = "| " + " | ".join(label for _, label, _ in specs) + " |"
    rule = "|" + "|".join("---" for _ in specs) + "|"
    rows = [header, rule]
    for _, row in frame.head(limit).iterrows():
        rows.append("| " + " | ".join(_fmt(row.get(col), kind) for col, _, kind in specs) + " |")
    return "\n".join(rows)


def _build_report(
    as_of: pd.Timestamp,
    start: pd.Timestamp,
    end: pd.Timestamp,
    weekly: pd.DataFrame,
    candidates: pd.DataFrame,
    recommendations: pd.DataFrame,
    backtest_summary: pd.DataFrame,
    warnings: list[str],
    coverage: dict[str, int],
) -> str:
    latest_week = weekly["week_end"].max()
    latest = weekly.loc[weekly["week_end"] == latest_week].sort_values("rank")
    sector_table = _markdown_table(
        latest,
        [
            ("industry", "行业", "plain"),
            ("main_net", "周主力净流入", "money"),
            ("positive_days", "净流入天数", "plain"),
            ("rank", "周排名", "plain"),
        ],
        15,
    )
    rec_table = _markdown_table(
        recommendations,
        [
            ("code", "代码", "plain"),
            ("name", "名称", "plain"),
            ("industry", "行业", "plain"),
            ("score", "综合分", "score"),
            ("growth_source", "成长口径", "plain"),
            ("growth_metric", "成长值", "plain"),
            ("rounded_bottom", "圆弧底", "plain"),
            ("accumulation_anomaly", "资金异动", "plain"),
            ("reason", "入选原因", "plain"),
        ],
        20,
    )
    backtest_table = _markdown_table(
        backtest_summary,
        [
            ("horizon", "最长持有日", "plain"),
            ("trades", "样本数", "plain"),
            ("win_rate", "胜率", "pct"),
            ("average_return", "平均净收益", "pct"),
            ("median_return", "中位净收益", "pct"),
            ("worst_trade", "最差单笔", "pct"),
        ],
        10,
    )
    warning_lines = "\n".join(f"- {item}" for item in warnings[:30]) or "- 无"
    return f"""# A 股日线筛选报告

- 生成基准日：{as_of.date()}
- 研究区间：{start.date()} 至 {end.date()}
- 最新周：截至 {pd.Timestamp(latest_week).date()}
- 行业资金流成功/总数：{coverage['industry_ok']}/{coverage['industry_total']}
- 股票日线成功/尝试数：{coverage['stock_ok']}/{coverage['stock_total']}
- 最终候选数：{len(recommendations)}

> 仅供研究，不构成投资建议。“主力净流入”是按成交单大小划分的代理指标，并非可验证的机构建仓。2026 年 9 月若尚未结束，本报告只包含基准日前已完成的交易日。

## 最新一周行业资金流

{sector_table}

## 1–3 日观察候选

{rec_table}

硬条件：最新周行业净流入为正、成长高于行业平均，并且圆弧底完成或资金流异动至少满足一项。计划为下一交易日开盘观察成交，默认止损 3%、止盈 6%、最多持有 3 个交易日。实际交易前必须检查公告、停复牌、涨跌停和流动性。

## 技术/资金触发样本回测

{backtest_table}

该回测只验证当前入围股票的技术面/资金面触发，不把当前财报倒填到历史，也没有重建历史行业成分；因此不是完整无偏的组合回测，仍有当前成分和当前入围造成的选择偏差。收益已扣除默认双边佣金及滑点；同一日同时触发止盈和止损时按止损处理。

## 数据质量提示

{warning_lines}
"""


def run_pipeline(config_path: str | Path, as_of: str | None = None, *, force: bool = False) -> Path:
    cfg = load_config(config_path)
    today = pd.Timestamp.now(tz="Asia/Shanghai").tz_localize(None).normalize()
    requested_as_of = pd.Timestamp(as_of).normalize() if as_of else today
    effective_as_of = min(requested_as_of, today)
    start = pd.Timestamp(cfg["period"]["start"]).normalize()
    configured_end = pd.Timestamp(cfg["period"]["end"]).normalize()
    end = min(configured_end, effective_as_of)
    if end < start:
        raise ValueError(f"有效结束日 {end.date()} 早于开始日 {start.date()}")

    download_cfg = cfg.get("download", {})
    client = AkshareClient(
        cache_hours=float(download_cfg.get("cache_hours", 18)),
        retries=int(download_cfg.get("retries", 4)),
        retry_base_seconds=float(download_cfg.get("retry_base_seconds", 1.5)),
        request_pause_seconds=float(download_cfg.get("request_pause_seconds", 0.15)),
    )
    max_workers = int(download_cfg.get("max_workers", 6))
    local_warnings: list[str] = []

    industries = client.industry_names(force=force)
    flows: list[pd.DataFrame] = []

    def get_industry_flow(name: str) -> tuple[str, pd.DataFrame | None, str | None]:
        try:
            frame = client.industry_flow(name, force=force)
            frame = frame.loc[(frame["date"] >= start) & (frame["date"] <= end)]
            return name, frame if not frame.empty else None, None
        except Exception as exc:
            return name, None, f"行业 {name} 资金流失败：{exc}"

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(get_industry_flow, name) for name in industries["industry"]]
        for future in as_completed(futures):
            _, frame, warning = future.result()
            if frame is not None:
                flows.append(frame)
            if warning:
                local_warnings.append(warning)

    if not flows:
        raise DataSourceError("所有行业资金流均获取失败，未生成空报告")
    daily_sector = pd.concat(flows, ignore_index=True)
    daily_sector["week_end"] = _week_end(daily_sector["date"])
    weekly = (
        daily_sector.groupby(["week_end", "industry"], as_index=False)
        .agg(
            main_net=("main_net", "sum"),
            positive_days=("main_net", lambda values: int((values > 0).sum())),
            trading_days=("main_net", "size"),
        )
        .sort_values(["week_end", "main_net"], ascending=[True, False])
    )
    weekly["rank"] = weekly.groupby("week_end")["main_net"].rank(method="min", ascending=False).astype(int)

    latest_week = weekly["week_end"].max()
    latest = weekly.loc[weekly["week_end"] == latest_week].sort_values(["main_net", "positive_days"], ascending=False)
    top_industries = int(cfg["universe"].get("top_industries", 10))
    selected = latest.head(top_industries).copy()
    if selected.empty:
        raise DataSourceError("最新周没有可用行业")

    try:
        fundamentals = client.fundamentals(str(cfg["period"]["report_date"]), force=force)
    except Exception as exc:
        local_warnings.append(f"财报数据失败，成长因子降级为 60 日价格动量：{exc}")
        fundamentals = pd.DataFrame(columns=["code", "revenue_yoy", "profit_yoy"])
    fundamental_map = fundamentals.set_index("code").to_dict("index") if not fundamentals.empty else {}

    member_rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    max_per_industry = int(cfg["universe"].get("max_stocks_per_industry", 25))
    exclude_patterns = tuple(str(item).upper() for item in cfg["universe"].get("exclude_name_patterns", ["ST", "退"]))
    sector_count = max(1, len(latest))

    for _, sector in selected.iterrows():
        industry = str(sector["industry"])
        try:
            members = client.industry_members(industry, force=force).sort_values(
                "latest_amount", ascending=False, na_position="last"
            )
        except Exception as exc:
            fallback = (
                fundamentals.loc[fundamentals.get("reported_industry", pd.Series(dtype=str)) == industry]
                if "reported_industry" in fundamentals
                else pd.DataFrame()
            )
            if fallback.empty:
                local_warnings.append(f"行业 {industry} 成分获取失败且无财报后备：{exc}")
                continue
            local_warnings.append(f"行业 {industry} 实时成分失败，使用 {len(fallback)} 只财报行业成分：{exc}")
            members = fallback.rename(columns={"name": "name"})[["code", "name"]].copy()
            members["latest_amount"] = np.nan
            members["turnover"] = np.nan
        if max_per_industry > 0:
            members = members.head(max_per_industry)
        for _, member in members.iterrows():
            code = str(member["code"]).zfill(6)
            name = str(member["name"])
            if code in seen or any(pattern in name.upper() for pattern in exclude_patterns):
                continue
            seen.add(code)
            member_rows.append(
                {
                    "code": code,
                    "name": name,
                    "industry": industry,
                    "sector_week_net": float(sector["main_net"]),
                    "sector_rank": int(sector["rank"]),
                    "sector_score": max(0.0, 1.0 - (int(sector["rank"]) - 1) / sector_count),
                    "constituent_latest_amount": member.get("latest_amount", np.nan),
                }
            )

    history_start = (start - pd.Timedelta(days=int(download_cfg.get("history_calendar_days", 180)))).strftime("%Y%m%d")
    price_end = end.strftime("%Y%m%d")
    min_history = int(cfg["universe"].get("min_history_days", 70))
    min_average_amount = float(cfg["universe"].get("min_average_amount", 30_000_000))
    price_store: dict[str, pd.DataFrame] = {}
    flow_store: dict[str, pd.DataFrame] = {}

    def analyze_stock(base: dict[str, Any]) -> tuple[dict[str, Any] | None, pd.DataFrame | None, pd.DataFrame | None, str | None]:
        code = base["code"]
        try:
            prices = client.price(
                code,
                history_start,
                price_end,
                adjust=str(download_cfg.get("adjust", "qfq")),
                force=force,
            )
            flows_for_stock = client.stock_flow(code, force=force)
            flows_for_stock = flows_for_stock.loc[flows_for_stock["date"] <= end]
            if len(prices) < min_history:
                return None, None, None, f"{code} 历史日线不足 {min_history} 日"
            average_amount = float(prices["amount"].tail(20).mean())
            if not np.isfinite(average_amount) or average_amount < min_average_amount:
                return None, None, None, None
            row = dict(base)
            row.update(fundamental_map.get(code, {}))
            row["average_amount_20d"] = average_amount
            row["momentum_60d"] = (
                float(prices["close"].iloc[-1] / prices["close"].iloc[-60] - 1.0)
                if len(prices) >= 60
                else np.nan
            )
            row.update(rounded_bottom_features(prices, cfg["rounded_bottom"]))
            row.update(accumulation_features(flows_for_stock, prices, cfg["accumulation"]))
            return row, prices, flows_for_stock, None
        except Exception as exc:
            return None, None, None, f"股票 {code} 分析失败：{exc}"

    candidate_rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(analyze_stock, base) for base in member_rows]
        for future in as_completed(futures):
            row, prices, stock_flow, warning = future.result()
            if row is not None and prices is not None and stock_flow is not None:
                candidate_rows.append(row)
                price_store[row["code"]] = prices
                flow_store[row["code"]] = stock_flow
            if warning:
                local_warnings.append(warning)

    if not candidate_rows:
        raise DataSourceError("行业成分中没有成功完成分析的股票，未生成空报告")
    candidates = add_growth_features(pd.DataFrame(candidate_rows))
    weights = cfg["strategy"]["weights"]
    candidates["score"] = (
        float(weights["sector"]) * candidates["sector_score"].fillna(0.0)
        + float(weights["growth"]) * candidates["growth_percentile"].fillna(0.0)
        + float(weights["rounded_bottom"]) * candidates["rounded_score"].fillna(0.0)
        + float(weights["accumulation"]) * candidates["accumulation_score"].fillna(0.0)
    )
    candidates["eligible"] = (
        (candidates["sector_week_net"] > 0)
        & candidates["growth_above_avg"].fillna(False)
        & (candidates["rounded_bottom"].fillna(False) | candidates["accumulation_anomaly"].fillna(False))
    )
    candidates["reason"] = candidates.apply(
        lambda row: "、".join(
            item
            for ok, item in [
                (bool(row["growth_above_avg"]), "成长高于行业均值"),
                (bool(row["rounded_bottom"]), "圆弧底回升确认"),
                (bool(row["accumulation_anomaly"]), "大单资金异动"),
            ]
            if ok
        ),
        axis=1,
    )
    candidates = candidates.sort_values(["eligible", "score"], ascending=False).reset_index(drop=True)
    top_n = int(cfg["strategy"].get("top_n", 10))
    recommendations = candidates.loc[candidates["eligible"]].head(top_n).copy()

    rules = TradeRules(
        stop_loss=float(cfg["strategy"].get("stop_loss", 0.03)),
        take_profit=float(cfg["strategy"].get("take_profit", 0.06)),
        max_holding_days=int(cfg["strategy"].get("max_holding_days", 3)),
        commission_bps_each_side=float(cfg["strategy"].get("commission_bps_each_side", 3.0)),
        slippage_bps_each_side=float(cfg["strategy"].get("slippage_bps_each_side", 5.0)),
    )
    backtest_parts: list[pd.DataFrame] = []
    backtest_universe = candidates.loc[
        candidates["rounded_bottom"].fillna(False) | candidates["accumulation_anomaly"].fillna(False)
    ].head(max(top_n, 10))
    for code in backtest_universe["code"]:
        trades = walk_forward_backtest(
            code,
            price_store[code],
            flow_store[code],
            cfg["rounded_bottom"],
            cfg["accumulation"],
            rules,
            start,
            end,
        )
        if not trades.empty:
            backtest_parts.append(trades)
    backtest_trades = pd.concat(backtest_parts, ignore_index=True) if backtest_parts else pd.DataFrame()
    backtest_summary = summarize_backtest(backtest_trades)

    output_dir = Path(cfg.get("output", {}).get("directory", "reports/latest"))
    output_dir.mkdir(parents=True, exist_ok=True)
    weekly.to_csv(output_dir / "weekly_sector_flows.csv", index=False, encoding="utf-8-sig")
    candidates.to_csv(output_dir / "candidates.csv", index=False, encoding="utf-8-sig")
    recommendations.to_csv(output_dir / "recommendations.csv", index=False, encoding="utf-8-sig")
    backtest_trades.to_csv(output_dir / "backtest_trades.csv", index=False, encoding="utf-8-sig")
    backtest_summary.to_csv(output_dir / "backtest_summary.csv", index=False, encoding="utf-8-sig")

    all_warnings = list(dict.fromkeys(client.warnings + local_warnings))
    coverage = {
        "industry_ok": int(daily_sector["industry"].nunique()),
        "industry_total": int(len(industries)),
        "stock_ok": int(len(candidates)),
        "stock_total": int(len(member_rows)),
    }
    report = _build_report(
        effective_as_of,
        start,
        end,
        weekly,
        candidates,
        recommendations,
        backtest_summary,
        all_warnings,
        coverage,
    )
    report_path = output_dir / "report.md"
    report_path.write_text(report, encoding="utf-8")
    LOG.info("报告已生成：%s", report_path)
    return report_path
