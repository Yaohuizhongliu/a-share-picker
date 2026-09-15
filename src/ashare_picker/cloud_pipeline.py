from __future__ import annotations

import argparse
import logging
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable

import akshare as ak
import numpy as np
import pandas as pd
import requests
import yaml

from .signals import (
    TradeRules,
    accumulation_features,
    add_growth_features,
    rounded_bottom_features,
    summarize_backtest,
    walk_forward_backtest,
)

LOG = logging.getLogger("ashare_picker.cloud")
REQUEST_TIMEOUT_SECONDS = 15
_ORIGINAL_REQUEST = requests.sessions.Session.request


def _request_with_timeout(self: requests.Session, method: str, url: str, **kwargs: Any):
    kwargs.setdefault("timeout", REQUEST_TIMEOUT_SECONDS)
    return _ORIGINAL_REQUEST(self, method, url, **kwargs)


def install_request_timeout() -> None:
    if requests.sessions.Session.request is not _request_with_timeout:
        requests.sessions.Session.request = _request_with_timeout


def retry(label: str, fn: Callable[[], Any], attempts: int = 3) -> Any:
    last: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except Exception as exc:  # public endpoints fail transiently
            last = exc
            if attempt < attempts:
                time.sleep(0.8 * attempt)
    raise RuntimeError(f"{label} failed after {attempts} attempts: {last}") from last


def stock_symbol(code: str) -> str:
    code = str(code).zfill(6)
    if code.startswith(("6", "9")):
        return f"sh{code}"
    return f"sz{code}"


def clean_price_frame(frame: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    out = frame.rename(columns={str(c).lower(): str(c).lower() for c in frame.columns}).copy()
    required = ["date", "open", "high", "low", "close", "volume", "amount"]
    if any(column not in out.columns for column in required):
        raise ValueError(f"missing price columns: {required}; got {list(out.columns)}")
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    for column in required[1:]:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    out = out.loc[
        out["date"].between(start, end)
        & out["close"].gt(0)
        & out["open"].gt(0)
        & out["amount"].ge(0),
        required,
    ]
    return out.drop_duplicates("date").sort_values("date").reset_index(drop=True)


def fetch_price(code: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    symbol = stock_symbol(code)
    frame = retry(
        f"Sina daily {code}",
        lambda: ak.stock_zh_a_daily(
            symbol=symbol,
            start_date=start.strftime("%Y%m%d"),
            end_date=end.strftime("%Y%m%d"),
            adjust="",
        ),
        attempts=2,
    )
    return clean_price_frame(frame, start, end)


def _nonempty(value: Any) -> bool:
    return value is not None and str(value).strip() not in {"", "nan", "None", "--"}


def fetch_industry(code: str, as_of: pd.Timestamp) -> dict[str, str]:
    frame = retry(
        f"CNInfo industry {code}",
        lambda: ak.stock_industry_change_cninfo(
            symbol=code,
            start_date="20000101",
            end_date=as_of.strftime("%Y%m%d"),
        ),
    )
    if frame.empty:
        return {"industry": "未分类", "industry_standard": "缺失", "industry_code": ""}
    out = frame.copy()
    if "变更日期" in out:
        out["变更日期"] = pd.to_datetime(out["变更日期"], errors="coerce")
        out = out.loc[out["变更日期"].isna() | out["变更日期"].le(as_of)]
    if out.empty:
        return {"industry": "未分类", "industry_standard": "缺失", "industry_code": ""}

    priorities = ["申银万国行业分类标准", "巨潮行业分类标准", "证监会行业分类标准"]
    chosen = None
    if "分类标准" in out:
        for standard in priorities:
            rows = out.loc[out["分类标准"].astype(str).eq(standard)]
            if not rows.empty:
                chosen = rows.sort_values("变更日期").iloc[-1] if "变更日期" in rows else rows.iloc[-1]
                break
    if chosen is None:
        chosen = out.sort_values("变更日期").iloc[-1] if "变更日期" in out else out.iloc[-1]

    industry = "未分类"
    for column in ("行业中类", "行业大类", "行业次类", "行业门类"):
        if column in chosen and _nonempty(chosen[column]):
            industry = str(chosen[column]).strip()
            break
    return {
        "industry": industry,
        "industry_standard": str(chosen.get("分类标准", "巨潮资讯")).strip(),
        "industry_code": str(chosen.get("行业编码", "")).strip(),
    }


def estimated_flow(prices: pd.DataFrame) -> pd.DataFrame:
    out = prices.copy()
    spread = out["high"] - out["low"]
    multiplier = np.where(
        spread.gt(0),
        ((out["close"] - out["low"]) - (out["high"] - out["close"])) / spread,
        0.0,
    )
    out["main_net"] = pd.Series(multiplier, index=out.index).clip(-1.0, 1.0) * out["amount"]
    out["flow_proxy_ratio"] = np.where(out["amount"].gt(0), out["main_net"] / out["amount"], 0.0)
    return out


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _build_industry_map(
    board_rows: list[tuple[int, str, str]],
    fetch_components: Callable[[str], pd.DataFrame],
    component_code_columns: tuple[str, ...],
    standard: str,
    max_workers: int,
) -> tuple[dict[str, dict[str, str]], list[str]]:
    results: list[tuple[int, str, str, pd.DataFrame]] = []
    errors: list[str] = []

    def worker(item: tuple[int, str, str]) -> tuple[int, str, str, pd.DataFrame]:
        idx, name, board_code = item
        frame = retry(
            f"{standard} constituents {name}",
            lambda: fetch_components(board_code or name),
            attempts=2,
        )
        return idx, name, board_code, frame

    with ThreadPoolExecutor(max_workers=min(max_workers, 12)) as executor:
        futures = {executor.submit(worker, item): item for item in board_rows}
        for future in as_completed(futures):
            item = futures[future]
            try:
                results.append(future.result())
            except Exception as exc:
                errors.append(f"industry {item[1]}: {exc}")

    mapping: dict[str, dict[str, str]] = {}
    for _, name, board_code, frame in sorted(results, key=lambda item: item[0]):
        stock_code_column = next(
            (column for column in component_code_columns if column in frame), None
        )
        if stock_code_column is None:
            continue
        for value in frame[stock_code_column]:
            code = str(value).split(".")[0].zfill(6)
            if code.startswith(("0", "3", "6")):
                mapping.setdefault(
                    code,
                    {
                        "industry": name,
                        "industry_standard": standard,
                        "industry_code": board_code,
                    },
                )
    return mapping, errors


def fetch_industry_board_map(max_workers: int) -> tuple[dict[str, dict[str, str]], list[str]]:
    try:
        boards = retry("Eastmoney industry boards", ak.stock_board_industry_name_em, attempts=2)
        name_column = "板块名称" if "板块名称" in boards else "行业名称"
        code_column = "板块代码" if "板块代码" in boards else None
        board_rows = [
            (idx, str(row[name_column]).strip(), str(row.get(code_column, "")) if code_column else "")
            for idx, row in boards.iterrows()
        ]
        mapping, errors = _build_industry_map(
            board_rows,
            lambda symbol: ak.stock_board_industry_cons_em(symbol=symbol),
            ("代码", "股票代码"),
            "东方财富行业板块",
            max_workers,
        )
        if len(mapping) >= 4000:
            return mapping, errors
        errors.append(f"Eastmoney industry coverage too low: {len(mapping)}")
    except Exception as exc:
        errors = [f"Eastmoney industry list unavailable: {exc}"]

    LOG.warning("Eastmoney industry mapping unavailable; falling back to Shenwan level-2")
    try:
        indexes = retry(
            "Shenwan level-2 industry list",
            lambda: ak.index_realtime_sw(symbol="二级行业"),
            attempts=3,
        )
        board_rows = [
            (idx, str(row["指数名称"]).strip(), str(row["指数代码"]).strip())
            for idx, row in indexes.iterrows()
        ]
        mapping, sw_errors = _build_industry_map(
            board_rows,
            lambda symbol: ak.index_component_sw(symbol=symbol),
            ("证券代码", "代码"),
            "申万二级行业",
            max_workers,
        )
        errors.extend(sw_errors)
        if len(mapping) < 3500:
            errors.append(f"Shenwan industry coverage low: {len(mapping)}")
        return mapping, errors
    except Exception as exc:
        errors.append(f"Shenwan industry mapping unavailable: {exc}")
        LOG.warning("industry mapping unavailable; continuing with 未分类")
        return {}, errors


def fetch_universe(
    as_of: pd.Timestamp,
    history_start: pd.Timestamp,
    max_workers: int,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], list[str]]:
    membership = retry("Shanghai-Shenzhen A-share list", ak.stock_info_a_code_name, attempts=3)
    if {"code", "name"}.issubset(membership.columns):
        members = membership[["code", "name"]].copy()
    elif {"代码", "名称"}.issubset(membership.columns):
        members = membership[["代码", "名称"]].copy()
        members.columns = ["code", "name"]
    else:
        raise ValueError(f"unexpected A-share list columns: {list(membership.columns)}")
    members["code"] = members["code"].astype(str).str.split(".").str[0].str.zfill(6)
    members["name"] = members["name"].astype(str).str.strip()
    members = members.loc[members["code"].str.startswith(("0", "3", "6"))]
    members = members.drop_duplicates("code").reset_index(drop=True)
    raw_universe_size = len(members)
    if raw_universe_size < 5000:
        raise RuntimeError(f"Shanghai-Shenzhen universe unexpectedly below 5000: {raw_universe_size}")
    members = members.loc[~members["name"].str.contains("ST|退", case=False, regex=True)].copy()
    members["exchange"] = np.where(members["code"].str.startswith("6"), "上海", "深圳")

    industry_map, industry_errors = fetch_industry_board_map(max_workers)
    prices: dict[str, pd.DataFrame] = {}
    metadata: list[dict[str, str]] = []
    errors: list[str] = list(industry_errors)

    def worker(row: dict[str, str]) -> tuple[dict[str, str], pd.DataFrame]:
        code = row["code"]
        industry = industry_map.get(
            code,
            {"industry": "未分类", "industry_standard": "缺失", "industry_code": ""},
        )
        bars = fetch_price(code, history_start, as_of)
        return {**row, **industry}, bars

    rows = members.to_dict("records")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(worker, row): row for row in rows}
        completed = 0
        for future in as_completed(futures):
            row = futures[future]
            completed += 1
            try:
                meta, bars = future.result()
                if len(bars) < 65:
                    raise ValueError(f"only {len(bars)} daily rows")
                metadata.append(meta)
                prices[row["code"]] = estimated_flow(bars)
            except Exception as exc:
                errors.append(f"{row['code']} {row['name']}: {exc}")
            if completed % 250 == 0 or completed == len(rows):
                LOG.info(
                    "full-market download %s/%s, success=%s, failed=%s",
                    completed, len(rows), len(prices), len(errors),
                )

    meta_df = pd.DataFrame(metadata)
    minimum_coverage = max(4500, int(len(rows) * 0.80))
    if meta_df.empty or len(prices) < minimum_coverage:
        raise RuntimeError(f"usable full-market coverage too low: {len(prices)}/{len(rows)}")
    meta_df["raw_universe_size"] = raw_universe_size
    return meta_df.sort_values("code").reset_index(drop=True), prices, errors


def build_weekly_flows(
    metadata: pd.DataFrame,
    prices: dict[str, pd.DataFrame],
    period_start: pd.Timestamp,
    as_of: pd.Timestamp,
) -> pd.DataFrame:
    chunks = []
    name_map = metadata.set_index("code")
    for code, frame in prices.items():
        meta = name_map.loc[code]
        sample = frame.loc[frame["date"].between(period_start, as_of), ["date", "main_net", "amount"]].copy()
        if sample.empty:
            continue
        sample["code"] = code
        sample["industry"] = meta["industry"]
        chunks.append(sample)
    if not chunks:
        return pd.DataFrame()

    daily_stocks = pd.concat(chunks, ignore_index=True)
    daily_sector = (
        daily_stocks.groupby(["industry", "date"], as_index=False)
        .agg(
            estimated_net_flow=("main_net", "sum"),
            total_amount=("amount", "sum"),
            constituent_observations=("code", "nunique"),
        )
    )
    daily_sector["week_end"] = (
        daily_sector["date"].dt.to_period("W-FRI").dt.end_time.dt.normalize()
    )
    weekly = (
        daily_sector.groupby(["industry", "week_end"], as_index=False)
        .agg(
            estimated_net_flow=("estimated_net_flow", "sum"),
            total_amount=("total_amount", "sum"),
            positive_days=("estimated_net_flow", lambda x: int((x > 0).sum())),
            trading_days=("date", "nunique"),
            constituents=("constituent_observations", "max"),
        )
    )
    weekly["positive_day_ratio"] = weekly["positive_days"] / weekly["trading_days"].clip(lower=1)
    weekly["flow_to_amount"] = np.where(
        weekly["total_amount"].gt(0),
        weekly["estimated_net_flow"] / weekly["total_amount"],
        0.0,
    )
    weekly["rank"] = weekly.groupby("week_end")["estimated_net_flow"].rank(method="min", ascending=False)
    weekly["week_complete"] = weekly["week_end"].le(as_of.normalize())
    return weekly.sort_values(["week_end", "rank", "industry"]).reset_index(drop=True)


def analyze_candidates(
    metadata: pd.DataFrame,
    prices: dict[str, pd.DataFrame],
    weekly: pd.DataFrame,
    cfg: dict[str, Any],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    meta = metadata.set_index("code")
    latest_week = weekly["week_end"].max()
    latest = weekly.loc[weekly["week_end"].eq(latest_week)].copy()
    latest["sector_percentile"] = latest["estimated_net_flow"].rank(pct=True)
    sector_map = latest.set_index("industry").to_dict("index")

    for code, frame in prices.items():
        if len(frame) < 65:
            continue
        info = meta.loc[code]
        close = frame["close"]
        momentum = float(close.iloc[-1] / close.iloc[-61] - 1.0)
        rounded = rounded_bottom_features(frame, cfg["rounded_bottom"])
        accumulation = accumulation_features(frame[["date", "main_net"]], frame, cfg["accumulation"])
        sector = sector_map.get(info["industry"], {})
        rows.append(
            {
                "code": code,
                "name": info["name"],
                "industry": info["industry"],
                "industry_standard": info["industry_standard"],
                "as_of": frame["date"].iloc[-1],
                "last_close": float(close.iloc[-1]),
                "average_amount_20d": float(frame["amount"].tail(20).mean()),
                "momentum_60d": momentum,
                "revenue_yoy": np.nan,
                "profit_yoy": np.nan,
                "latest_week_end": latest_week,
                "sector_estimated_net_flow": float(sector.get("estimated_net_flow", 0.0)),
                "sector_flow_to_amount": float(sector.get("flow_to_amount", 0.0)),
                "sector_score": float(sector.get("sector_percentile", 0.0)),
                **rounded,
                **accumulation,
            }
        )

    candidates = add_growth_features(pd.DataFrame(rows))
    weights = cfg["strategy"]["weights"]
    candidates["composite_score"] = (
        float(weights["sector"]) * candidates["sector_score"]
        + float(weights["growth"]) * candidates["growth_percentile"]
        + float(weights["rounded_bottom"]) * candidates["rounded_score"]
        + float(weights["accumulation"]) * candidates["accumulation_score"]
    )
    candidates["hard_signal"] = (
        candidates["sector_estimated_net_flow"].gt(0)
        & candidates["growth_above_avg"]
        & (candidates["rounded_bottom"] | candidates["accumulation_anomaly"])
    )
    candidates["watch_signal"] = (
        candidates["sector_estimated_net_flow"].gt(0)
        & candidates["growth_above_avg"]
    )
    return candidates.sort_values(["hard_signal", "composite_score"], ascending=[False, False]).reset_index(drop=True)


def make_recommendations(candidates: pd.DataFrame, cfg: dict[str, Any], preferred_horizon: int) -> pd.DataFrame:
    top_n = int(cfg["strategy"].get("top_n", 10))
    selected = candidates.loc[candidates["hard_signal"]].head(top_n).copy()
    if len(selected) < top_n:
        supplement = candidates.loc[
            candidates["watch_signal"] & ~candidates["code"].isin(selected["code"])
        ].head(top_n - len(selected))
        selected = pd.concat([selected, supplement], ignore_index=True)
    if len(selected) < top_n:
        supplement = candidates.loc[~candidates["code"].isin(selected["code"])].head(top_n - len(selected))
        selected = pd.concat([selected, supplement], ignore_index=True)

    selected["recommendation_status"] = np.where(
        selected["hard_signal"], "触发候选", np.where(selected["watch_signal"], "观察候选", "备选")
    )
    selected["planned_holding_days"] = preferred_horizon
    selected["entry_rule"] = "下一交易日开盘；若较参考收盘高开超过3%则跳过"
    selected["reference_stop"] = selected["last_close"] * (1 - float(cfg["strategy"]["stop_loss"]))
    selected["reference_target"] = selected["last_close"] * (1 + float(cfg["strategy"]["take_profit"]))
    selected["exit_rule"] = f"止损{float(cfg['strategy']['stop_loss']):.0%} / 止盈{float(cfg['strategy']['take_profit']):.0%} / 最长{preferred_horizon}日"
    return selected


def run_backtest(
    candidates: pd.DataFrame,
    prices: dict[str, pd.DataFrame],
    cfg: dict[str, Any],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    strategy = cfg["strategy"]
    rules = TradeRules(
        stop_loss=float(strategy["stop_loss"]),
        take_profit=float(strategy["take_profit"]),
        max_holding_days=int(strategy["max_holding_days"]),
        commission_bps_each_side=float(strategy["commission_bps_each_side"]),
        slippage_bps_each_side=float(strategy["slippage_bps_each_side"]),
    )
    trades = []
    industry_map = candidates.set_index("code")["industry"].to_dict()
    for code, frame in prices.items():
        result = walk_forward_backtest(
            code=code,
            prices=frame,
            flow=frame[["date", "main_net"]],
            rounded_cfg=cfg["rounded_bottom"],
            accumulation_cfg=cfg["accumulation"],
            rules=rules,
            start=start,
            end=end,
        )
        if not result.empty:
            result["industry"] = industry_map.get(code, "")
            trades.append(result)
    all_trades = pd.concat(trades, ignore_index=True) if trades else pd.DataFrame()
    return all_trades, summarize_backtest(all_trades)


def markdown_table(frame: pd.DataFrame, columns: list[str], limit: int = 12) -> str:
    if frame.empty:
        return "暂无数据。"
    data = frame.loc[:, [column for column in columns if column in frame]].head(limit).copy()
    for column in data.columns:
        if pd.api.types.is_float_dtype(data[column]):
            if column in {"estimated_net_flow", "total_amount", "main_net_10d", "average_amount_20d"}:
                data[column] = data[column].map(lambda x: f"{x / 1e8:.2f}亿" if pd.notna(x) else "")
            elif column in {
                "flow_to_amount", "positive_day_ratio", "win_rate", "average_return",
                "median_return", "worst_trade", "best_trade",
            } or "ratio" in column or "return" in column or "rate" in column:
                data[column] = data[column].map(lambda x: f"{x:.2%}" if pd.notna(x) else "")
            else:
                data[column] = data[column].map(lambda x: f"{x:.3f}" if pd.notna(x) else "")
        else:
            data[column] = data[column].astype(str)
    header = "| " + " | ".join(data.columns) + " |"
    rule = "| " + " | ".join(["---"] * len(data.columns)) + " |"
    rows = ["| " + " | ".join(str(value).replace("|", "/") for value in row) + " |" for row in data.to_numpy()]
    return "\n".join([header, rule, *rows])


def write_outputs(
    output_dir: Path,
    metadata: pd.DataFrame,
    prices: dict[str, pd.DataFrame],
    weekly: pd.DataFrame,
    candidates: pd.DataFrame,
    recommendations: pd.DataFrame,
    trades: pd.DataFrame,
    summary: pd.DataFrame,
    errors: list[str],
    as_of: pd.Timestamp,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata.to_csv(output_dir / "universe_industries.csv", index=False, encoding="utf-8-sig")
    weekly.to_csv(output_dir / "weekly_sector_flows.csv", index=False, encoding="utf-8-sig")
    candidates.to_csv(output_dir / "candidates.csv", index=False, encoding="utf-8-sig")
    recommendations.to_csv(output_dir / "recommendations.csv", index=False, encoding="utf-8-sig")
    trades.to_csv(output_dir / "backtest_trades.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(output_dir / "backtest_summary.csv", index=False, encoding="utf-8-sig")

    raw = []
    for code, frame in prices.items():
        item = frame[["date", "open", "high", "low", "close", "volume", "amount", "main_net"]].copy()
        item.insert(0, "code", code)
        raw.append(item)
    pd.concat(raw, ignore_index=True).to_csv(
        output_dir / "all_market_prices.csv.gz",
        index=False,
        encoding="utf-8-sig",
        compression="gzip",
    )
    (output_dir / "download_errors.txt").write_text("\n".join(errors), encoding="utf-8")

    latest_week = weekly["week_end"].max()
    latest = weekly.loc[weekly["week_end"].eq(latest_week)].sort_values("rank")
    last_market_date = max(frame["date"].max() for frame in prices.values()).date()
    report = f"""# A股云端选股报告

报告运行日：{as_of.date()}；实际行情截至：{last_market_date}。  
股票池：沪深全部A股 {int(metadata['raw_universe_size'].max())} 只（排除ST/退市并要求至少65根日线）；成功日线 {len(prices)} 只，行业分类 {metadata['industry'].ne('未分类').sum()} 只，失败 {len(errors)} 只。  
研究区间：2026-07-01 至 {last_market_date}；9月未结束时，本报告不会填充未来行情。

## 最新一周细分行业资金方向

{markdown_table(latest, ['industry', 'week_end', 'estimated_net_flow', 'flow_to_amount', 'positive_days', 'trading_days', 'week_complete', 'rank'])}

## 下一交易日 1–3日候选

{markdown_table(recommendations, ['code', 'name', 'industry', 'recommendation_status', 'composite_score', 'growth_metric', 'growth_source', 'rounded_bottom', 'accumulation_anomaly', 'planned_holding_days'], 10)}

“触发候选”满足行业估算资金为正、成长高于行业平均，且圆弧底或吸筹代理信号成立；“观察候选”尚未满足全部技术硬条件。开盘较参考收盘高开超过3%时跳过，止损/止盈以实际成交价重新计算。

## 1–3日策略回测摘要

{markdown_table(summary, ['horizon', 'trades', 'win_rate', 'average_return', 'median_return', 'worst_trade', 'best_trade'])}

回测采用信号后下一交易日开盘买入，最长1–3日，止损3%、止盈6%，单边佣金3bp、滑点5bp；同日同时触及止损与止盈时按先止损处理。当前成分股回测存在幸存者偏差，结果不代表未来收益。

## 口径与限制

- 日线来自新浪财经公开HTTPS接口；股票池来自沪深京A股代码表并限定沪深代码；行业归属按东方财富行业板块成分批量映射。
- 免费公开源没有可审计的逐笔“大单/主力”历史。本报告的 `estimated_net_flow` 使用 Chaikin 价位乘数 × 成交额估算买卖压力；它是量价代理，不是机构持仓，也不应表述为真实“主力建仓”。
- “成长高于平均”在当前云端口径中使用60日价格动量与同一细分行业比较；`growth_source` 已明确标注。财报同比字段保留为空，避免把价格上涨伪装成财务增长。
- 全市场原始日线以 all_market_prices.csv.gz 保存于当次云端产物；行业映射、所有候选、逐笔回测和下载失败清单同步保存，便于审计。
- 本项目仅用于研究，不构成投资建议。A股存在涨跌停、停牌和实际成交偏差，任何候选都可能亏损。
"""
    (output_dir / "report.md").write_text(report, encoding="utf-8")


def run(config_path: str, as_of_text: str | None) -> Path:
    install_request_timeout()
    cfg = load_config(config_path)
    today = pd.Timestamp.now(tz="Asia/Shanghai").tz_localize(None).normalize()
    as_of = pd.Timestamp(as_of_text).normalize() if as_of_text else today
    as_of = min(as_of, today)
    period_start = pd.Timestamp(cfg["period"]["start"])
    configured_end = pd.Timestamp(cfg["period"]["end"])
    as_of = min(as_of, configured_end)
    if as_of < period_start:
        raise ValueError(f"as-of {as_of.date()} is before period start {period_start.date()}")

    history_days = int(cfg["download"].get("history_calendar_days", 260))
    history_start = min(period_start, as_of) - pd.Timedelta(days=max(history_days, 260))
    metadata, prices, errors = fetch_universe(
        as_of=as_of,
        history_start=history_start,
        max_workers=int(cfg["download"].get("max_workers", 8)),
    )
    weekly = build_weekly_flows(metadata, prices, period_start, as_of)
    if weekly.empty:
        raise RuntimeError("weekly sector flow table is empty")
    candidates = analyze_candidates(metadata, prices, weekly, cfg)
    trades, summary = run_backtest(candidates, prices, cfg, period_start, as_of)
    preferred_horizon = 3
    if not summary.empty:
        eligible = summary.loc[summary["trades"].ge(10)]
        if not eligible.empty:
            preferred_horizon = int(eligible.sort_values("average_return", ascending=False).iloc[0]["horizon"])
    recommendations = make_recommendations(candidates, cfg, preferred_horizon)
    output_dir = Path(cfg["output"]["directory"])
    write_outputs(
        output_dir, metadata, prices, weekly, candidates, recommendations,
        trades, summary, errors, as_of
    )
    LOG.info("report written to %s", output_dir)
    return output_dir


def main() -> int:
    parser = argparse.ArgumentParser(description="Verified HTTPS cloud A-share research pipeline")
    parser.add_argument("--config", default="config/default.yml")
    parser.add_argument("--as-of", help="YYYY-MM-DD; defaults to today in Asia/Shanghai")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    try:
        path = run(args.config, args.as_of)
        print(f"完成：{path}")
        return 0
    except Exception:
        LOG.exception("cloud pipeline failed")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
