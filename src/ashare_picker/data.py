from __future__ import annotations

import hashlib
import logging
import random
import threading
import time
from pathlib import Path
from typing import Callable

import pandas as pd
import requests

LOG = logging.getLogger(__name__)


class DataSourceError(RuntimeError):
    """Raised when a required upstream dataset cannot be obtained."""


def _number(series: pd.Series) -> pd.Series:
    text = series.astype(str).str.replace(",", "", regex=False).str.replace("%", "", regex=False)
    return pd.to_numeric(text, errors="coerce")


def _codes(series: pd.Series) -> pd.Series:
    return series.astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(6)


class AkshareClient:
    """Small, cached and retrying adapter around the AKShare endpoints we use."""

    def __init__(
        self,
        cache_dir: str | Path = "data/cache",
        cache_hours: float = 18,
        retries: int = 4,
        retry_base_seconds: float = 1.5,
        request_pause_seconds: float = 0.15,
        catalog_path: str | Path | None = None,
    ) -> None:
        try:
            import akshare as ak
        except ImportError as exc:
            raise DataSourceError("缺少 akshare，请先执行 pip install -e .") from exc
        self.ak = ak
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_seconds = cache_hours * 3600
        self.retries = max(1, retries)
        self.retry_base_seconds = retry_base_seconds
        self.request_pause_seconds = request_pause_seconds
        self.catalog_path = (
            Path(catalog_path)
            if catalog_path
            else Path(__file__).resolve().parents[2] / "data" / "industry_catalog.csv"
        )
        self.warnings: list[str] = []
        self._warning_lock = threading.Lock()

    def _warn(self, message: str) -> None:
        LOG.warning(message)
        with self._warning_lock:
            self.warnings.append(message)

    def _cache_path(self, key: str) -> Path:
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:20]
        return self.cache_dir / f"{digest}.csv"

    def _cached_call(
        self,
        key: str,
        fetch: Callable[[], pd.DataFrame],
        *,
        force: bool = False,
    ) -> pd.DataFrame:
        path = self._cache_path(key)
        fresh = path.exists() and (time.time() - path.stat().st_mtime) <= self.cache_seconds
        if fresh and not force:
            return pd.read_csv(path)

        last_error: Exception | None = None
        for attempt in range(self.retries):
            try:
                if self.request_pause_seconds:
                    time.sleep(self.request_pause_seconds + random.random() * 0.1)
                frame = fetch()
                if frame is None or frame.empty:
                    raise DataSourceError(f"{key} 返回空数据")
                tmp = path.with_suffix(".tmp")
                frame.to_csv(tmp, index=False)
                tmp.replace(path)
                return frame.copy()
            except Exception as exc:  # upstream exceptions vary by AKShare release
                last_error = exc
                if attempt + 1 < self.retries:
                    time.sleep(self.retry_base_seconds * (2**attempt) + random.random())

        if path.exists():
            self._warn(f"{key} 刷新失败，使用过期缓存：{last_error}")
            return pd.read_csv(path)
        raise DataSourceError(f"{key} 获取失败：{last_error}") from last_error

    def industry_names(self, *, force: bool = False) -> pd.DataFrame:
        try:
            raw = self._cached_call(
                "industry_names",
                self.ak.stock_board_industry_name_em,
                force=force,
            )
            required = {"板块名称", "板块代码"}
            if not required.issubset(raw.columns):
                raise DataSourceError(f"行业列表字段变化：{list(raw.columns)}")
            out = raw.rename(columns={"板块名称": "industry", "板块代码": "industry_code"})
        except Exception as exc:
            if not self.catalog_path.exists():
                raise
            self._warn(f"实时行业列表不可用，使用仓库内行业目录快照：{exc}")
            out = pd.read_csv(self.catalog_path, dtype=str)
        return out[["industry", "industry_code"]].dropna().drop_duplicates().reset_index(drop=True)

    def _industry_code(self, industry: str) -> str:
        catalog = pd.read_csv(self.catalog_path, dtype=str)
        match = catalog.loc[catalog["industry"] == industry, "industry_code"]
        if match.empty:
            raise DataSourceError(f"本地行业目录缺少 {industry}")
        return str(match.iloc[0])

    def _direct_industry_flow(self, industry: str) -> pd.DataFrame:
        code = self._industry_code(industry)
        url = "https://push2his.eastmoney.com/api/qt/stock/fflow/daykline/get"
        params = {
            "lmt": "0",
            "klt": "101",
            "fields1": "f1,f2,f3,f7",
            "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61,f62,f63,f64,f65",
            "secid": f"90.{code}",
        }
        response = requests.get(
            url,
            params=params,
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=20,
        )
        response.raise_for_status()
        payload = response.json()
        klines = (payload.get("data") or {}).get("klines") or []
        frame = pd.DataFrame([item.split(",") for item in klines])
        if frame.empty:
            raise DataSourceError(f"{industry} 历史资金流直连返回空数据")
        frame.columns = [
            "日期",
            "主力净流入-净额",
            "小单净流入-净额",
            "中单净流入-净额",
            "大单净流入-净额",
            "超大单净流入-净额",
            "主力净流入-净占比",
            "小单净流入-净占比",
            "中单净流入-净占比",
            "大单净流入-净占比",
            "超大单净流入-净占比",
            "_1",
            "_2",
            "_3",
            "_4",
        ]
        return frame

    def _fetch_industry_flow(self, industry: str) -> pd.DataFrame:
        try:
            return self.ak.stock_sector_fund_flow_hist(symbol=industry)
        except Exception as exc:
            self._warn(f"{industry} 的 AKShare 行业资金流映射失败，改用本地代码直连历史接口：{exc}")
            return self._direct_industry_flow(industry)

    def industry_members(self, industry: str, *, force: bool = False) -> pd.DataFrame:
        raw = self._cached_call(
            f"industry_members:{industry}",
            lambda: self.ak.stock_board_industry_cons_em(symbol=industry),
            force=force,
        )
        if not {"代码", "名称"}.issubset(raw.columns):
            raise DataSourceError(f"{industry} 成分字段变化：{list(raw.columns)}")
        out = pd.DataFrame(
            {
                "code": _codes(raw["代码"]),
                "name": raw["名称"].astype(str),
                "latest_amount": _number(raw["成交额"]) if "成交额" in raw else float("nan"),
                "turnover": _number(raw["换手率"]) if "换手率" in raw else float("nan"),
            }
        )
        return out.drop_duplicates("code").reset_index(drop=True)

    def price(
        self,
        symbol: str,
        start: str,
        end: str,
        *,
        adjust: str = "qfq",
        force: bool = False,
    ) -> pd.DataFrame:
        symbol = str(symbol).zfill(6)
        raw = self._cached_call(
            f"price:{symbol}:{start}:{end}:{adjust}",
            lambda: self.ak.stock_zh_a_hist(
                symbol=symbol,
                period="daily",
                start_date=start.replace("-", ""),
                end_date=end.replace("-", ""),
                adjust=adjust,
                timeout=20,
            ),
            force=force,
        )
        mapping = {
            "日期": "date",
            "开盘": "open",
            "收盘": "close",
            "最高": "high",
            "最低": "low",
            "成交量": "volume",
            "成交额": "amount",
            "换手率": "turnover",
        }
        missing = set(mapping) - set(raw.columns)
        if missing:
            raise DataSourceError(f"{symbol} 日线缺少字段 {sorted(missing)}")
        out = raw.rename(columns=mapping)[list(mapping.values())].copy()
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
        for col in set(mapping.values()) - {"date"}:
            out[col] = _number(out[col])
        out["code"] = symbol
        return out.dropna(subset=["date", "open", "close"]).sort_values("date").reset_index(drop=True)

    def industry_flow(self, industry: str, *, force: bool = False) -> pd.DataFrame:
        raw = self._cached_call(
            f"industry_flow:{industry}",
            lambda: self._fetch_industry_flow(industry),
            force=force,
        )
        mapping = {
            "日期": "date",
            "主力净流入-净额": "main_net",
            "主力净流入-净占比": "main_ratio",
            "超大单净流入-净额": "xl_net",
            "大单净流入-净额": "large_net",
        }
        required = {"日期", "主力净流入-净额"}
        if not required.issubset(raw.columns):
            raise DataSourceError(f"{industry} 资金流字段变化：{list(raw.columns)}")
        present = {key: value for key, value in mapping.items() if key in raw.columns}
        out = raw.rename(columns=present)[list(present.values())].copy()
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
        for col in set(out.columns) - {"date"}:
            out[col] = _number(out[col])
        out["industry"] = industry
        return out.dropna(subset=["date", "main_net"]).sort_values("date").reset_index(drop=True)

    @staticmethod
    def market_for(symbol: str) -> str:
        symbol = str(symbol).zfill(6)
        if symbol.startswith(("4", "8", "9")):
            return "bj"
        if symbol.startswith(("5", "6", "7")):
            return "sh"
        return "sz"

    def stock_flow(self, symbol: str, *, force: bool = False) -> pd.DataFrame:
        symbol = str(symbol).zfill(6)
        market = self.market_for(symbol)
        raw = self._cached_call(
            f"stock_flow:{symbol}:{market}",
            lambda: self.ak.stock_individual_fund_flow(stock=symbol, market=market),
            force=force,
        )
        mapping = {
            "日期": "date",
            "收盘价": "close",
            "涨跌幅": "pct_change",
            "主力净流入-净额": "main_net",
            "主力净流入-净占比": "main_ratio",
            "超大单净流入-净额": "xl_net",
            "大单净流入-净额": "large_net",
        }
        required = {"日期", "主力净流入-净额"}
        if not required.issubset(raw.columns):
            raise DataSourceError(f"{symbol} 个股资金流字段变化：{list(raw.columns)}")
        present = {key: value for key, value in mapping.items() if key in raw.columns}
        out = raw.rename(columns=present)[list(present.values())].copy()
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
        for col in set(out.columns) - {"date"}:
            out[col] = _number(out[col])
        out["code"] = symbol
        return out.dropna(subset=["date", "main_net"]).sort_values("date").reset_index(drop=True)

    def fundamentals(self, report_date: str, *, force: bool = False) -> pd.DataFrame:
        compact = report_date.replace("-", "")
        raw = self._cached_call(
            f"fundamentals:{compact}",
            lambda: self.ak.stock_yjbb_em(date=compact),
            force=force,
        )
        aliases = {
            "股票代码": "code",
            "股票简称": "name",
            "营业总收入-同比增长": "revenue_yoy",
            "净利润-同比增长": "profit_yoy",
            "净资产收益率": "roe",
            "销售毛利率": "gross_margin",
            "所处行业": "reported_industry",
        }
        required = {"股票代码", "营业总收入-同比增长", "净利润-同比增长"}
        if not required.issubset(raw.columns):
            raise DataSourceError(f"业绩报表字段变化：{list(raw.columns)}")
        present = {key: value for key, value in aliases.items() if key in raw.columns}
        out = raw.rename(columns=present)[list(present.values())].copy()
        out["code"] = _codes(out["code"])
        for col in {"revenue_yoy", "profit_yoy", "roe", "gross_margin"} & set(out.columns):
            out[col] = _number(out[col])
        return out.drop_duplicates("code").reset_index(drop=True)
