from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .data import AkshareClient, DataSourceError
from .pipeline import load_config, run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="A 股日线行业资金流与短线候选研究工具")
    parser.add_argument("--verbose", action="store_true", help="输出详细日志")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="运行完整筛选、回测并生成报告")
    run.add_argument("--config", default="config/default.yml", help="YAML 配置文件")
    run.add_argument("--as-of", help="报告基准日，格式 YYYY-MM-DD；默认今天")
    run.add_argument("--force", action="store_true", help="忽略成功缓存并重新请求")

    download = subparsers.add_parser("download", help="下载单只股票前复权日线 CSV")
    download.add_argument("--symbol", required=True, help="六位股票代码")
    download.add_argument("--start", required=True, help="开始日期 YYYYMMDD 或 YYYY-MM-DD")
    download.add_argument("--end", required=True, help="结束日期 YYYYMMDD 或 YYYY-MM-DD")
    download.add_argument("--config", default="config/default.yml", help="YAML 配置文件")
    download.add_argument("--output", default="data/downloads", help="输出目录")
    download.add_argument("--force", action="store_true", help="忽略成功缓存并重新请求")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    try:
        if args.command == "run":
            path = run_pipeline(args.config, args.as_of, force=args.force)
            print(f"完成：{path}")
            return 0

        cfg = load_config(args.config)
        download_cfg = cfg.get("download", {})
        client = AkshareClient(
            cache_hours=float(download_cfg.get("cache_hours", 18)),
            retries=int(download_cfg.get("retries", 4)),
            retry_base_seconds=float(download_cfg.get("retry_base_seconds", 1.5)),
            request_pause_seconds=float(download_cfg.get("request_pause_seconds", 0.15)),
        )
        frame = client.price(
            args.symbol,
            args.start,
            args.end,
            adjust=str(download_cfg.get("adjust", "qfq")),
            force=args.force,
        )
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        target = output_dir / f"{str(args.symbol).zfill(6)}_{args.start.replace('-', '')}_{args.end.replace('-', '')}.csv"
        frame.to_csv(target, index=False, encoding="utf-8-sig")
        print(f"已下载 {len(frame)} 条日线：{target}")
        return 0
    except (DataSourceError, ValueError, KeyError) as exc:
        logging.getLogger(__name__).error("%s", exc)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
