# A-share Picker

A 股日线研究、细分行业资金方向和 1–3 日候选仓库。项目已经部署为 GitHub 云端计算，用户无需在本机安装或执行。

最新成品：

- [中文报告](https://github.com/Yaohuizhongliu/a-share-picker/blob/main/reports/latest/report.md)
- [每日候选 CSV](https://github.com/Yaohuizhongliu/a-share-picker/blob/main/reports/latest/recommendations.csv)
- [逐周行业资金方向 CSV](https://github.com/Yaohuizhongliu/a-share-picker/blob/main/reports/latest/weekly_sector_flows.csv)
- [1–3 日回测摘要](https://github.com/Yaohuizhongliu/a-share-picker/blob/main/reports/latest/backtest_summary.csv)
- [完整沪深300日线数据](https://github.com/Yaohuizhongliu/a-share-picker/blob/main/reports/latest/daily_prices.csv)

> 仅用于数据研究和策略验证，不构成投资建议。任何候选都可能亏损。免费公开源没有可审计的逐笔机构持仓，本项目绝不把量价估算写成真实“主力建仓”。

## 已运行的云端方案

GitHub Actions 工作日北京时间约 16:30 自动执行，也可在仓库的 [Actions 页面](https://github.com/Yaohuizhongliu/a-share-picker/actions/workflows/daily.yml) 点击 **Run workflow** 立即刷新。工作流会：

1. 运行离线测试。
2. 从中证指数公司取得当前沪深300成分表。
3. 从新浪财经 HTTPS 接口下载日线 OHLCV 和成交额。
4. 从巨潮资讯取得申银万国细分行业归属。
5. 计算 2026 年 7、8、9 月逐周行业资金压力、成长相对值、圆弧底和吸筹异动代理。
6. 回测信号后下一交易日开盘买入、1–3 日退出的规则。
7. 上传可下载的报告压缩产物，并把最新 CSV/Markdown 自动提交到主分支。

这条数据链已在 GitHub 托管算力实测通过：沪深300日线 300/300、行业分类 300/300，失败 0 只。

## 输出文件

| 文件 | 内容 |
|---|---|
| `reports/latest/report.md` | 中文摘要、最新一周行业方向、当日候选、回测 |
| `reports/latest/weekly_sector_flows.csv` | 每周细分行业估算净流、成交额、正向天数、排名 |
| `reports/latest/candidates.csv` | 300 只股票的全部因子、行业均值、综合分和信号 |
| `reports/latest/recommendations.csv` | 下一交易日 1–3 日触发/观察候选与退出规则 |
| `reports/latest/backtest_trades.csv` | 历史交易明细 |
| `reports/latest/backtest_summary.csv` | 1、2、3 日持有期的交易数、胜率、收益分布 |
| `reports/latest/daily_prices.csv` | 日线 OHLCV、成交额和资金压力代理 |
| `reports/latest/universe_industries.csv` | 股票与细分行业映射 |
| `reports/latest/download_errors.txt` | 数据下载失败清单 |

## 研究口径

- **行业资金方向**：用 Chaikin Money Flow 的价位乘数 `((2×收盘-最高-最低)/(最高-最低))` 乘以成交额，先按股票、交易日计算，再按细分行业和周汇总。字段名为 `estimated_net_flow`，是量价压力代理，不是真实资金账户净流入。
- **成长高于平均**：云端默认用 60 日价格动量在同一申万细分行业内比较，并在 `growth_source` 明确标注。财报同比未取得时保持为空，不伪造财务成长。
- **圆弧底**：对 60 日对数收盘价拟合开口向上的二次曲线，结合拟合优度、底部位置、回升幅度和近 5 日方向。
- **吸筹异动代理**：近 10 日估算资金压力为正、正向天数、相对 60 日基线 z-score、量比和价格涨幅共同判断。
- **每日候选**：行业周度估算资金为正、成长高于行业平均，并且圆弧底或吸筹代理至少一项触发时标记为“触发候选”；其余为观察/备选。
- **交易模拟**：信号后下一交易日开盘买入；止损 3%、止盈 6%、最多持有 3 日；单边佣金 3bp、滑点 5bp。同日同时触及止损和止盈时按先止损处理。

当前成分股回测存在幸存者偏差；日线数据无法确定盘中先后顺序；A 股还存在涨跌停、停牌和实际成交偏差。回测结果不能代表未来表现。

## 代码结构

- `src/ashare_picker/cloud_pipeline.py`：云端可用的已验证 HTTPS 主流程
- `src/ashare_picker/data.py`：AKShare/东方财富详细资金流适配器（部分公网机房可能拒绝）
- `src/ashare_picker/signals.py`：圆弧底、吸筹代理和 1–3 日回测
- `src/ashare_picker/pipeline.py`：详细资金流模式
- `.github/workflows/daily.yml`：每日自动计算、产物上传和提交
- `tests/`：离线单元测试
- `config/default.yml`：所有阈值和交易成本

如需命令行复现云端口径：

~~~bash
pip install -e ".[dev]"
python -m ashare_picker.cloud_pipeline --config config/default.yml --as-of 2026-09-14
~~~

## 数据来源

- [中证指数公司](https://www.csindex.com.cn/)：沪深300成分表
- [新浪财经](https://finance.sina.com.cn/)：A股日线行情
- [巨潮资讯 WebAPI](https://webapi.cninfo.com.cn/)：上市公司行业归属
- [AKShare](https://github.com/akfamily/akshare)：MIT 许可的第三方数据接口封装

代码采用 MIT License。数据使用还需遵守各数据网站和接口提供方条款。
