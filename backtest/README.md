# 回测模块说明（`backtest/`）

SMC 策略回测框架，支持参数网格/随机搜索、多进程并行、断点续跑、结果审计与诊断。

---

## 核心脚本

### `engine.py` — 回测引擎

定义 `BacktestParams`（参数 dataclass）、`BacktestResult`（结果 dataclass）和 `run_backtest()`（单组合回测入口）。

- `BacktestParams`：描述一次回测的所有参数（TF、FVG 过滤器、风控、趋势方法等），提供 `label()` 用于日志/报告展示，`to_dict()` / `from_dict()` 用于序列化。
- `BacktestResult`：持有 Trade 列表及衍生统计指标（Sharpe / Sortino / max_drawdown_r 等）。
- `run_backtest()`：接受 klines 字典 + `BacktestParams`，返回 `BacktestResult`。

---

### `run.py` — 网格/随机搜索 CLI

批量搜索参数空间，并行运行所有 `(TF pair × 参数组合)` 的回测。

```bash
# 全量网格搜索
uv run backtest/run.py --config config/backtest/default_smc_v2.json

# 随机搜索（固定 seed 保证可复现，跨股票 combo 一致）
uv run backtest/run.py --config config/backtest/default_smc_v2.json --random 200 --seed 42

# 覆盖 config 里的 codes
uv run backtest/run.py --config config/backtest/default_smc_v2.json --codes US.NVDA

# 快速冒烟测试
uv run backtest/run.py --fast --no-viz
```

**断点续跑**：结果每 `--save-every`（默认 500）次完成写一次 checkpoint（`backtest/results/checkpoints/`），下次相同 config 运行自动跳过已完成的 combo。用 `--no-resume` 强制重跑。

**输出目录**格式：`backtest/results/<时间戳>_<config名>_grid/` 或 `…_random_<N>/`。

---

### `aggregate_random.py` — 跨股票 Random 结果聚合

在用相同 `--seed` 对多支股票跑完 random 搜索后，读取所有 `results_*.csv`，按参数组合做跨股票汇总，输出 HTML 报告并生成缩范围 config。

```bash
# 基本用法（top-30 combos，min_freq=25%）
uv run backtest/aggregate_random.py \
    --run-dir backtest/results/<timestamp>_default_smc_v2_random_200/ \
    --src-config config/backtest/default_smc_v2.json

# 自定义参数
uv run backtest/aggregate_random.py \
    --run-dir <dir> \
    --top-n 40 \
    --min-freq 0.20 \
    --min-trades 15 \
    --out-config cross_stock_grid_v2.json
```

**关键参数**：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--top-n` | 30 | 分析参数频率时考虑的 top-N 数量 |
| `--min-freq` | 0.25 | 参数值在 top-N 中出现频率 ≥ 此值才保留 |
| `--min-trades` | 10 | 任一股票 trades < N 的 combo 直接排除 |
| `--out-config` | `cross_stock_grid_v1.json` | 生成的缩范围 config 文件名 |
| `--src-config` | — | 源 config，用于复制 codes/start/end 等顶层字段 |

**输出**：
- `<run-dir>/agg_report.html`：Top-N combo 表格、各股票 Sharpe 分解、参数频率分析表、建议的 `param_grid` JSON 预览
- `config/backtest/<out-config>`：可直接传给 `run.py --config` 的缩范围 config

---

### `audit.py` — 单组合交易审计报告

对指定参数组合生成完整的审计 HTML，包含净值曲线、10 项 KPI、全交易列表（含 trade_id）、最长连胜/连亏图、Top 盈利/亏损交易 K 线图。同时将交易写入 `review_trades.duckdb` 供 Trade Viewer 查询。

```bash
# 从结果 CSV 自动选最优参数
uv run backtest/audit.py \
    --from-csv backtest/results/.../results_US_SNDK.csv \
    --code US.SNDK --start 2025-05-22 --end 2026-05-22
```

---

### `fvg_inspect.py` — FVG 过滤诊断

在指定时间窗口内，逐条列出引擎检测到的每个 FVG 触碰事件及过滤步骤（未达深度、LTF 未确认、位移过滤、SL/RR 不足等）。输出自包含 HTML。

```bash
uv run backtest/fvg_inspect.py \
    --from-csv backtest/results/.../results_US_SNDK.csv \
    --code US.SNDK --start 2025-05-22 --end 2026-05-22 \
    --inspect-start 2025-11-03 --inspect-end 2025-11-07
```

---

### `screener.py` — 策略适配性筛选器

对一批股票计算 SMC 策略适配特征（FVG Bounce%/Overfill%、KD 清晰度、ATR%、日均成交额），输出评分排行榜 + 相关性矩阵 HTML。

```bash
uv run backtest/screener.py --codes US.SNDK US.NVDA US.AMD --start 2025-01-01
```

---

### `db.py` — DuckDB 读写

封装 `BacktestDB`（写 runs / trades / run_stats / live_trades）和 `ReviewTradesDB`（写 review_trades 供 Trade Viewer 查询）。两个数据库文件分开，避免长时间 backtest 持锁冲突。

---

### `stats.py` — 统计函数

`sharpe_ratio()`、`sortino_ratio()`、`max_drawdown()`、`heatmap_data()`、`parameter_importance()` 等。不依赖 DuckDB，纯 pandas/numpy 计算。

---

### `report.py` — HTML 报告生成

`generate_report()` 接受 results DataFrame，生成含 KPI 卡片、热力图、参数重要性柱状图的 Plotly 交互 HTML。

---

### `viz.py` — matplotlib 可视化

`plot_backtest_results()` / `plot_from_csv()`：参数网格结果的 Sharpe 分布图和 top-N 净值曲线（保存为 PNG）。

---

### `logger.py` — 多进程安全日志

基于 `logging.handlers.QueueHandler` / `QueueListener` 的日志工厂，保证多 worker 进程的日志不乱序写入。

---

### `migrate_algo_version.py` — Algo 版本迁移工具

一次性脚本，用于将旧版 `backtest.duckdb` 中 `algo_version` 字段迁移到新格式。通常无需手动运行。

---

## 典型工作流

```
1. random 探索 → 确定参数区间
   run.py --random 200 --seed 42 → aggregate_random.py → cross_stock_grid_v1.json

2. 全量 grid 搜索
   run.py --config cross_stock_grid_v1.json

3. 审计最优组合
   audit.py --from-csv results.csv → agg_report.html + review_trades.duckdb

4. 诊断 FVG 过滤
   fvg_inspect.py --from-csv results.csv → inspect_*.html
```

详见：
- [`config/backtest/README.md`](../config/backtest/README.md) — config JSON 字段说明
- [`strategy/smc/STRATEGY.md`](../strategy/smc/STRATEGY.md) — 策略逻辑与参数完整说明
