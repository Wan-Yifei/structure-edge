# moomoo-project

Trading analysis and backtesting tools built on the [moomoo OpenAPI](https://openapi.moomoo.com/) Python SDK.

## Prerequisites

| Requirement | Version |
|-------------|---------|
| Python | >= 3.12 |
| [uv](https://docs.astral.sh/uv/) | any |
| moomoo OpenD | >= 10.5.6508 (running locally) |

```bash
uv sync
```

---

## Project Structure

```
moomoo/
├── analysis/                    # GUI 工具
│   ├── trade_viewer.py          #   K 线图表 + Order Flow + Trade Review
│   ├── tick_collector.py        #   实时 tick 采集（写入 db/ticks.db）
│   └── scheduler.py             #   定时任务调度器
│
├── backtest/                    # 回测框架
│   ├── engine.py                #   回测引擎：run_backtest(), BacktestParams, Trade
│   ├── run.py                   #   批量网格/随机搜索 CLI（自动写入 review_trades.duckdb）
│   ├── aggregate_random.py      #   跨股票 random 结果聚合：对比 + HTML 报告 + 生成缩范围 config
│   ├── audit.py                 #   单组合交易审计报告（K 线图 + 统计，自包含 HTML）
│   ├── fvg_inspect.py           #   FVG 过滤诊断工具：显示指定时段内每个 FVG 触碰被过滤的原因
│   ├── screener.py              #   策略适配性筛选器：FVG/KD/ATR/换手率特征评分 + 相关性矩阵
│   ├── db.py                    #   DuckDB 读写（runs / trades / run_stats / live_trades / review_trades）
│   ├── stats.py                 #   统计函数：Sharpe / Sortino / heatmap
│   ├── report.py                #   HTML 报告生成（Plotly）
│   ├── viz.py                   #   matplotlib 可视化（结果图表）
│   └── logger.py                #   多进程安全日志（QueueHandler）
│
├── config/                      # 配置文件
│   ├── backtest/                #   回测参数配置（见 config/backtest/README.md）
│   ├── trade_viewer.toml        #   Trade Viewer 默认显示参数
│   ├── chart.json               #   K 线图表布局配置
│   └── schedule.json            #   screener.py 默认股票列表
│
├── strategy/                    # 策略逻辑
│   ├── base.py                  #   BaseStrategy ABC + SMCStrategy 适配器
│   └── smc/                     #   Smart Money Concepts 实现
│       ├── fvg.py               #     Fair Value Gap 检测
│       ├── market_structure.py  #     摆动高低点 / BOS / CHoCH
│       ├── kd_trend.py          #     KD 通道趋势指标（快/慢 EMA 通道 + 自适应分段）
│       ├── confirmation.py      #     LTF 入场确认
│       └── order_blocks.py      #     Order Block 检测
│
├── feeds/                       # 数据层
│   ├── fetcher.py               #   从 moomoo OpenD 拉取 K 线（含本地缓存）
│   ├── kline_store.py           #   K 线缓存读写（backtest_klines.duckdb）
│   └── tick_store.py            #   Tick 读写（ticks.db）
│
├── core/                        # 跨模块工具
│   └── time_utils.py            #   candle_start() 等时间对齐函数
│
├── db/                          # 本地数据库文件（见 db/README.md）
│   ├── ticks.db                 #   实时 tick（~770 万行，SQLite）
│   ├── backtest_klines.duckdb   #   K 线缓存（DuckDB）
│   ├── backtest.duckdb          #   回测结果 + 交易记录（DuckDB）
│   └── review_trades.duckdb     #   交易索引（DuckDB，trade_viewer 查询用，独立文件避免锁冲突）
│
├── tests/                       # 单元测试（163 tests）
├── main.py                      # 统一入口
└── pyproject.toml
```

---

## 启动方式

```bash
# K 线图表 / Order Flow 分析工具
uv run main.py trade_viewer

# 回测（快速冒烟测试，内置精简参数空间）
uv run backtest/run.py --fast --no-viz

# 全量网格搜索（config 指定股票、日期、参数空间）
uv run backtest/run.py --config config/backtest/default_smc_v2.json

# 随机搜索（每个 TF pair 采样 300 个参数组合）
uv run backtest/run.py --config config/backtest/default_smc_v2.json --random 300

# 单元测试
uv run pytest tests/ -v
```

---

## Trade Viewer (`analysis/trade_viewer.py`)

K 线图表工具，支持三种使用场景：

| 模式 | 说明 |
|------|------|
| **Live** | 订阅实时 tick，Order Flow profile 每 N 秒刷新 |
| **Historical** | 加载历史 OHLCV，叠加 SMC 结构（FVG / BOS / OB）|
| **Trade Review** | 勾选 checkbox 后输入 Trade ID，自动跳转到该交易的 K 线，只显示入场相关结构（FVG + BOS），隐藏其他 overlay |

```bash
# 交互式启动（默认：US.SNDK，15m，Live）
uv run main.py trade_viewer

# 直接定位到指定交易
uv run main.py trade_viewer --trade-id <uuid>

# 命令行预填参数
uv run analysis/trade_viewer.py --code US.SNDK --mode Historical --date 2026-05-15
```

**可选 SMC overlay（历史模式）：**

| Overlay | 说明 |
|---------|------|
| FVG | Fair Value Gap 区间（半透明色块）|
| BOS/CHoCH | 结构突破 / 换手 |
| OB | Order Block |

---

## 回测系统 (`backtest/`)

### 运行一次回测

```bash
# 快速冒烟测试（内置精简参数空间，约 8 个组合）
uv run backtest/run.py --fast --no-viz

# 全量网格搜索（stocks、日期、参数空间全部由 config 指定）
uv run backtest/run.py --config config/backtest/default_smc_v2.json

# 随机搜索（比穷举网格更高效，每个 TF pair 采样 300 个组合）
uv run backtest/run.py --config config/backtest/default_smc_v2.json --random 300 --no-viz

# 覆盖 config 中的股票列表（只回测指定标的）
uv run backtest/run.py --config config/backtest/default_smc_v2.json --codes US.NVDA

# 强制重跑（忽略断点续跑缓存）
uv run backtest/run.py --config config/backtest/default_smc_v2.json --no-resume

# 强制重跑（忽略 DB 已有的日期段，不从数据库复用交易数据）
uv run backtest/run.py --config config/backtest/default_smc_v2.json --no-reuse
```

### 输出

| 文件 | 说明 |
|------|------|
| `backtest/results/<时间戳>/results_<代码>.csv` | 所有组合的统计指标（含 sharpe / sortino）|
| `backtest/results/<时间戳>/report_<代码>.html` | 交互式 Plotly 报告（KPI 卡片 / 热力图 / 参数重要性）|
| `db/backtest.duckdb` | 每笔模拟交易持久化，支持按 trade_id 回溯 |

### 断点续跑 & 日期段复用

- **断点续跑**：相同参数组合的结果自动缓存在 `backtest/results/checkpoints/`。下次运行时已完成的组合直接跳过。
- **DB 日期段复用**：若某个 `(股票, 参数组合)` 的部分日期已在 `backtest.duckdb` 中，只运行缺口日期段，并把新旧交易合并。使用 `--no-reuse` 可禁用此行为。

### 单组合交易审计报告（`audit.py`）

```bash
# 从网格结果 CSV 中选最优参数组合生成审计报告
uv run backtest/audit.py --from-csv backtest/results/.../results_US_SNDK.csv \
    --code US.SNDK --start 2025-05-22 --end 2026-05-22

# 手动指定参数
uv run backtest/audit.py --code US.SNDK --start 2025-05-22 --end 2026-05-22 \
    --trend-tf 15m --entry-tf 3m
```

报告内容：10 项 KPI 卡片（WR / PF / Sharpe / Sortino / DD 等）、净值曲线、全部交易列表（含 trade_id）、最长连胜/连亏 K 线图、Top-5 亏损与 Top-3 盈利交易图。同时将交易记录写入 `db/review_trades.duckdb`，供 Trade Viewer 按 trade_id 查询。

### 策略适配性筛选器（`screener.py`）

对一批股票计算 SMC 策略适配特征（FVG 有效性、KD 清晰度、ATR、日均成交金额），输出评分排行榜、收益率相关矩阵和成交量相关矩阵。

```bash
# 使用 config/schedule.json 中的股票列表
uv run backtest/screener.py --start 2025-01-01

# 指定股票列表和输出路径
uv run backtest/screener.py --codes US.SNDK US.NVDA US.AMD --start 2025-01-01 --out backtest/results/my_screen.html
```

**输出特征说明：**

| 特征 | 含义 |
|------|------|
| `Touch%` | FVG 区间被价格影线触及的比例（越高 = 价格越会回测缺口）|
| `Bounce%` | 触及后收盘在区间正确侧的比例（关键评分指标）|
| `Overfill%` | 触及后穿越另一侧的比例（越低越好）|
| `FVG/100bar` | 每 100 HTF bar 的 FVG 数量（信号密度）|
| `KD clarity%` | KD 通道清晰度：40-60% 区间最优 |
| `ATR%` | 归一化波动率，甜点约 1.0% |
| `AvgDV($M)` | 日均成交金额（百万美元），衡量流动性 |

### FVG 过滤诊断（`fvg_inspect.py`）

用于手工对照：在指定时间窗口内，引擎检测到的每个 FVG 触碰在哪一步被过滤（或成功入场）。

```bash
# 从结果 CSV 选最优参数，检查某一周的所有 FVG 事件
uv run backtest/fvg_inspect.py \
    --from-csv backtest/results/.../results_US_SNDK.csv \
    --code US.SNDK --start 2025-05-22 --end 2026-05-22 \
    --inspect-start 2025-11-03 --inspect-end 2025-11-07
```

输出一份自包含 HTML，每行一个 FVG 触碰事件，列出：触碰时间、FVG 区间、方向、深度、过滤步骤及原因。颜色区分：绿色 = 入场，灰色 = 深度未达到，橙/红 = 各过滤器拦截。

**过滤步骤说明：**

| 结果 | 含义 |
|------|------|
| `direction_mismatch` | FVG 方向与当前趋势方向不一致（如牛市趋势中出现熊向 FVG），直接跳过 |
| `depth_never_reached` | 影线进入 FVG 但从未达到 `fvg_entry_depth_pct` 阈值，区间失效 |
| `ltf_confirmation` | 深度已达到，但区间失效前始终未完成 LTF CHoCH+BOS 确认 |
| `lvn_filter` | FVG 区间不在低成交量节点（LVN）内 |
| `displacement_filter` | FVG 中间蜡烛不满足位移蜡烛条件 |
| `no_sl_tp` | 无法找到有效的止损/止盈摆动位（含 KD fallback 也失败）|
| `max_sl_pct` | 止损距离超过 `max_sl_pct` 上限 |
| `min_rr` | 盈亏比不足 `min_rr` |
| `entered` | 所有条件通过，入场交易（显示 trade_id）|

### 统计指标说明

| 指标 | 缩写 | 公式 / 定义 | 参考值 |
|------|------|------------|--------|
| `n_trades` | — | 回测期间总交易笔数。过少（< 30）时统计结论不可靠。 | ≥ 30 较可信 |
| `total_r` | 总R | 所有交易的 R 加总。**1R = 1 倍止损距离**（例如止损 $1，盈利 $2 即 +2R）。与账户规模无关，便于跨股票对比。 | > 0 为净盈利 |
| `avg_r` | avgR | `total_r ÷ n_trades`，平均每笔交易的期望值 | > 0 为正期望 |
| `win_rate` | WR | 盈利笔数 ÷ 总笔数。单独看无意义，需结合 `avg_r` 或 `profit_factor`（低胜率 + 高盈亏比也可盈利）。 | 视策略而定 |
| `profit_factor` | PF | 总赢利 ÷ 总亏损（绝对值之比）。= 1 表示不赔不赚，> 1.5 通常认为具有实用价值。 | > 1.5 较好 |
| `sharpe` | — | `mean(R) / std(R)`，衡量每单位总波动所获得的回报。越高表示收益相对波动越稳定。基于每笔交易的 R 序列计算，非时间加权。 | > 1.0 较好 |
| `sortino` | — | 与 Sharpe 类似，但分母只用**下行**波动（亏损的标准差），对上行波动不惩罚。右偏策略（少亏多赢）的 Sortino 会明显高于 Sharpe。无亏损时为 ∞。 | > 1.5 较好，通常 ≥ Sharpe |
| `max_drawdown_r` | DD | 净值从峰值到谷值的最大跌幅（以 R 计）。反映资金曲线最坏的连续亏损段，影响实盘心理承受能力。若每笔风险 1% 账户，DD=10R 即账户最大回撤 10%。 | 越小越好，< 10R 较安全 |
| `max_loss_r` | — | 单笔最大亏损（绝对值）。正常情况应接近 −1R；若明显更大，说明止损执行存在滑点或跳空风险。 | 接近 1.0 为正常 |

---

## 文档

| 文件 | 内容 |
|------|------|
| [`backtest/README.md`](backtest/README.md) | 回测模块各脚本功能说明 + 典型工作流 |
| [`config/backtest/README.md`](config/backtest/README.md) | 回测 config JSON 字段说明 + 各配置文件用途 |
| [`strategy/smc/STRATEGY.md`](strategy/smc/STRATEGY.md) | SMC 策略逻辑 & 参数完整说明（Pipeline、过滤器、风控、KD 趋势方法）|
| [`doc/smc_v2_strategy.md`](doc/smc_v2_strategy.md) | smc_v2 / smc_v2.1 变更说明（KD 趋势、over-refill guard、自适应分段、与 v1 对比）|
| [`doc/smc_v1_strategy.md`](doc/smc_v1_strategy.md) | smc_v1 策略归档文档 |
| [`db/README.md`](db/README.md) | 数据库文件说明 + 完整 schema |
| [`doc/BACKTEST_SYSTEM_GUIDE.md`](doc/BACKTEST_SYSTEM_GUIDE.md) | 回测系统设计指南（架构决策、性能优化、实施状态、参考文献）|

## 数据库

详见 [`db/README.md`](db/README.md)。简要概述：

| 数据库 | 格式 | 内容 |
|--------|------|------|
| `ticks.db` | SQLite | 实时逐笔成交（~770 万行）|
| `backtest_klines.duckdb` | DuckDB | K 线缓存，供回测离线使用 |
| `backtest.duckdb` | DuckDB | 回测结果 + 实盘 / 模拟盘交易记录 |
| `review_trades.duckdb` | DuckDB | 交易索引（run.py + audit.py 写入，trade_viewer 读取）|

---

## 单元测试

```bash
uv run pytest tests/ -v        # 全部运行
uv run pytest tests/backtest/  # 只跑回测模块
uv run pytest tests/strategy/  # 只跑策略模块
```

| 测试文件 | 覆盖内容 |
|----------|---------|
| `tests/backtest/test_engine.py` | BacktestResult 指标计算、run_backtest() |
| `tests/backtest/test_db.py` | DB schema、CRUD、断点续跑、live_trades |
| `tests/backtest/test_stats.py` | Sharpe / Sortino / heatmap / parameter importance |
| `tests/backtest/test_logger.py` | 多进程日志 QueueHandler / QueueListener |
| `tests/strategy/test_base.py` | BaseStrategy ABC、SMCStrategy zone / signal schema |
| `tests/strategy/test_smc.py` | FVG 检测、BOS/CHoCH、swing points、volume profile |
| `tests/analysis/test_orderflow.py` | candle_start() 时间对齐、OHLCV profile |
