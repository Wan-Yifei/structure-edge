# db/ 文件夹说明

这个文件夹存放所有本地数据库文件。共有三个数据库，职责各不相同。

---

## 文件一览

| 文件 | 大小（约） | 用途 |
|------|-----------|------|
| `ticks.db` | ~1 GB | 实时 tick 数据（逐笔成交，约 770 万行）|
| `ticks.db-shm` / `ticks.db-wal` | 几 MB | SQLite WAL 模式的配套文件，不用管 |
| `ticks.duckdb.migrated` / `.wal` | ~20 MB | 旧版迁移前的残留文件，可删除 |
| `backtest_klines.duckdb` | ~22 MB | K 线缓存（供回测使用）|
| `backtest.duckdb` | ~3 MB | 回测结果 + 实盘/模拟盘交易记录 |
| `scheduler.log` | 很小 | 定时任务的运行日志 |
| `__init__.py` | 0 字节 | Python 包标记，不含实际内容 |

---

## ticks.db — 实时 Tick 数据库

**格式**：SQLite（WAL 模式，支持同时读写）

由 `analysis/tick_collector.py` 写入，由 `analysis/trade_viewer.py` 读取用于 Order Flow 图表。

### 表：`ticks`

| 列 | 类型 | 说明 |
|----|------|------|
| `code` | TEXT | 股票代码，如 `US.SNDK` |
| `ts` | TEXT | 成交时间，ISO-8601 格式，如 `2026-05-18 09:30:00.123456` |
| `price` | REAL | 成交价格 |
| `volume` | INTEGER | 成交量（股数） |
| `direction` | TEXT | 方向：`BUY` / `SELL` / `NEUTRAL` |

**唯一约束**：`(code, ts, price, volume)` 防止重复写入。

**索引**：`(code, ts)` 加速按时间段查询。

---

## backtest_klines.duckdb — K 线缓存

**格式**：DuckDB

由 `feeds/fetcher.py` 从 moomoo API 拉取后写入缓存，回测时直接读取，避免每次重新请求 API。

### 表：`klines`

| 列 | 类型 | 说明 |
|----|------|------|
| `code` | TEXT | 股票代码，如 `US.SNDK` |
| `ktype` | TEXT | 时间框架：`1m` / `5m` / `15m` / `60m` / `1d` 等 |
| `time_key` | TEXT | K 线时间（moomoo 约定：bar 结束时间） |
| `open` | DOUBLE | 开盘价 |
| `high` | DOUBLE | 最高价 |
| `low` | DOUBLE | 最低价 |
| `close` | DOUBLE | 收盘价 |
| `volume` | BIGINT | 成交量 |

**主键**：`(code, ktype, time_key)`，同一根 K 线只存一条。

---

## backtest.duckdb — 回测结果与交易记录

**格式**：DuckDB

由 `backtest/run.py` 写入（回测结果），将来由 moomoo 执行模块写入（实盘/模拟盘）。Schema 定义在 `backtest/db.py`。

包含四张表，相互关联：

```
runs ──┬── trades
       └── run_stats

live_trades（独立）
```

---

### 表：`runs` — 每次回测任务

每个参数组合（如 4h/15m + 特定参数）对应一行。

| 列 | 类型 | 说明 |
|----|------|------|
| `run_id` | VARCHAR (UUID) | 主键，唯一标识一次回测 |
| `config_hash` | VARCHAR | 参数的 MD5 哈希，用于去重/断点续跑 |
| `config_json` | JSON | 完整参数快照，方便回溯 |
| `symbol` | VARCHAR | 股票代码，如 `US.SNDK` |
| `trend_tf` | VARCHAR | 趋势时间框架，如 `4h` |
| `entry_tf` | VARCHAR | 入场时间框架，如 `15m` |
| `start_date` | VARCHAR | 回测开始日期 |
| `end_date` | VARCHAR | 回测结束日期 |
| `status` | VARCHAR | `pending` / `running` / `done` / `failed` |
| `created_at` | TIMESTAMP | 创建时间 |
| `finished_at` | TIMESTAMP | 完成时间 |

---

### 表：`trades` — 每笔模拟交易

| 列 | 类型 | 说明 |
|----|------|------|
| `trade_id` | VARCHAR (UUID) | 主键 |
| `run_id` | VARCHAR | 关联 `runs.run_id` |
| `symbol` | VARCHAR | 股票代码 |
| `direction` | VARCHAR | `bull`（做多）/ `bear`（做空）|
| `entry_time` | VARCHAR | 入场时间 |
| `entry_price` | DOUBLE | 入场价 |
| `sl_price` | DOUBLE | 止损价 |
| `tp_price` | DOUBLE | 止盈价 |
| `exit_time` | VARCHAR | 出场时间 |
| `exit_price` | DOUBLE | 出场价 |
| `result` | VARCHAR | `win` / `loss` / `timeout` |
| `r_multiple` | DOUBLE | 实现盈亏（以 R 为单位，1R = 1 倍止损距离）|
| `planned_rr` | DOUBLE | 计划风险回报比 |

---

### 表：`run_stats` — 每次回测的汇总统计

| 列 | 类型 | 说明 |
|----|------|------|
| `run_id` | VARCHAR | 主键，关联 `runs.run_id` |
| `n_trades` | INTEGER | 总交易笔数 |
| `win_rate` | DOUBLE | 胜率（0–1） |
| `total_r` | DOUBLE | 累计 R（总盈亏） |
| `avg_r` | DOUBLE | 平均每笔 R |
| `profit_factor` | DOUBLE | 盈利因子 = 总赢利 ÷ 总亏损 |
| `max_drawdown_r` | DOUBLE | 最大回撤（以 R 计） |
| `max_loss_r` | DOUBLE | 单笔最大亏损（绝对值） |
| `sharpe` | DOUBLE | 夏普比率（基于 R 序列） |
| `sortino` | DOUBLE | 索提诺比率（只考虑下行波动） |
| `computed_at` | TIMESTAMP | 写入时间 |

---

### 表：`live_trades` — 实盘 / 模拟盘交易记录

记录通过 moomoo 实际执行的交易，与回测模拟交易严格分开。

| 列 | 类型 | 说明 |
|----|------|------|
| `trade_id` | VARCHAR (UUID) | 主键 |
| `account_type` | VARCHAR | **`LIVE`（实盘）或 `PAPER`（模拟盘）**，应用层强制校验 |
| `account_id` | VARCHAR | moomoo 账户 ID |
| `symbol` | VARCHAR | 股票代码 |
| `direction` | VARCHAR | `LONG` / `SHORT` |
| `order_id` | VARCHAR | 入场订单 ID（来自 moomoo） |
| `exit_order_id` | VARCHAR | 出场订单 ID |
| `entry_time` | TIMESTAMP | 入场时间 |
| `entry_price` | DOUBLE | 入场价 |
| `qty` | DOUBLE | 持仓数量（股） |
| `sl_price` | DOUBLE | 止损价 |
| `tp_price` | DOUBLE | 止盈价 |
| `planned_rr` | DOUBLE | 计划风险回报比 |
| `exit_time` | TIMESTAMP | 出场时间 |
| `exit_price` | DOUBLE | 出场价 |
| `result` | VARCHAR | `win` / `loss` / `breakeven` / `manual` / `open` |
| `pnl_gross` | DOUBLE | 毛利润（账户货币） |
| `commission` | DOUBLE | 手续费 |
| `pnl_net` | DOUBLE | 净利润 = 毛利润 − 手续费 |
| `r_multiple` | DOUBLE | 实现 R 倍数 |
| `strategy` | VARCHAR | 策略名称，如 `SMC_v1` |
| `run_id` | VARCHAR | 关联的回测 run_id（可选，用于对比回测与实盘表现）|
| `signal_params` | JSON | 产生信号时使用的策略参数快照 |
| `notes` | TEXT | 手工备注 |
| `tags` | VARCHAR | 标签，逗号分隔，如 `news_risk,missed_entry` |
| `created_at` | TIMESTAMP | 写入时间 |
| `updated_at` | TIMESTAMP | 最近更新时间 |

---

## 常用查询示例

```python
from backtest.db import BacktestDB

db = BacktestDB()

# 查看最近 10 次最优回测结果
df = db.get_run_stats(top_n=10)
print(df[["symbol", "trend_tf", "entry_tf", "profit_factor", "sharpe"]])

# 查看某次回测的所有交易
trades = db.get_trades(run_id="<uuid>")

# 查看所有未平仓的实盘交易
open_live = db.get_open_live_trades(account_type="LIVE")

# 查看某笔交易的完整信息（含策略参数快照）
record = db.fetch_trade("<trade_id>")
```
