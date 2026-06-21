# 回测 Config 说明

每个 JSON 文件定义一次回测实验的**标的、时间范围、并发度**以及**参数搜索空间**。运行时通过 `--config` 传入：

```bash
uv run backtest/run.py --config config/backtest/default_smc_v2.json
uv run backtest/run.py --config config/backtest/default_smc_v2.json --random 300
```

---

## JSON 字段说明

### 顶层字段

| 字段 | 类型 | 说明 |
|------|------|------|
| `codes` | `string[]` | 回测标的列表，格式 `"US.<代码>"` |
| `start` | `string` | 回测开始日期（含），格式 `"YYYY-MM-DD"` |
| `end` | `string` | 回测结束日期（含），格式 `"YYYY-MM-DD"` |
| `workers` | `int` | 并行 worker 进程数 |
| `top_n` | `int` | 结果报告中展示的最优参数组合数量 |
| `tf_pairs` | `[string, string][]` | 全量/随机搜索使用的时间框架对列表，格式 `[HTF, LTF]`，如 `["15m", "3m"]` |
| `tf_pairs_fast` | `[string, string][]` | `--fast` 冒烟测试使用的时间框架对（通常只保留一对以加快速度）|
| `param_grid` | `object` | 参数搜索空间（见下方说明）|

### `param_grid` 字段

`param_grid` 是一个字典，每个键对应一个 `BacktestParams` 字段，值为待搜索的候选值列表。引擎对所有键的候选值做**笛卡尔积**（全量模式）或**随机采样**（`--random N` 模式）。

| 字段 | 类型 | 说明 |
|------|------|------|
| `htf_trend_methods` | `string[][]` | HTF 趋势判断方法组合。每个元素是一个方法列表，如 `["bos_choch"]`、`["kd"]`、`["bos_choch", "kd"]`（两者同时需满足）。**注意：JSON 为 list-of-lists，引擎内部会转为 tuple。** |
| `htf_trend_params` | `object[]` | KD 趋势指标参数字典列表（仅在 `htf_trend_methods` 包含 `"kd"` 时生效），见下方参数说明 |
| `htf_window_bars` | `int[]` | HTF 回望窗口长度（bar 数），用于 BOS/CHoCH 的历史结构上下文 |
| `swing_lookback` | `int[]` | 摆动高低点识别的左右回望 bar 数 |
| `bos_count` | `int[]` | 触发趋势确认所需的 BOS 次数 |
| `fvg_min_width_pct` | `float[]` | FVG 最小宽度（相对价格百分比），过滤过小的缺口 |
| `fvg_entry_depth_pct` | `float[]` | 入场深度阈值：价格需进入 FVG 区间该比例才触发信号 |
| `fvg_max_age_bars` | `int[]` | FVG 最大有效期（LTF bar 数），超出则作废（可选字段）|
| `require_ltf_confirmation` | `bool[]` | 是否要求 LTF CHoCH + BOS 完成才入场 |
| `displacement_required` | `bool[]` | 是否要求形成 FVG 的中间蜡烛满足"位移"条件 |
| `sl_buffer_pct` | `float[]` | 止损点在摆动位之外的额外缓冲（相对价格百分比）|
| `max_sl_pct` | `float[]` | 止损最大允许距离上限；超出则放弃该入场机会 |
| `min_rr` | `float[]` | 最低盈亏比要求；低于该值不入场 |
| `kd_sl_fallback` | `bool[]` | 当正常摆动位止损超过 `max_sl_pct` 时，是否尝试用 KD 通道边界作为备用止损 |

### `htf_trend_params` 子字段

KD 趋势指标支持两种参数模式，字段名称有所不同：

**窗口模式**（`kd_window` + `kd_flat_threshold`）：

| 字段 | 说明 |
|------|------|
| `kd_fast` | 快速 EMA 周期 |
| `kd_slow` | 慢速 EMA 周期 |
| `kd_window` | 回望窗口：用最近 N 个方向一致的 bar 判断趋势 |
| `kd_flat_threshold` | 通道宽度低于此 ATR 倍数时视为震荡（trend=0）|

**平滑/最小 bar 模式**（`kd_smooth` + `kd_min_bars` + `kd_atr_threshold`）：

| 字段 | 说明 |
|------|------|
| `kd_fast` | 快速 EMA 周期 |
| `kd_slow` | 慢速 EMA 周期 |
| `kd_smooth` | KD 信号平滑 EMA 周期 |
| `kd_min_bars` | 趋势需持续至少 N bar 才确认 |
| `kd_atr_threshold` | 通道宽度低于此 ATR 倍数时视为震荡 |

---

## 各配置文件用途

| 文件 | 标的 | 趋势方法 | 说明 |
|------|------|---------|------|
| `default_smc_v2.json` | CSCO / AMD / NVDA / QCOM | bos_choch（默认）| 通用全参数空间搜索，不含 KD 趋势方法 |
| `sndk_15m1m_smc_v2.json` | SNDK | bos_choch（默认）| SNDK 专用，15m/1m TF 对，全参数空间 |
| `kd_smc_v2.json` | SNDK | `["kd"]` | 仅 KD 趋势，8 种 KD 参数组合（窗口模式）|
| `combined_smc_v2.json` | SNDK | `["bos_choch", "kd"]` | BOS/CHoCH + KD 共识趋势，8 种 KD 参数组合 |
| `focused_smc_v2.json` | SNDK | `["bos_choch", "kd"]` | 同上，但 KD 使用平滑/最小 bar 模式，含 15m/1m TF 对 |
| `mu_smc_v2.json` | MU | bos_choch（默认）| MU 专用，收窄参数空间，测试 `fvg_max_age_bars` |

---

## 运行模式

```bash
# 全量网格搜索（笛卡尔积穷举所有组合）
uv run backtest/run.py --config config/backtest/default_smc_v2.json

# 随机搜索（每个 TF pair 随机采样 300 个组合，适合大参数空间初探）
uv run backtest/run.py --config config/backtest/default_smc_v2.json --random 300

# 覆盖 config 中的 codes（仅回测指定标的，其他字段沿用 config）
uv run backtest/run.py --config config/backtest/default_smc_v2.json --codes US.NVDA

# 快速冒烟测试（不使用 config 的 param_grid，改用内置精简参数空间）
uv run backtest/run.py --fast --no-viz
```

结果目录名称格式：`backtest/results/<时间戳>_<config文件名>_grid/` 或 `…_random_<N>/`，例如：

```
backtest/results/20260526_1000_default_smc_v2_grid/
backtest/results/20260526_1010_default_smc_v2_random_200/
```

---

## 推荐工作流：Random → 聚合 → Grid

```bash
# 1. 多股票 random（固定 seed，所有股票跑相同的参数组合）
uv run backtest/run.py \
    --config config/backtest/default_smc_v2.json \
    --random 200 --seed 42 --no-viz

# 2. 聚合分析（生成 HTML 报告 + 缩范围 config）
uv run backtest/aggregate_random.py \
    --run-dir backtest/results/<timestamp>_default_smc_v2_random_200/ \
    --src-config config/backtest/default_smc_v2.json \
    --out-config cross_stock_grid_v1.json

# 3. 全量 grid（参数空间已缩小，运行时间大幅缩短）
uv run backtest/run.py --config config/backtest/cross_stock_grid_v1.json
```

`aggregate_random.py` 详细说明见 [`backtest/README.md`](../../backtest/README.md)。

---

## 非交易类工具配置

`fvg_width_default.json` 不属于以上交易策略回测体系——它给 [`backtest/fvg_width_sweep.py`](../../backtest/fvg_width_sweep.py) 使用，
只调 `strategy/smc/fvg.py` 自身的 FVG 检测参数（宽度/位移过滤），不产生交易、不挂 `smc_v` 算法版本号。
字段形状也不同：`tfs` 是单时间框架列表（FVG 检测不需要 HTF/LTF 配对），没有 `tf_pairs`/`workers`。
