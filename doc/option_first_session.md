# 期权逐笔采集：第一场怎么跑

**2026-11-23（周一）。标的 SOXL。**

这一场的目的不是"开始采数据"，是**回答六个今天不存在的数**。整合进 scheduler
的每一个参数都要用到它们，照猜的值写完明天重写，比等一天贵。

> **期权逐笔事后补不回来。** `get_rt_ticker` 每合约上限 1000 笔 —— 活跃合约约
> 25 分钟。这一场漏了就是漏了。

---

## 时间表（美东，本机时区就是 ET）

| 时刻 | 做什么 | 怎么触发 |
|---|---|---|
| **09:15** | 启动采集器 | 计划任务 `moomoo option collect (first session)`，会弹出一个**可见的控制台** |
| 09:30 | 开盘，盯 `DATA RECEIVED` | 看那个窗口 |
| 全天 | 每约 30 分钟一行 `telemetry:` | 同上 |
| **16:25** | 在控制台里按 **Ctrl-C**（不要点叉关窗） | 手动 |
| **16:45** | 对账 | 计划任务 `moomoo option reconcile (first session)` |

两个任务都是**一次性**的，不是每周重复 —— 第一场必须有人盯着。确认跑得住之后
再谈自动化。

**为什么 09:15 而不是 09:35**：开盘后头五分钟的流量最重，尤其在周五 −31.8%
之后。代价是选合约时 `last_price` 还是周五收盘价（30.88，而夜盘已到 35+），
但 ±80% 的窗口是 $6.18–$55.58，宽到覆盖所有会成交的行权价 —— 排序可能略偏，
窗口不会漏。

**为什么 16:45 而不是收盘即跑**：**5.6% 的笔数、7.8% 的张数在 16:00 之后**
（迟报，拖到约 16:15），券商的当日 `volume` 那之前没结清。跑太早比率会读错。

**为什么必须 Ctrl-C**：Windows 上父进程的 `terminate()` 调 `TerminateProcess`，
**不走 handler 也不走 `finally`** —— 退订、对账、prune、WAL checkpoint 全部
跳过。前台控制台的 Ctrl-C（SIGINT）是能跑到清理代码的。

---

## 盯什么

控制台里三类行：

```
DATA RECEIVED  US.SOXL261127C33000  ts=...  px=...  vol=...  BUY
    开盘一分钟内必须出现。没出现 = 推送没通，立刻查。

telemetry: subscribed 200 of 200 requested  option_used=6 option_remain=194
    ★ 最关键的一行。200 掉下去 = 订阅被踢。
      周日探测见过 210 → 60（加订第二个标的之后），但同一轮里
      unsubscribe_all 也没生效、数字还会自己衰减，所以分不清是真踢了
      还是收市时的脏读。这一场是唯一能分清的机会。

telemetry: greeks lag min  p50=... p90=... p99=... max=...
    ★ 盘中 now − update_time 的分布。收盘后的快照只给排序不给绝对值。
      这个数决定 delta 能错多少，进而决定采样频率该是多少。
```

---

## 收盘后记下这六个数

| # | 数 | 从哪读 | 它决定什么 |
|---|---|---|---|
| 1 | 覆盖率 % | 对账输出 | 这一场的数据能不能用 |
| 2 | `subscribed N of 200` 全天最低值 | telemetry 行 | 单标的 / 多标的；要不要做订阅自检 |
| 3 | greeks lag 的 p50 / p90 | **不要用 telemetry 那行**，用下面的 SQL | 采样频率、以及陈旧 delta 的修正要不要做 |
| 4 | ticks/min，以及**最长无成交间隔** | telemetry 行 + `option_ticks` | 僵尸判定阈值 |
| 5 | 两张表的行数、文件大小 | `option_ticks` / `option_greeks` | 保留期、eps 要不要调 |
| 6 | `dropped` / `snap_fail` / `handler_err` | 退出时的 session totals | 有没有静默丢数据 |

现成的查询：

```bash
uv run python -c "
import sqlite3
c = sqlite3.connect('db/option_ticks.db')
for q, lbl in [
    (\"select count(*), sum(volume) from option_ticks where substr(ts,1,10)='2026-11-23'\", 'ticks / contracts'),
    (\"select count(*) from option_greeks where substr(ts,1,10)='2026-11-23'\", 'greeks rows'),
    (\"select count(*) from option_greeks where nogreeks=1 and substr(ts,1,10)='2026-11-23'\", 'rows with no greeks'),
]:
    print(f'{lbl:22}', c.execute(q).fetchone())
print('coverage:', c.execute(
    \"select count(*), sum(our_vol), sum(feed_vol) from option_coverage where session='2026-11-23'\").fetchone())
print('longest gap between prints (minutes):', c.execute('''
    select max(gap) from (
      select (julianday(ts) - julianday(lag(ts) over (order by ts))) * 1440 as gap
      from option_ticks where substr(ts,1,10)='2026-11-23')''').fetchone())
"
```

### greeks 滞后要用这个查，不要用 telemetry 那一行

telemetry 读的是写缓存，**最多高估一个心跳（300 秒），而且偏差正好落在活跃
合约那一端** —— 真实滞后一分钟的合约可能被报成六分钟。它只能看方向。

无偏的数在库里：对**真正发生变化的行**（排除纯心跳行），`ts − upd` 就是那一刻
的滞后。

```bash
uv run python -c "
import sqlite3
c = sqlite3.connect('db/option_ticks.db')
rows = c.execute('''
  WITH g AS (
    SELECT code, ts, upd, delta, oi,
           LAG(delta) OVER (PARTITION BY code ORDER BY ts) pd,
           LAG(oi)    OVER (PARTITION BY code ORDER BY ts) po
    FROM option_greeks
    WHERE substr(ts,1,10)='2026-11-23' AND nogreeks=0 AND upd<>'')
  SELECT (julianday(ts)-julianday(upd))*1440 FROM g
  WHERE pd IS NULL OR delta<>pd OR oi<>po''').fetchall()
v = sorted(r[0] for r in rows if r[0] is not None)
q = lambda p: v[min(len(v)-1, int(p*len(v)))]
print(f'n={len(v)}  p50={q(.5):.2f}  p90={q(.9):.2f}  p99={q(.99):.2f}  max={v[-1]:.2f} min')
"
```

---

## 出事了怎么办

| 症状 | 判断 | 动作 |
|---|---|---|
| 开盘一分钟没有 `DATA RECEIVED` | **先查库再下判断** | 见下面那段。**不要先重起** |
| `subscribed N of 200` 里 N 在掉，但 N ≥ 150 | 订阅在被踢 | **记录 15 分钟**，然后重起一次看会不会再掉。衰减曲线看更久没有新信息，而「只掉过一次」和「总是会掉」是两回事 |
| N 掉到 **150 以下** | 已经在丢数据 | **立刻重起**，别等满 15 分钟 |
| `handler_err` 在涨 | 处理器在抛异常 | 控制台里有前 5 次的 traceback，抄下来 |
| 对账报 `NOT COMPARABLE` | 日期对不上 | `--session 2026-11-23` 显式指定 |
| 对账报 `caught NOTHING` 的合约很多 | 订阅掉了 | 这一场的数据作废，但**这个结论本身就是收获** |

### 「没有 DATA RECEIVED」不等于没在采

**这一条是用代价换来的。** 开发时有过一个 bug：handler 里一句
`time.time()` 因为导入遮蔽抛了 `AttributeError`，被 try/except 接住。
症状是 `DATA RECEIVED` 永远不打印、telemetry 永远报 `0 ticks = 0.0/min`
—— **而数据库一直在正常写入**，因为落库发生在抛异常之前。

按「没有 DATA RECEIVED → 推送没通 → 重起」处理，会去重启一个完全健康的
采集器，甚至可能得出「200 路订阅根本不工作」的错误结论。

**所以先查库，别信日志：**

```bash
uv run python -c "
import sqlite3; c = sqlite3.connect('db/option_ticks.db')
print(c.execute(\"select count(*), max(ts) from option_ticks where substr(ts,1,10)='2026-11-23'\").fetchone())
"
```

- **行数在涨** → 采集器是好的，坏的是计数器/日志。**不要重起**，记下来收盘后查。
- **行数是 0 且不动** → 推送真的没通。Ctrl-C 重起一次。还不行就记下来，
  **不要反复重起** —— 每次重起要跑 6–8 次 `get_option_chain`，会撞限流。

**一条提醒：** 周日探测时在 OpenD 上留下过几路 SOXL/QQQ 期权订阅，显式退订
没生效（数字自己从 130 衰减到 30）。周一起采集器时如果看到订阅数不对，先想到
这件事，重启 OpenD 可以清掉。

---

## 这一场之后才做的事

1. **指标**（期权版 CVD）—— 覆盖率站不住的话写了也白写
2. **接进 scheduler** —— 必须用 cron 机制（`09:15 起 / 16:25 停 / 16:45 对账`），
   **不能挂会话状态机**：实测那个状态机一天 24 小时没有空档，挂上去的采集器
   永远不会被停、因此**永远不会重选合约**，周五还握着周一的近月，而那个近月
   已经到期了
3. 采样频率 / eps / 保留期按第 3、5 两个数调
