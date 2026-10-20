/* Load the drill page the way a browser does and exercise the paths that only
   run at runtime: data arriving through <script> tags, a question rendering,
   and a trade settling into the verdict panel. A syntax check cannot catch a
   field renamed out from under the template. */
import fs from "fs";
import path from "path";

const dir = "E:/project/moomoo/strategy/drill";
const html = fs.readFileSync(path.join(dir, "index.html"), "utf8");
const code = html.match(/<script>([\s\S]*)<\/script>/)[1];

const els = {};
/* Drawing calls are recorded so a test can ask what the chart painted.
   Only fillText is kept: it is the only call whose arguments identify what
   was drawn rather than merely where. */
const ctxCalls = [];
const ctxText = [];
/* Fills in the order they were issued, each with the alpha in force.
   Draw order is the only way to catch one thing painted over another. */
const ctxOps = [];
let curAlpha = 1;
const alphaStack = [];
const ctx = new Proxy({}, {
  get: (t, k) => {
    if (k === "measureText") return () => ({ width: 20 });
    if (k === "canvas") return {};
    if (k === "globalAlpha") return curAlpha;
    /* save/restore are modelled for alpha only, which is all the overpaint
       check needs; a real context restores much more. */
    if (k === "save") return () => { alphaStack.push(curAlpha); };
    if (k === "restore")
      return () => { if (alphaStack.length) curAlpha = alphaStack.pop(); };
    if (k === "fillText")
      return (txt, x, y) => {
        ctxCalls.push("fillText:" + txt);
        /* Where, not just what: label collisions are a geometry bug and
           the string alone cannot show one. */
        ctxText.push({ txt: String(txt), x, y });
        ctxOps.push({ i: ctxOps.length, op: "fillText", alpha: curAlpha,
                      txt: String(txt), x, y });
      };
    if (k === "fillRect")
      return (x, y, w, h) => {
        ctxOps.push({ i: ctxOps.length, op: "fillRect", alpha: curAlpha,
                      x, y, w, h });
      };
    return () => {};
  },
  set: (t, k, v) => { if (k === "globalAlpha") curAlpha = v; return true; },
});
const mkEl = (id) => ({
  id, textContent: "", innerHTML: "", hidden: false, value: "",
  disabled: false, style: {}, className: "",
  onclick: null, oninput: null, onchange: null,
  focus() {}, setAttribute() {}, appendChild() {},
  getContext: () => ctx, width: 0, height: 0,
  /* The curve charts map clientX back through this to a question index. */
  getBoundingClientRect: () => ({ left: 0, top: 0, width: 860, height: 230 }),
});

const loaded = [];
global.window = { devicePixelRatio: 1 };
global.localStorage = {
  _d: {},
  getItem(k) { return this._d[k] ?? null; },
  setItem(k, v) { this._d[k] = v; },
  removeItem(k) { delete this._d[k]; },
  /* badge() walks the keys to count each set's answers. */
  get length() { return Object.keys(this._d).length; },
  key(i) { return Object.keys(this._d)[i] ?? null; },
};
global.document = {
  getElementById: (id) => (els[id] ||= mkEl(id)),
  documentElement: {},
  addEventListener() {},
  createElement: () => ({ src: "", onload: null, onerror: null }),
  head: {
    appendChild(el) {
      loaded.push(el.src);
      const txt = fs.readFileSync(path.join(dir, el.src), "utf8");
      const eq = txt.indexOf("=");
      const name = txt.slice("window.".length, eq);
      global.window[name] = JSON.parse(txt.slice(eq + 1).replace(/;\s*$/, ""));
      setTimeout(() => el.onload(), 0);
    },
  },
};
global.getComputedStyle = () => ({ getPropertyValue: () => "#ffffff" });
global.matchMedia = () => ({ addEventListener() {} });
global.atob = (s) => Buffer.from(s, "base64").toString("binary");
global.fetch = () => Promise.reject(new Error("fetch must not be used"));

global.setInterval = () => 0;
global.clearInterval = () => {};
const api = new Function(
  code + "\n;return {choose, submit, levels, settle, ruleLight, stepTo, loadSet, report, next," +
         "         prev, goto1, isAnswered: (id) => !!ans[id], getIdx: () => idx," +
         "         getStep: () => step, getQS: () => QS};")();

await new Promise((r) => setTimeout(r, 400));

console.log("loaded files :", loaded.join(", "));
console.log("questions    :", global.window.DRILL_c1.length);
console.log("progress     :", els.prog.textContent);
if (!/第 \d+ \/ \d+ 题/.test(els.prog.textContent))
  throw new Error("progress not rendered");
if (!els.meta.innerHTML.includes("5m"))
  throw new Error("5m delta missing from the meta bar");
console.log("meta has 5m  : true");

/* The stop must not drag the target. Setting them in ATR made R a derived
   number; this is the property that was broken before. */
const q0 = global.window.DRILL_c1[0];
const a1 = api.levels(q0, 0.5, 1.5);
const a2 = api.levels(q0, 1.5, 1.5);
if (Math.abs(a1.target - a2.target) > 1e-9)
  throw new Error("target moved when only the stop changed");
if (Math.abs(a1.stop - a2.stop) < 1e-9)
  throw new Error("stop did not move");
if (Math.abs(a1.rr - 3.0) > 1e-9 || Math.abs(a2.rr - 1.0) > 1e-9)
  throw new Error(`R should be derived as tgt/stop, got ${a1.rr} / ${a2.rr}`);
console.log("ok   止损不再拖动目标，R 由两者相除得出");

els["s-stop"].value = "1.0";
els["s-tgt"].value = "2.0";
api.choose("enter");
api.submit();

/* Playback: the future starts hidden and advances one bar at a time. */
if (api.getStep() !== 0) throw new Error("future should start hidden");
api.stepTo(1);
if (api.getStep() !== 1) throw new Error("step did not advance");
api.stepTo(999);
const cur = api.getQS()[0];
const maxStep = cur.c.length - 1 - cur.mark;
if (api.getStep() !== maxStep)
  throw new Error(`step should clamp to ${maxStep}, got ${api.getStep()}`);
api.stepTo(-5);
if (api.getStep() !== 0) throw new Error("step should clamp at 0");
console.log(`ok   逐根播放 0..${maxStep}，两端都收住`);
api.stepTo(0);

const v = els.vd.innerHTML;
const need = [
  ["delta_r 面板", "delta_r"],
  ["15m 窗口", "15m"],
  ["结构读数", "结构读数"],
  ["基准率出处", "自助法"],
  ["确认那一根的代价", "等这根确认"],
];
for (const [label, token] of need) {
  const ok = v.includes(token);
  console.log(`${ok ? "ok  " : "MISS"} ${label}`);
  if (!ok) throw new Error(`verdict missing: ${label}`);
}
const sep = v.includes("并没有分开");
console.log(`${sep ? "ok  " : "MISS"} 结构三档未分开的说明`);
if (!sep) throw new Error("expected the 'not separated' note for these rates");

console.log("\nSMOKE OK");


/* The ledger set takes a different path: no rates file, nobase on every
   question, and the trader's own stop/target revealed only after the answer.
   A missing rates file must not break the load. */
await api.loadSet("mine");
await new Promise((r) => setTimeout(r, 200));
const mineQ = api.getQS();
if (!mineQ.length) throw new Error("ledger set did not load");
if (!mineQ.every((q) => q.nobase))
  throw new Error("every ledger question must be flagged nobase");
if (!mineQ.every((q) => q.myStop != null && q.myTarget != null))
  throw new Error("ledger questions must carry the stop and target used");
console.log(`ok   我的交易：${mineQ.length} 题，全部 nobase，带当时的止损止盈`);

els["s-stop"].value = "1.0";
els["s-tgt"].value = "2.0";
api.choose("enter");
api.submit();
const mv = els.vd.innerHTML;
for (const tok of ["你当时", "没有基准率"]) {
  if (!mv.includes(tok)) throw new Error(`ledger reveal missing: ${tok}`);
}
if (mv.includes("深回落") && mv.includes("到 +2ATR"))
  throw new Error("base rates must not be quoted at a nobase question");
console.log("ok   揭晓显示当时参数，且不套用基准率");

console.log("\nMINE OK");


/* The scorecard. Its per-class rows were keyed on the old schema's names
   (real/decoy/mine) long after the generator started writing 深回落 /
   浅回调 / 我的交易, so they matched nothing and silently never rendered --
   a table that is simply missing rows looks like a table. */
for (let k = 0; k < 4; k++) {
  els["s-stop"].value = "1.0";
  els["s-tgt"].value = "2.0";
  api.choose(k % 2 ? "pass" : "enter");
  api.submit();
  api.next();
}
api.report();
const sc = els.score.innerHTML;
for (const tok of ["你的总 R", "入场后胜率", "按实际盈亏算的盈亏平衡胜率",
                   "实际 平均盈利", "我的交易"]) {
  if (!sc.includes(tok)) throw new Error(`scorecard missing row: ${tok}`);
}
if (!/入场后胜率[\s\S]{0,80}\d+ \/ \d+/.test(sc))
  throw new Error("win rate must show the count, not just a percentage");
console.log("ok   成绩页：总 R、胜率带分母、盈亏平衡线、分类统计都在");

/* The curves. drawCurves() runs inside a try/catch on the page so a chart
   that throws cannot take the scorecard with it; that makes a silent
   failure possible, so check the catch did not fire and that something was
   actually painted. Enter enough trades to pass curveStats' n >= 5 floor,
   which is what gates the reshuffle paragraphs. */
/* Twelve entries: inside the band where the old n >= 10 gate drew the
   line and the correct n > 20 gate does not, so this discriminates
   between them instead of passing under either. */
for (let k = 0; k < 10; k++) {
  els["s-stop"].value = "1.0";
  els["s-tgt"].value = "2.0";
  api.choose("enter");
  api.submit();
  api.next();
}
/* With 20 entries or fewer, the last-20 window is every entry, so the
   rolling rate equals the cumulative one exactly and the line is not
   drawn -- it would hide the teal one under an identical orange one. */
ctxText.length = 0;
api.report();
if (/近20笔/.test(els["cv-leg1"].innerHTML.replace(/最近 20 笔/g, "")))
  throw new Error("the rolling line must not be drawn under 21 entries");
if (!/与累计完全重合/.test(els["cv-leg1"].innerHTML))
  throw new Error("the legend must say why the rolling line is absent");
if (ctxText.some((t) => t.txt === "近20笔"))
  throw new Error("a label was drawn for a line that is not there");

/* Past the gate, so both lines -- and both end labels -- exist below. */
for (let k = 0; k < 25; k++) {
  els["s-stop"].value = "1.0";
  els["s-tgt"].value = "2.0";
  api.choose("enter");
  api.submit();
  api.next();
}
ctxCalls.length = 0;
ctxText.length = 0;
api.report();

if (/画不出来/.test(els["cv-note"].textContent))
  throw new Error("drawCurves threw: " + els["cv-note"].textContent);
if (!/答题顺序/.test(els["cv-note"].textContent))
  throw new Error("the curve caption must say what the x axis is");
if (!/累计胜率/.test(els["cv-leg1"].innerHTML))
  throw new Error("win-rate legend missing");
if (!/累计 R/.test(els["cv-leg2"].innerHTML))
  throw new Error("equity legend missing");
if (!ctxCalls.some((c) => /^fillText:平衡 \d+%$/.test(c)))
  throw new Error("the break-even line was never labelled");
if (!ctxCalls.some((c) => /^fillText:答题序号$/.test(c)))
  throw new Error("the x axis was never labelled");
console.log("ok   曲线：两张图都画了，横轴和平衡线都标了");

const cr = els["cv-read"].innerHTML;
for (const tok of ["重排", "置换检验", "95% 区间"]) {
  if (!cr.includes(tok))
    throw new Error(`curve notes missing: ${tok}`);
}
/* The whole point of the section: the shape is one draw, and the page has
   to say so rather than let a rising line read as progress. */
if (!/形状是抽签/.test(cr))
  throw new Error("the notes must say the curve's shape is a draw");
console.log("ok   曲线读法：重排百分位、趋势置换、总 R 区间都在");

if (!ctxText.some((t) => t.txt === "近20笔"))
  throw new Error("past 20 entries the rolling line must be labelled");
if (!/最近 20 笔/.test(els["cv-leg1"].innerHTML))
  throw new Error("the legend must carry the rolling swatch once drawn");
console.log("ok   近20笔：满 21 笔才画，之前说明它与累计重合");

/* Label geometry. These all shipped broken once: the two line-end labels
   printed on top of each other whenever the lines finished near the same
   value, the break-even caption competed for the same gutter, and the axis
   caption sat under the last tick. */
{
  const at = (re) => ctxText.filter((t) => re.test(t.txt));
  /* Both charts paint into the one stub context, and frame() writes the
     axis caption first on each, so the captions mark the boundary. Without
     splitting here the test would compare a y on one canvas against a y on
     the other. */
  const caps = ctxText
    .map((t, i) => (t.txt === "答题序号" ? i : -1))
    .filter((i) => i >= 0);
  if (caps.length !== 2)
    throw new Error(`expected 2 charts, saw ${caps.length} axis captions`);
  const chart1 = ctxText.slice(caps[0], caps[1]);
  /* x > 700 is the right gutter: the y-axis ticks read "100%" too, and
     those sit at x = CVL - 6. */
  const ends = chart1.filter((t) => t.x > 700 && /%|近20笔/.test(t.txt));
  if (ends.length < 2)
    throw new Error("both line ends must be labelled, got " + ends.length);
  const ys = ends.map((e) => e.y).sort((a, b) => a - b);
  for (let i = 1; i < ys.length; i++)
    if (ys[i] - ys[i - 1] < 14)
      throw new Error(`end labels overlap: y ${ys[i - 1]} and ${ys[i]}`);

  const be = at(/^平衡 \d+%$/)[0];
  if (!be) throw new Error("break-even label missing");
  if (be.x > 200)
    throw new Error(`break-even label is in the right gutter (x=${be.x})`);

  /* CVW 860 minus CVR 96 is where the caption starts; a tick drawn inside
     58px of it would print under the caption. */
  const cap = at(/^答题序号$/)[0];
  if (!cap) throw new Error("axis caption missing");
  for (const t of ctxText)
    if (/^\d+$/.test(t.txt) && t.y === cap.y && t.x > cap.x - 58)
      throw new Error(`tick ${t.txt} collides with the axis caption`);
  console.log("ok   曲线标注：两端不重叠、平衡线靠左、刻度避开轴名");
}

/* The crosshair. The question chart above has had one all along; these two
   shipped inert. */
{
  ctxText.length = 0;
  els["cv-wr"].onmousemove({ clientX: 400 });
  const read = ctxText.filter((t) => /^第 \d+ 题/.test(t.txt));
  if (read.length < 2)
    throw new Error("hover must put a readout on BOTH charts, got " +
                    read.length);
  if (!/已入场 \d+ 笔/.test(read.map((r) => r.txt).join(" ")))
    throw new Error("the win-rate readout must say how many trades so far");
  if (!/累计 [+−]\d/.test(read.map((r) => r.txt).join(" ")))
    throw new Error("the equity readout must show the running total");
  ctxText.length = 0;
  els["cv-wr"].onmouseleave();
  if (ctxText.some((t) => /^第 \d+ 题/.test(t.txt)))
    throw new Error("the readout must clear when the cursor leaves");
  console.log("ok   曲线十字线：悬停两张图都出读数，移开就清掉");
}

/* Order comes from `at` now. Without it the sequence is unrecoverable from
   the page, since `ans` is keyed by id. */
const saved = JSON.parse(localStorage.getItem("drill:" + api.getQS()[0].id));
if (!saved || !saved.at)
  throw new Error("answers must record when they were given");
console.log("ok   答题记录带时间戳");

/* Signs. "入场 — +0.50 R" read as negative: the em dash sat exactly where a
   minus would, and the ASCII hyphen used for negatives is visually lighter
   than the plus beside it. Every signed R now goes through one formatter. */
const vd = els.vd.innerHTML;
if (/入场\s*—|放过\s*—/.test(vd))
  throw new Error("verdict heading still separates with a dash");
if (!/入场：|放过：/.test(vd))
  throw new Error("verdict heading should separate with a colon");
const allText = vd + els.score.innerHTML;
if (!/[+\u2212]\d/.test(allText))
  throw new Error("expected signed numbers in the verdict or scorecard");
const bad = allText.match(/(^|[\s>(])-\d+\.\d+[^<]{0,24}/);
if (bad)
  throw new Error("ASCII hyphen on a negative: " + JSON.stringify(bad[0]));
console.log("ok   正负号：冒号分隔，负号用 \u2212，加减视觉等重");

/* Order. Questions ship sorted by day and time, so in file order the first
   few come from adjacent minutes of one session -- the chart is already
   familiar by the second and the drill stops testing judgement. */
await api.loadSet("1");
await new Promise((r) => setTimeout(r, 300));
const ord = api.getQS();
if (ord.length < 50) throw new Error("expected the full generated set");

const fileOrder = [...ord].sort((a, b) =>
  a.day === b.day ? a.tm.localeCompare(b.tm) : a.day.localeCompare(b.day));
const sameAsFile = ord.every((q, i) => q.id === fileOrder[i].id);
if (sameAsFile) throw new Error("the set is still in file order");

let adjacent = 0;
for (let i = 1; i < ord.length; i++)
  if (ord[i].day === ord[i - 1].day) adjacent++;
if (adjacent > 0)
  throw new Error(`${adjacent} neighbouring pairs share a session`);
console.log(`ok   出题顺序：${ord.length} 题已打乱，相邻同日 0 对`);

/* Stable across reloads: "jump to the first unanswered" has to mean the same
   place tomorrow, so the seed is kept. */
await api.loadSet("1");
await new Promise((r) => setTimeout(r, 300));
const again = api.getQS();
if (!again.every((q, i) => q.id === ord[i].id))
  throw new Error("order changed on reload -- the seed is not being kept");
console.log("ok   重新加载顺序不变（种子已保存）");

/* Going back. An answered question must come back REVEALED -- serving it
   blank again would let a second answer overwrite the first, and the first
   is the one being scored. */
api.loadSet("1");
await new Promise((r) => setTimeout(r, 300));
els["s-stop"].value = "1.0";
els["s-tgt"].value = "2.0";
api.choose("enter");
api.submit();
const firstId = api.getQS()[0].id;
api.next();
if (els.vd.hidden !== true && els["b-go"].hidden !== false) {
  // a fresh question shows the submit button and hides the verdict
}
api.prev();
if (els["b-go"].hidden !== true)
  throw new Error("a revisited answered question must not offer 提交 again");
if (els.vd.hidden)
  throw new Error("a revisited answered question must show its verdict");
console.log("ok   上一题：已答的题回看时带结果，不能重答");

api.goto1(7);
if (!/第 7 \//.test(els.prog.textContent))
  throw new Error(`jump failed: ${els.prog.textContent}`);
api.goto1(99999);
if (!new RegExp(`第 ${api.getQS().length} \\/`).test(els.prog.textContent))
  throw new Error("jump should clamp to the last question");
api.goto1(-3);
if (!/第 1 \//.test(els.prog.textContent))
  throw new Error("jump should clamp to the first question");
console.log("ok   跳题：越界两端都收住");

/* The strip must not extend past the decision line. Its width is derived
   from the decision bar's x, so a bubble can never be drawn over the
   chart's hidden future -- which is what made it LOOK like a leak even
   though the window (decision-30min, decision] never held one. */
const src = fs.readFileSync(path.join(dir, "index.html"), "utf8");
if (!/const FW = mx - PAD_L;/.test(src))
  throw new Error("the flow strip no longer ends at the decision line");
if (/FX = c => PAD_L \+ \(c \/ NC\) \* W;/.test(src))
  throw new Error("the flow strip is spanning the full plot width again");
console.log("ok   订单流条带右边缘对齐决策线，未来区下方留空");

console.log("\nREPORT OK");

/* Layout. Two lines of caption used to be drawn INSIDE the strip's box, in
   the corner the volume profile and the bubbles occupy, so text and data
   overprinted. The caption sits above the frame now; this pins the
   arithmetic so a height change cannot quietly put it back over the data or
   clip the axis labels off the canvas. */
{
  // Parsed without a regex on purpose: every escape in this file has to
  // survive a shell heredoc to get here, and the backslashes did not.
  const num = (name) => {
    const i = src.indexOf(name + " = ");
    if (i < 0) throw new Error("layout constant missing: " + name);
    const v = parseInt(src.slice(i + name.length + 3), 10);
    if (!Number.isFinite(v)) throw new Error("unreadable constant: " + name);
    return v;
  };
  const ry0 = num("PAD_T") + num("PRICE_H") + num("GAP_H");
  const fy0 = ry0 + num("RSI_H") + num("GAP_H") + num("CAP_H");
  const CH = 520 + num("FLOW_H") + num("GAP_H") + num("CAP_H") + num("AXIS_H");
  if (num("CAP_H") < 12)
    throw new Error("no room above the strip for its caption");
  if (fy0 <= ry0 + num("RSI_H"))
    throw new Error("the strip overlaps the RSI panel");
  if (fy0 + num("FLOW_H") + 14 >= CH)
    throw new Error("the strip's axis labels fall off the canvas");
  console.log(`ok   版面：字幕在框外，底部刻度 ${fy0 + num("FLOW_H") + 14} < 画布 ${CH}`);
}

console.log("\nLAYOUT OK");

/* Answered questions belong at the front. next() is idx++ and does not look
   at whether a question is answered, so without the partition, walking
   forward from anywhere lands on ones already done -- answered in an earlier
   session, or reached by a jump. */
{
  api.loadSet("1");
  await new Promise((r) => setTimeout(r, 300));
  const qs = api.getQS();
  // Answer a scatter of them, then reload so the partition runs on restore.
  for (const i of [3, 11, 25, 40, 61]) {
    api.goto1(i + 1);
    els["s-stop"].value = "1.0";
    els["s-tgt"].value = "2.0";
    api.choose(i % 2 ? "pass" : "enter");
    api.submit();
  }
  api.loadSet("1");
  await new Promise((r) => setTimeout(r, 300));
  const after = api.getQS();
  const answered = after.map((q) => api.isAnswered(q.id));
  const firstTodo = answered.indexOf(false);
  if (firstTodo < 5)
    throw new Error(`expected 5 answered at the front, got ${firstTodo}`);
  if (answered.slice(firstTodo).some(Boolean))
    throw new Error("an answered question is sitting behind the boundary");
  console.log(`ok   已答的 ${firstTodo} 题排在最前，之后全是新题`);

  // The unanswered run must still keep sessions apart.
  const todo = after.slice(firstTodo);
  let adj = 0;
  for (let i = 1; i < todo.length; i++)
    if (todo[i].day === todo[i - 1].day) adj++;
  if (adj > 0)
    throw new Error(`${adj} same-session neighbours in the unanswered run`);
  console.log("ok   未答区间内相邻同日仍为 0");
}

console.log("\nPARTITION OK");

/* Jumping by identity. A position number is only stable within one load --
   the next one moves answered questions to the front -- so the box also
   takes the question's own date and time, which does not move. */
{
  api.loadSet("1");
  await new Promise((r) => setTimeout(r, 300));
  const qs = api.getQS();
  const target = qs[40];
  const label = target.day.slice(5) + " " + target.tm;   // MM-DD HH:MM

  api.goto1(label);
  if (api.getQS()[api.getIdx()].id !== target.id)
    throw new Error(`jump by "${label}" landed on the wrong question`);
  console.log(`ok   按时间跳题：「${label}」命中`);

  // And it survives a reorder, which a position number does not.
  els["s-stop"].value = "1.0";
  els["s-tgt"].value = "2.0";
  api.goto1(1); api.choose("pass"); api.submit();
  api.goto1(2); api.choose("pass"); api.submit();
  api.loadSet("1");
  await new Promise((r) => setTimeout(r, 300));
  api.goto1(label);
  if (api.getQS()[api.getIdx()].id !== target.id)
    throw new Error("jump by date/time broke after the order changed");
  console.log("ok   重排之后按时间跳题仍然命中同一题");

  api.goto1("99-99 99:99");
  if (!/找不到/.test(els.saved.textContent))
    throw new Error("an unmatched time should say so, not jump silently");
  console.log("ok   跳到不存在的时间会提示，而不是静默乱跳");
}

console.log("\nJUMP OK");

/* The settlement horizon. The chart draws AFTER bars of future, the rule
   settles on FWD of them, and a level reached between the two is visible
   while the verdict says it never happened. That looked like a bug until
   the horizon was drawn and the verdict named the late bar. */
{
  const q = global.window.DRILL_c1.find(
    x => x.day === "2026-10-02" && x.tm === "14:03");
  if (!q) throw new Error("the 2026-10-02 14:03 question is missing");
  /* The levels from the report: stop 162.42, target 164.34, R 1.52.
     Derived from the question so the test keeps meaning if the
     question's ATR is ever recomputed. */
  const stopAtr = (q.entry - 162.42) / q.atr;
  const tgtAtr  = (164.34 - q.entry) / q.atr;
  const s = api.settle(q, stopAtr, tgtAtr);
  if (!/没碰到/.test(s.how))
    throw new Error("expected a time-stop settlement, got: " + s.how);
  if (!/已出 30 根结算窗/.test(s.how))
    throw new Error("the late touch is not named: " + s.how);
  const late = Number(s.how.match(/第 (\d+) 根才/)[1]);
  if (!(late > 30))
    throw new Error("a touch inside the window must not be called late");
  // And the window itself must still be the thing that decided it.
  const within = q.h.slice(q.mark + 1, q.mark + 1 + 30);
  if (Math.max(...within) >= s.target)
    throw new Error("target was reachable inside the window after all");
  console.log("ok   结算视野：窗外才触及的目标会说明，而不是静默当作没发生");
}

/* The rule light. It is the measured rule rendered as three states, and
   it must stay hidden until the answer is in -- a light visible while you
   are still deciding turns the exercise into following a button. */
{
  const qs = global.window.DRILL_c1.filter(x => x.drs1 && x.dir === "long");
  const pick = (f) => qs.find(q => {
    const n = q.drs1[q.mark], p = q.mark >= 1 ? q.drs1[q.mark - 1] : null;
    return n != null && f(n, p);
  });

  const red = pick((n) => n < 0);
  const green = pick((n, p) => n >= 0.4 && p != null && p < 0);
  if (!red || !green)
    throw new Error("the question set has no red/green example to check");

  const L = (q) => api.ruleLight(q);
  if (L(red).k !== "red") throw new Error("dr1<0 must be red: " + L(red).k);
  if (L(green).k !== "green")
    throw new Error("threshold+flip must be green: " + L(green).k);

  // Amber is the in-between, and must not be reachable from either extreme.
  const amber = pick((n, p) => n >= 0 && !(n >= 0.4 && p != null && p < 0));
  if (amber && L(amber).k !== "amber")
    throw new Error("in-between must be amber: " + L(amber).k);

  // A bar with no reading says so rather than guessing a side.
  const na = qs.find(q => q.drs1[q.mark] == null);
  if (na && L(na).k !== "na")
    throw new Error("a bar with no delta_r must not get a colour");

  // And it is absent before the answer, present after.
  // The page hides the panel without clearing it, so the content of a
  // previous reveal lingers -- `hidden` is what decides visibility.
  // Answered questions are sorted to the front and re-show their verdict,
  // which is correct -- so this has to land on one that is not answered.
  const all = api.getQS();
  const fresh = all.findIndex(x => !api.isAnswered(x.id));
  if (fresh < 0) throw new Error("every question is already answered");
  api.goto1(fresh + 1);
  if (els.vd.hidden !== true)
    throw new Error("the verdict panel is visible before answering");
  api.choose("pass"); api.submit();
  if (els.vd.hidden !== false)
    throw new Error("the verdict panel stayed hidden after answering");
  if (!/规则：/.test(els.vd.innerHTML))
    throw new Error("the light is missing from the reveal");
  console.log("ok   规则红绿灯：三态正确，且答题前不显示");
}

/* The same light on the chart. It is drawn, not written into the DOM, so
   the check is on the canvas calls the stub records -- and the point being
   pinned is that it is absent until the answer is in, exactly as in the
   panel. */
{
  const all = api.getQS();
  const fresh = all.findIndex(x => !api.isAnswered(x.id));
  if (fresh < 0) throw new Error("every question is already answered");

  ctxCalls.length = 0;
  api.goto1(fresh + 1);
  const before = ctxCalls.filter(c => c.startsWith("fillText:"))
                         .some(c => /做多|别做|不触发|无数据/.test(c));
  if (before) throw new Error("the chart light is drawn before answering");

  api.choose("pass"); api.submit();
  ctxCalls.length = 0;
  api.stepTo(0);                       // force a redraw of the revealed chart
  const after = ctxCalls.filter(c => c.startsWith("fillText:"))
                        .some(c => /做多|别做|不触发|无数据/.test(c));
  if (!after) throw new Error("the chart light is missing after answering");
  console.log("ok   主图红绿灯：答题后才画出");

/* ...and that nothing paints over it afterwards. The settlement-horizon
   veil covers the price panel from the horizon to its right edge at 0.62
   alpha; the light sits in that corner, and was drawn first, so it shipped
   dimmed along with the bars behind it. Draw order is the bug, so draw
   order is what this checks. */
{
  ctxOps.length = 0;
  api.stepTo(60);                      // all the way out, so the veil is up
  const light = ctxOps.find(
    (o) => o.op === "fillText" && /^(做多|别做|不触发|无数据)$/.test(o.txt));
  if (!light) throw new Error("the rule light was not drawn");
  if (light.alpha < 1)
    throw new Error(`the light is drawn at alpha ${light.alpha}`);

  const over = ctxOps.filter(
    (o) => o.op === "fillRect" && o.alpha < 1 && o.i > light.i &&
           o.x <= light.x && light.x <= o.x + o.w &&
           o.y <= light.y && light.y <= o.y + o.h);
  if (over.length)
    throw new Error(
      `${over.length} translucent fill(s) painted over the rule light ` +
      `(alpha ${over[0].alpha})`);
  console.log("ok   主图红绿灯：结算灰罩不会盖在它上面");
}
}


/* Answering the last question used to be a dead end: 跳到未答的 hid itself
   with nothing left unanswered, and that was the only control that puts a
   question back -- prev and next are inside the card the score page hides.
   The jump box could not rescue it either, because `change` does not fire
   when the typed value matches what the field already shows, and the field
   shows the current position.

   This runs last on purpose: it answers everything, which would wreck the
   partition and jump tests above. */
{
  const all = api.getQS();
  for (let i = 0; i < all.length; i++) {
    if (api.isAnswered(all[i].id)) continue;
    api.goto1(i + 1);
    api.choose("pass");
    api.submit();
  }
  const left = all.filter((q) => !api.isAnswered(q.id)).length;
  if (left) throw new Error(`${left} questions left unanswered`);

  if (els["b-jump"].hidden)
    throw new Error("with everything answered there must still be a way " +
                    "back to the questions");
  if (els["b-jump"].textContent !== "回到题目")
    throw new Error("the button must say what it now does, got: " +
                    els["b-jump"].textContent);

  api.report();
  if (!els.qcard.hidden) throw new Error("report() should hide the card");
  els["b-jump"].onclick();
  if (els.qcard.hidden)
    throw new Error("回到题目 must bring the question card back");

  /* The jump box, on the position it is already showing. */
  api.report();
  els["b-goto"].value = String(api.getIdx() + 1);
  els["b-goto"].onkeydown({ key: "Enter", preventDefault() {} });
  if (els.qcard.hidden)
    throw new Error("Enter on the current position must bring the " +
                    "question back");

  /* A key that is not Enter must not jump. */
  api.report();
  els["b-goto"].onkeydown({ key: "7", preventDefault() {} });
  if (!els.qcard.hidden)
    throw new Error("only Enter should jump");
  console.log("ok   做完之后：回到题目可用，跳到当前序号回车也有效");
}


/* Switching decision position. The sets keep separate progress on purpose
   -- the ids carry _c0 / _c1 / mine_, and the low bar and the bar after it
   are different questions -- so the counter dropping to 0 is correct. It
   read as a wipe, with 清空重做 next to it and both buttons disappearing at
   the same moment.

   Runs after the block above, which answered the whole c1 set. */
{
  const beforeN = api.getQS().length;
  els.conf.value = "0";
  await api.loadSet("0");

  if (api.getQS().length === 0) throw new Error("c0 set failed to load");
  if (api.getQS().some((q) => api.isAnswered(q.id)))
    throw new Error("c1 answers must not count as c0 answers");

  const saved = els.saved.textContent;
  if (!/已保存 0 \//.test(saved))
    throw new Error("a fresh set should read zero: " + saved);
  if (!/仍在/.test(saved))
    throw new Error("it must say the other set's answers are still there: " +
                    saved);
  if (!new RegExp(`${beforeN} 题`).test(saved))
    throw new Error(`it should name the other set's count (${beforeN}): ` +
                    saved);

  if (els["b-jump"].hidden)
    throw new Error("the jump control must survive a set switch");
  if (els.qcard.hidden)
    throw new Error("switching sets must leave a question on screen");
  console.log("ok   切换决策位置：计数归零但说明另一套仍在，题目和跳转都在");
}
