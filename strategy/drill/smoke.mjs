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
let curAlpha = 1, curStroke = "", curFill = "";
/* y values of the path under construction: a stroke takes no
   arguments, so without them a stroke op cannot say WHAT it drew. */
let curYs = [];
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
    /* Path fills and strokes carry no arguments, so only their count and
       the alpha in force say anything -- which is enough to tell whether a
       band was painted. */
    if (k === "beginPath") return () => { curYs = []; };
    if (k === "moveTo" || k === "lineTo")
      return (x, y) => { curYs.push(y); };
    if (k === "fill")
      return () => { ctxOps.push({ i: ctxOps.length, op: "fill",
                                   alpha: curAlpha, color: curFill }); };
    if (k === "stroke")
      return () => {
        const ys = curYs.filter((v) => typeof v === "number" && isFinite(v));
        ctxOps.push({ i: ctxOps.length, op: "stroke", alpha: curAlpha,
                      color: curStroke, n: ys.length,
                      meanY: ys.length
                        ? ys.reduce((a, b) => a + b, 0) / ys.length : null });
      };
    return () => {};
  },
  set: (t, k, v) => {
    if (k === "globalAlpha") curAlpha = v;
    if (k === "strokeStyle") curStroke = v;
    if (k === "fillStyle") curFill = v;
    return true;
  },
});
/* A real little classList, not a no-op: the keyboard outline on the live
   slider is state a test wants to read back. */
const mkCls = () => {
  const set = new Set();
  return {
    _set: set,
    add: (c) => set.add(c),
    remove: (c) => set.delete(c),
    contains: (c) => set.has(c),
    toggle: (c, on) => {
      const want = on === undefined ? !set.has(c) : !!on;
      if (want) set.add(c); else set.delete(c);
      return want;
    },
  };
};
/* Attributes come from the page rather than from a copy kept here: the
   sliders' min/max/step are what the keyboard nudge clamps and rounds
   against, and a stub that guessed them would test arithmetic nobody
   ships. */
const attrsOf = (id) => {
  const tag = html.match(
    new RegExp('<[a-zA-Z]+[^>]*[ ]id="' + id + '"[^>]*>'));
  const out = {};
  if (tag)
    for (const m of tag[0].matchAll(/([a-zA-Z-]+)="([^"]*)"/g))
      out[m[1]] = m[2];
  return out;
};
const mkEl = (id) => ({
  id, textContent: "", innerHTML: "", hidden: false,
  disabled: false, checked: true, style: {}, className: "",
  classList: mkCls(),
  type: "", step: "", min: "", max: "", value: "",
  ...attrsOf(id),
  onclick: null, oninput: null, onchange: null,
  focus() {}, setAttribute() {}, appendChild() {},
  getContext: () => ctx, width: 0, height: 0,
  /* The curve charts map clientX back through this to a question index. */
  getBoundingClientRect: () => ({ left: 0, top: 0, width: 860, height: 230 }),
});

const loaded = [];
global.window = { devicePixelRatio: 1 };
/* orderFor draws a fresh seed from Math.random() when localStorage has
   none, which made this harness non-deterministic: a check that depends on
   WHICH questions get answered passed on one run and failed on the next.
   Seeding both sets up front makes a failure reproducible, which is the
   minimum a harness owes you. Override with SMOKE_SEED to sweep orders. */
const SEED = process.env.SMOKE_SEED || "20261025";
global.localStorage = {
  _d: { "drill:seed:0": SEED, "drill:seed:1": SEED },
  getItem(k) { return this._d[k] ?? null; },
  setItem(k, v) { this._d[k] = v; },
  removeItem(k) { delete this._d[k]; },
  /* badge() walks the keys to count each set's answers. */
  get length() { return Object.keys(this._d).length; },
  key(i) { return Object.keys(this._d)[i] ?? null; },
};
/* Keydown handlers were dropped on the floor, so nothing bound to the
   keyboard could be tested at all. They are collected instead, and
   fireKey() replays one the way the browser would. */
const keyHandlers = [];
const fireKey = (key, opts = {}) => {
  let prevented = false;
  const ev = {
    key, shiftKey: !!opts.shift, ctrlKey: false, altKey: false,
    metaKey: false, target: opts.target || { tagName: "BODY" },
    preventDefault(){ prevented = true; },
  };
  for (const h of keyHandlers) h(ev);
  return prevented;
};
global.document = {
  getElementById: (id) => (els[id] ||= mkEl(id)),
  documentElement: {},
  addEventListener(type, fn) { if (type === "keydown") keyHandlers.push(fn); },
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
/* Echo custom properties back as their own name so a test can assert
   WHICH token a mark was drawn in; anything else keeps a real colour. */
global.getComputedStyle = () => ({
  getPropertyValue: (k) => (String(k).startsWith("--") ? String(k) : "#ffffff"),
});
global.matchMedia = () => ({ addEventListener() {} });
global.atob = (s) => Buffer.from(s, "base64").toString("binary");
global.fetch = () => Promise.reject(new Error("fetch must not be used"));

global.setInterval = () => 0;
global.clearInterval = () => {};
const api = new Function(
  code + "\n;return {choose, submit, levels, settle, ruleLight, stepTo, loadSet, report, next," +
         "         prev, goto1, isAnswered: (id) => !!ans[id], getIdx: () => idx," +
         "         cellOf, nextByDeficit," +
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
/* Where you stand against the gate, not just the rule: 12 entries in, it
   must name the count and the gap, or a pending line reads as a broken
   one. */
{
  const m = els["cv-leg1"].innerHTML.match(/现在 (\d+) 笔入场，还差 (\d+) 笔/);
  if (!m)
    throw new Error("the legend must report the count and the gap: " +
                    els["cv-leg1"].innerHTML);
  /* Checked as arithmetic rather than against a literal, so the numbers
     stay consistent with the gate when the blocks above change. */
  if (Number(m[1]) + Number(m[2]) !== 21)
    throw new Error(
      `${m[1]} entries plus ${m[2]} to go does not reach the gate of 21`);
}
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
if (bad){
  /* Print what surrounds it: the text is assembled from several notes and
     the match alone does not say which one produced it. */
  const at = allText.indexOf(bad[0]);
  throw new Error("ASCII hyphen on a negative: " + JSON.stringify(bad[0]) +
                  " | context: " +
                  JSON.stringify(allText.slice(Math.max(0, at - 260), at + 100)));
}
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

/* The per-period panel: grouped by the question's month, calendar order,
   with the excess over the mechanical entry stated per month. */
{
  /* Re-render: later blocks in this file answered more questions, and the
     panel left over from the earlier report() call describes the set as
     it stood then. Comparing that against the current answers compares
     two different moments. */
  api.report();
  const rowsHtml = els["period"].innerHTML || "";
  const months = [...rowsHtml.matchAll(/(\d{4}-\d{2})（/g)].map(m => m[1]);
  if (months.length < 1)
    throw new Error("the period table rendered no rows");
  const sorted = months.slice().sort();
  if (String(months) !== String(sorted))
    throw new Error(`periods are not in calendar order: ${months}`);
  if (new Set(months).size !== months.length)
    throw new Error("a month appears twice -- grouping is not by month");

  /* Every month present among the ANSWERED questions has to appear; a
     panel that quietly dropped one would understate exactly the kind of
     period skew it exists to expose. */
  const answered = api.getQS().filter(q => api.isAnswered(q.id));
  const want = [...new Set(answered.map(q => q.day.slice(0, 7)))].sort();
  if (String(months) !== String(want))
    throw new Error(`months shown ${months} but answered span ${want}`);

  if (!/超额/.test(rowsHtml))
    throw new Error("the period rows should state the excess over the " +
                    "mechanical entry, not the raw R");
  console.log("ok   分时期：按题目月份分组、日历序，并给出相对机械的超额");
}

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

/* The DK channel ribbon. It is a path fill plus two strokes, so it leaves
   no text and no fillRect -- the suite passed it for a week by never
   running it, because the question files on disk had no dk1/dk2 and the
   guard skipped the block entirely. */
{
  const q = api.getQS()[api.getIdx()];
  const n = q.c.length;

  const count = () => {
    ctxOps.length = 0;
    api.stepTo(0);
    return {
      fills: ctxOps.filter((o) => o.op === "fill").length,
      strokes: ctxOps.filter((o) => o.op === "stroke").length,
      faint: ctxOps.filter((o) => o.op === "fill" && o.alpha < 0.5).length,
    };
  };

  /* Baseline FIRST, with the channel removed. Taking the file's current
     state as the baseline only worked while the question sets had no
     dk1/dk2; once they did, the baseline already contained a ribbon and
     the comparison measured which synthetic channel had more crossings. */
  const saved1 = q.dk1, saved2 = q.dk2;
  q.dk1 = null; q.dk2 = null;
  const before = count();

  /* A channel that crosses once, so the fill has to be split and both
     colours used -- a single segment would pass even if the crossing
     logic were dropped. */
  q.dk1 = []; q.dk2 = [];
  for (let i = 0; i < n; i++) {
    q.dk1.push(q.c[i] + (i < n / 2 ? 0.5 : -0.5));
    q.dk2.push(q.c[i]);
  }
  const after = count();

  if (after.fills <= before.fills)
    throw new Error(`the ribbon drew no fill (${before.fills} -> ` +
                    `${after.fills})`);
  if (after.strokes < before.strokes + 2)
    throw new Error("both midlines should be stroked, got " +
                    (after.strokes - before.strokes));
  if (after.faint <= before.faint)
    throw new Error("the ribbon fill should be translucent, so it sits " +
                    "behind the candles rather than over them");

  /* The lines wear their OWN tokens, not the fill's. The viewer pens the
     fast midline blue and the slow one gold, while the fill is gold/purple
     by which side leads -- four roles, and the first version of this had
     the pair swapped AND borrowed the fill's two colours for the lines. */
  const strokedIn = (tok) =>
    ctxOps.some((o) => o.op === "stroke" && o.color === tok);
  const filledIn = (tok) =>
    ctxOps.some((o) => o.op === "fill" && o.color === tok);
  if (!strokedIn("--dk-fast"))
    throw new Error("the fast midline should be stroked in --dk-fast");
  if (!strokedIn("--dk-slow"))
    throw new Error("the slow midline should be stroked in --dk-slow");
  if (!filledIn("--dk-up") && !filledIn("--dk-dn"))
    throw new Error("the ribbon fill should wear --dk-up / --dk-dn");
  if (strokedIn("--dk-up") || strokedIn("--dk-dn"))
    throw new Error("a midline is wearing a fill colour");

  /* Which line wears which. Both tokens being present somewhere passes
     when the two lines simply trade colours, which is the bug that got
     shipped -- so feed a channel whose fast line is above the slow one
     everywhere and check the blue one really is the upper line. */
  for (let i = 0; i < n; i++){ q.dk1[i] = q.c[i] + 0.5; q.dk2[i] = q.c[i]; }
  count();
  const meanOf = (tok) => {
    const s = ctxOps.filter((o) => o.op === "stroke" && o.color === tok &&
                                   o.n > 1);
    if (!s.length) throw new Error(`no midline stroked in ${tok}`);
    return s.reduce((a, o) => a + o.meanY, 0) / s.length;
  };
  /* Smaller y is higher on the canvas. */
  if (!(meanOf("--dk-fast") < meanOf("--dk-slow")))
    throw new Error("fast and slow midlines are wearing each other's " +
                    "colour -- fast is blue, slow is gold");

  /* And the real data on file draws one too -- the point of regenerating
     the sets. Checked against the same cleared baseline. */
  q.dk1 = saved1; q.dk2 = saved2;
  if (saved1 && saved2 && count().fills <= before.fills)
    throw new Error("the channel stored on the question drew nothing");
  console.log("ok   DK 多空通道：有数据才画，填充半透明，快线蓝、慢线金不互换");

/* The arrow marking the bar the trade actually ended on. */
{
  const qs = api.getQS();
  /* A question whose stop or target is actually reached inside the window;
     one that times out has no bar to point at and would pass vacuously. */
  let found = null;
  for (let k = 0; k < qs.length && !found; k++){
    const st = api.settle(qs[k], 1.0, 2.0);
    if (st.hit === "tp" || st.hit === "sl" || st.hit === "both")
      found = {k, st};
  }
  if (!found) throw new Error("no settling question in the set");

  api.goto1(found.k + 1);
  els["s-stop"].value = "1.0";
  els["s-tgt"].value = "2.0";
  api.choose("enter");
  api.submit();

  const tok = found.st.hit === "tp" ? "--tgt" : "--stop";
  const count = (step) => {
    ctxOps.length = 0;
    api.stepTo(step);
    return ctxOps.filter((o) => o.op === "fill" && o.color === tok).length;
  };

  /* One bar short of the hit: the level's own label is already drawn, so
     this is the baseline the arrow has to rise above. */
  const before = count(found.st.bar - 1);
  const after = count(found.st.bar);
  if (after <= before)
    throw new Error(`the ${found.st.hit} arrow did not appear ON the ` +
                    `settling bar (${before} -> ${after}) -- either it is ` +
                    `not drawn at all, or it is drawn on every bar and was ` +
                    `already in the baseline`);

  /* And nothing of the sort while the bar is still hidden. */
  if (count(0) !== before)
    throw new Error("the arrow is drawn before its bar is revealed -- " +
                    "that answers the question early");
  console.log("ok   成交箭头：结算那根才画，之前不画");

/* The green light's flip may be one bar back. */
{
  const DR = 0.4;
  const at = (q, k) => {
    const s = q.drs1, m = q.mark;
    return (s && m - k >= 0) ? s[m - k] : null;
  };
  const qs = api.getQS();
  const pick = (f) => qs.find((q) => {
    const n = at(q, 0), p1 = at(q, 1), p2 = at(q, 2);
    return n != null && p1 != null && p2 != null && f(n, p1, p2);
  });

  /* One bar late: the flip is at m-2, m-1 is positive but short of the
     threshold, m clears it. Amber under the old rule, green now. */
  const late = pick((n, p1, p2) => n >= DR && p1 >= 0 && p1 < DR && p2 < 0);
  if (!late) throw new Error("no one-bar-late flip in the set to test");
  const L = api.ruleLight(late);
  if (L.k !== "green")
    throw new Error(`a flip one bar back should be green, got ${L.k}`);

  /* No flip behind it at all stays amber -- the widening is about WHEN the
     threshold is reached, not about dropping the flip requirement. */
  const noFlip = pick((n, p1, p2) => n >= DR && p1 >= DR && p2 >= 0);
  if (noFlip && api.ruleLight(noFlip).k !== "amber")
    throw new Error("0.4 with no flip behind it must not be green");

  /* And the plain cases are untouched. */
  const onBar = pick((n, p1) => n >= DR && p1 < 0);
  if (onBar && api.ruleLight(onBar).k !== "green")
    throw new Error("a flip on the bar itself should still be green");
  const neg = pick((n) => n < 0);
  if (neg && api.ruleLight(neg).k !== "red")
    throw new Error("dr1 < 0 should still be red");

  /* The continuation case: there IS a flip two bars back, but the bar
     right after it already cleared the threshold, so the green fired
     there. Saying "身后两根都不是负的" here is simply false -- the
     negative is on the chart, two bars back. */
  const cont = pick((n, p1, p2) => n >= DR && p1 >= DR && p2 < 0);
  if (cont){
    const C = api.ruleLight(cont);
    if (C.k !== "amber")
      throw new Error(`a continuation bar should be amber, got ${C.k}`);
    if (/都不是负的/.test(C.d))
      throw new Error("the amber text claims there is no flip behind a " +
                      "bar that has one two back: " + C.d);
    if (!/延续/.test(C.d))
      throw new Error("a continuation bar should say so: " + C.d);
  }

  console.log("ok   规则红绿灯：翻转晚一根也算，延续那根不谎称没有翻转");
}
}
}
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


/* ---- serving the bucket that is furthest behind ---------------------- *
 * Measured over the 800 answers on file, coverage by kind was 浅回调 48%,
 * 深回落 52% and 夹角 10% -- with zero 夹角 answered in August or
 * September, where 50 and 42 of them exist. The kind the trader named as
 * their weakness was the one a fixed order never reached.
 *
 * Everything above has already answered the whole set, and orderFor draws
 * a fresh seed from Math.random() when localStorage has none, so this
 * starts from a cleared, seeded state rather than from the leftovers.   */
{
  for (const k of Object.keys(localStorage._d))
    if (/^drill:\d{4}-/.test(k) || /^drill:mine/.test(k))
      delete localStorage._d[k];
  localStorage.setItem("drill:seed:1", "20261025");
  api.loadSet("1");
  await new Promise((r) => setTimeout(r, 300));
  els["b-bal"].checked = true;

  const QS = api.getQS();
  const cells = new Map();
  for (let i = 0; i < QS.length; i++){
    const c = api.cellOf(QS[i]);
    if (!cells.has(c)) cells.set(c, []);
    cells.get(c).push(i);
  }
  if (cells.size < 2)
    throw new Error("need at least two buckets to balance between");

  /* Build the imbalance deliberately: fill the largest bucket part way
     and leave the rest untouched. */
  const big = [...cells.entries()].sort((a,b) => b[1].length - a[1].length)[0];
  const fill = Math.min(40, big[1].length - 1);
  for (let k = 0; k < fill; k++){
    api.goto1(String(big[1][k] + 1));
    els["s-stop"].value = "1.0";
    els["s-tgt"].value = "2.0";
    api.choose("pass");
    api.submit();
  }
  const covOf = (c) => {
    const idxs = cells.get(c);
    const done = idxs.filter(i => api.isAnswered(QS[i].id)).length;
    return done / idxs.length;
  };
  let want = null, best = Infinity;
  for (const c of cells.keys()){
    if (covOf(c) >= 1) continue;
    if (covOf(c) < best){ best = covOf(c); want = c; }
  }
  if (want === big[0])
    throw new Error("the fed bucket is still the emptiest; bad fixture");

  api.next();
  const got = QS[api.getIdx()];
  if (api.isAnswered(got.id))
    throw new Error("served a question that was already answered");
  if (api.cellOf(got) !== want)
    throw new Error(`served ${api.cellOf(got)} at coverage ` +
                    `${(100*covOf(api.cellOf(got))).toFixed(0)}%, but ` +
                    `${want} was behind at ${(100*best).toFixed(0)}%`);
  console.log(`ok   按缺口出题：喂饱 ${big[0]} 之后，转到落后的 ${want}`);

  /* Keep feeding and the deficit has to move on, or a bucket that starts
     behind would be served for ever. */
  const first = api.cellOf(got);
  let n = 0;
  while (api.cellOf(QS[api.getIdx()]) === first && n++ < 500){
    els["s-stop"].value = "1.0";
    els["s-tgt"].value = "2.0";
    api.choose("pass");
    api.submit();
    api.next();
  }
  if (n >= 500)
    throw new Error("the same bucket kept coming back after 500 answers");
  console.log(`ok   喂饱之后换桶：${n} 题后从 ${first} 移开`);

  /* Off, it is the plain walk it always was. */
  els["b-bal"].checked = false;
  const at = api.getIdx();
  api.next();
  if (api.getIdx() !== at + 1)
    throw new Error(`toggle off should step to ${at+1}, went to ` +
                    api.getIdx());
  els["b-bal"].checked = true;
  console.log("ok   关掉开关：恢复成按顺序走");

  if (els["l-bal"].hidden !== els["b-jump"].hidden)
    throw new Error("the toggle must appear with the jump control");
  console.log("ok   开关和「跳到未答的」一起出现");
}

console.log("\nBALANCE OK");

/* ---- the monthly table says how uncertain it is ---------------------- *
 * The win rates by month were 50/56/70/52/49/43% on 4/18/20/61/45/23
 * entries, and every Wilson interval covers the pooled 53%. The panel
 * presented those as if they were differences.                          */
{
  /* Needs entries, and in more than one month, or there is nothing to be
     uncertain about. */
  const QS = api.getQS();
  const byMonth = new Map();
  for (let i = 0; i < QS.length; i++){
    const m = QS[i].day.slice(0, 7);
    if (!byMonth.has(m)) byMonth.set(m, []);
    byMonth.get(m).push(i);
  }
  const months = [...byMonth.keys()].sort();
  if (months.length < 2)
    throw new Error("the set spans one month; nothing to compare");
  // A deliberately lopsided pair: many entries in one month, few in another.
  // Skip what the balance section already answered -- submit() refuses a
  // second answer -- and take the two months with the most left, so the
  // fixture does not depend on which months that section happened to eat.
  const free = m => byMonth.get(m).filter(i => !api.isAnswered(QS[i].id));
  const roomy = months.slice().sort((a, b) => free(b).length - free(a).length);
  if (free(roomy[1]).length < 25)
    throw new Error("not enough unanswered questions left for the fixture");
  const plan = [[roomy[0], 25], [roomy[1], 4]];
  for (const [m, k] of plan)
    for (const i of free(m).slice(0, k)){
      api.goto1(String(i + 1));
      els["s-stop"].value = "1.0";
      els["s-tgt"].value = "2.0";
      api.choose("enter");
      api.submit();
    }

  api.report();
  const tbl = els.period.innerHTML;
  if (!/\[\d+%, \d+%\]/.test(tbl))
    throw new Error("no confidence interval beside the monthly win rate");
  console.log("ok   月度胜率带上了 95% 区间");

  const read = els["pd-read"].innerHTML;
  if (!/区间|噪声/.test(read))
    throw new Error("the reading notes say nothing about the uncertainty");
  console.log("ok   读法里点明了区间/噪声");

  /* The interval must widen as the sample shrinks, or it is decoration. */
  // Per row: a single regex over the whole table runs past a month with
  // no entries and picks up the NEXT month's interval.
  const w = tbl.split("</tr>").map(row => {
    const h = row.match(/(\d+) 题，进 (\d+)）/);
    const ci = row.match(/\[(\d+)%, (\d+)%\]/);
    return h && ci ? { ent: +h[2], width: +ci[2] - +ci[1] } : null;
  }).filter(x => x && x.ent > 0);
  if (w.length < 2)
    throw new Error("expected two months with entries, got " + w.length +
                    " | table: " + JSON.stringify(tbl.slice(0, 600)));
  const bySize = w.slice().sort((a, b) => a.ent - b.ent);
  if (bySize[0].width <= bySize[bySize.length - 1].width)
    throw new Error("the smallest month must have the widest interval");
  console.log(`ok   样本越小区间越宽（${bySize[0].ent} 笔 ` +
              `${bySize[0].width} 点 vs ${bySize[bySize.length-1].ent} 笔 ` +
              `${bySize[bySize.length-1].width} 点）`);
}

console.log("\nINTERVAL OK");


/* ---- answering from the keyboard ------------------------------------ *
 * 1 放过  2 入场  3 调止损  4 调目标  5 提交，W/S 加减，空格或回车确认.
 * The two bindings worth guarding are the collisions: digits typed into
 * a text box must stay in the box, and space after the reveal must keep
 * driving the playback rather than jumping to the next question.       */
{
  for (const k of Object.keys(localStorage._d))
    if (/^drill:\d{4}-/.test(k) || /^drill:mine/.test(k))
      delete localStorage._d[k];
  api.loadSet("1");
  await new Promise((r) => setTimeout(r, 300));

  const stop = els["s-stop"], tgt = els["s-tgt"];
  const inStop = () => stop.classList.contains("kbfocus");
  const inTgt = () => tgt.classList.contains("kbfocus");

  if (!fireKey("1")) throw new Error("1 should be handled");
  if (!els.sl.hidden) throw new Error("1 放过 must not show the sliders");
  if (els["b-go"].disabled) throw new Error("1 should arm 提交");
  console.log("ok   键盘 1：放过");

  fireKey("2");
  if (els.sl.hidden) throw new Error("2 入场 must show the sliders");
  if (inStop() || inTgt())
    throw new Error("2 must not jump into a slider by itself");
  console.log("ok   键盘 2：入场，且不自作主张进入调整");

  fireKey("3");
  if (!inStop()) throw new Error("3 should drive the stop slider");
  if (!/调止损/.test(els["kb-hint"].textContent))
    throw new Error("the hint must name the mode: " +
                    els["kb-hint"].textContent);
  const v0 = +stop.value;
  fireKey("w");
  if (Math.abs(+stop.value - (v0 + 0.05)) > 1e-9)
    throw new Error(`W should add one step, ${v0} -> ${stop.value}`);
  fireKey("s"); fireKey("s");
  if (Math.abs(+stop.value - (v0 - 0.05)) > 1e-9)
    throw new Error(`S should subtract, got ${stop.value}`);
  fireKey("w", { shift: true });
  if (Math.abs(+stop.value - (v0 + 0.20)) > 1e-9)
    throw new Error(`Shift+W should be five steps, got ${stop.value}`);
  console.log(`ok   键盘 W/S：一步 0.05，Shift 五步（${v0} → ${stop.value}）`);

  /* Floating point: 1.00 + 3*0.05 is 1.1500000000000001 and the ATR
     readout would print it. */
  for (let i = 0; i < 7; i++) fireKey("w");
  if (!/^\d+\.\d{2}$/.test(String(stop.value)))
    throw new Error("the value drifted off the grid: " + stop.value);
  console.log(`ok   不会漂出两位小数（${stop.value}）`);

  for (let i = 0; i < 400; i++) fireKey("w");
  if (+stop.value > +stop.max)
    throw new Error(`ran past max: ${stop.value} > ${stop.max}`);
  for (let i = 0; i < 400; i++) fireKey("s");
  if (+stop.value < +stop.min)
    throw new Error(`ran past min: ${stop.value} < ${stop.min}`);
  console.log(`ok   夹在 [${stop.min}, ${stop.max}] 之内`);

  fireKey(" ");
  if (inStop()) throw new Error("space should leave the slider");
  console.log("ok   空格：退出调整状态");

  const held = +stop.value;
  fireKey("4");
  if (!inTgt() || inStop())
    throw new Error("4 should move the outline to the target, alone");
  const t0 = +tgt.value;
  fireKey("w");
  if (Math.abs(+tgt.value - (t0 + 0.05)) > 1e-9)
    throw new Error("W should now drive the target");
  if (+stop.value !== held)
    throw new Error(`the stop moved while the target was selected: ` +
                    `${held} -> ${stop.value}`);
  fireKey("Enter");
  if (inTgt()) throw new Error("enter should leave the slider too");
  console.log("ok   键盘 4：只有一个滑块在被驱动，回车同样退出");

  /* A text box keeps its digits, or the goto field stops working. */
  const box = { tagName: "INPUT", type: "text" };
  const before = els["b-go"].disabled;
  fireKey("1", { target: box });
  if (els["b-go"].disabled !== before)
    throw new Error("a digit typed into a text field reached the drill");
  console.log("ok   输入框里的数字不会被抢走");

  fireKey("5");
  if (els.vd.hidden) throw new Error("5 should submit and reveal");
  console.log("ok   键盘 5：提交");

  /* Space after the reveal belongs to the playback; that binding is older. */
  const at = api.getIdx();
  fireKey(" ");
  if (api.getIdx() !== at)
    throw new Error("space after the reveal must not change question");
  fireKey("Enter");
  if (api.getIdx() === at)
    throw new Error("enter after the reveal should move on");
  console.log("ok   揭晓后：空格留给播放，回车才翻题");
}

console.log("\nKEYS OK");
