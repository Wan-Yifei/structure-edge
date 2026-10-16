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
const ctx = new Proxy({}, {
  get: (t, k) => {
    if (k === "measureText") return () => ({ width: 20 });
    if (k === "canvas") return {};
    return () => {};
  },
  set: () => true,
});
const mkEl = (id) => ({
  id, textContent: "", innerHTML: "", hidden: false, value: "",
  disabled: false, style: {}, className: "",
  onclick: null, oninput: null, onchange: null,
  focus() {}, setAttribute() {}, appendChild() {},
  getContext: () => ctx, width: 0, height: 0,
});

const loaded = [];
global.window = { devicePixelRatio: 1 };
global.localStorage = {
  _d: {},
  getItem(k) { return this._d[k] ?? null; },
  setItem(k, v) { this._d[k] = v; },
  removeItem(k) { delete this._d[k]; },
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
  code + "\n;return {choose, submit, levels, stepTo, loadSet, report, next," +
         "         prev, goto1," +
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
  ["确认让掉的幅度", "让掉"],
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
for (const tok of ["你的总 R", "入场后胜率", "盈亏平衡胜率", "我的交易"]) {
  if (!sc.includes(tok)) throw new Error(`scorecard missing row: ${tok}`);
}
if (!/入场后胜率[\s\S]{0,80}\d+ \/ \d+/.test(sc))
  throw new Error("win rate must show the count, not just a percentage");
console.log("ok   成绩页：总 R、胜率带分母、盈亏平衡线、分类统计都在");

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
if (/(^|[\s>(])-\d+\.\d+/.test(allText))
  throw new Error("a negative number still uses an ASCII hyphen, not U+2212");
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
