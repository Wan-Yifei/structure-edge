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

const api = new Function(code + "\n;return {choose, submit, QS};")();

await new Promise((r) => setTimeout(r, 400));

console.log("loaded files :", loaded.join(", "));
console.log("questions    :", global.window.DRILL_c1.length);
console.log("progress     :", els.prog.textContent);
if (!/第 \d+ \/ \d+ 题/.test(els.prog.textContent))
  throw new Error("progress not rendered");
if (!els.meta.innerHTML.includes("5m"))
  throw new Error("5m delta missing from the meta bar");
console.log("meta has 5m  : true");

els["s-stop"].value = "1.0";
els["s-tgt"].value = "2.0";
api.choose("enter");
api.submit();

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
