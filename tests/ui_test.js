const { JSDOM } = require("jsdom");
const BASE = "http://127.0.0.1:5055";
const sleep = (ms) => new Promise(r => setTimeout(r, ms));
let fails = 0, calls = [], delayFirst = 0, forceError = false;
const ok = (c, l, d) => { console.log((c ? "  PASS  " : "  FAIL  ") + l + (!c && d ? "  [" + d + "]" : "")); if (!c) fails++; };
(async () => {
  const dom = await JSDOM.fromURL(BASE + "/", { runScripts: "dangerously", resources: "usable", pretendToBeVisual: true,
    beforeParse(w) {
      w.fetch = async (url, opts) => {
        const body = JSON.parse(opts.body); calls.push(body);
        const my = calls.length; const nc = new AbortController();
        opts.signal.addEventListener("abort", () => nc.abort());
        if (forceError) return new w.Response ? null : { ok: false, json: async () => ({ status: "error", error: "boom" }) };
        const r = await fetch(BASE + url, { method: "POST", headers: opts.headers, body: opts.body, signal: nc.signal });
        const data = await r.json();
        if (delayFirst && my === delayFirst) await sleep(700);
        return { ok: r.ok, status: r.status, json: async () => data };
      };
    } });
  const w = dom.window, d = w.document;
  await sleep(1200);
  const ed = d.getElementById("editor"), chips = () => [...d.querySelectorAll(".chip")], ghost = () => d.querySelector(".ghost");
  const type = (t) => { ed.value = t; ed.setSelectionRange(t.length, t.length); ed.dispatchEvent(new w.Event("input", { bubbles: true })); };
  const status = () => d.getElementById("status").textContent;
  ok(!d.getElementById("predict"), "no Predict button in DOM");
  ok(d.querySelector('.seg[aria-checked="true"]').dataset.model === "LSTM", "LSTM selected by default");

  console.log("[typing + debounce]");
  for (const t of ["a", "a n", "a ne", "a neural", "a neural network learns"]) { type(t); await sleep(30); }
  await sleep(900);
  ok(calls.length === 1, "5 rapid keystrokes -> 1 request", calls.length);
  ok(calls[0].text === "a neural network learns" && calls[0].model === "LSTM", "request carries text+model");
  ok(chips().length === 5, "5 chips shown", chips().length);
  ok(chips()[0].querySelector("kbd"), "top chip has Tab badge");
  const top1 = chips()[0].querySelector(".w").textContent;
  ok(ghost() && ghost().textContent === " " + top1, "inline ghost = ' ' + top word", ghost() && ghost().textContent);

  console.log("[Tab accept]");
  let prevented = false; const ev = new w.KeyboardEvent("keydown", { key: "Tab", bubbles: true, cancelable: true });
  ed.dispatchEvent(ev); prevented = ev.defaultPrevented;
  ok(prevented, "Tab default prevented");
  ok(ed.value === "a neural network learns " + top1 + " ", "word inserted once, followed by a space", JSON.stringify(ed.value));
  await sleep(900);
  ok(calls.length === 2 && calls[1].text === ed.value, "exactly one new prediction after accept", calls.length);
  ok(chips().length === 5, "new suggestions shown");
  ok(ghost() && !ghost().textContent.startsWith(" "), "no double space in new ghost");
  await sleep(600); ok(calls.length === 2, "no prediction loop");

  console.log("[click chip]");
  const second = chips()[1].querySelector(".w").textContent, before = ed.value;
  chips()[1].dispatchEvent(new w.MouseEvent("click", { bubbles: true }));
  ok(ed.value === before + second + " ", "clicked word inserted", JSON.stringify(ed.value));
  await sleep(900); ok(calls.length === 3, "prediction after click");

  console.log("[escape / caret]");
  ed.dispatchEvent(new w.KeyboardEvent("keydown", { key: "Escape", bubbles: true }));
  ok(!ghost(), "Esc hides ghost");
  type(ed.value + "x"); await sleep(900);
  ed.setSelectionRange(3, 3); ed.dispatchEvent(new w.Event("mouseup", { bubbles: true }));
  ok(!ghost(), "no ghost when caret is mid-text");

  console.log("[model switch]");
  type("the training loss"); await sleep(900);
  const n = calls.length;
  d.querySelector('.seg[data-model="RNN"]').click(); await sleep(900);
  ok(calls.length === n + 1 && calls[n].model === "RNN", "switching model sends RNN request without reload");
  ok(/RNN/.test(status()), "status mentions RNN", status());

  console.log("[sentence start + empty]");
  type("Hello world."); await sleep(900);
  ok(ghost() && /^ [A-Z]/.test(ghost().textContent), "capitalised after full stop", ghost() && ghost().textContent);
  const m = calls.length; type(""); await sleep(600);
  ok(calls.length === m && chips().length === 0 && !ghost(), "empty text: no request, no suggestions");

  console.log("[partial word]");
  type("the gradient desc"); await sleep(900);
  const g = ghost() && ghost().textContent;
  ok(g && !g.startsWith(" ") && ("desc" + g).length > 4, "ghost completes the fragment", g);
  ed.dispatchEvent(new w.KeyboardEvent("keydown", { key: "Tab", bubbles: true, cancelable: true }));
  ok(/^the gradient desc\w+ $/.test(ed.value), "fragment completed without duplication", ed.value);

  console.log("[stale response]");
  await sleep(600); delayFirst = calls.length + 1;
  type("a neural network learns"); await sleep(400);   // request 1 fired, held 700ms
  type("the gradient"); await sleep(1600);
  const exp = await (await fetch(BASE + "/api/predict", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ text: "the gradient", model: "RNN", top_k: 5 }) })).json();
  ok(chips()[0].querySelector(".w").textContent === exp.predictions[0].word, "UI shows result for latest text, not the slow earlier one");
  delayFirst = 0;

  console.log("[server error]");
  w.fetch = async () => ({ ok: false, status: 500, json: async () => ({ status: "error", error: "The prediction service had a problem." }) });
  type("a model is"); await sleep(900);
  ok(d.querySelector(".status-row").classList.contains("error") && chips().length === 0, "error shown, suggestions cleared");
  console.log(fails ? `\n${fails} FAILED` : "\nALL UI CHECKS PASSED"); process.exit(fails ? 1 : 0);
})();
