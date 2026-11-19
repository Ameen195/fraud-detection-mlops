// FraudGuard console. Numbers come only from eval/results (served at results/) and the API.
// Safe rendering: data reaches the DOM through text nodes / textContent, never innerHTML.
import { disposeCharts, lineChart, responsive, scale, svgEl } from "./charts.js";

/* ---------- formatting (same precision as the eval/results markdown) ---------- */
const nf = (d) => new Intl.NumberFormat("en-US", { minimumFractionDigits: d, maximumFractionDigits: d });
const F = { 0: nf(0), 2: nf(2), 3: nf(3), 4: nf(4) };
const fmt = {
  p: (v) => (v < 0.0001 ? "<0.0001" : F[4].format(v)), // probabilities, thresholds
  m4: (v) => F[4].format(v), // metrics
  m2: (v) => F[2].format(v),
  m3: (v) => F[3].format(v),
  pct: (v) => `${F[2].format(v * 100)}%`,
  pct0: (v) => `${F[0].format(v * 100)}%`,
  int: (v) => F[0].format(v),
  amt: (v) => F[2].format(v),
  // raw input amounts have up to 3 decimals; show them exactly instead of rounding a tie like 92.835
  amtIn: (v) => new Intl.NumberFormat("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 3 }).format(v),
  shap: (v) => `${v >= 0 ? "+" : "−"}${F[3].format(Math.abs(v))}`,
};
const LEVEL = { normal: "Normal", elevated: "Elevated", highest: "Highest" };
const ACTION = {
  normal: ["Allow", "Below the Elevated threshold: approve without friction."],
  elevated: ["Step-up check", "Ask for a one-time passcode (OTP) before approving."],
  highest: ["Block", "At or above the Highest threshold: decline and send to manual review."],
};

/* ---------- DOM helpers ---------- */
function el(tag, attrs = {}, ...kids) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v == null || v === false) continue;
    if (k === "class") e.className = v;
    else if (k.startsWith("on")) e.addEventListener(k.slice(2), v);
    else e.setAttribute(k, v === true ? "" : v);
  }
  for (const k of kids.flat()) if (k != null && k !== false) e.append(k instanceof Node ? k : String(k));
  return e;
}
const num = (v, cls = "") => el("span", { class: `num ${cls}` }, v);
const pill = (lvl) => el("span", { class: `pill lvl-${lvl}` }, LEVEL[lvl]);
const tile = (label, value, sub) =>
  el("div", { class: "tile" }, el("div", { class: "tile-label" }, label),
    el("div", { class: "tile-value" }, value), sub && el("div", { class: "tile-sub" }, sub));
const head = (title, sub, aside) =>
  el("header", { class: "head" },
    el("div", {}, el("h1", { tabindex: -1 }, title), sub && el("p", { class: "sub" }, sub)), aside);
const panel = (title, desc, ...kids) =>
  el("section", { class: "panel", "aria-label": title },
    el("div", { class: "panel-head" }, el("h2", {}, title), desc && el("p", {}, desc)), ...kids);
const table = (cols, rows, rowAttrs = () => ({})) =>
  el("div", { class: "table-wrap" }, el("table", {},
    el("thead", {}, el("tr", {}, cols.map((c) => el("th", { class: c.num ? "num" : null, scope: "col" }, c.label)))),
    el("tbody", {}, rows.map((r) => el("tr", rowAttrs(r),
      cols.map((c) => el("td", { class: c.num ? "num" : null }, c.get(r))))))));

/* ---------- data ---------- */
const cache = {};
function getJSON(url, opts) {
  return (cache[url] ??= fetch(url, opts).then(async (r) => {
    if (!r.ok) {
      const body = await r.json().catch(() => ({}));
      throw new Error(`${url}: HTTP ${r.status}${body.detail ? ` · ${body.detail}` : ""}`);
    }
    return r.json();
  }).catch((e) => { delete cache[url]; throw e; }));
}
const results = (name) => getJSON(`results/${name}`);
const model = () => getJSON("../model");

async function queue() {
  const samples = await getJSON("samples");
  const { results: out } = await getJSON("../score/batch", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ transactions: samples.map((s) => s.payload) }),
  });
  return samples.map((s, i) => ({ ...s, ...out[i] })).sort((a, b) => b.probability - a.probability);
}

/* ---------- honest feature labels (IEEE-CIS columns are masked by the data provider) ---------- */
const ENGINEERED = {
  card_n_past: "Earlier transactions on this card",
  card_amt_mean_past: "Card's average earlier amount",
  amt_to_card_mean: "Amount ÷ card's average earlier amount",
  amt_z_card: "Amount vs card's earlier amounts (z-score)",
  secs_since_last: "Seconds since card's previous transaction",
  new_device_for_card: "Device never seen on this card before",
  new_email_for_card: "Email domain never seen on this card before",
  new_addr_for_card: "Billing region never seen on this card before",
};
const RAW = {
  TransactionAmt: "Transaction amount",
  ProductCD: "Product code (masked category)",
  card4: "Card network",
  card6: "Card type (credit / debit)",
  addr1: "Billing region (masked code)",
  addr2: "Billing country (masked code)",
  P_emaildomain: "Purchaser email domain",
  R_emaildomain: "Recipient email domain",
  DeviceType: "Device type",
  DeviceInfo: "Device / browser string",
};
const MASKED = [
  [/^C(\d+)$/, "Anonymized count C$1", "a masked counter, e.g. addresses linked to the card; meaning not disclosed"],
  [/^D(\d+)$/, "Anonymized time delta D$1", "days between events, e.g. since a previous transaction; meaning not disclosed"],
  [/^M(\d+)$/, "Anonymized match flag M$1", "e.g. name on card vs billing address; meaning not disclosed"],
  [/^card(\d)$/, "Card attribute card$1 (masked)", "issuer / category style field; meaning not disclosed"],
  [/^dist(\d)$/, "Anonymized distance dist$1", "a masked distance between addresses; meaning not disclosed"],
  [/^id_(\d+)$/, "Identity field id_$1 (masked)", "device / network / identity attribute; meaning not disclosed"],
];
const WINDOW = { "1h": "past hour", "24h": "past 24 h", "7d": "past 7 days" };

function describe(f, payload) {
  const has = Object.hasOwn(payload, f);
  const value = has ? `value ${payload[f]}` : "missing in this transaction";
  let m;
  if (ENGINEERED[f]) return [ENGINEERED[f], "computed from the card's earlier transactions"];
  if ((m = f.match(/^card_(n|amt)_(1h|24h|7d)$/))) {
    return [`${m[1] === "n" ? "Transactions" : "Spend"} on this card in the ${WINDOW[m[2]]}`,
      "computed from the card's earlier transactions"];
  }
  if ((m = f.match(/^freq_(.+)$/))) {
    const base = RAW[m[1]] ?? (m[1] === "card1" ? "card attribute card1 (masked)" : m[1]);
    return [`How often this ${base.toLowerCase()} was seen before`, "computed from earlier transactions"];
  }
  if (RAW[f]) return [RAW[f], value];
  for (const [re, label, note] of MASKED) if (re.test(f)) return [f.replace(re, label), `${value} · ${note}`];
  return [f, value];
}

/* ---------- pages ---------- */
const PBAR_LO = 0.001;
function probCell(p, lvl) {
  // log scale [0.001, 1], same as the detail page risk bar
  const frac = Math.log10(Math.min(1, Math.max(PBAR_LO, p)) / PBAR_LO) / -Math.log10(PBAR_LO);
  const fill = el("i");
  fill.style.width = `${(frac * 100).toFixed(1)}%`;
  return el("span", { class: "pcell" },
    el("span", { class: `pbar lvl-${lvl}`, "aria-hidden": "true" }, fill), fmt.p(p));
}

async function pageQueue() {
  const [rows, info] = await Promise.all([queue(), model()]);
  const count = (l) => rows.filter((r) => r.risk_level === l).length;
  return el("div", {},
    head("Review queue",
      `${rows.length} sample test transactions, scored live by the API and sorted by risk. Select one to see why.`,
      el("div", { class: "counts", "aria-label": "Transactions per risk level" },
        ["highest", "elevated", "normal"].map((l) => el("span", { class: `pill lvl-${l}` }, `${LEVEL[l]} ${count(l)}`)))),
    el("div", { class: "meta", style: "margin-bottom:14px" },
      el("span", {}, "Model ", el("b", { class: "num" }, `${info.name} v${info.version}`)),
      el("span", {}, "Elevated ≥ ", el("b", { class: "num" }, fmt.p(info.thresholds.elevated))),
      el("span", {}, "Highest ≥ ", el("b", { class: "num" }, fmt.p(info.thresholds.highest)))),
    table([
      { label: "Risk", get: (r) => pill(r.risk_level) },
      { label: "Probability", num: true, get: (r) => probCell(r.probability, r.risk_level) },
      { label: "Amount", num: true, get: (r) => fmt.amtIn(r.payload.TransactionAmt) },
      { label: "Product", get: (r) => r.payload.ProductCD ?? "—" },
      { label: "Card type", get: (r) => r.payload.card6 ?? "—" },
      { label: "Action", get: (r) => el("span", { class: "dim" }, ACTION[r.risk_level][0]) },
      { label: "Transaction", num: true, get: (r) => el("a", { href: `#/tx/${r.id}` }, String(r.id)) },
    ], rows, (r) => ({ class: "link", onclick: (e) => { if (e.target.tagName !== "A") location.hash = `#/tx/${r.id}`; } })));
}

function riskBar(p, elev, high) {
  const lo = 0.001, H = 92;
  return responsive((W) => {
    const m = { l: 10, r: 10 };
    const x = scale("log", [lo, 1], [m.l, W - m.r]);
    const cx = x(Math.min(1, Math.max(lo, p)));
    const svg = svgEl("svg", {
      class: "chart", width: W, height: H, viewBox: `0 0 ${W} ${H}`, role: "img",
      "aria-label": `Fraud probability ${fmt.p(p)}. Elevated threshold ${fmt.p(elev)}, Highest threshold ${fmt.p(high)}. Log scale.`,
    });
    const add = (t, a, s) => svg.appendChild(svgEl(t, a, s));
    const by = 36, bh = 12;
    add("rect", { class: "zone lvl-normal", x: x(lo), y: by, width: x(elev) - x(lo), height: bh, rx: 2 });
    add("rect", { class: "zone lvl-elevated", x: x(elev), y: by, width: x(high) - x(elev), height: bh });
    add("rect", { class: "zone lvl-highest", x: x(high), y: by, width: x(1) - x(high), height: bh, rx: 2 });
    for (const [t, lvl] of [[elev, "elevated"], [high, "highest"]]) {
      add("line", { class: `thr lvl-${lvl}`, x1: x(t), x2: x(t), y1: by - 4, y2: by + bh + 4 });
      add("text", { class: `thr-label lvl-${lvl}`, x: x(t), y: by + bh + 22, "text-anchor": "middle" },
        `${LEVEL[lvl]} ≥ ${fmt.p(t)}`);
    }
    add("text", { class: "tick", x: m.l, y: by + bh + 22 }, "0.001");
    add("text", { class: "tick", x: W - m.r, y: by + bh + 22, "text-anchor": "end" }, "1");
    add("line", { class: "needle", x1: cx, x2: cx, y1: by - 10, y2: by + bh + 6 });
    add("text", {
      class: "needle-label", x: Math.min(W - m.r - 30, Math.max(m.l + 30, cx)), y: by - 16, "text-anchor": "middle",
    }, p < lo ? `<${lo} (${fmt.p(p)})` : fmt.p(p));
    return svg;
  });
}

async function pageTx(id) {
  const [rows, info] = await Promise.all([queue(), model()]);
  const i = rows.findIndex((r) => String(r.id) === id);
  if (i < 0) throw new Error(`Transaction ${id} is not in the sample set.`);
  const r = rows[i], t = r.payload, lvl = r.risk_level, [word, why] = ACTION[lvl];
  const max = Math.max(...r.reasons.map((x) => Math.abs(x.shap)));
  const outcome = r.label
    ? (lvl === "normal" ? "Fraud · missed by the model" : "Fraud · caught")
    : (lvl === "normal" ? "Legitimate · correctly allowed" : `Legitimate · false alarm (${ACTION[lvl][0].toLowerCase()})`);
  const reveal = el("div", { class: "reveal", "aria-live": "polite" });
  reveal.append(el("button", {
    type: "button",
    onclick: () => reveal.replaceChildren(
      el("span", { class: "outcome" }, outcome),
      el("span", { class: "dim" }, `isFraud = ${r.label} in the dataset`)),
  }, "Reveal true label"));
  const nav = (j, text) => (rows[j]
    ? el("a", { class: "btn", href: `#/tx/${rows[j].id}` }, text)
    : el("span", { class: "btn dim", "aria-disabled": "true" }, text));

  return el("div", {},
    el("div", { class: "row", style: "justify-content:space-between;margin-bottom:18px" },
      el("a", { class: "btn", href: "#/queue" }, "← Review queue"),
      el("div", { class: "pager" }, nav(i - 1, "Previous"), nav(i + 1, "Next"))),
    head(el("span", {}, "Transaction ", num(String(r.id))), null, pill(lvl)),
    panel("Transaction", null,
      el("dl", { class: "facts" }, [
        ["Amount", num(fmt.amtIn(t.TransactionAmt))], ["Product", t.ProductCD ?? "—"],
        ["Card type", t.card6 ?? "—"], ["Card network", t.card4 ?? "—"],
        ["Purchaser email", t.P_emaildomain ?? "—"], ["Device", t.DeviceType ?? "—"],
      ].map(([k, v]) => el("div", {}, el("dt", {}, k), el("dd", {}, v))))),
    panel("Risk", `Model ${info.name} v${info.version} · log scale`,
      riskBar(r.probability, info.thresholds.elevated, info.thresholds.highest),
      el("div", { class: `decision lvl-${lvl}` },
        el("span", { class: "decision-word" }, word), el("p", {}, why))),
    panel("Risk insights", "Top 3 SHAP contributions (log-odds). Positive values push toward fraud.",
      el("div", { class: "legend", style: "margin-bottom:16px" },
        el("span", {}, el("i", { style: "background:var(--text)" }), "raises risk"),
        el("span", {}, el("i", { style: "background:var(--faint)" }), "lowers risk")),
      el("div", { class: "reasons" }, r.reasons.map(({ feature, shap }) => {
        const [name, note] = describe(feature, t);
        const bar = el("div", { class: `bar ${shap >= 0 ? "up" : "down"}` });
        bar.style.width = `${(Math.abs(shap) / max) * 50}%`;
        return el("div", { class: "reason" },
          el("div", {}, el("div", { class: "reason-name" }, name),
            el("div", { class: "reason-note" }, el("span", { class: "mono" }, feature), ` · ${note}`)),
          el("div", { class: "track", role: "img", "aria-label": `${shap >= 0 ? "raises" : "lowers"} risk by ${fmt.shap(shap)}` }, bar),
          el("div", { class: "reason-val num" }, fmt.shap(shap)));
      })),
      el("p", { class: "note" },
        "Names marked masked or anonymized are columns the data provider (Vesta, IEEE-CIS) never explained; the labels describe only what is publicly known.")),
    panel("Ground truth", "Hidden while you review, as it would be in production (labels arrive weeks later).", reveal));
}

async function pagePolicy() {
  const c = await results("threshold_curve.json");
  const pts = c.points;
  const iE = pts.findIndex((p) => p.threshold === c.elevated);
  const iH = pts.findIndex((p) => p.threshold === c.highest);
  const tiles = el("div", { class: "tiles" }); // the slider's aria-valuetext announces changes
  const value = el("div", { class: "slider-value" });
  const tag = el("span", {});
  const slider = el("input", {
    type: "range", min: 0, max: pts.length - 1, step: 1, value: iE, id: "thr",
    "aria-label": "Decision threshold",
  });
  const fa = (p) => [p.false_alarm_rate, p.recall];
  const chartOpts = (i) => ({
    label: "Fraud caught versus false-alarm rate across thresholds on validation",
    height: 300,
    x: { domain: [0, Math.ceil(pts[0].false_alarm_rate * 10) / 10], ticks: [0, 0.1, 0.2, 0.3, 0.4, 0.5].filter((v) => v <= pts[0].false_alarm_rate + 0.05), fmt: fmt.pct0, label: "false-alarm rate (share of legitimate transactions flagged)" },
    y: { domain: [0, 1], ticks: [0, 0.25, 0.5, 0.75, 1], fmt: fmt.pct0, label: "fraud caught (recall)" },
    series: [{ points: pts.map(fa) }],
    markers: [
      { x: pts[iE].false_alarm_rate, y: pts[iE].recall, cls: "lvl-elevated", label: "Elevated", dy: 18 },
      { x: pts[iH].false_alarm_rate, y: pts[iH].recall, cls: "lvl-highest", label: "Highest", dy: 18 },
      { x: pts[i].false_alarm_rate, y: pts[i].recall, cls: "current", label: "Selected", dy: -12, dx: -10, anchor: "end" },
    ],
    tip: (j) => [`Threshold ${fmt.p(pts[j].threshold)}`,
      `Caught ${fmt.int(pts[j].caught_n)} (${fmt.pct(pts[j].recall)})`,
      `False alarms ${fmt.int(pts[j].false_alarms)} (${fmt.pct(pts[j].false_alarm_rate)})`],
  });
  const chart = lineChart(chartOpts(iE));
  const update = () => {
    const i = Number(slider.value), p = pts[i];
    value.textContent = fmt.p(p.threshold);
    tag.replaceChildren(i === iE ? pill("elevated") : i === iH ? pill("highest") : "");
    slider.setAttribute("aria-valuetext", `threshold ${fmt.p(p.threshold)}, ${fmt.pct(p.recall)} of fraud caught, ${fmt.int(p.false_alarms)} false alarms`);
    tiles.replaceChildren(
      tile("Fraud caught", num(fmt.int(p.caught_n)), `${fmt.pct(p.recall)} of ${fmt.int(c.fraud_n)} · amount ${fmt.amt(p.fraud_caught)}`),
      tile("Fraud missed", num(fmt.int(p.missed_n)), `${fmt.pct(1 - p.recall)} · amount ${fmt.amt(p.fraud_missed)}`),
      tile("False alarms", num(fmt.int(p.false_alarms)), `precision ${fmt.pct(p.precision)}`),
      tile("False-alarm rate", num(fmt.pct(p.false_alarm_rate)), `of ${fmt.int(c.legit_n)} legitimate`));
    chart.set(chartOpts(i));
  };
  slider.addEventListener("input", update);
  update();

  return el("div", {},
    head("Policy", `Pick a threshold on the ${fmt.int(c.fraud_n + c.legit_n)} validation transactions. The test set is never used here; both risk levels were chosen on validation.`),
    panel("Threshold", "Transactions with probability ≥ threshold are flagged.",
      el("div", { class: "row", style: "justify-content:space-between" },
        el("div", { class: "row", style: "gap:14px" }, el("label", { for: "thr", class: "dim" }, "Threshold"), value, tag),
        el("div", { class: "row" },
          el("button", { type: "button", onclick: () => { slider.value = iE; slider.dispatchEvent(new Event("input")); } }, `Elevated · ${fmt.p(c.elevated)}`),
          el("button", { type: "button", onclick: () => { slider.value = iH; slider.dispatchEvent(new Event("input")); } }, `Highest · ${fmt.p(c.highest)}`))),
      slider,
      el("div", { class: "row dim", style: "justify-content:space-between;font-size:12px" },
        el("span", {}, "← flag more"), el("span", {}, "flag fewer →"))),
    el("div", { class: "section-label" }, "At this threshold (validation)"),
    tiles,
    panel("Caught vs false alarms", "Each point is one threshold. Hover or focus the chart and use ← → to read values.", chart),
    el("p", { class: "note" }, `Amounts are in dataset currency. Elevated is the threshold that minimises missed fraud amount + ${c.cost_per_false_alarm} per false alarm; Highest is the lowest threshold with a false-alarm rate ≤ 1%.`));
}

async function pageModel() {
  const [ft, pr, th] = await Promise.all([results("final_test.json"), results("pr_curve.json"), results("threshold.json")]);
  const m = ft.metrics, a = ft.at_threshold, op = pr.operating_points;
  return el("div", {},
    head("Model", null, el("div", { class: "meta" },
      el("span", {}, "Model ", el("b", { class: "num" }, ft.model)),
      el("span", {}, "Test rows ", el("b", { class: "num" }, fmt.int(ft.test_rows))))),
    el("div", { class: "tiles" },
      tile("PR-AUC", num(fmt.m4(m.pr_auc)), "primary metric · test"),
      tile("ROC-AUC", num(fmt.m4(m.roc_auc)), "test"),
      tile("Recall @ 1% FPR", num(fmt.m4(m.recall_at_1pct_fpr)), "test"),
      tile("Recall @ 5% FPR", num(fmt.m4(m.recall_at_5pct_fpr)), "test")),
    el("div", { class: "section-label" }, `At the Elevated threshold ${fmt.p(a.threshold)} (test)`),
    el("div", { class: "tiles" },
      tile("Fraud amount caught", num(fmt.amt(a.fraud_caught)), `recall ${fmt.m4(a.recall)}`),
      tile("Fraud amount missed", num(fmt.amt(a.fraud_missed))),
      tile("False alarms", num(fmt.int(a.false_alarms)), `${fmt.pct(a.false_alarm_rate)} of ${fmt.int(ft.legit_rows)} legitimate`),
      tile("Net saving", num(fmt.amt(a.net_saving)), `caught − ${ft.cost_per_false_alarm} per false alarm`)),
    el("div", { class: "grid", style: "margin-top:16px" },
      panel("Precision–recall curve", `test · PR-AUC ${fmt.m4(pr.pr_auc)}`,
        lineChart({
          label: "Precision-recall curve on the test set with Elevated and Highest operating points",
          height: 320,
          x: { domain: [0, 1], ticks: [0, 0.2, 0.4, 0.6, 0.8, 1], fmt: fmt.m2, label: "recall" },
          y: { domain: [0, 1], ticks: [0, 0.25, 0.5, 0.75, 1], fmt: fmt.m2, label: "precision" },
          series: [{ points: pr.points }],
          hlines: [{ y: pr.base_rate, label: `base rate ${fmt.m4(pr.base_rate)}`, anchor: "start" }],
          markers: [
            { x: op.elevated.recall, y: op.elevated.precision, cls: "lvl-elevated", label: "Elevated", dy: -14 },
            { x: op.highest.recall, y: op.highest.precision, cls: "lvl-highest", label: "Highest" },
          ],
          tip: (i) => ["Test", `Recall ${fmt.m4(pr.points[i][0])}`, `Precision ${fmt.m4(pr.points[i][1])}`],
        })),
      panel("Operating points", "Thresholds chosen on validation, applied once to test.",
        table([
          { label: "Level", get: ([l]) => pill(l) },
          { label: "Threshold", num: true, get: ([, o]) => fmt.p(o.threshold) },
          { label: "Precision", num: true, get: ([, o]) => fmt.m4(o.precision) },
          { label: "Recall", num: true, get: ([, o]) => fmt.m4(o.recall) },
          { label: "FA rate", num: true, get: ([, o]) => fmt.pct(o.false_alarm_rate) },
        ], [["elevated", op.elevated], ["highest", op.highest]]),
        el("p", { class: "note" },
          `Highest was set to a 1% false-alarm rate on validation (${fmt.pct(th.highest.false_alarm_rate)} there); on test the same threshold gives ${fmt.pct(op.highest.false_alarm_rate)}. It is not re-tuned on test.`),
        el("p", { class: "note" }, "The test set was scored once, in Phase 4. Every number on this page comes from that single run."))));
}

async function pageMonitoring() {
  const mo = await results("monitoring.json");
  const w = mo.weeks, s = mo.settings;
  const noLabels = w.filter((x) => x.labels_arrived_for_weeks.length === 0).map((x) => x.week);
  const firstTest = w.find((x) => x.split === "test")?.week;
  const days = (x) => `days ${x.day_start.toFixed(0)}–${x.day_end.toFixed(0)}`;
  const base = (label, height = 170) => ({
    label, height, margin: { t: 14, b: 30 },
    x: { domain: [-0.5, w.length - 0.5], ticks: w.map((x) => x.week), fmt: String },
    bands: noLabels.length ? [{ x0: noLabels[0] - 0.5, x1: noLabels.at(-1) + 0.5 }] : [],
    vlines: firstTest != null ? [{ x: firstTest - 0.5, label: "test →" }] : [],
  });
  // multiples of step inside [a, b]
  const ticks = ([a, b], step) => {
    const out = [];
    for (let t = Math.ceil(a / step - 1e-9) * step; t <= b + 1e-9; t += step) out.push(+t.toFixed(6));
    return out;
  };
  const prv = w.flatMap((x) => [x.static.pr_auc, x.retrain.pr_auc]);
  // 0.05 of headroom below the lowest week so it never sits on the axis
  const prd = [Math.floor((Math.min(...prv) - 0.05) / 0.05) * 0.05, Math.ceil(Math.max(...prv) / 0.1) * 0.1];
  const frd = [0, Math.max(0.06, Math.ceil(Math.max(...w.map((x) => x.fraud_rate)) * 100) / 100)];
  const drd = [0, Math.max(0.4, Math.ceil(Math.max(...w.map((x) => x.drift_share)) * 10) / 10)];
  const wk = (x) => `Week ${x.week} · ${x.split} · ${days(x)}`;
  const first = base("Weekly PR-AUC, static model versus retrained model");
  first.margin = { t: 30, b: 30 };
  first.height = 200;
  if (first.bands.length) first.bands[0].label = "No labels yet · retraining not possible";

  return el("div", {},
    head("Monitoring",
      `Validation and test replayed as ${w.length} weekly production batches. Labels arrive ${s.label_delay_days} days after a transaction; drift is checked on the top ${s.drift_features} SHAP features.`,
      el("div", { class: "meta" },
        el("span", {}, "Alarm weeks ", el("b", { class: "num" }, String(w.filter((x) => x.alarm).length))),
        el("span", {}, "Static model ", el("b", { class: "num" }, mo.static_model)))),
    panel("Weekly replay", "Shaded: weeks before any labels arrived, so no retraining was possible.",
      el("div", { class: "legend", style: "margin-bottom:6px" },
        el("span", {}, el("i", { style: "background:var(--text)" }), "retrain (live champion)"),
        el("span", {}, el("i", { style: "background:var(--normal)" }), "static model (dashed)")),
      el("div", { class: "multiples" },
        el("div", { class: "multiple" }, el("h3", {}, "PR-AUC"),
          lineChart({
            ...first,
            y: { domain: prd, ticks: ticks(prd, 0.05), fmt: fmt.m2 },
            series: [
              { points: w.map((x) => [x.week, x.retrain.pr_auc]), dots: true },
              { points: w.map((x) => [x.week, x.static.pr_auc]), cls: "secondary", dots: true },
            ],
            tip: (i) => [wk(w[i]), `Retrain ${fmt.m4(w[i].retrain.pr_auc)}`, `Static ${fmt.m4(w[i].static.pr_auc)}`, `Live model v${w[i].live_champion_version}`],
          })),
        el("div", { class: "multiple" }, el("h3", {}, "Drift share (share of monitored features drifting)"),
          lineChart({
            ...base("Weekly drift share with the alarm line"),
            y: { domain: drd, ticks: ticks(drd, 0.1), fmt: fmt.m2 },
            hlines: [{ y: s.drift_share_alarm, label: `alarm ≥ ${fmt.m2(s.drift_share_alarm)}` }],
            series: [{ points: w.map((x) => [x.week, x.drift_share]), dots: true }],
            tip: (i) => [wk(w[i]), `Drift share ${fmt.m2(w[i].drift_share)}`, `Score PSI ${fmt.m3(w[i].score_psi)}`, `Alarm ${w[i].alarm ? "yes" : "no"}`],
          })),
        el("div", { class: "multiple" }, el("h3", {}, "Fraud rate"),
          lineChart({
            ...base("Weekly fraud rate"),
            y: { domain: frd, ticks: ticks(frd, 0.02), fmt: fmt.pct0 },
            series: [{ points: w.map((x) => [x.week, x.fraud_rate]), dots: true }],
            tip: (i) => [wk(w[i]), `Fraud rate ${fmt.pct(w[i].fraud_rate)}`, `Rows ${fmt.int(w[i].rows)}`],
          }))),
      el("p", { class: "note" }, "Hover or focus a chart and use ← → to read weekly values.")),
    panel("Retrain decisions", `Scheduled every ${s.retrain_every_weeks} weeks; a challenger is promoted only if it beats the champion PR-AUC by ≥ ${s.min_promotion_gain} on the newest labelled ${s.eval_window_days} days.`,
      table([
        { label: "End of week", num: true, get: (d) => String(d.week) },
        { label: "Trigger", get: (d) => d.reason },
        { label: "Decision", get: (d) => d.decision },
        { label: "Champion PR-AUC", num: true, get: (d) => (d.champion ? `v${d.champion_version} ${fmt.m4(d.champion.pr_auc)}` : `v${d.champion_version}`) },
        { label: "Challenger PR-AUC", num: true, get: (d) => (d.challenger ? `v${d.challenger_version} ${fmt.m4(d.challenger.pr_auc)}` : "—") },
        { label: "Why", get: (d) => d.why ?? (d.eval_days ? `eval days ${d.eval_days.map((v) => v.toFixed(0)).join("–")}` : "") },
      ], mo.retrain_decisions)),
    panel("Weekly table", null,
      table([
        { label: "Week", num: true, get: (x) => String(x.week) },
        { label: "Split", get: (x) => x.split },
        { label: "Days", num: true, get: (x) => `${x.day_start.toFixed(0)}–${x.day_end.toFixed(0)}` },
        { label: "Rows", num: true, get: (x) => fmt.int(x.rows) },
        { label: "Fraud rate", num: true, get: (x) => fmt.m4(x.fraud_rate) },
        { label: "Drift share", num: true, get: (x) => fmt.m2(x.drift_share) },
        { label: "Score PSI", num: true, get: (x) => fmt.m3(x.score_psi) },
        { label: "Alarm", get: (x) => (x.alarm ? "yes" : "no") },
        { label: "Labels at", num: true, get: (x) => (x.labels_known_at_week ?? "after replay") },
        { label: "PR-AUC static", num: true, get: (x) => fmt.m4(x.static.pr_auc) },
        { label: "PR-AUC retrain", num: true, get: (x) => fmt.m4(x.retrain.pr_auc) },
        { label: "Live model", num: true, get: (x) => `v${x.live_champion_version}` },
      ], w)));
}

/* ---------- router ---------- */
const ROUTES = { queue: pageQueue, tx: pageTx, policy: pagePolicy, model: pageModel, monitoring: pageMonitoring };
const TITLES = { queue: "Review queue", tx: "Transaction", policy: "Policy", model: "Model", monitoring: "Monitoring" };
let current = 0, firstLoad = true;

async function route() {
  const [, name0, arg] = location.hash.split("/");
  const name = ROUTES[name0] ? name0 : "queue";
  const token = ++current;
  const main = document.getElementById("main");
  for (const a of document.querySelectorAll("[data-nav]")) {
    if (a.dataset.nav === (name === "tx" ? "queue" : name)) a.setAttribute("aria-current", "page");
    else a.removeAttribute("aria-current");
  }
  document.title = `${TITLES[name]} · FraudGuard`;
  disposeCharts();
  main.replaceChildren(el("p", { class: "loading", role: "status" }, "Loading…"));
  let view;
  try {
    view = await ROUTES[name](arg && decodeURIComponent(arg));
  } catch (e) {
    view = el("div", { class: "error", role: "alert" }, el("strong", {}, "Could not load this page"), e.message);
  }
  if (token !== current) return;
  view.classList.add("view");
  main.replaceChildren(view);
  window.scrollTo(0, 0);
  if (!firstLoad) (main.querySelector("h1") ?? main).focus({ preventScroll: true });
  firstLoad = false;
}

addEventListener("hashchange", route);
route();
