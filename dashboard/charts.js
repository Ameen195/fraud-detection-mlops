// Hand-drawn SVG charts. Every label goes in through textContent / setAttribute, never innerHTML.
const NS = "http://www.w3.org/2000/svg";
const observers = new Set();

export function svgEl(tag, attrs = {}, text) {
  const e = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) if (v != null) e.setAttribute(k, v);
  if (text != null) e.textContent = text;
  return e;
}

export function scale(kind, [d0, d1], [r0, r1]) {
  const f = kind === "log" ? Math.log10 : (v) => v;
  const a = f(d0), span = f(d1) - a;
  return (v) => r0 + ((f(v) - a) / span) * (r1 - r0);
}

export function showTip(lines, cx, cy) {
  const t = document.getElementById("tip");
  t.replaceChildren(...lines.map((l, i) => {
    const d = document.createElement("div");
    d.textContent = l;
    if (i === 0) d.className = "tip-head";
    return d;
  }));
  t.hidden = false;
  const r = t.getBoundingClientRect();
  let x = cx + 14, y = cy - r.height - 12;
  if (x + r.width > innerWidth - 8) x = cx - r.width - 14;
  if (y < 8) y = cy + 16;
  t.style.transform = `translate(${Math.max(8, x)}px, ${y}px)`;
}

export function hideTip() {
  document.getElementById("tip").hidden = true;
}

/** A div that (re)renders render(width) -> svg whenever its width changes. */
export function responsive(render) {
  const host = document.createElement("div");
  host.className = "chart-host";
  let w = 0;
  const draw = () => host.replaceChildren(render(w));
  const ro = new ResizeObserver(() => {
    const nw = Math.round(host.clientWidth);
    if (nw > 0 && nw !== w) { w = nw; draw(); }
  });
  ro.observe(host);
  observers.add(ro);
  host.update = (r) => { render = r; if (w) draw(); };
  return host;
}

export function disposeCharts() {
  for (const ro of observers) ro.disconnect();
  observers.clear();
  hideTip();
}

/**
 * Line chart. o = {label, height, margin, x: {kind, domain, ticks, fmt, label}, y: {...},
 * series: [{points: [[x, y]], cls, dots}], bands: [{x0, x1, label}], hlines: [{y, label}],
 * vlines: [{x, label}], markers: [{x, y, cls, label, anchor, dx, dy}], tip: (i) => [lines]}.
 * Hover / arrow keys step through the indices of series[0]; all series share those indices.
 */
export function lineChart(o) {
  const host = responsive((W) => draw(o, W));
  host.set = (next) => host.update((W) => draw(next, W));
  return host;
}

function draw(o, W) {
  const H = o.height ?? 220;
  const m = { t: 26, r: 20, b: 36, l: 56, ...o.margin };
  const x = scale(o.x.kind, o.x.domain, [m.l, W - m.r]);
  const y = scale(o.y.kind, o.y.domain, [H - m.b, m.t]);
  const svg = svgEl("svg", {
    class: "chart", width: W, height: H, viewBox: `0 0 ${W} ${H}`,
    role: "img", "aria-label": o.label, tabindex: o.tip ? 0 : null,
  });
  const add = (tag, a, t) => svg.appendChild(svgEl(tag, a, t));

  for (const b of o.bands ?? []) {
    add("rect", { class: "band", x: x(b.x0), y: m.t, width: x(b.x1) - x(b.x0), height: H - m.t - m.b });
    if (b.label) add("text", { class: "band-label", x: x(b.x0) + 8, y: m.t + 16 }, b.label);
  }
  for (const t of o.y.ticks) {
    add("line", { class: "grid", x1: m.l, x2: W - m.r, y1: y(t), y2: y(t) });
    add("text", { class: "tick", x: m.l - 10, y: y(t) + 4, "text-anchor": "end" }, o.y.fmt(t));
  }
  for (const t of o.x.ticks) {
    add("text", { class: "tick", x: x(t), y: H - m.b + 18, "text-anchor": "middle" }, o.x.fmt(t));
  }
  add("line", { class: "axis", x1: m.l, x2: W - m.r, y1: H - m.b, y2: H - m.b });
  if (o.x.label) add("text", { class: "axis-label", x: W - m.r, y: H - 2, "text-anchor": "end" }, o.x.label);
  if (o.y.label) add("text", { class: "axis-label", x: m.l - 10, y: m.t - 12 }, o.y.label);
  for (const v of o.vlines ?? []) {
    add("line", { class: "vref", x1: x(v.x), x2: x(v.x), y1: m.t, y2: H - m.b });
    if (v.label) add("text", { class: "band-label", x: x(v.x) + 6, y: m.t + 16 }, v.label);
  }
  for (const h of o.hlines ?? []) {
    add("line", { class: "ref", x1: m.l, x2: W - m.r, y1: y(h.y), y2: y(h.y) });
    const left = h.anchor === "start";
    if (h.label) {
      add("text", {
        class: "ref-label", x: left ? m.l + 6 : W - m.r, y: y(h.y) - 6, "text-anchor": left ? "start" : "end",
      }, h.label);
    }
  }
  for (const s of o.series) {
    const d = s.points.map(([a, b], i) => `${i ? "L" : "M"}${x(a).toFixed(1)},${y(b).toFixed(1)}`);
    add("path", { class: `line ${s.cls ?? ""}`, d: d.join("") });
    if (s.dots) for (const [a, b] of s.points) add("circle", { class: `dot ${s.cls ?? ""}`, cx: x(a), cy: y(b), r: 2.5 });
  }
  for (const k of o.markers ?? []) {
    add("circle", { class: `marker ${k.cls}`, cx: x(k.x), cy: y(k.y), r: 5.5 });
    if (k.label) {
      add("text", {
        class: `marker-label ${k.cls}`, x: x(k.x) + (k.dx ?? 10), y: y(k.y) + (k.dy ?? -10),
        "text-anchor": k.anchor ?? "start",
      }, k.label);
    }
  }
  if (o.tip) hover(svg, o, x, y, m, W, H);
  return svg;
}

function hover(svg, o, x, y, m, W, H) {
  const pts = o.series[0].points;
  const xs = pts.map((p) => x(p[0]));
  const rule = svg.appendChild(svgEl("line", { class: "hover-rule", y1: m.t, y2: H - m.b, visibility: "hidden" }));
  const dots = o.series.map((s) =>
    svg.appendChild(svgEl("circle", { class: `hover-dot ${s.cls ?? ""}`, r: 4, visibility: "hidden" })));
  let idx = -1;
  const show = (i, cx, cy) => {
    idx = i;
    rule.setAttribute("x1", xs[i]);
    rule.setAttribute("x2", xs[i]);
    rule.setAttribute("visibility", "visible");
    o.series.forEach((s, j) => {
      dots[j].setAttribute("cx", x(s.points[i][0]));
      dots[j].setAttribute("cy", y(s.points[i][1]));
      dots[j].setAttribute("visibility", "visible");
    });
    const r = svg.getBoundingClientRect();
    showTip(o.tip(i), cx ?? r.left + xs[i], cy ?? r.top + y(pts[i][1]));
  };
  const hide = () => {
    idx = -1;
    rule.setAttribute("visibility", "hidden");
    dots.forEach((d) => d.setAttribute("visibility", "hidden"));
    hideTip();
  };
  svg.addEventListener("pointermove", (e) => {
    const px = e.clientX - svg.getBoundingClientRect().left;
    if (px < m.l - 12 || px > W - m.r + 12) return hide();
    let b = 0;
    for (let i = 1; i < xs.length; i++) if (Math.abs(xs[i] - px) < Math.abs(xs[b] - px)) b = i;
    show(b, e.clientX, e.clientY);
  });
  svg.addEventListener("pointerleave", hide);
  svg.addEventListener("blur", hide);
  svg.addEventListener("keydown", (e) => {
    const n = xs.length, step = e.shiftKey ? 10 : 1;
    const next = { ArrowRight: idx + step, ArrowLeft: idx - step, Home: 0, End: n - 1 }[e.key];
    if (e.key === "Escape") return hide();
    if (next === undefined) return;
    e.preventDefault();
    show(idx < 0 ? 0 : Math.min(n - 1, Math.max(0, next)));
  });
}
