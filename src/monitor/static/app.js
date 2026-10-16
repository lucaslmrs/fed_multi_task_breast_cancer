"use strict";
// Training monitor front-end (plano E2). Read-only: everything comes from /api/*.

const COLORS = ["#08519C", "#9F2B2C", "#6F6357", "#3182BD", "#584A47", "#6BAED6", "#CABD91",
  "#9ECAE1", "#2F7D4F", "#B7791F"];
const PRIMARY = { seg: "dice_positive", cls: "balanced_accuracy" };
const SEVERITY_ORDER = { critical: 0, high: 1, info: 2 };

const state = {
  config: { refresh_seconds: 10 },
  catalog: { runs: [], studies: [] },
  selected: null,       // path of the open run
  compare: new Set(),   // paths ticked for comparison
  view: "empty",
  timer: null,
  curves: null,
  controls: {},         // remembered selections per section
  diagnostics: null,
  openGroups: new Set(),
};

// ---- helpers --------------------------------------------------------------------------------

const $ = (selector, root = document) => root.querySelector(selector);
function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (value === undefined || value === null || value === false) continue;
    if (key === "class") node.className = value;
    else if (key.startsWith("on")) node.addEventListener(key.slice(2), value);
    else if (key === "html") node.innerHTML = value;
    else node.setAttribute(key, value === true ? "" : value);
  }
  for (const child of children.flat()) {
    if (child === null || child === undefined || child === false) continue;
    node.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return node;
}
async function api(path, options) {
  const response = await fetch(path, options);
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error || response.statusText);
  return payload;
}
const q = (path) => encodeURIComponent(path);
function fmt(value, digits = 4) {
  if (value === null || value === undefined || value === "") return "—";
  if (typeof value !== "number") return String(value);
  if (Number.isInteger(value)) return String(value);
  return Math.abs(value) >= 1000 ? value.toFixed(0) : value.toFixed(digits);
}
function pill(text, cls) { return el("span", { class: `pill ${cls || text}` }, text); }
function css(name) { return getComputedStyle(document.documentElement).getPropertyValue(name).trim(); }
function plotReady() { return typeof window.Plotly !== "undefined"; }
function shortName(entry) {
  return entry.arm_id || entry.name;
}

function plot(node, traces, layout = {}) {
  if (!plotReady()) {
    node.replaceChildren(el("p", { class: "note" }, "Gráfico indisponível (Plotly não carregou)."));
    return;
  }
  const base = {
    paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
    font: { color: css("--text"), size: 12 },
    margin: { l: 52, r: 12, t: 34, b: 30 },
    legend: { orientation: "h", x: 0, y: -0.18, yanchor: "top", font: { size: 11 } },
    height: 340,
    xaxis: { gridcolor: css("--grid"), zeroline: false, ...(layout.xaxis || {}) },
    yaxis: { gridcolor: css("--grid"), zeroline: false, ...(layout.yaxis || {}) },
    hovermode: "x unified",
  };
  window.Plotly.react(node, traces, { ...base, ...layout, xaxis: base.xaxis, yaxis: base.yaxis },
    { responsive: true, displaylogo: false });
}

function table(columns, rows, opts = {}) {
  const head = el("tr", {}, columns.map((c) => el("th", {}, c.label)));
  const body = rows.map((row) => el("tr", {}, columns.map((c) => {
    const value = c.value ? c.value(row) : row[c.key];
    const cls = [c.num ? "num" : "", c.cls ? c.cls(row) : ""].join(" ").trim();
    return el("td", { class: cls || undefined }, value instanceof Node ? value : (c.num ? fmt(value) : (value ?? "—")));
  })));
  return el("div", { class: "table-wrap", style: opts.maxHeight ? `max-height:${opts.maxHeight}px` : undefined },
    el("table", {}, el("thead", {}, head), el("tbody", {}, body)));
}

function select(options, value, onChange, label) {
  const node = el("select", { onchange: (event) => onChange(event.target.value) },
    options.map((o) => {
      const [val, text] = Array.isArray(o) ? o : [o, o];
      return el("option", { value: val, selected: String(val) === String(value) }, text);
    }));
  return label ? el("label", {}, `${label} `, node) : node;
}

function card(title, ...children) {
  return el("section", { class: "card" }, el("h2", {}, title), ...children);
}
function unavailable(payload) {
  return el("p", { class: "note" }, `Indisponível: ${payload.reason || "sem dados"}`);
}

// ---- catalog / sidebar ----------------------------------------------------------------------

async function loadCatalog() {
  try {
    state.catalog = await api("/api/runs");
  } catch (error) {
    $("#run-tree").replaceChildren(el("p", { class: "banner critical" }, String(error)));
    return;
  }
  renderTree();
}

function matches(entry, text) {
  if (!text) return true;
  const hay = [entry.name, entry.path, entry.arm_id, entry.study_id, entry.kind, entry.status,
    ...(entry.datasets || [])].join(" ").toLowerCase();
  return text.split(/\s+/).every((t) => hay.includes(t));
}

function runItem(entry) {
  const pending = entry.status === "pending";
  const checkbox = el("input", {
    type: "checkbox", disabled: pending, checked: state.compare.has(entry.path),
    title: "Marcar para comparar",
    onchange: (event) => {
      if (event.target.checked) state.compare.add(entry.path); else state.compare.delete(entry.path);
      updateCompareButton();
    },
  });
  const diag = entry.diagnostics && entry.diagnostics.status;
  return el("div", { class: `run-item${state.selected === entry.path ? " selected" : ""}`, title: entry.path || "planned, not started" },
    checkbox,
    el("button", { class: "name", disabled: pending, onclick: () => openRun(entry.path) },
      shortName(entry)),
    entry.kind && entry.kind !== "federated" ? el("span", { class: "muted", style: "font-size:11px" }, entry.kind) : null,
    entry.status !== "complete" ? pill(entry.status) : null,
    diag ? pill(diag) : null);
}

function group(key, title, children, extra) {
  const details = el("details", {
    class: "group", open: state.openGroups.has(key) || undefined,
    ontoggle: (event) => {
      if (event.target !== details) return;
      if (details.open) state.openGroups.add(key); else state.openGroups.delete(key);
    },
  }, el("summary", {}, title), extra || null, children);
  return details;
}

function reportLinks(reports) {
  if (!reports || !reports.length) return null;
  return el("div", { class: "group-links" }, reports.map((r) =>
    el("a", { href: `/files/${r}`, target: "_blank", rel: "noopener" }, r.split("/").pop()), " "));
}

function renderTree() {
  const text = $("#filter").value.trim().toLowerCase();
  const runs = state.catalog.runs.filter((e) => matches(e, text));
  const tree = [];
  for (const study of state.catalog.studies) {
    const arms = runs.filter((e) => e.group === "study" && e.study_id === study.study_id);
    if (!arms.length) continue;
    const seeds = [...new Set(arms.map((a) => a.seed))].sort();
    const done = arms.filter((a) => a.status === "complete").length;
    const live = arms.some((a) => a.status === "running");
    const title = el("span", {}, `${study.study_id} `, el("span", { class: "muted" }, `${done}/${arms.length}`),
      live ? " " : null, live ? pill("running") : null,
      study.diagnostics && study.diagnostics.status ? " " : null,
      study.diagnostics && study.diagnostics.status ? pill(study.diagnostics.status) : null);
    const children = seeds.length > 1
      ? seeds.map((seed) => group(`${study.study_id}/${seed}`, `seed ${seed}`, arms.filter((a) => a.seed === seed).map(runItem)))
      : arms.map(runItem);
    tree.push(group(`study:${study.study_id}`, title, children, reportLinks(study.reports)));
  }
  const loose = { runs: "Runs avulsos", baseline: "Baselines (B0)", validation: "Validação / smokes" };
  for (const [key, title] of Object.entries(loose)) {
    const entries = runs.filter((e) => e.group === key).sort((a, b) => b.last_activity - a.last_activity);
    if (entries.length) tree.push(group(`loose:${key}`, `${title} (${entries.length})`, entries.map(runItem)));
  }
  if (text && !state.openGroups.has("__filter")) {
    tree.forEach((d) => { d.open = true; d.querySelectorAll("details").forEach((x) => { x.open = true; }); });
  }
  $("#run-tree").replaceChildren(...(tree.length ? tree : [el("p", { class: "muted" }, "Nenhum run encontrado.")]));
}

function updateCompareButton() {
  const button = $("#compare-button");
  button.textContent = `Comparar (${state.compare.size})`;
  button.disabled = state.compare.size < 2;
}

// ---- run view -------------------------------------------------------------------------------

function stopTimer() {
  if (state.timer) clearInterval(state.timer);
  state.timer = null;
  $("#live-indicator").classList.add("hidden");
}

async function openRun(path) {
  stopTimer();
  state.selected = path;
  state.view = "run";
  state.diagnostics = null;
  renderTree();
  const main = $("#main");
  main.replaceChildren(el("p", { class: "muted" }, "Carregando…"));
  let run;
  try {
    run = await api(`/api/run?path=${q(path)}`);
  } catch (error) {
    main.replaceChildren(el("p", { class: "banner critical" }, String(error)));
    return;
  }
  const sections = {
    header: el("div"), final: el("div"), curves: el("div"), aggregation: el("div"),
    conflict: el("div"), resources: el("div"), diagnostics: el("div"),
  };
  main.replaceChildren(...Object.values(sections));
  state.sections = sections;
  renderHeader(run.entry);
  renderFinal(run.final);
  await Promise.all([loadCurves(), loadSection("aggregation", renderAggregation),
    loadSection("conflict", renderConflict), loadSection("resources", renderResources),
    loadSection("diagnostics", (p) => { state.diagnostics = p; renderDiagnostics(); })]);
  if (run.entry.status === "running") startLive();
}

function startLive() {
  $("#live-indicator").classList.remove("hidden");
  const period = Math.max(2, Number(state.config.refresh_seconds) || 10) * 1000;
  state.timer = setInterval(refreshLive, period);
}

async function refreshLive() {
  const path = state.selected;
  if (!path || state.view !== "run") return stopTimer();
  try {
    const run = await api(`/api/run?path=${q(path)}`);
    if (path !== state.selected) return;
    renderHeader(run.entry);
    renderFinal(run.final);
    await loadCurves();
    if (run.entry.status !== "running") {
      // Finished (or went stale): load the end-of-fold artifacts once and stop polling.
      stopTimer();
      await Promise.all([loadSection("aggregation", renderAggregation),
        loadSection("conflict", renderConflict), loadSection("resources", renderResources),
        loadSection("diagnostics", (p) => { state.diagnostics = p; renderDiagnostics(); })]);
      loadCatalog();
    } else {
      loadSection("resources", renderResources);
    }
  } catch (error) {
    console.warn("live refresh failed", error);
  }
}

async function loadSection(name, render) {
  const path = state.selected;
  try {
    const payload = await api(`/api/${name}?path=${q(path)}`);
    if (path === state.selected) render(payload);
  } catch (error) {
    if (path === state.selected) state.sections[name].replaceChildren(card(name, el("p", { class: "banner critical" }, String(error))));
  }
}

function renderHeader(entry) {
  const meta = [
    ["tipo", entry.kind], ["status", entry.status], ["datasets", (entry.datasets || []).join(", ")],
    ["budget", entry.budget_mode || "—"], [entry.kind === "baseline" ? "épocas" : "rodadas", entry.rounds],
    ["folds", (entry.folds || []).join(", ") || "—"],
    ["topologia", Object.entries(entry.topology || {}).map(([d, t]) => `${d}:${t}`).join(", ") || "—"],
  ];
  if (entry.study_id) meta.unshift(["estudo", `${entry.study_id} / seed ${entry.seed} / ${entry.arm_id}`]);
  const updated = new Date(entry.last_activity * 1000).toLocaleString();
  state.sections.header.replaceChildren(el("section", { class: "card" },
    el("div", { class: "run-header" }, el("h2", {}, entry.name), pill(entry.status),
      entry.diagnostics && entry.diagnostics.status ? pill(entry.diagnostics.status) : null),
    el("div", { class: "meta" }, meta.map(([k, v]) => el("span", {}, `${k}: `, el("b", {}, v ?? "—"))),
      el("span", {}, "última atividade: ", el("b", {}, updated))),
    el("div", { class: "meta" }, el("span", {}, `caminho: ${entry.path}`),
      entry.master ? el("span", {}, `master: ${entry.master}`) : null),
    entry.reports && entry.reports.length
      ? el("div", { class: "meta" }, "relatórios estáticos: ", entry.reports.map((r) =>
        el("a", { href: `/files/${r}`, target: "_blank", rel: "noopener" }, r.slice(entry.path.length + 1))))
      : null));
}

function renderFinal(final) {
  if (!final.available) {
    state.sections.final.replaceChildren(card("Métricas finais (teste)", unavailable(final)));
    return;
  }
  const summary = table([
    { label: "dataset", key: "dataset" }, { label: "tarefa", key: "task" },
    { label: "métrica", key: "metric" }, { label: "média", key: "mean", num: true },
    { label: "dp", key: "std", num: true }, { label: "n (cliente × fold)", key: "n", num: true },
  ], final.summary);
  const clientColumns = [{ label: "fold", key: "fold" }, { label: "cliente", key: "client_id" },
    { label: "dataset", key: "dataset" }, { label: "tarefa", key: "task" },
    ...final.metrics.map((m) => ({ label: m, key: m, num: true }))];
  const clients = el("details", {}, el("summary", { class: "muted" }, `Por cliente (${final.clients.length} linhas)`),
    table(clientColumns, final.clients, { maxHeight: 360 }));
  state.sections.final.replaceChildren(card("Métricas finais (teste)", summary, clients,
    el("p", { class: "note" }, "Média ± dp entre cliente × fold, que não são réplicas independentes (descritivo).")));
}

// ---- curves ---------------------------------------------------------------------------------

async function loadCurves() {
  const path = state.selected;
  try {
    const payload = await api(`/api/curves?path=${q(path)}`);
    if (path !== state.selected) return;
    state.curves = payload;
    renderCurves();
  } catch (error) {
    state.sections.curves.replaceChildren(card("Curvas", el("p", { class: "banner critical" }, String(error))));
  }
}

function defaultMetric(keys) {
  for (const k of ["val/seg/dice_positive", "val/cls/balanced_accuracy", "val/seg/loss", "val/combined/loss"]) {
    if (keys.includes(k)) return k;
  }
  return keys[0];
}

function renderCurves() {
  const payload = state.curves;
  const container = state.sections.curves;
  if (!payload.available) {
    container.replaceChildren(card("Curvas de treino", unavailable(payload)));
    return;
  }
  const keys = new Set();
  const folds = new Set();
  for (const s of payload.series) {
    folds.add(String(s.fold));
    for (const split of ["val", "train"]) for (const k of Object.keys(s[split])) keys.add(`${split}/${k}`);
  }
  const allKeys = [...keys].sort();
  const c = state.controls.curves || {};
  if (!allKeys.includes(c.metric)) c.metric = defaultMetric(allKeys);
  if (c.fold !== "all" && !folds.has(String(c.fold))) c.fold = folds.size > 1 ? "all" : [...folds][0];
  state.controls.curves = c;
  const [split, ...rest] = c.metric.split("/");
  const metricKey = rest.join("/");
  const datasets = [...new Set(payload.series.map((s) => s.dataset))].sort();
  const charts = el("div", { class: "charts" });
  const controls = el("div", { class: "controls" },
    select(allKeys, c.metric, (v) => { c.metric = v; renderCurves(); }, "métrica"),
    folds.size > 1 ? select([["all", "todos"], ...[...folds].sort().map((f) => [f, f])], c.fold,
      (v) => { c.fold = v; renderCurves(); }, "fold") : null,
    el("span", { class: "muted" }, `fonte: ${payload.source}; eixo: ${payload.unit}`));
  const notes = [];
  if (payload.source === "execution.log") notes.push("Histórico antigo sem curvas por cliente: valores por dataset/tarefa da linha de log do servidor.");
  if (split === "val" && payload.unit === "round") notes.push("Validação = post_aggregation_round (modelo após a agregação); post_local_round é ignorado (A8).");
  if (split === "train" && payload.unit === "round") notes.push("Treino = média dos passos locais da rodada.");
  container.replaceChildren(card("Curvas de treino", controls, charts,
    ...notes.map((n) => el("p", { class: "note" }, n))));
  for (const dataset of datasets) {
    const node = el("div", { class: "chart" });
    charts.append(node);
    const traces = [];
    payload.series.filter((s) => s.dataset === dataset && (c.fold === "all" || String(s.fold) === String(c.fold)))
      .forEach((s, index) => {
        const xy = s[split][metricKey];
        if (!xy) return;
        traces.push({ x: xy.x, y: xy.y, mode: "lines", type: "scatter",
          name: c.fold === "all" && folds.size > 1 ? `${s.client_id} f${s.fold}` : s.client_id,
          line: { color: COLORS[index % COLORS.length], width: 1.6 } });
      });
    if (!traces.length) { node.remove(); continue; }
    plot(node, traces, { title: { text: `${dataset} — ${c.metric}`, font: { size: 13 } },
      xaxis: { title: payload.unit }, yaxis: { title: metricKey.split("/").pop() } });
  }
  if (!charts.children.length) charts.replaceWith(el("p", { class: "note" }, "Nenhuma série com essa métrica."));
}

// ---- aggregation ----------------------------------------------------------------------------

function renderAggregation(payload) {
  const container = state.sections.aggregation;
  if (!payload.available) {
    container.replaceChildren(card("Agregação e cancelamento", unavailable(payload)));
    return;
  }
  const folds = Object.keys(payload.folds).sort();
  const c = state.controls.aggregation || {};
  if (!folds.includes(c.fold)) c.fold = folds[0];
  state.controls.aggregation = c;
  const fold = payload.folds[c.fold];
  const weights = el("div", { class: "chart" });
  const transfer = el("div", { class: "chart" });
  const notes = (payload.pending || []).map((p) => el("p", { class: "note" }, p));
  if (payload.negative_transfer_reason) notes.push(el("p", { class: "note" }, `Cancelamento: ${payload.negative_transfer_reason}.`));
  container.replaceChildren(card("Agregação e cancelamento",
    el("div", { class: "controls" }, folds.length > 1 ? select(folds, c.fold, (v) => { c.fold = v; renderAggregation(payload); }, "fold") : null),
    el("div", { class: "charts" }, weights, transfer), ...notes,
    el("p", { class: "note" }, "Leia o cancelamento 'overall' contra 'orthogonal_reference': só o excesso sobre a referência indica conflito.")));
  const clients = Object.keys(fold.weights).sort();
  plot(weights, clients.map((client, i) => ({
    x: fold.rounds, y: fold.weights[client], name: client, stackgroup: "w", mode: "lines",
    line: { width: 0.5, color: COLORS[i % COLORS.length] },
  })), { title: { text: "Peso final por cliente", font: { size: 13 } }, xaxis: { title: "rodada" }, yaxis: { title: "peso" } });
  if (fold.negative_transfer) {
    const keys = Object.keys(fold.negative_transfer).filter((k) => !k.startsWith("intra_"));
    plot(transfer, keys.map((k, i) => ({
      x: fold.rounds.slice(0, fold.negative_transfer[k].length), y: fold.negative_transfer[k], name: k, mode: "lines",
      line: { color: k === "orthogonal_reference" ? css("--muted") : COLORS[i % COLORS.length], dash: k === "orthogonal_reference" ? "dash" : "solid" },
    })), { title: { text: "Cancelamento 1 − ‖Σwᵢ Δᵢ‖ / Σwᵢ‖Δᵢ‖", font: { size: 13 } }, xaxis: { title: "rodada" } });
  } else {
    transfer.replaceChildren(el("p", { class: "note", style: "padding:10px" }, "Sem negative_transfer neste histórico."));
  }
}

// ---- conflict -------------------------------------------------------------------------------

function heatmap(node, matrix, title) {
  plot(node, [{ type: "heatmap", z: matrix.values, x: matrix.targets || matrix.classes,
    y: matrix.sources || matrix.blocks, colorscale: "RdBu", zmid: 0, reversescale: false,
    hovertemplate: "%{y} → %{x}: %{z:.3g}<extra></extra>" }],
  { title: { text: title, font: { size: 13 } }, hovermode: "closest", margin: { l: 110, r: 12, t: 34, b: 80 } });
}

function renderConflict(payload) {
  const container = state.sections.conflict;
  if (!payload.available) {
    container.replaceChildren(card("Conflito de gradientes (E0)", unavailable(payload)));
    return;
  }
  const charts = el("div", { class: "charts" });
  container.replaceChildren(card("Conflito de gradientes (E0)", charts,
    el("p", { class: "note" }, `M2: ${payload.caveat}.`)));
  if (payload.m1) {
    const node = el("div", { class: "chart" });
    charts.append(node);
    plot(node, Object.entries(payload.m1).map(([cls, xy], i) => ({ x: xy.x, y: xy.y, name: cls, mode: "lines+markers",
      line: { color: COLORS[i % COLORS.length] } })),
    { title: { text: "M1: cosseno médio por classe de par", font: { size: 13 } }, xaxis: { title: "rodada" } });
  }
  if (payload.m1_blocks) {
    const node = el("div", { class: "chart" });
    charts.append(node);
    heatmap(node, payload.m1_blocks, "M1: cosseno médio por bloco × classe de par");
  }
  for (const [split, matrix] of Object.entries(payload.m2 || {})) {
    const node = el("div", { class: "chart" });
    charts.append(node);
    heatmap(node, matrix, `M2 (${split}): Z médio, origem → alvo`);
  }
}

// ---- resources ------------------------------------------------------------------------------

function renderResources(payload) {
  const container = state.sections.resources;
  if (!payload.available) {
    container.replaceChildren(card("Recursos", unavailable(payload)));
    return;
  }
  const charts = el("div", { class: "charts" });
  const parts = [charts];
  for (const [index, gpu] of Object.entries(payload.gpu || {})) {
    const node = el("div", { class: "chart" });
    charts.append(node);
    const x = gpu.elapsed_seconds.map((s) => s / 60);
    plot(node, [
      { x, y: gpu.utilization_gpu_percent, name: "utilização %", mode: "lines", line: { color: COLORS[0], width: 1 } },
      { x, y: gpu.memory_used_mb.map((m, i) => 100 * m / (gpu.memory_total_mb[i] || 1)), name: "memória %", mode: "lines", line: { color: COLORS[1], width: 1 } },
    ], { title: { text: `GPU ${index}`, font: { size: 13 } }, xaxis: { title: "minutos" }, yaxis: { title: "%", range: [0, 100] } });
  }
  if (payload.phases) {
    parts.push(el("h3", {}, "Tempo por fase (runtime_events.csv)"), table([
      { label: "fase", key: "phase" }, { label: "eventos", key: "count", num: true },
      { label: "total (min)", value: (r) => fmt(r.total_seconds / 60, 1), num: true },
    ], payload.phases));
  }
  parts.push(el("p", { class: "note" }, "A memória RSS por ator Ray não é registrada (pendência E0.4 (e))."));
  container.replaceChildren(card("Recursos", ...parts));
}

// ---- diagnostics ----------------------------------------------------------------------------

async function recomputeDiagnostics(button) {
  const path = state.selected;
  button.disabled = true;
  button.textContent = "Calculando…";
  try {
    const payload = await api(`/api/diagnose?path=${q(path)}`, { method: "POST" });
    if (path === state.selected) { state.diagnostics = payload; renderDiagnostics(); }
  } catch (error) {
    button.textContent = `Falhou: ${error}`;
  }
}

function renderDiagnostics() {
  const payload = state.diagnostics;
  const container = state.sections.diagnostics;
  const button = el("button", { onclick: (event) => recomputeDiagnostics(event.target) }, "Recalcular em memória");
  if (!payload || !payload.available) {
    container.replaceChildren(card("Diagnóstico (E1)", el("div", { class: "controls" }, button), unavailable(payload || {})));
    return;
  }
  const f = state.controls.findings || (state.controls.findings = {});
  const findings = payload.findings || [];
  const options = (key) => [["", "todos"], ...[...new Set(findings.map((r) => r[key]).filter((v) => v !== null && v !== ""))].sort().map((v) => [v, v])];
  const shown = findings.filter((r) => ["check_id", "severity", "dataset", "client_id"].every((k) => !f[k] || String(r[k]) === f[k]))
    .sort((a, b) => (SEVERITY_ORDER[a.severity] ?? 9) - (SEVERITY_ORDER[b.severity] ?? 9));
  const rerender = (key) => (value) => { f[key] = value; renderDiagnostics(); };
  const checks = Object.entries(payload.checks || {}).map(([id, c]) => ({ id, ...c }));
  container.replaceChildren(card("Diagnóstico (E1)",
    el("div", { class: "controls" },
      pill(payload.status || "—"),
      el("span", { class: "muted" }, Object.entries(payload.counts || {}).map(([k, v]) => `${k}=${v}`).join("  ")),
      el("span", { class: "muted" }, payload.computed ? "recalculado agora, em memória (nada foi gravado)" : "salvo em diagnostics/"),
      el("span", { class: "spacer" }), button),
    el("details", {}, el("summary", { class: "muted" }, "Estado das checagens D1–D15"),
      table([{ label: "id", key: "id" }, { label: "checagem", key: "name" },
        { label: "estado", value: (r) => pill(r.state === "triggered" ? "triggered" : r.state, r.state === "triggered" ? "high" : r.state === "ok" ? "ok" : "pending") },
        { label: "motivo", key: "reason" }], checks)),
    el("h3", {}, `Achados (${shown.length} de ${findings.length})`),
    el("div", { class: "controls" },
      select(options("check_id"), f.check_id || "", rerender("check_id"), "check"),
      select(options("severity"), f.severity || "", rerender("severity"), "severidade"),
      select(options("dataset"), f.dataset || "", rerender("dataset"), "dataset"),
      select(options("client_id"), f.client_id || "", rerender("client_id"), "cliente")),
    table([
      { label: "sev.", value: (r) => pill(r.severity) }, { label: "id", key: "check_id" },
      { label: "fold", key: "fold" }, { label: "dataset", key: "dataset" }, { label: "cliente", key: "client_id" },
      { label: "tarefa", key: "task" }, { label: "métrica", key: "metric" },
      { label: "valor", key: "value", num: true }, { label: "limiar", key: "threshold" },
      { label: "evidência", key: "evidence" },
    ], shown, { maxHeight: 420 }),
    el("p", { class: "note" }, "Sinalizações descritivas de um run (uma seed); não são testes de significância.")));
}

// ---- compare --------------------------------------------------------------------------------

async function openCompare() {
  stopTimer();
  state.view = "compare";
  state.selected = null;
  renderTree();
  const paths = [...state.compare];
  const main = $("#main");
  main.replaceChildren(el("p", { class: "muted" }, "Carregando comparação…"));
  let payload;
  try {
    payload = await api(`/api/compare?${paths.map((p) => `path=${q(p)}`).join("&")}`);
  } catch (error) {
    main.replaceChildren(el("p", { class: "banner critical" }, String(error)));
    return;
  }
  state.comparePayload = payload;
  renderCompare();
}

function renderCompare() {
  const payload = state.comparePayload;
  const main = $("#main");
  const runs = payload.runs;
  const labels = runs.map((r, i) => `${i === 0 ? "★ " : ""}${shortName(r.entry)}`);
  const header = card("Comparação",
    el("p", { class: "note" }, `Referência (★): ${runs[0].entry.path}. ${payload.note}`),
    ...payload.warnings.map((w) => el("div", { class: `banner ${w.severity}` }, w.message)),
    table([{ label: "", value: (r) => labels[runs.indexOf(r)] }, { label: "caminho", value: (r) => r.entry.path },
      { label: "tipo", value: (r) => r.entry.kind }, { label: "status", value: (r) => pill(r.entry.status) },
      { label: "budget", value: (r) => r.entry.budget_mode || "—" },
      { label: "diag.", value: (r) => r.entry.diagnostics && r.entry.diagnostics.status ? pill(r.entry.diagnostics.status) : "—" }], runs));

  const keys = new Map();
  runs.forEach((run) => run.final.forEach((row) => keys.set(`${row.dataset}|${row.task}|${row.metric}`, row)));
  const rows = [...keys.keys()].sort().map((key) => {
    const [dataset, task, metric] = key.split("|");
    const row = { dataset, task, metric };
    runs.forEach((run, i) => {
      const hit = run.final.find((r) => r.dataset === dataset && r.task === task && r.metric === metric);
      row[`v${i}`] = hit ? `${fmt(hit.mean)} ± ${fmt(hit.std, 3)}` : "—";
      row[`d${i}`] = hit ? hit.delta_vs_reference : null;
    });
    return row;
  });
  const lowerIsBetter = (metric) => metric === "empty_fp_image_rate";
  const columns = [{ label: "dataset", key: "dataset" }, { label: "tarefa", key: "task" }, { label: "métrica", key: "metric" }];
  runs.forEach((run, i) => {
    columns.push({ label: labels[i], key: `v${i}`, num: true });
    if (i > 0) columns.push({ label: "Δ vs ★", key: `d${i}`, num: true, cls: (r) => {
      const d = r[`d${i}`];
      if (d === null || d === undefined || d === 0) return "";
      return (d > 0) !== lowerIsBetter(r.metric) ? "pos" : "neg";
    } });
  });
  const finals = card("Métricas finais lado a lado (média ± dp entre cliente × fold)",
    rows.length ? table(columns, rows) : el("p", { class: "note" }, "Nenhum run selecionado tem métricas finais."));

  const datasets = [...new Set(runs.flatMap((r) => Object.keys(r.curves)))].sort();
  const curveKeys = [...new Set(runs.flatMap((r) => Object.values(r.curves).flatMap((d) => Object.keys(d))))].sort();
  const c = state.controls.compare || (state.controls.compare = {});
  if (!curveKeys.includes(c.metric)) c.metric = defaultMetric(curveKeys);
  const charts = el("div", { class: "charts" });
  const curves = card("Curvas médias por dataset (média entre clientes × folds)",
    el("div", { class: "controls" }, curveKeys.length ? select(curveKeys, c.metric, (v) => { c.metric = v; renderCompare(); }, "métrica") : null),
    charts);
  main.replaceChildren(header, finals, curves);
  for (const dataset of datasets) {
    const traces = runs.map((run, i) => {
      const xy = (run.curves[dataset] || {})[c.metric];
      return xy ? { x: xy.x, y: xy.y, name: labels[i], mode: "lines", line: { color: COLORS[i % COLORS.length], width: 1.8 } } : null;
    }).filter(Boolean);
    if (!traces.length) continue;
    const node = el("div", { class: "chart" });
    charts.append(node);
    plot(node, traces, { title: { text: `${dataset} — ${c.metric}`, font: { size: 13 } }, xaxis: { title: "rodada / época" } });
  }
  if (!charts.children.length) charts.replaceWith(el("p", { class: "note" }, "Sem curvas para essa métrica."));
}

// ---- boot -----------------------------------------------------------------------------------

function initTheme() {
  let theme = null;
  try { theme = localStorage.getItem("monitor-theme"); } catch (error) { /* storage blocked */ }
  if (theme) document.documentElement.dataset.theme = theme;
  $("#theme-button").addEventListener("click", () => {
    const dark = document.documentElement.dataset.theme
      ? document.documentElement.dataset.theme === "dark"
      : matchMedia("(prefers-color-scheme: dark)").matches;
    const next = dark ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    try { localStorage.setItem("monitor-theme", next); } catch (error) { /* storage blocked */ }
    if (state.view === "run" && state.sections) openRun(state.selected);
    if (state.view === "compare") renderCompare();
  });
}

async function boot() {
  initTheme();
  $("#filter").addEventListener("input", renderTree);
  $("#compare-button").addEventListener("click", openCompare);
  try {
    state.config = await api("/api/config");
    $("#root-label").textContent = state.config.root;
  } catch (error) { /* keep defaults */ }
  await loadCatalog();
  setTimeout(() => { if (!plotReady()) $("#plotly-warning").classList.remove("hidden"); }, 4000);
  // The run list itself refreshes slowly, so new and finished runs show up without a reload.
  setInterval(loadCatalog, Math.max(10, Number(state.config.refresh_seconds) || 10) * 3000);
}

boot();
