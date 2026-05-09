"use strict";

const els = {
  historyMeta: document.getElementById("historyMeta"),
  historyType: document.getElementById("historyType"),
  historyNamespace: document.getElementById("historyNamespace"),
  historyLimit: document.getElementById("historyLimit"),
  historyRefreshBtn: document.getElementById("historyRefreshBtn"),
  historyCards: document.getElementById("historyCards"),
  historyCharts: document.getElementById("historyCharts"),
  historyTableBody: document.getElementById("historyTableBody"),
  compareA: document.getElementById("compareA"),
  compareB: document.getElementById("compareB"),
  compareSummary: document.getElementById("compareSummary")
};

const CHART_PALETTE = ["#5db2ff", "#24c38e", "#f0b429", "#f05c7a", "#9a7cff"];
let historyRows = [];

function asArray(value) {
  return Array.isArray(value) ? value : [];
}

function asObject(value) {
  return value && typeof value === "object" && !Array.isArray(value) ? value : {};
}

function escapeHtml(value) {
  return String(value == null ? "" : value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}

function formatNumber(value, digits = 2) {
  const n = Number(value);
  if (!Number.isFinite(n)) {
    return "-";
  }
  return n.toFixed(digits);
}

function buildLineChartSvg(series, { width = 920, height = 210 } = {}) {
  const rows = asArray(series)
    .filter((row) => row && row.name && asArray(row.values).length)
    .map((row) => ({
      name: String(row.name),
      values: asArray(row.values).map((value) => Number(value)).map((n) => (Number.isFinite(n) ? n : 0)),
      color: row.color || null
    }));
  if (!rows.length) {
    return `<div class="muted">No chart data.</div>`;
  }

  const maxLen = rows.reduce((max, row) => Math.max(max, row.values.length), 0);
  const allValues = rows.flatMap((row) => row.values);
  const yMin = Math.min(0, ...allValues);
  const yMax = Math.max(1, ...allValues);
  const span = Math.max(1e-9, yMax - yMin);
  const left = 42;
  const right = width - 14;
  const top = 12;
  const bottom = height - 28;
  const plotW = Math.max(10, right - left);
  const plotH = Math.max(10, bottom - top);
  const xForIndex = (idx) => (maxLen <= 1 ? left + plotW / 2 : left + (idx / (maxLen - 1)) * plotW);
  const yForValue = (value) => bottom - ((value - yMin) / span) * plotH;
  const pointsFor = (values) =>
    values
      .map((value, idx) => `${xForIndex(idx).toFixed(2)},${yForValue(value).toFixed(2)}`)
      .join(" ");
  const yTicks = [0, 0.25, 0.5, 0.75, 1].map((r) => ({
    label: formatNumber(yMin + span * (1 - r), Math.abs(yMin + span * (1 - r)) < 10 ? 2 : 1),
    y: top + plotH * r
  }));

  return `
    <svg class="timeseries-svg" viewBox="0 0 ${width} ${height}" preserveAspectRatio="none">
      <rect x="0" y="0" width="${width}" height="${height}" fill="#0d1430"></rect>
      ${yTicks
        .map(
          (tick) => `
          <line x1="${left}" y1="${tick.y.toFixed(2)}" x2="${right}" y2="${tick.y.toFixed(2)}" stroke="#223058" stroke-width="1" />
          <text x="${left - 6}" y="${(tick.y + 4).toFixed(2)}" text-anchor="end" fill="#93a0c7" font-size="10">${tick.label}</text>
        `
        )
        .join("")}
      ${rows
        .map((row, idx) => {
          const color = row.color || CHART_PALETTE[idx % CHART_PALETTE.length];
          const points = pointsFor(row.values);
          const markers = row.values
            .map((value, valueIdx) => {
              const x = xForIndex(valueIdx).toFixed(2);
              const y = yForValue(value).toFixed(2);
              return `<circle cx="${x}" cy="${y}" r="3" fill="${color}" stroke="#d9e2ff" stroke-width="0.6" />`;
            })
            .join("");
          return `<polyline fill="none" stroke="${color}" stroke-width="2" points="${points}" />${markers}`;
        })
        .join("")}
    </svg>
    <div class="chart-legend">
      ${rows
        .map((row, idx) => {
          const color = row.color || CHART_PALETTE[idx % CHART_PALETTE.length];
          return `<span class="chart-legend-item"><span class="chart-legend-swatch" style="background:${color}"></span>${escapeHtml(
            row.name
          )}</span>`;
        })
        .join("")}
    </div>
  `;
}

function renderCards() {
  const total = historyRows.length;
  const baselineCount = historyRows.filter((row) => asObject(row.profile).baseline).length;
  const moduleCount = total - baselineCount;
  const avgLatency =
    total > 0
      ? historyRows.reduce((sum, row) => sum + Number(asObject(row.metrics).latencyMs || 0), 0) / total
      : 0;
  const avgRatio =
    total > 0
      ? historyRows.reduce((sum, row) => sum + Number(asObject(row.metrics).transferComputeRatio || 0), 0) / total
      : 0;
  const lastTs = total ? new Date(historyRows[0].createdAt).toLocaleString() : "-";
  const cards = [
    { k: "Runs", v: String(total) },
    { k: "Baseline Runs", v: String(baselineCount) },
    { k: "Module Runs", v: String(moduleCount) },
    { k: "Avg Latency (ms)", v: formatNumber(avgLatency, 1) },
    { k: "Avg Comm/Comp", v: formatNumber(avgRatio, 3) },
    { k: "Latest Run", v: lastTs }
  ];
  els.historyCards.innerHTML = cards
    .map(
      (card) => `
      <div class="invoke-metric-card">
        <div class="k">${escapeHtml(card.k)}</div>
        <div class="v">${escapeHtml(card.v)}</div>
      </div>
    `
    )
    .join("");
}

function runLabel(row) {
  const config = asObject(row.config);
  const metrics = asObject(row.metrics);
  return `#${row.id} ${new Date(row.createdAt).toLocaleString()} ${row.type} tok=${metrics.generatedTokens || 0} max=${
    config.maxNewTokens || 0
  }`;
}

function renderCompareOptions() {
  const options = historyRows
    .map((row) => `<option value="${row.id}">${escapeHtml(runLabel(row))}</option>`)
    .join("");
  els.compareA.innerHTML = options;
  els.compareB.innerHTML = options;
  if (historyRows.length >= 2) {
    els.compareA.value = String(historyRows[1].id);
    els.compareB.value = String(historyRows[0].id);
  } else if (historyRows.length === 1) {
    els.compareA.value = String(historyRows[0].id);
    els.compareB.value = String(historyRows[0].id);
  }
}

function getRunById(idValue) {
  return historyRows.find((row) => String(row.id) === String(idValue)) || null;
}

function renderCompareSummary() {
  const runA = getRunById(els.compareA.value);
  const runB = getRunById(els.compareB.value);
  if (!runA || !runB) {
    els.compareSummary.innerHTML = `<div class="muted">Select two runs.</div>`;
    return;
  }

  const mA = asObject(runA.metrics);
  const mB = asObject(runB.metrics);
  const cA = asObject(runA.config);
  const cB = asObject(runB.config);
  const pA = asObject(runA.profile);
  const pB = asObject(runB.profile);

  const deltas = [
    ["Latency (ms)", mB.latencyMs, mA.latencyMs],
    ["Tokens/sec", mB.tokensPerSecond, mA.tokensPerSecond],
    ["Compute (ms)", mB.computeMs, mA.computeMs],
    ["Transfer (ms)", mB.transferMs, mA.transferMs],
    ["Comm/Comp", mB.transferComputeRatio, mA.transferComputeRatio],
    ["Token p95 (ms)", mB.tokenP95Ms, mA.tokenP95Ms],
    ["Transfer p95 (ms)", mB.transferP95Ms, mA.transferP95Ms],
    ["Net (MiB)", mB.networkMib, mA.networkMib],
    ["Payload (MiB)", mB.payloadMib, mA.payloadMib]
  ];

  const deltaRows = deltas
    .map(([name, bVal, aVal]) => {
      const delta = Number(bVal || 0) - Number(aVal || 0);
      return `<tr><td>${escapeHtml(name)}</td><td>${formatNumber(aVal, 2)}</td><td>${formatNumber(
        bVal,
        2
      )}</td><td>${delta >= 0 ? "+" : ""}${formatNumber(delta, 2)}</td></tr>`;
    })
    .join("");

  els.compareSummary.innerHTML = `
    <div class="invoke-subpanel">
      <h3>Config</h3>
      <table class="metrics-table">
        <thead>
          <tr><th>Field</th><th>A</th><th>B</th></tr>
        </thead>
        <tbody>
          <tr><td>Date</td><td>${escapeHtml(new Date(runA.createdAt).toLocaleString())}</td><td>${escapeHtml(
            new Date(runB.createdAt).toLocaleString()
          )}</td></tr>
          <tr><td>Target</td><td>${escapeHtml(`${runA.target} ${runA.path}`)}</td><td>${escapeHtml(
            `${runB.target} ${runB.path}`
          )}</td></tr>
          <tr><td>Prompt Tokens</td><td>${escapeHtml(cA.promptTokens || "-")}</td><td>${escapeHtml(
            cB.promptTokens || "-"
          )}</td></tr>
          <tr><td>Prompt Chars</td><td>${escapeHtml(cA.promptChars || "-")}</td><td>${escapeHtml(
            cB.promptChars || "-"
          )}</td></tr>
          <tr><td>Max New Tokens</td><td>${escapeHtml(cA.maxNewTokens || "-")}</td><td>${escapeHtml(
            cB.maxNewTokens || "-"
          )}</td></tr>
          <tr><td>Min New Tokens</td><td>${escapeHtml(cA.minNewTokens || "-")}</td><td>${escapeHtml(
            cB.minNewTokens || "-"
          )}</td></tr>
          <tr><td>Temperature</td><td>${escapeHtml(cA.temperature || "-")}</td><td>${escapeHtml(
            cB.temperature || "-"
          )}</td></tr>
          <tr><td>Timeout (ms)</td><td>${escapeHtml(cA.timeoutMs || "-")}</td><td>${escapeHtml(
            cB.timeoutMs || "-"
          )}</td></tr>
          <tr><td>Topology Hash</td><td>${escapeHtml(cA.topologyHash || "-")}</td><td>${escapeHtml(
            cB.topologyHash || "-"
          )}</td></tr>
          <tr><td>Profile</td><td>${pA.baseline ? "baseline" : "modules"}</td><td>${pB.baseline ? "baseline" : "modules"}</td></tr>
          <tr><td>Modules</td><td>${escapeHtml(asArray(pA.enabledModules).join(", ") || "-")}</td><td>${escapeHtml(
            asArray(pB.enabledModules).join(", ") || "-"
          )}</td></tr>
        </tbody>
      </table>
    </div>
    <div class="invoke-subpanel">
      <h3>Metric Delta (B - A)</h3>
      <table class="metrics-table">
        <thead>
          <tr><th>Metric</th><th>A</th><th>B</th><th>Delta</th></tr>
        </thead>
        <tbody>${deltaRows}</tbody>
      </table>
    </div>
  `;
}

function renderCharts() {
  const rowsAsc = [...historyRows].reverse();
  const latency = rowsAsc.map((row) => Number(asObject(row.metrics).latencyMs || 0));
  const ratio = rowsAsc.map((row) => Number(asObject(row.metrics).transferComputeRatio || 0));
  const p50 = rowsAsc.map((row) => Number(asObject(row.metrics).tokenP50Ms || 0));
  const p95 = rowsAsc.map((row) => Number(asObject(row.metrics).tokenP95Ms || 0));
  const p99 = rowsAsc.map((row) => Number(asObject(row.metrics).tokenP99Ms || 0));
  const transferP95 = rowsAsc.map((row) => Number(asObject(row.metrics).transferP95Ms || 0));
  const tps = rowsAsc.map((row) => Number(asObject(row.metrics).tokensPerSecond || 0));
  const compute = rowsAsc.map((row) => Number(asObject(row.metrics).computeMs || 0));
  const transfer = rowsAsc.map((row) => Number(asObject(row.metrics).transferMs || 0));

  els.historyCharts.innerHTML = `
    <div class="chart-card">
      <h4>Latency Trend (ms)</h4>
      ${buildLineChartSvg([{ name: "Latency", values: latency }])}
    </div>
    <div class="chart-card">
      <h4>Comm/Compute Ratio Trend</h4>
      ${buildLineChartSvg([{ name: "Ratio", values: ratio }])}
    </div>
    <div class="chart-card">
      <h4>Token Latency p50/p95/p99</h4>
      ${buildLineChartSvg([
        { name: "p50", values: p50 },
        { name: "p95", values: p95 },
        { name: "p99", values: p99 }
      ])}
    </div>
    <div class="chart-card">
      <h4>Transfer p95 + Tokens/sec</h4>
      ${buildLineChartSvg([
        { name: "transfer p95", values: transferP95 },
        { name: "tokens/sec", values: tps }
      ])}
    </div>
    <div class="chart-card">
      <h4>Transfer vs Compute</h4>
      ${buildLineChartSvg([
        { name: "transfer", values: transfer },
        { name: "compute", values: compute }
      ])}
    </div>
  `;
}

function renderTable() {
  if (!historyRows.length) {
    els.historyTableBody.innerHTML = "<tr><td colspan='20'>No history rows.</td></tr>";
    return;
  }

  els.historyTableBody.innerHTML = historyRows
    .map((row) => {
      const config = asObject(row.config);
      const metrics = asObject(row.metrics);
      const profile = asObject(row.profile);
      const modules = asArray(profile.enabledModules);
      const profileHtml = profile.baseline
        ? `<span class="history-badge baseline">baseline</span>`
        : `<span class="history-badge modules">${escapeHtml(modules.join(", ") || "modules")}</span>`;
      return `
        <tr>
          <td>${row.id}</td>
          <td>${escapeHtml(new Date(row.createdAt).toLocaleString())}</td>
          <td>${escapeHtml(row.type)}</td>
          <td>${escapeHtml(row.namespace)}</td>
          <td>${escapeHtml(row.target)}</td>
          <td>${escapeHtml(row.path)}</td>
          <td>${formatNumber(config.promptChars, 0)}</td>
          <td>${formatNumber(config.promptTokens, 0)}</td>
          <td>${formatNumber(metrics.generatedTokens, 0)}</td>
          <td>${formatNumber(config.maxNewTokens, 0)}</td>
          <td>${formatNumber(config.minNewTokens, 0)}</td>
          <td>${formatNumber(config.temperature, 2)}</td>
          <td>${formatNumber(config.timeoutMs, 0)}</td>
          <td>${escapeHtml(config.topologyHash || "-")}</td>
          <td>${formatNumber(metrics.latencyMs, 1)}</td>
          <td>${formatNumber(metrics.tokensPerSecond, 3)}</td>
          <td>${formatNumber(metrics.transferComputeRatio, 3)}</td>
          <td>${formatNumber(metrics.maxMemoryMb, 1)}</td>
          <td>${formatNumber(metrics.networkMib, 3)}</td>
          <td>${profileHtml}</td>
        </tr>
      `;
    })
    .join("");
}

async function fetchJson(url) {
  const res = await fetch(url, { cache: "no-store" });
  if (!res.ok) {
    throw new Error(`${res.status} ${res.statusText}`);
  }
  return res.json();
}

async function refreshHistory() {
  const type = String(els.historyType.value || "all");
  const namespace = String(els.historyNamespace.value || "").trim();
  const limit = Math.max(10, Math.min(5000, Number.parseInt(els.historyLimit.value || "300", 10)));
  const query = new URLSearchParams();
  if (type !== "all") {
    query.set("type", type);
  }
  if (namespace) {
    query.set("namespace", namespace);
  }
  query.set("limit", String(limit));

  try {
    const payload = await fetchJson(`/api/history?${query.toString()}`);
    historyRows = asArray(payload.entries).filter((row) => row && typeof row === "object");
    els.historyMeta.textContent = `stored=${payload.totalStored} shown=${payload.returned} updated=${new Date(
      payload.generatedAt
    ).toLocaleString()}`;
    renderCards();
    renderCharts();
    renderTable();
    renderCompareOptions();
    renderCompareSummary();
  } catch (error) {
    els.historyMeta.textContent = `History error: ${error.message}`;
    historyRows = [];
    renderCards();
    renderCharts();
    renderTable();
    renderCompareOptions();
    renderCompareSummary();
  }
}

els.historyRefreshBtn.addEventListener("click", refreshHistory);
els.historyType.addEventListener("change", refreshHistory);
els.historyNamespace.addEventListener("change", refreshHistory);
els.historyLimit.addEventListener("change", refreshHistory);
els.compareA.addEventListener("change", renderCompareSummary);
els.compareB.addEventListener("change", renderCompareSummary);

const initialNs = new URLSearchParams(window.location.search).get("namespace");
if (initialNs) {
  els.historyNamespace.value = initialNs;
}

refreshHistory();
