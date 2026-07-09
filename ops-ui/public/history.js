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
  compareSummary: document.getElementById("compareSummary"),
  historyExportJsonBtn: document.getElementById("historyExportJsonBtn"),
  historyExportCsvBtn: document.getElementById("historyExportCsvBtn"),
  historyDeleteSelectedBtn: document.getElementById("historyDeleteSelectedBtn"),
  historyClearAllBtn: document.getElementById("historyClearAllBtn"),
  historySelectAll: document.getElementById("historySelectAll"),
  historyChartsExportPngBtn: document.getElementById("historyChartsExportPngBtn"),
  customChartMetrics: document.getElementById("customChartMetrics"),
  customChartXAxis: document.getElementById("customChartXAxis"),
  customChartFilter: document.getElementById("customChartFilter"),
  customChartSortByX: document.getElementById("customChartSortByX"),
  customChartRenderBtn: document.getElementById("customChartRenderBtn"),
  customChartExportPngBtn: document.getElementById("customChartExportPngBtn"),
  customChart: document.getElementById("customChart"),
};

const CHART_PALETTE = ["#5db2ff", "#24c38e", "#f0b429", "#f05c7a", "#9a7cff"];
let historyRows = [];
let selectedHistoryIds = new Set();

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

function describeFeatureVariants(profileValue) {
  const profile = asObject(profileValue);
  const explicit = asArray(profile.moduleVariants)
    .map((item) => String(item || "").trim())
    .filter(Boolean);
  if (explicit.length) {
    return explicit.join(", ");
  }
  const flags = asObject(profile.featureFlags);
  const pairs = [];
  if ("transport_mode" in flags) {
    pairs.push(`transport=${String(flags.transport_mode)}`);
  }
  if ("activation_precision" in flags) {
    pairs.push(`precision=${String(flags.activation_precision)}`);
  }
  if ("rebalance_profile" in flags) {
    pairs.push(`rebalance=${String(flags.rebalance_profile)}`);
  }
  if ("kv_cache_enabled" in flags) {
    pairs.push(`kv_cache=${flags.kv_cache_enabled ? "on" : "off"}`);
  }
  if ("forward_dedupe_enabled" in flags) {
    pairs.push(`dedupe=${flags.forward_dedupe_enabled ? "on" : "off"}`);
  }
  if ("topology_aware_routing" in flags) {
    pairs.push(`topology=${flags.topology_aware_routing ? "on" : "off"}`);
  }
  if ("persistent_sessions_enabled" in flags) {
    pairs.push(`sessions=${flags.persistent_sessions_enabled ? "on" : "off"}`);
  }
  if ("backpressure_enabled" in flags) {
    pairs.push(`backpressure=${flags.backpressure_enabled ? "on" : "off"}`);
  }
  if ("backpressure_queue_size" in flags) {
    pairs.push(`queue=${String(flags.backpressure_queue_size)}`);
  }
  return pairs.join(", ");
}

function getOutputText(rowValue) {
  const output = asObject(asObject(rowValue).output);
  const generated = String(output.generatedText || "").trim();
  const assistant = String(output.assistantMessage || "").trim();
  const preview = String(output.preview || "").trim();
  return generated || assistant || preview || "";
}

function getOutputTermination(rowValue) {
  const output = asObject(asObject(rowValue).output);
  return String(output.terminationReason || "").trim();
}

function formatOutputPreview(text, maxLen = 180) {
  const clean = String(text || "").trim();
  if (!clean) {
    return "-";
  }
  if (clean.length <= maxLen) {
    return clean;
  }
  return `${clean.slice(0, maxLen - 1)}…`;
}

function updateDeleteControls() {
  const selectedCount = selectedHistoryIds.size;

  if (els.historyExportJsonBtn) {
    els.historyExportJsonBtn.disabled = selectedCount === 0;
    els.historyExportJsonBtn.textContent =
      selectedCount > 0 ? `Export selected JSON (${selectedCount})` : "Export selected JSON";
  }

  if (els.historyExportCsvBtn) {
    els.historyExportCsvBtn.disabled = selectedCount === 0;
    els.historyExportCsvBtn.textContent =
      selectedCount > 0 ? `Export selected CSV (${selectedCount})` : "Export selected CSV";
  }

  if (els.historyDeleteSelectedBtn) {
    els.historyDeleteSelectedBtn.disabled = selectedCount === 0;
    els.historyDeleteSelectedBtn.textContent =
      selectedCount > 0
        ? `Delete selected (${selectedCount})`
        : "Delete selected";
  }

  if (els.historySelectAll) {
    const visibleIds = historyRows.map((row) => Number(row.id)).filter(Number.isFinite);
    const allSelected =
      visibleIds.length > 0 && visibleIds.every((id) => selectedHistoryIds.has(id));

    els.historySelectAll.checked = allSelected;
    els.historySelectAll.indeterminate =
      !allSelected && visibleIds.some((id) => selectedHistoryIds.has(id));
  }
}

function selectedHistoryRows() {
  return historyRows.filter((row) => selectedHistoryIds.has(Number(row.id)));
}

// Raw, per-token/per-step traces captured at request time. These already get
// summarized into row.metrics and dashboard.runSummary/cards, so we strip them
// from exports — otherwise a single run drags ~1.5 MB of redundant data along.
const HEAVY_RESPONSE_FIELDS = [
  "token_metrics",
  "steps",
  "prompt_token_ids",
  "generated_token_ids",
];

// Dashboard sub-fields stripped from exports because they are either pure UI
// render state, redundant across rows, or raw per-token/per-sample traces whose
// aggregates already live in runSummary/cards/perStage/perNode:
//   - sessionRunEvolution embeds the whole session history into every entry.
//   - tokenRows is one row per generated token, so a 512-token run alone adds
//     ~150 KB; the p50/p95/p99 + sums in runSummary already summarize it.
//   - metricCatalogRows is a derived min/avg/max catalog recomputable on demand.
const HEAVY_DASHBOARD_FIELDS = [
  "responseJson",
  "sessionRunEvolution",
  "tokenRows",
  "metricCatalogRows",
];

// Compact token-level summary kept in place of the full tokenRows array so the
// export still carries per-run token aggregates without the row-per-token bulk.
function summarizeTokenRowsForExport(tokenRowsValue) {
  const rows = asArray(tokenRowsValue);
  if (!rows.length) {
    return undefined;
  }
  const latencies = rows.map((row) => Number(asObject(row).latencyMs || 0));
  const computeSum = rows.reduce((sum, row) => sum + Number(asObject(row).computeMs || 0), 0);
  const transferSum = rows.reduce((sum, row) => sum + Number(asObject(row).transferMs || 0), 0);
  const sorted = [...latencies].sort((a, b) => a - b);
  const pct = (p) => {
    if (!sorted.length) {
      return 0;
    }
    const rank = p * (sorted.length - 1);
    const lo = Math.floor(rank);
    const hi = Math.min(lo + 1, sorted.length - 1);
    return sorted[lo] + (sorted[hi] - sorted[lo]) * (rank - lo);
  };
  return {
    count: rows.length,
    latencyMsP50: Number(pct(0.5).toFixed(3)),
    latencyMsP95: Number(pct(0.95).toFixed(3)),
    latencyMsP99: Number(pct(0.99).toFixed(3)),
    latencyMsMax: sorted.length ? Number(sorted[sorted.length - 1].toFixed(3)) : 0,
    computeMsSum: Number(computeSum.toFixed(3)),
    transferMsSum: Number(transferSum.toFixed(3)),
  };
}

function slimResponseJson(responseJsonValue) {
  const response = asObject(responseJsonValue);
  if (!Object.keys(response).length) {
    return undefined;
  }
  const slim = { ...response };
  for (const field of HEAVY_RESPONSE_FIELDS) {
    delete slim[field];
  }
  return slim;
}

function sanitizeDashboardForExport(dashboardValue) {
  const dashboard = asObject(dashboardValue);
  if (!Object.keys(dashboard).length) {
    return undefined;
  }
  const slim = { ...dashboard };
  for (const field of HEAVY_DASHBOARD_FIELDS) {
    delete slim[field];
  }
  // Replace the dropped per-token rows with a compact aggregate.
  const tokenSummary = summarizeTokenRowsForExport(dashboard.tokenRows);
  if (tokenSummary) {
    slim.tokenSummary = tokenSummary;
  }
  // Keep a trimmed response payload (summaries + generated text) without the
  // heavy per-token/per-step arrays.
  const slimResponse = slimResponseJson(dashboard.responseJson);
  if (slimResponse) {
    slim.responseJson = slimResponse;
  }
  return slim;
}

function sanitizeHistoryEntryForExport(row) {
  const entry = { ...row };
  const dashboard = sanitizeDashboardForExport(row.dashboard);
  if (dashboard) {
    entry.dashboard = dashboard;
  } else {
    delete entry.dashboard;
  }
  return entry;
}

function downloadTextFile(fileName, text, mimeType) {
  const blob = new Blob([text], { type: mimeType });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = fileName;
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  URL.revokeObjectURL(url);
}

function csvCell(value) {
  if (value == null) {
    return "";
  }
  const text = typeof value === "string" ? value : JSON.stringify(value);
  return `"${String(text).replaceAll('"', '""')}"`;
}

function historyRowToCsvRecord(row) {
  const config = asObject(row.config);
  const metrics = asObject(row.metrics);
  const profile = asObject(row.profile);
  const output = asObject(row.output);
  const dashboard = sanitizeDashboardForExport(row.dashboard) || {};
  return {
    id: row.id,
    label: row.label || "",
    created_at: row.createdAt,
    type: row.type,
    namespace: row.namespace,
    target: row.target,
    method: row.method,
    path: row.path,
    prompt_chars: config.promptChars,
    prompt_tokens: config.promptTokens,
    generated_tokens: metrics.generatedTokens,
    max_new_tokens: config.maxNewTokens,
    min_new_tokens: config.minNewTokens,
    temperature: config.temperature,
    top_k: config.topK,
    top_p: config.topP,
    seed: config.seed,
    timeout_ms: config.timeoutMs,
    model_name: config.modelName,
    topology_hash: config.topologyHash,
    baseline: profile.baseline,
    enabled_modules: asArray(profile.enabledModules).join(";"),
    module_variants: asArray(profile.moduleVariants).join(";"),
    feature_flags_json: asObject(profile.featureFlags),
    latency_ms: metrics.latencyMs,
    tokens_per_second: metrics.tokensPerSecond,
    compute_ms: metrics.computeMs,
    matmul_ms: metrics.matmulMs,
    attention_ms: metrics.attentionMs,
    ggml_threads: metrics.ggmlThreads,
    transfer_ms: metrics.transferMs,
    true_comm_ms: metrics.trueCommMs,
    rpc_compute_ratio: metrics.rpcComputeRatio,
    true_comm_compute_ratio: metrics.trueCommComputeRatio,
    payload_mib: metrics.payloadMib,
    network_mib: metrics.networkMib,
    max_memory_mb: metrics.maxMemoryMb,
    token_p50_ms: metrics.tokenP50Ms,
    token_p95_ms: metrics.tokenP95Ms,
    token_p99_ms: metrics.tokenP99Ms,
    transfer_p50_ms: metrics.transferP50Ms,
    transfer_p95_ms: metrics.transferP95Ms,
    transfer_p99_ms: metrics.transferP99Ms,
    termination_reason: output.terminationReason,
    generated_text: getOutputText(row),
    dashboard_json: dashboard
  };
}

function exportSelectedHistoryJson() {
  const rows = selectedHistoryRows();
  if (!rows.length) {
    return;
  }
  const payload = {
    exportedAt: new Date().toISOString(),
    count: rows.length,
    entries: rows.map(sanitizeHistoryEntryForExport)
  };
  downloadTextFile(
    `dli-history-selected-${Date.now()}.json`,
    JSON.stringify(payload, null, 2),
    "application/json"
  );
}

function exportSelectedHistoryCsv() {
  const rows = selectedHistoryRows();
  if (!rows.length) {
    return;
  }
  const records = rows.map(historyRowToCsvRecord);
  const headers = Object.keys(records[0]);
  const csv = [
    headers.map(csvCell).join(","),
    ...records.map((record) => headers.map((header) => csvCell(record[header])).join(","))
  ].join("\n");
  downloadTextFile(
    `dli-history-selected-${Date.now()}.csv`,
    `${csv}\n`,
    "text/csv"
  );
}

async function deleteHistory(ids, { all = false } = {}) {
  const payload = all ? { all: true } : { ids };

  const res = await fetch("/api/history", {
    method: "DELETE",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });

  if (!res.ok) {
    const text = await res.text();
    throw new Error(`${res.status} ${res.statusText}: ${text}`);
  }

  return res.json();
}

async function deleteSelectedHistoryRuns() {
  const ids = [...selectedHistoryIds];
  if (!ids.length) {
    return;
  }

  const confirmed = window.confirm(
    `Delete ${ids.length} selected history run${ids.length === 1 ? "" : "s"}?`
  );
  if (!confirmed) {
    return;
  }

  try {
    await deleteHistory(ids);
    selectedHistoryIds.clear();
    await refreshHistory();
  } catch (error) {
    els.historyMeta.textContent = `Delete error: ${error.message}`;
  }
}

async function clearAllHistoryRuns() {
  const confirmed = window.confirm(
    "Clear all stored history runs? This cannot be undone."
  );
  if (!confirmed) {
    return;
  }

  try {
    await deleteHistory([], { all: true });
    selectedHistoryIds.clear();
    await refreshHistory();
  } catch (error) {
    els.historyMeta.textContent = `Clear error: ${error.message}`;
  }
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
  const avgRpcRatio =
    total > 0
      ? historyRows.reduce(
          (sum, row) =>
            sum +
            Number(
              asObject(row.metrics).rpcComputeRatio ??
                asObject(row.metrics).transferComputeRatio ??
                0
            ),
          0
        ) / total
      : 0;
  const avgTrueCommRatio =
    total > 0
      ? historyRows.reduce(
          (sum, row) =>
            sum +
            Number(
              asObject(row.metrics).trueCommComputeRatio ??
                asObject(row.metrics).transferComputeRatio ??
                0
            ),
          0
        ) / total
      : 0;
  const lastTs = total ? new Date(historyRows[0].createdAt).toLocaleString() : "-";
  const cards = [
    { k: "Runs", v: String(total) },
    { k: "Baseline Runs", v: String(baselineCount) },
    { k: "Module Runs", v: String(moduleCount) },
    { k: "Avg Latency (ms)", v: formatNumber(avgLatency, 1) },
    { k: "Avg RPC/Comp", v: formatNumber(avgRpcRatio, 3) },
    { k: "Avg True Comm/Comp", v: formatNumber(avgTrueCommRatio, 3) },
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
  const variantA = describeFeatureVariants(pA) || "-";
  const variantB = describeFeatureVariants(pB) || "-";
  const outputA = getOutputText(runA);
  const outputB = getOutputText(runB);
  const outputTermA = getOutputTermination(runA);
  const outputTermB = getOutputTermination(runB);

  const deltas = [
    ["Latency (ms)", mB.latencyMs, mA.latencyMs],
    ["Tokens/sec", mB.tokensPerSecond, mA.tokensPerSecond],
    ["Compute (ms)", mB.computeMs, mA.computeMs],
    ["RPC Wall (ms)", mB.transferMs, mA.transferMs],
    ["RPC/Comp", mB.rpcComputeRatio, mA.rpcComputeRatio],
    [
      "True Comm/Comp",
      mB.trueCommComputeRatio ?? mB.transferComputeRatio,
      mA.trueCommComputeRatio ?? mA.transferComputeRatio
    ],
    ["Token p95 (ms)", mB.tokenP95Ms, mA.tokenP95Ms],
    ["RPC Wall p95 (ms)", mB.transferP95Ms, mA.transferP95Ms],
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
          <tr><td>Label</td><td>${escapeHtml(runA.label || "-")}</td><td>${escapeHtml(
            runB.label || "-"
          )}</td></tr>
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
          <tr><td>Top-k</td><td>${escapeHtml(cA.topK ?? "-")}</td><td>${escapeHtml(
            cB.topK ?? "-"
          )}</td></tr>
          <tr><td>Top-p</td><td>${escapeHtml(cA.topP ?? "-")}</td><td>${escapeHtml(
            cB.topP ?? "-"
          )}</td></tr>
          <tr><td>Seed</td><td>${escapeHtml(cA.seed == null ? "random" : cA.seed)}</td><td>${escapeHtml(
            cB.seed == null ? "random" : cB.seed
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
          <tr><td>Feature Variants</td><td>${escapeHtml(variantA)}</td><td>${escapeHtml(variantB)}</td></tr>
          <tr><td>Output Preview</td><td>${escapeHtml(formatOutputPreview(outputA, 220))}</td><td>${escapeHtml(
            formatOutputPreview(outputB, 220)
          )}</td></tr>
          <tr><td>Termination</td><td>${escapeHtml(outputTermA || "-")}</td><td>${escapeHtml(
            outputTermB || "-"
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
  const rpcRatio = rowsAsc.map((row) => Number(asObject(row.metrics).rpcComputeRatio || 0));
  const ratio = rowsAsc.map((row) =>
    Number(asObject(row.metrics).trueCommComputeRatio ?? asObject(row.metrics).transferComputeRatio ?? 0)
  );
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
      <h4>RPC/Compute + True Comm/Compute</h4>
      ${buildLineChartSvg([
        { name: "rpc/compute", values: rpcRatio },
        { name: "true comm/compute", values: ratio }
      ])}
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
      <h4>RPC Wall p95 + Tokens/sec</h4>
      ${buildLineChartSvg([
        { name: "RPC wall p95", values: transferP95 },
        { name: "tokens/sec", values: tps }
      ])}
    </div>
    <div class="chart-card">
      <h4>RPC Wall vs Compute</h4>
      ${buildLineChartSvg([
        { name: "RPC wall", values: transfer },
        { name: "compute", values: compute }
      ])}
    </div>
  `;
}

function renderTable() {
  if (!historyRows.length) {
    els.historyTableBody.innerHTML = "<tr><td colspan='24'>No history rows.</td></tr>";
    updateDeleteControls();
    return;
  }

  els.historyTableBody.innerHTML = historyRows
    .map((row) => {
      const config = asObject(row.config);
      const metrics = asObject(row.metrics);
      const profile = asObject(row.profile);
      const modules = asArray(profile.enabledModules);
      const variantText = describeFeatureVariants(profile) || "-";
      const outputText = getOutputText(row);
      const outputPreview = formatOutputPreview(outputText, 180);
      const outputTermination = getOutputTermination(row);
      const profileHtml = profile.baseline
        ? `<span class="history-badge baseline">baseline</span>`
        : `<span class="history-badge modules">${escapeHtml(modules.join(", ") || "modules")}</span>`;
      const outputHtml = outputText
        ? `
          <details class="history-output-details">
            <summary>${escapeHtml(outputPreview)}</summary>
            <pre class="history-output-pre">${escapeHtml(outputText)}</pre>
            ${
              outputTermination
                ? `<div class="muted">stop: ${escapeHtml(outputTermination)}</div>`
                : ""
            }
          </details>
        `
        : "-";

      return `
        <tr>
          <td>
            <input
              class="history-row-select"
              type="checkbox"
              data-history-id="${row.id}"
              ${selectedHistoryIds.has(Number(row.id)) ? "checked" : ""}
              title="Select run #${row.id}"
            />
          </td>
          <td>${row.id}</td>
          <td>${escapeHtml(row.label || "")}</td>
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
          <td>${formatNumber(config.topK, 0)}</td>
          <td>${formatNumber(config.topP != null ? config.topP : 1, 2)}</td>
          <td>${config.seed == null ? "rnd" : escapeHtml(String(config.seed))}</td>
          <td>${formatNumber(config.timeoutMs, 0)}</td>
          <td>${escapeHtml(config.topologyHash || "-")}</td>
          <td>${formatNumber(metrics.latencyMs, 1)}</td>
          <td>${formatNumber(metrics.tokensPerSecond, 3)}</td>
          <td>${formatNumber(metrics.rpcComputeRatio ?? metrics.transferComputeRatio, 3)}</td>
          <td>${formatNumber(metrics.trueCommComputeRatio ?? metrics.transferComputeRatio, 3)}</td>
          <td>${formatNumber(metrics.maxMemoryMb, 1)}</td>
          <td>${formatNumber(metrics.networkMib, 3)}</td>
          <td>${profileHtml}</td>
          <td>${escapeHtml(variantText)}</td>
          <td>${outputHtml}</td>
        </tr>
      `;
    })
    .join("");

  document.querySelectorAll(".history-row-select").forEach((checkbox) => {
    checkbox.addEventListener("change", () => {
      const id = Number.parseInt(String(checkbox.dataset.historyId || ""), 10);
      if (!Number.isFinite(id)) {
        return;
      }

      if (checkbox.checked) {
        selectedHistoryIds.add(id);
      } else {
        selectedHistoryIds.delete(id);
      }

      updateDeleteControls();
    });
  });

  updateDeleteControls();
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
    const visibleIds = new Set(historyRows.map((row) => Number(row.id)));
    selectedHistoryIds = new Set(
        [...selectedHistoryIds].filter((id) => visibleIds.has(Number(id)))
    );
    els.historyMeta.textContent = `stored=${payload.totalStored} shown=${payload.returned} updated=${new Date(
      payload.generatedAt
    ).toLocaleString()}`;
    renderCards();
    renderCharts();
    renderTable();
    renderCompareOptions();
    renderCompareSummary();
    populateCustomChartMetrics();
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

// ---------------------------------------------------------------------------
// Custom chart builder
// ---------------------------------------------------------------------------

// Numeric metrics selectable as chart series. Each accessor reads a single
// value from a history row (metrics + config), returning a finite number.
const METRIC_CATALOG = [
  { key: "latencyMs", label: "Latency (ms)", get: (r) => Number(asObject(r.metrics).latencyMs || 0) },
  { key: "tokensPerSecond", label: "Tokens/sec", get: (r) => Number(asObject(r.metrics).tokensPerSecond || 0) },
  { key: "computeMs", label: "Compute (ms)", get: (r) => Number(asObject(r.metrics).computeMs || 0) },
  { key: "matmulMs", label: "Matmul (ms)", get: (r) => Number(asObject(r.metrics).matmulMs || 0) },
  { key: "attentionMs", label: "Attention (ms)", get: (r) => Number(asObject(r.metrics).attentionMs || 0) },
  { key: "ggmlThreads", label: "GGML Threads", get: (r) => Number(asObject(r.metrics).ggmlThreads || 0) },
  { key: "transferMs", label: "RPC Wall (ms)", get: (r) => Number(asObject(r.metrics).transferMs || 0) },
  { key: "rpcComputeRatio", label: "RPC/Compute", get: (r) => Number(asObject(r.metrics).rpcComputeRatio ?? asObject(r.metrics).transferComputeRatio ?? 0) },
  { key: "trueCommComputeRatio", label: "True Comm/Compute", get: (r) => Number(asObject(r.metrics).trueCommComputeRatio ?? asObject(r.metrics).transferComputeRatio ?? 0) },
  { key: "payloadMib", label: "Payload (MiB)", get: (r) => Number(asObject(r.metrics).payloadMib || 0) },
  { key: "networkMib", label: "Net (MiB)", get: (r) => Number(asObject(r.metrics).networkMib || 0) },
  { key: "maxMemoryMb", label: "Max Memory (MB)", get: (r) => Number(asObject(r.metrics).maxMemoryMb || 0) },
  { key: "tokenP50Ms", label: "Token p50 (ms)", get: (r) => Number(asObject(r.metrics).tokenP50Ms || 0) },
  { key: "tokenP95Ms", label: "Token p95 (ms)", get: (r) => Number(asObject(r.metrics).tokenP95Ms || 0) },
  { key: "tokenP99Ms", label: "Token p99 (ms)", get: (r) => Number(asObject(r.metrics).tokenP99Ms || 0) },
  { key: "transferP95Ms", label: "RPC Wall p95 (ms)", get: (r) => Number(asObject(r.metrics).transferP95Ms || 0) },
  { key: "generatedTokens", label: "Generated tokens", get: (r) => Number(asObject(r.metrics).generatedTokens || 0) },
  { key: "maxNewTokens", label: "Max new tokens", get: (r) => Number(asObject(r.config).maxNewTokens || 0) },
  { key: "minNewTokens", label: "Min new tokens", get: (r) => Number(asObject(r.config).minNewTokens || 0) },
  { key: "temperature", label: "Temperature", get: (r) => Number(asObject(r.config).temperature || 0) },
  { key: "topK", label: "Top-k", get: (r) => Number(asObject(r.config).topK || 0) },
  { key: "topP", label: "Top-p", get: (r) => { const v = asObject(r.config).topP; return v != null ? Number(v) : 1; } },
  { key: "seed", label: "Seed", get: (r) => { const v = asObject(r.config).seed; return v != null ? Number(v) : 0; } },
  { key: "promptTokens", label: "Prompt tokens", get: (r) => Number(asObject(r.config).promptTokens || 0) },
];

const METRIC_BY_KEY = new Map(METRIC_CATALOG.map((m) => [m.key, m]));

function readXValue(row, xKey) {
  if (xKey === "createdAt") {
    return new Date(row.createdAt).getTime();
  }
  const metric = METRIC_BY_KEY.get(xKey);
  return metric ? metric.get(row) : 0;
}

function matchesCustomFilter(row, filter) {
  if (!filter) {
    return true;
  }
  const profile = asObject(row.profile);
  const haystack = [
    row.type,
    row.namespace,
    row.target,
    row.path,
    asArray(profile.enabledModules).join(" "),
    asArray(profile.moduleVariants).join(" "),
    JSON.stringify(asObject(profile.featureFlags)),
  ]
    .join(" ")
    .toLowerCase();
  return haystack.includes(filter);
}

let customChartMetricsInitialized = false;
function populateCustomChartMetrics() {
  if (customChartMetricsInitialized || !els.customChartMetrics) {
    return;
  }
  els.customChartMetrics.innerHTML = METRIC_CATALOG.map(
    (m) => `<option value="${m.key}">${escapeHtml(m.label)}</option>`
  ).join("");
  // Sensible default selection.
  for (const option of els.customChartMetrics.options) {
    if (option.value === "latencyMs") {
      option.selected = true;
    }
  }
  customChartMetricsInitialized = true;
}

function renderCustomChart() {
  if (!els.customChart) {
    return;
  }
  const selectedKeys = [...els.customChartMetrics.selectedOptions].map((o) => o.value);
  if (!selectedKeys.length) {
    els.customChart.innerHTML = `<div class="muted">Select at least one metric, then click Render.</div>`;
    els.customChartExportPngBtn.disabled = true;
    return;
  }

  const filter = String(els.customChartFilter.value || "").trim().toLowerCase();
  const xKey = String(els.customChartXAxis.value || "index");

  // historyRows arrives newest-first; ascending order reads better on a trend.
  let rows = [...historyRows].reverse().filter((row) => matchesCustomFilter(row, filter));
  if (xKey !== "index" && els.customChartSortByX.checked) {
    rows = rows.sort((a, b) => readXValue(a, xKey) - readXValue(b, xKey));
  }

  if (!rows.length) {
    els.customChart.innerHTML = `<div class="muted">No runs match the current filter.</div>`;
    els.customChartExportPngBtn.disabled = true;
    return;
  }

  const series = selectedKeys
    .map((key) => METRIC_BY_KEY.get(key))
    .filter(Boolean)
    .map((metric) => ({ name: metric.label, values: rows.map((row) => metric.get(row)) }));

  const xLabel =
    xKey === "index"
      ? "run order"
      : (METRIC_BY_KEY.get(xKey)?.label || xKey);
  const title = `${series.map((s) => s.name).join(", ")} vs ${xLabel} — ${rows.length} run${rows.length === 1 ? "" : "s"}`;

  els.customChart.innerHTML = `<h4>${escapeHtml(title)}</h4>${buildLineChartSvg(series)}`;
  els.customChartExportPngBtn.disabled = false;
}

// ---------------------------------------------------------------------------
// Chart PNG export (SVG -> canvas -> PNG)
// ---------------------------------------------------------------------------

function svgDimensions(svgEl, fallbackW = 920, fallbackH = 210) {
  const viewBox = (svgEl.getAttribute("viewBox") || "").split(/\s+/).map(Number);
  if (viewBox.length === 4 && viewBox[2] > 0 && viewBox[3] > 0) {
    return { width: viewBox[2], height: viewBox[3] };
  }
  const rect = svgEl.getBoundingClientRect();
  return { width: rect.width || fallbackW, height: rect.height || fallbackH };
}

function loadSvgAsImage(svgEl, width, height) {
  const clone = svgEl.cloneNode(true);
  clone.setAttribute("width", String(width));
  clone.setAttribute("height", String(height));
  clone.setAttribute("xmlns", "http://www.w3.org/2000/svg");
  const svgString = new XMLSerializer().serializeToString(clone);
  const blob = new Blob([svgString], { type: "image/svg+xml;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => {
      URL.revokeObjectURL(url);
      resolve(img);
    };
    img.onerror = () => {
      URL.revokeObjectURL(url);
      reject(new Error("Failed to rasterize chart SVG"));
    };
    img.src = url;
  });
}

function downloadCanvasPng(canvas, fileName) {
  canvas.toBlob((blob) => {
    if (!blob) {
      return;
    }
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = fileName;
    document.body.appendChild(anchor);
    anchor.click();
    anchor.remove();
    URL.revokeObjectURL(url);
  }, "image/png");
}

// Stitch every chart inside `container` into one tall PNG, drawing each card's
// <h4> title above its plot. Works for the trends grid and the custom chart.
async function exportContainerChartsToPng(container, fileName, scale = 2) {
  if (!container) {
    return;
  }
  const cards = [...container.querySelectorAll(".chart-card")];
  const items = (cards.length ? cards : [container])
    .map((card) => ({
      title: (card.querySelector("h4")?.textContent || "").trim(),
      svg: card.querySelector("svg.timeseries-svg"),
    }))
    .filter((item) => item.svg);

  if (!items.length) {
    return;
  }

  const titleH = 22;
  const gap = 14;
  const pad = 16;
  const dims = items.map((item) => svgDimensions(item.svg));
  const contentW = Math.max(...dims.map((d) => d.width));
  const totalH =
    pad * 2 + items.reduce((sum, _item, i) => sum + titleH + dims[i].height + gap, 0) - gap;

  const canvas = document.createElement("canvas");
  canvas.width = (contentW + pad * 2) * scale;
  canvas.height = totalH * scale;
  const ctx = canvas.getContext("2d");
  ctx.scale(scale, scale);
  ctx.fillStyle = "#0d1430";
  ctx.fillRect(0, 0, contentW + pad * 2, totalH);

  let y = pad;
  for (let i = 0; i < items.length; i += 1) {
    const { width, height } = dims[i];
    if (items[i].title) {
      ctx.fillStyle = "#d9e2ff";
      ctx.font = "600 13px system-ui, sans-serif";
      ctx.fillText(items[i].title, pad, y + 15);
    }
    y += titleH;
    // eslint-disable-next-line no-await-in-loop
    const img = await loadSvgAsImage(items[i].svg, width, height);
    ctx.drawImage(img, pad, y, width, height);
    y += height + gap;
  }

  downloadCanvasPng(canvas, fileName);
}

async function exportHistoryTrendCharts() {
  try {
    await exportContainerChartsToPng(els.historyCharts, `dli-history-charts-${Date.now()}.png`);
  } catch (error) {
    els.historyMeta.textContent = `Chart export error: ${error.message}`;
  }
}

async function exportCustomChart() {
  try {
    await exportContainerChartsToPng(els.customChart, `dli-custom-chart-${Date.now()}.png`);
  } catch (error) {
    els.historyMeta.textContent = `Chart export error: ${error.message}`;
  }
}

els.historyRefreshBtn.addEventListener("click", refreshHistory);
els.historyType.addEventListener("change", refreshHistory);
els.historyNamespace.addEventListener("change", refreshHistory);
els.historyLimit.addEventListener("change", refreshHistory);
els.compareA.addEventListener("change", renderCompareSummary);
els.compareB.addEventListener("change", renderCompareSummary);
els.historyExportJsonBtn.addEventListener("click", exportSelectedHistoryJson);
els.historyExportCsvBtn.addEventListener("click", exportSelectedHistoryCsv);
if (els.historyChartsExportPngBtn) {
  els.historyChartsExportPngBtn.addEventListener("click", exportHistoryTrendCharts);
}
if (els.customChartRenderBtn) {
  els.customChartRenderBtn.addEventListener("click", renderCustomChart);
}
if (els.customChartExportPngBtn) {
  els.customChartExportPngBtn.addEventListener("click", exportCustomChart);
}
els.historyDeleteSelectedBtn.addEventListener("click", deleteSelectedHistoryRuns);
els.historyClearAllBtn.addEventListener("click", clearAllHistoryRuns);

els.historySelectAll.addEventListener("change", () => {
  const visibleIds = historyRows.map((row) => Number(row.id)).filter(Number.isFinite);

  if (els.historySelectAll.checked) {
    for (const id of visibleIds) {
      selectedHistoryIds.add(id);
    }
  } else {
    for (const id of visibleIds) {
      selectedHistoryIds.delete(id);
    }
  }

  renderTable();
});

const initialNs = new URLSearchParams(window.location.search).get("namespace");
if (initialNs) {
  els.historyNamespace.value = initialNs;
}

refreshHistory();
