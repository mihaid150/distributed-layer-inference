"use strict";

const els = {
  clusterMeta: document.getElementById("clusterMeta"),
  overviewCards: document.getElementById("overviewCards"),
  pipeline: document.getElementById("pipeline"),
  workloadsTable: document.getElementById("workloadsTable"),
  namespaceInput: document.getElementById("namespaceInput"),
  refreshSeconds: document.getElementById("refreshSeconds"),
  autoRefresh: document.getElementById("autoRefresh"),
  refreshTopologyBtn: document.getElementById("refreshTopologyBtn"),
  logTarget: document.getElementById("logTarget"),
  logSource: document.getElementById("logSource"),
  logTail: document.getElementById("logTail"),
  logPrevious: document.getElementById("logPrevious"),
  refreshLogsBtn: document.getElementById("refreshLogsBtn"),
  logSummary: document.getElementById("logSummary"),
  logOutput: document.getElementById("logOutput"),
  endpointPreset: document.getElementById("endpointPreset"),
  endpointTarget: document.getElementById("endpointTarget"),
  endpointMethod: document.getElementById("endpointMethod"),
  endpointPath: document.getElementById("endpointPath"),
  endpointTimeout: document.getElementById("endpointTimeout"),
  endpointPrompt: document.getElementById("endpointPrompt"),
  endpointMaxTokens: document.getElementById("endpointMaxTokens"),
  endpointMinTokens: document.getElementById("endpointMinTokens"),
  endpointTemperature: document.getElementById("endpointTemperature"),
  endpointBody: document.getElementById("endpointBody"),
  invokeEndpointBtn: document.getElementById("invokeEndpointBtn"),
  invokeMeta: document.getElementById("invokeMeta"),
  invokeMetricsDashboard: document.getElementById("invokeMetricsDashboard"),
  invokeOutput: document.getElementById("invokeOutput"),
  chatTimeout: document.getElementById("chatTimeout"),
  chatMaxTokens: document.getElementById("chatMaxTokens"),
  chatMinTokens: document.getElementById("chatMinTokens"),
  chatTemperature: document.getElementById("chatTemperature"),
  chatMessages: document.getElementById("chatMessages"),
  chatInput: document.getElementById("chatInput"),
  chatSendBtn: document.getElementById("chatSendBtn"),
  chatClearBtn: document.getElementById("chatClearBtn"),
  chatExportBtn: document.getElementById("chatExportBtn"),
  chatMeta: document.getElementById("chatMeta"),
  chatContextInfo: document.getElementById("chatContextInfo"),
  chatMetricsTimeline: document.getElementById("chatMetricsTimeline")
};

let refreshTimer = null;
let endpointCatalog = { targets: [], presets: [] };
let chatMessages = [];
let chatTurns = [];
let chatBusy = false;
let invokeBusy = false;
let podToNodeMap = new Map();
let invokeRuns = [];

const SESSION_STORAGE_KEY = "dli_ops_ui_session_v2";
const INVOKE_TIMEOUT_MIN_MS = 1000;
const INVOKE_TIMEOUT_DEFAULT_MS = 15000;
const INVOKE_TIMEOUT_LONG_DEFAULT_MS = 300000;
const INVOKE_TIMEOUT_MAX_MS = 600000;
const CHART_PALETTE = [
  "#5db2ff",
  "#24c38e",
  "#f0b429",
  "#f05c7a",
  "#9a7cff",
  "#ff8a65",
  "#4dd0e1",
  "#ffd54f"
];

function statusClass(status) {
  const s = String(status || "").toLowerCase();
  if (s.includes("running") || s.includes("healthy") || s.includes("ready")) {
    return "ok";
  }
  if (s.includes("init") || s.includes("pending") || s.includes("degraded")) {
    return "warn";
  }
  return "err";
}

function safeText(value) {
  return value == null ? "-" : String(value);
}

function parseEndpointTimeoutMs(rawValue, fallbackMs = INVOKE_TIMEOUT_DEFAULT_MS) {
  const parsed = Number.parseInt(String(rawValue ?? ""), 10);
  if (Number.isNaN(parsed)) {
    return fallbackMs;
  }
  if (parsed === 0) {
    return 0;
  }
  return Math.max(INVOKE_TIMEOUT_MIN_MS, Math.min(INVOKE_TIMEOUT_MAX_MS, parsed));
}

async function fetchJson(url) {
  const res = await fetch(url, { cache: "no-store" });
  if (!res.ok) {
    const text = await res.text();
    let payload = null;
    try {
      payload = JSON.parse(text);
    } catch {
      payload = null;
    }
    if (!payload) {
      throw new Error(`${res.status} ${res.statusText}: ${text}`);
    }
    const pieces = [`${res.status} ${res.statusText}`];
    if (payload.error) {
      pieces.push(payload.error);
    }
    if (payload.hint) {
      pieces.push(payload.hint);
    }
    if (Array.isArray(payload.availableNamespaces) && payload.availableNamespaces.length) {
      pieces.push(`available: ${payload.availableNamespaces.join(", ")}`);
    }
    if (payload.matchedBy) {
      pieces.push(`matchedBy: ${payload.matchedBy}`);
    }
    if (Array.isArray(payload.visiblePods) && payload.visiblePods.length) {
      pieces.push(`pods: ${payload.visiblePods.join(", ")}`);
    }
    throw new Error(pieces.join(": "));
  }
  return res.json();
}

function getNamespace() {
  const value = (els.namespaceInput.value || "").trim();
  return value || "inference";
}

function prettyJson(value) {
  try {
    return JSON.stringify(value, null, 2);
  } catch {
    return String(value);
  }
}

function escapeHtml(value) {
  return String(value)
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

function clampNumber(value, min, max, fallback) {
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) {
    return fallback;
  }
  return Math.max(min, Math.min(max, parsed));
}

function parseJsonOrNull(text) {
  try {
    return JSON.parse(text);
  } catch {
    return null;
  }
}

function isGenerationEndpoint() {
  const method = String(els.endpointMethod.value || "").toUpperCase();
  const path = String(els.endpointPath.value || "").trim();
  return method === "POST" && /^\/generate(?:\/|$)/i.test(path);
}

function syncGenerationControlsEnabledState() {
  const enabled = isGenerationEndpoint();
  const controls = [
    els.endpointPrompt,
    els.endpointMaxTokens,
    els.endpointMinTokens,
    els.endpointTemperature
  ];
  for (const control of controls) {
    control.disabled = !enabled;
  }
}

function syncGenerationFieldsFromBody() {
  const parsed = parseJsonOrNull(els.endpointBody.value || "");
  if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
    return;
  }

  if (parsed.prompt != null) {
    els.endpointPrompt.value = String(parsed.prompt);
  }
  if (parsed.max_new_tokens != null) {
    els.endpointMaxTokens.value = String(
      clampNumber(parsed.max_new_tokens, 1, 512, 24)
    );
  }
  if (parsed.min_new_tokens != null) {
    els.endpointMinTokens.value = String(
      clampNumber(parsed.min_new_tokens, 1, 512, 8)
    );
  }
  if (parsed.temperature != null) {
    els.endpointTemperature.value = String(
      clampNumber(parsed.temperature, 0, 2, 0.2)
    );
  }
}

function syncBodyFromGenerationFields() {
  if (!isGenerationEndpoint()) {
    return;
  }

  const current = parseJsonOrNull(els.endpointBody.value || "");
  if (!current || typeof current !== "object" || Array.isArray(current)) {
    return;
  }

  const maxNewTokens = clampNumber(els.endpointMaxTokens.value, 1, 512, 24);
  const minNewTokens = clampNumber(els.endpointMinTokens.value, 1, maxNewTokens, 8);
  const temperature = clampNumber(els.endpointTemperature.value, 0, 2, 0.2);

  els.endpointMinTokens.value = String(minNewTokens);

  const nextBody = {
    ...current,
    prompt: String(els.endpointPrompt.value || ""),
    max_new_tokens: Math.round(maxNewTokens),
    min_new_tokens: Math.round(minNewTokens),
    temperature
  };
  els.endpointBody.value = prettyJson(nextBody);
}

function getSelectedPreset() {
  const presetId = els.endpointPreset.value;
  return endpointCatalog.presets.find((p) => p.id === presetId) || null;
}

function applyEndpointPreset(preset) {
  if (!preset) {
    return;
  }
  els.endpointTarget.value = preset.target || "gateway";
  els.endpointMethod.value = (preset.method || "GET").toUpperCase();
  els.endpointPath.value = preset.path || "/health";
  if (preset.timeoutMs != null) {
    els.endpointTimeout.value = String(parseEndpointTimeoutMs(preset.timeoutMs));
  }
  const bodyValue = preset.body == null ? {} : preset.body;
  els.endpointBody.value = prettyJson(bodyValue);
  syncGenerationControlsEnabledState();
  syncGenerationFieldsFromBody();
}

async function loadEndpointCatalog() {
  const namespace = getNamespace();
  const catalog = await fetchJson(
    `/api/endpoint-catalog?namespace=${encodeURIComponent(namespace)}`
  );
  endpointCatalog = catalog;

  els.endpointTarget.innerHTML = (catalog.targets || [])
    .map((target) => `<option value="${escapeHtml(target)}">${escapeHtml(target)}</option>`)
    .join("");

  els.endpointPreset.innerHTML = (catalog.presets || [])
    .map((preset) => `<option value="${escapeHtml(preset.id)}">${escapeHtml(preset.label)}</option>`)
    .join("");

  if (catalog.presets && catalog.presets.length) {
    els.endpointPreset.value = catalog.presets[0].id;
    applyEndpointPreset(catalog.presets[0]);
  }
  syncGenerationControlsEnabledState();
  syncGenerationFieldsFromBody();
}

function renderOverview(data) {
  const totalNodes = data.nodes.length;
  const readyNodes = data.nodes.filter((n) => n.ready).length;
  const totalPods = data.pods.length;
  const runningPods = data.pods.filter((p) => p.phase === "Running").length;
  const totalDeployments = data.deployments.length;
  const healthyDeployments = data.deployments.filter((d) => d.status === "healthy").length;

  const cards = [
    { label: "Nodes Ready", value: `${readyNodes}/${totalNodes}` },
    { label: "Pods Running", value: `${runningPods}/${totalPods}` },
    { label: "Deployments Healthy", value: `${healthyDeployments}/${totalDeployments}` },
    { label: "Namespace", value: data.namespace }
  ];

  els.overviewCards.innerHTML = cards
    .map((c) => `<div class="card"><div class="k">${c.label}</div><div class="v">${c.value}</div></div>`)
    .join("");
}

function renderPipeline(data) {
  const gateway = data.orchestration.gateway;
  const stages = data.orchestration.stages;
  const parts = [];

  function nodeHtml(title, status, meta) {
    return `
      <div class="stage-node">
        <div class="name">${title}</div>
        <div class="meta ${statusClass(status)}">${safeText(status)}</div>
        <div class="meta">${meta.join("<br/>")}</div>
      </div>
    `;
  }

  parts.push(
    nodeHtml("Gateway", gateway.deploymentStatus, [
      `pod: ${safeText(gateway.podName)}`,
      `phase: ${safeText(gateway.podPhase)}`,
      `node: ${safeText(gateway.node)}`,
      `replicas: ${gateway.readyReplicas}/${gateway.desiredReplicas}`
    ])
  );

  stages.forEach((stage, idx) => {
    parts.push(`<div class="arrow">→</div>`);
    parts.push(
      nodeHtml(`Stage ${stage.id}`, stage.deploymentStatus, [
        `pod: ${safeText(stage.podName)}`,
        `phase: ${safeText(stage.podPhase)}`,
        `node: ${safeText(stage.node)}`,
        `replicas: ${stage.readyReplicas}/${stage.desiredReplicas}`
      ])
    );
  });

  els.pipeline.innerHTML = parts.join("");
}

function renderWorkloads(data) {
  const rows = data.pods
    .map(
      (p) => `
      <tr>
        <td>${safeText(p.name)}</td>
        <td>${safeText(p.app)}</td>
        <td class="${statusClass(p.phase)}">${safeText(p.phase)}</td>
        <td>${safeText(p.node)}</td>
        <td>${safeText(p.podIp)}</td>
        <td>${safeText(p.readyContainers)}/${safeText(p.totalContainers)}</td>
        <td>${safeText(p.restarts)}</td>
      </tr>
    `
    )
    .join("");

  els.workloadsTable.innerHTML = `
    <table class="table">
      <thead>
        <tr>
          <th>Pod</th>
          <th>App</th>
          <th>Phase</th>
          <th>Node</th>
          <th>IP</th>
          <th>Ready</th>
          <th>Restarts</th>
        </tr>
      </thead>
      <tbody>${rows || "<tr><td colspan='7'>No pods</td></tr>"}</tbody>
    </table>
  `;
}

function renderLogs(payload) {
  const s = payload.summary || {};
  const hints = Array.isArray(s.hints) ? s.hints : [];

  els.logSummary.innerHTML = `
    <div class="log-pill">lines: ${safeText(s.totalLines || 0)}</div>
    <div class="log-pill line-err">errors: ${safeText(s.errorLines || 0)}</div>
    <div class="log-pill line-warn">warnings: ${safeText(s.warningLines || 0)}</div>
    <div class="log-pill line-info">info: ${safeText(s.infoLines || 0)}</div>
    <div class="hints">
      <strong>Interpretation</strong>
      ${
        hints.length
          ? `<ul>${hints.map((h) => `<li>${h}</li>`).join("")}</ul>`
          : "<ul><li>No specific patterns detected.</li></ul>"
      }
    </div>
  `;

  const lines = (payload.log || "").split(/\r?\n/);
  const colored = lines
    .map((line) => {
      const escaped = line
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;");
      if (/\b(ERROR|Traceback|Exception|CRITICAL)\b/.test(line)) {
        return `<span class="line-err">${escaped}</span>`;
      }
      if (/\b(WARN|Warning)\b/.test(line)) {
        return `<span class="line-warn">${escaped}</span>`;
      }
      if (/\bINFO\b/.test(line)) {
        return `<span class="line-info">${escaped}</span>`;
      }
      return escaped;
    })
    .join("\n");

  els.logOutput.innerHTML = colored;
  els.logOutput.scrollTop = els.logOutput.scrollHeight;
}

function toFiniteNumber(value) {
  const n = Number(value);
  return Number.isFinite(n) ? n : null;
}

function asObject(value) {
  return value && typeof value === "object" && !Array.isArray(value) ? value : {};
}

function asArray(value) {
  return Array.isArray(value) ? value : [];
}

function percentile(values, p) {
  const nums = asArray(values)
    .map((value) => Number(value))
    .filter((value) => Number.isFinite(value))
    .sort((a, b) => a - b);
  if (!nums.length) {
    return 0;
  }
  const rank = Math.max(0, Math.min(1, p)) * (nums.length - 1);
  const low = Math.floor(rank);
  const high = Math.ceil(rank);
  if (low === high) {
    return nums[low];
  }
  const weight = rank - low;
  return nums[low] * (1 - weight) + nums[high] * weight;
}

function buildCriticalPathBreakdown(stageSamples, totalLatencyMs) {
  let gatewaySerializationMs = 0;
  let gatewayWaitMs = 0;
  let stageComputeMs = 0;
  let stageTransferMs = 0;

  for (const sample of stageSamples) {
    const stageId = Number(sample?.stage_id ?? -1);
    const computeMs = Number(sample?.compute_time_ms || 0);
    const transferMs = Number(sample?.transfer_time_ms || 0);
    if (stageId === 0) {
      gatewaySerializationMs += computeMs;
      gatewayWaitMs += transferMs;
    } else {
      stageComputeMs += computeMs;
      stageTransferMs += transferMs;
    }
  }

  const stageResidualMs = Math.max(0, gatewayWaitMs - stageComputeMs - stageTransferMs);
  const pipelineVisibleMs = gatewaySerializationMs + gatewayWaitMs;
  const overheadMs = Math.max(0, Number(totalLatencyMs || 0) - pipelineVisibleMs);

  return {
    gatewaySerializationMs,
    gatewayWaitMs,
    stageComputeMs,
    stageTransferMs,
    stageResidualMs,
    pipelineVisibleMs,
    overheadMs
  };
}

function formatDurationCompact(ms) {
  const n = Number(ms);
  if (!Number.isFinite(n) || n < 0) {
    return "-";
  }
  if (n < 1000) {
    return `${n.toFixed(1)} ms`;
  }
  return `${(n / 1000).toFixed(2)} s`;
}

function buildLineChartSvg(series, { width = 920, height = 220, yLabel = "" } = {}) {
  const normalizedSeries = asArray(series)
    .filter((row) => row && row.name && asArray(row.values).length)
    .map((row) => ({
      name: String(row.name),
      values: asArray(row.values).map((value) => Number(value)).map((n) => (Number.isFinite(n) ? n : 0)),
      color: row.color || null
    }));

  if (!normalizedSeries.length) {
    return `<div class="muted">No chart data.</div>`;
  }

  const maxLen = normalizedSeries.reduce((max, row) => Math.max(max, row.values.length), 0);
  const allValues = normalizedSeries.flatMap((row) => row.values);
  const yMin = Math.min(0, ...allValues);
  const yMax = Math.max(1, ...allValues);
  const span = Math.max(1e-9, yMax - yMin);

  const left = 42;
  const right = width - 14;
  const top = 12;
  const bottom = height - 28;
  const plotW = Math.max(10, right - left);
  const plotH = Math.max(10, bottom - top);

  const xForIndex = (idx) => {
    if (maxLen <= 1) {
      return left + plotW / 2;
    }
    return left + (idx / (maxLen - 1)) * plotW;
  };

  const yForValue = (value) => bottom - ((value - yMin) / span) * plotH;

  const pointsFor = (values) =>
    values
      .map((value, idx) => {
        const x = xForIndex(idx);
        const y = yForValue(value);
        return `${x.toFixed(2)},${y.toFixed(2)}`;
      })
      .join(" ");

  const lines = normalizedSeries
    .map((row, idx) => {
      const color = row.color || CHART_PALETTE[idx % CHART_PALETTE.length];
      return `<polyline fill="none" stroke="${color}" stroke-width="2" points="${pointsFor(row.values)}" />`;
    })
    .join("");

  const markers = normalizedSeries
    .map((row, idx) => {
      const color = row.color || CHART_PALETTE[idx % CHART_PALETTE.length];
      return row.values
        .map((value, valueIdx) => {
          const x = xForIndex(valueIdx);
          const y = yForValue(value);
          return `<circle cx="${x.toFixed(2)}" cy="${y.toFixed(2)}" r="3.2" fill="${color}" stroke="#d9e2ff" stroke-width="0.6" />`;
        })
        .join("");
    })
    .join("");

  const yTicks = [0, 0.25, 0.5, 0.75, 1].map((r) => {
    const yValue = yMin + span * (1 - r);
    const y = top + plotH * r;
    return {
      label: formatNumber(yValue, Math.abs(yValue) < 10 ? 2 : 1),
      y
    };
  });

  const yTickMarkup = yTicks
    .map(
      (tick) => `
      <line x1="${left}" y1="${tick.y.toFixed(2)}" x2="${right}" y2="${tick.y.toFixed(2)}" stroke="#223058" stroke-width="1" />
      <text x="${left - 6}" y="${(tick.y + 4).toFixed(2)}" text-anchor="end" fill="#93a0c7" font-size="10">${tick.label}</text>
    `
    )
    .join("");

  const xTickMarkup = [0, Math.max(0, Math.floor((maxLen - 1) / 2)), Math.max(0, maxLen - 1)]
    .filter((value, idx, arr) => arr.indexOf(value) === idx)
    .map((idx) => {
      const x = xForIndex(idx);
      return `
        <line x1="${x.toFixed(2)}" y1="${bottom}" x2="${x.toFixed(2)}" y2="${(bottom + 4).toFixed(2)}" stroke="#6073a6" stroke-width="1" />
        <text x="${x.toFixed(2)}" y="${(bottom + 16).toFixed(2)}" text-anchor="middle" fill="#93a0c7" font-size="10">${idx + 1}</text>
      `;
    })
    .join("");

  const legendMarkup = normalizedSeries
    .map((row, idx) => {
      const color = row.color || CHART_PALETTE[idx % CHART_PALETTE.length];
      return `
        <span class="chart-legend-item">
          <span class="chart-legend-swatch" style="background:${color}"></span>
          ${escapeHtml(row.name)}
        </span>
      `;
    })
    .join("");

  return `
    <svg class="timeseries-svg" viewBox="0 0 ${width} ${height}" preserveAspectRatio="none" aria-label="${escapeHtml(yLabel)}">
      <rect x="0" y="0" width="${width}" height="${height}" fill="#0d1430"></rect>
      ${yTickMarkup}
      <line x1="${left}" y1="${bottom}" x2="${right}" y2="${bottom}" stroke="#6073a6" stroke-width="1.2" />
      ${xTickMarkup}
      ${lines}
      ${markers}
    </svg>
    <div class="chart-legend">${legendMarkup}</div>
  `;
}

function persistSessionState() {
  const state = {
    savedAt: new Date().toISOString(),
    chatMessages,
    chatTurns,
    invokeRuns
  };
  try {
    sessionStorage.setItem(SESSION_STORAGE_KEY, JSON.stringify(state));
  } catch {
    // Ignore storage errors silently.
  }
}

function restoreSessionState() {
  let raw = null;
  try {
    raw = sessionStorage.getItem(SESSION_STORAGE_KEY);
  } catch {
    raw = null;
  }
  if (!raw) {
    return;
  }
  const parsed = parseJsonOrNull(raw);
  if (!parsed || typeof parsed !== "object") {
    return;
  }

  chatMessages = asArray(parsed.chatMessages).filter(
    (item) => item && typeof item === "object" && item.role && item.content != null
  );
  chatTurns = asArray(parsed.chatTurns).filter((item) => item && typeof item === "object");
  invokeRuns = asArray(parsed.invokeRuns).filter((item) => item && typeof item === "object");
}

function updateChatContextInfo() {
  const totalMessages = chatMessages.length;
  const turns = chatTurns.length;
  if (els.chatContextInfo) {
    els.chatContextInfo.textContent = `Context: ${totalMessages} messages, ${turns} completed turns (sent on each /chat request).`;
  }
}

function stageLabelFromMetric(metric) {
  const m = asObject(metric);
  const service = m.service_name ? String(m.service_name) : "unknown";
  const stageId = toFiniteNumber(m.stage_id);
  if (stageId == null || stageId === 0) {
    return service;
  }
  return `stage_${stageId}`;
}

function metricNetworkDeltaBytes(metric) {
  const networkDelta = asObject(metric.network_delta);
  const bytesTotal = toFiniteNumber(networkDelta.bytes_total);
  if (bytesTotal != null) {
    return bytesTotal;
  }
  const sent = toFiniteNumber(networkDelta.bytes_sent) || 0;
  const recv = toFiniteNumber(networkDelta.bytes_recv) || 0;
  return sent + recv;
}

function metricPayloadBytes(metric) {
  const m = asObject(metric);
  const keys = [
    "outbound_payload_b64_bytes",
    "inbound_payload_b64_bytes",
    "stage1_request_payload_bytes",
    "stage1_response_payload_bytes",
    "request_payload_bytes",
    "response_payload_bytes"
  ];
  let total = 0;
  for (const key of keys) {
    const value = toFiniteNumber(m[key]);
    if (value != null) {
      total += value;
    }
  }
  return total;
}

function flattenNumericMetrics(value, prefix = "", out = []) {
  if (value == null) {
    return out;
  }
  if (Array.isArray(value)) {
    return out;
  }
  if (typeof value !== "object") {
    const num = toFiniteNumber(value);
    if (num != null) {
      out.push({ key: prefix, value: num });
    }
    return out;
  }

  for (const [rawKey, rawValue] of Object.entries(value)) {
    const key = prefix ? `${prefix}.${rawKey}` : rawKey;
    if (rawValue == null) {
      continue;
    }
    if (Array.isArray(rawValue)) {
      continue;
    }
    if (typeof rawValue === "object") {
      flattenNumericMetrics(rawValue, key, out);
      continue;
    }
    const num = toFiniteNumber(rawValue);
    if (num != null) {
      out.push({ key, value: num });
    }
  }
  return out;
}

function buildMetricCatalog(stageSamples) {
  const statsByKey = new Map();
  for (const sample of stageSamples) {
    const flattened = flattenNumericMetrics(sample);
    for (const item of flattened) {
      if (!item.key) {
        continue;
      }
      const prev = statsByKey.get(item.key) || {
        key: item.key,
        count: 0,
        min: Number.POSITIVE_INFINITY,
        max: Number.NEGATIVE_INFINITY,
        sum: 0
      };
      prev.count += 1;
      prev.sum += item.value;
      prev.min = Math.min(prev.min, item.value);
      prev.max = Math.max(prev.max, item.value);
      statsByKey.set(item.key, prev);
    }
  }

  return [...statsByKey.values()]
    .map((item) => ({
      key: item.key,
      count: item.count,
      min: item.min,
      avg: item.count ? item.sum / item.count : 0,
      max: item.max
    }))
    .sort((a, b) => a.key.localeCompare(b.key));
}

function aggregateStageMetrics(stageSamples, perStageSummary) {
  const result = new Map();

  for (const [stageKey, summary] of Object.entries(asObject(perStageSummary))) {
    const s = asObject(summary);
    result.set(stageKey, {
      stageKey,
      stageId: toFiniteNumber(s.stage_id),
      serviceName: safeText(s.service_name),
      samples: Number(s.samples || 0),
      computeMs: Number(s.compute_time_ms_sum || 0),
      transferMs: Number(s.transfer_time_ms_sum || 0),
      payloadBytes: Number(s.payload_b64_bytes_sum || 0),
      networkBytes: Number(s.network_delta_bytes_sum || 0),
      maxMemMb: Number(s.max_process_memory_mb || 0),
      maxCpuPct: Number(s.max_cpu_percent || 0),
      maxSystemCpuPct: 0,
      maxBandwidthMbps: 0,
      maxThreads: 0,
      ctxSwitches: 0
    });
  }

  for (const sample of stageSamples) {
    const stageKey = stageLabelFromMetric(sample);
    const curr = result.get(stageKey) || {
      stageKey,
      stageId: toFiniteNumber(sample.stage_id),
      serviceName: safeText(sample.service_name),
      samples: 0,
      computeMs: 0,
      transferMs: 0,
      payloadBytes: 0,
      networkBytes: 0,
      maxMemMb: 0,
      maxCpuPct: 0,
      maxSystemCpuPct: 0,
      maxBandwidthMbps: 0,
      maxThreads: 0,
      ctxSwitches: 0
    };

    curr.samples += 1;
    curr.computeMs += Number(sample.compute_time_ms || 0);
    curr.transferMs += Number(sample.transfer_time_ms || 0);
    curr.payloadBytes += metricPayloadBytes(sample);
    curr.networkBytes += metricNetworkDeltaBytes(sample);
    curr.maxMemMb = Math.max(curr.maxMemMb, Number(sample.process_memory_mb || 0));
    curr.maxCpuPct = Math.max(
      curr.maxCpuPct,
      Number(sample.process_cpu_percent ?? sample.cpu_percent ?? 0)
    );
    curr.maxSystemCpuPct = Math.max(
      curr.maxSystemCpuPct,
      Number(sample.system_cpu_percent || 0)
    );
    curr.maxBandwidthMbps = Math.max(
      curr.maxBandwidthMbps,
      Number(
        sample.estimated_link_mbps ??
          sample.network_delta?.bandwidth_mbps_total ??
          0
      )
    );
    curr.maxThreads = Math.max(curr.maxThreads, Number(sample.process_threads || 0));
    curr.ctxSwitches += Number(sample.process_context_switches?.voluntary || 0);
    curr.ctxSwitches += Number(sample.process_context_switches?.involuntary || 0);
    result.set(stageKey, curr);
  }

  return [...result.values()].sort((a, b) => {
    const aId = a.stageId == null ? -1 : a.stageId;
    const bId = b.stageId == null ? -1 : b.stageId;
    if (aId !== bId) {
      return aId - bId;
    }
    return a.stageKey.localeCompare(b.stageKey);
  });
}

function aggregateNodeMetrics(stageSamples, podNodeMap = null) {
  const result = new Map();

  for (const sample of stageSamples) {
    const podHostname = String(sample?.hostname || "unknown");
    const mappedNodeName = podNodeMap instanceof Map ? podNodeMap.get(podHostname) : null;
    const nodeName = mappedNodeName || podHostname;
    const key = nodeName;
    const curr = result.get(key) || {
      nodeName,
      podHostnames: new Set(),
      services: new Set(),
      stages: new Set(),
      samples: 0,
      computeMs: 0,
      transferMs: 0,
      payloadBytes: 0,
      networkBytes: 0,
      maxMemMb: 0,
      maxCpuPct: 0,
      maxSystemCpuPct: 0,
      maxBandwidthMbps: 0,
      maxThreads: 0,
      ctxSwitches: 0,
      ioReadCountDelta: 0,
      ioWriteCountDelta: 0
    };

    curr.podHostnames.add(podHostname);
    curr.services.add(String(sample?.service_name || "unknown"));
    if (sample?.stage_id != null) {
      curr.stages.add(String(sample.stage_id));
    }
    curr.samples += 1;
    curr.computeMs += Number(sample.compute_time_ms || 0);
    curr.transferMs += Number(sample.transfer_time_ms || 0);
    curr.payloadBytes += metricPayloadBytes(sample);
    curr.networkBytes += metricNetworkDeltaBytes(sample);
    curr.maxMemMb = Math.max(curr.maxMemMb, Number(sample.process_memory_mb || 0));
    curr.maxCpuPct = Math.max(
      curr.maxCpuPct,
      Number(sample.process_cpu_percent ?? sample.cpu_percent ?? 0)
    );
    curr.maxSystemCpuPct = Math.max(curr.maxSystemCpuPct, Number(sample.system_cpu_percent || 0));
    curr.maxBandwidthMbps = Math.max(
      curr.maxBandwidthMbps,
      Number(sample.estimated_link_mbps ?? sample.network_delta?.bandwidth_mbps_total ?? 0)
    );
    curr.maxThreads = Math.max(curr.maxThreads, Number(sample.process_threads || 0));
    curr.ctxSwitches += Number(sample.process_context_switches?.voluntary || 0);
    curr.ctxSwitches += Number(sample.process_context_switches?.involuntary || 0);
    curr.ioReadCountDelta += Number(sample.process_io_delta?.read_count || 0);
    curr.ioWriteCountDelta += Number(sample.process_io_delta?.write_count || 0);

    result.set(key, curr);
  }

  return [...result.values()]
    .map((row) => ({
      ...row,
      nodeKey: row.nodeName,
      podHostnamesCsv: [...row.podHostnames].sort().join(", "),
      servicesCsv: [...row.services].sort().join(", "),
      stagesCsv: [...row.stages].sort().join(", ")
    }))
    .sort((a, b) =>
      String(a.nodeName).localeCompare(String(b.nodeName)) ||
      String(a.podHostnamesCsv).localeCompare(String(b.podHostnamesCsv))
    );
}

function summarizeTokenRows(tokenMetrics) {
  return asArray(tokenMetrics).map((tokenStep) => {
    const samples = asArray(tokenStep.stage_metrics);
    let computeMs = 0;
    let transferMs = 0;
    let maxMemMb = 0;
    let maxCpu = 0;
    for (const sample of samples) {
      computeMs += Number(sample.compute_time_ms || 0);
      transferMs += Number(sample.transfer_time_ms || 0);
      maxMemMb = Math.max(maxMemMb, Number(sample.process_memory_mb || 0));
      maxCpu = Math.max(maxCpu, Number(sample.process_cpu_percent ?? sample.cpu_percent ?? 0));
    }
    return {
      tokenIndex: Number(tokenStep.token_index || 0),
      tokenId: safeText(tokenStep.token_id),
      tokenText: safeText(tokenStep.token_text),
      latencyMs: Number(tokenStep.latency_ms || 0),
      stageHopCount: samples.length,
      computeMs,
      transferMs,
      maxMemMb,
      maxCpu
    };
  });
}

function buildInvokeRunSummary({
  responseJson,
  nodeRows,
  tokenRows,
  criticalPath,
  totalLatencyMs
}) {
  return {
    timestamp: new Date().toISOString(),
    totalLatencyMs: Number(totalLatencyMs || 0),
    tokensPerSecond: Number(responseJson?.tokens_per_second || 0),
    generatedTokens: Number(asArray(responseJson?.generated_token_ids).length || 0),
    computeSumMs: tokenRows.reduce((sum, row) => sum + Number(row.computeMs || 0), 0),
    transferSumMs: tokenRows.reduce((sum, row) => sum + Number(row.transferMs || 0), 0),
    maxMemoryMb: tokenRows.reduce((max, row) => Math.max(max, Number(row.maxMemMb || 0)), 0),
    p95TokenLatencyMs: percentile(tokenRows.map((row) => row.latencyMs), 0.95),
    p95TransferLatencyMs: percentile(tokenRows.map((row) => row.transferMs), 0.95),
    criticalPath,
    perNode: nodeRows.map((row) => ({
      node: row.nodeName,
      computeMs: row.computeMs,
      transferMs: row.transferMs,
      maxMemMb: row.maxMemMb,
      networkDeltaMib: row.networkBytes / (1024 * 1024)
    }))
  };
}

function renderInvokeRunEvolution() {
  if (!invokeRuns.length) {
    return `<div class="muted">No endpoint run history in this browser session yet.</div>`;
  }
  const runs = invokeRuns.slice(-40);
  const latencySeries = [{ name: "Latency ms", values: runs.map((r) => Number(r.totalLatencyMs || 0)) }];
  const transferSeries = [{ name: "Transfer ms", values: runs.map((r) => Number(r.transferSumMs || 0)) }];
  const computeSeries = [{ name: "Compute ms", values: runs.map((r) => Number(r.computeSumMs || 0)) }];
  const tpsSeries = [{ name: "Tokens/sec", values: runs.map((r) => Number(r.tokensPerSecond || 0)) }];

  return `
    <div class="invoke-subpanel">
      <h3>Endpoint Run Evolution (session, last ${runs.length})</h3>
      <div class="chart-grid">
        <div class="chart-card">
          <h4>Latency per Run (ms)</h4>
          ${buildLineChartSvg(latencySeries, { yLabel: "Latency ms" })}
        </div>
        <div class="chart-card">
          <h4>Transfer vs Compute per Run</h4>
          ${buildLineChartSvg(
            [
              { name: "Transfer ms", values: transferSeries[0].values },
              { name: "Compute ms", values: computeSeries[0].values }
            ],
            { yLabel: "ms" }
          )}
        </div>
        <div class="chart-card">
          <h4>Throughput per Run</h4>
          ${buildLineChartSvg(tpsSeries, { yLabel: "tokens/sec" })}
        </div>
      </div>
    </div>
  `;
}

function renderInvokeMetrics(responseJson, callMeta) {
  if (!responseJson || typeof responseJson !== "object") {
    els.invokeMetricsDashboard.innerHTML = `<div class="muted">No structured JSON response available for dashboard rendering.</div>`;
    return;
  }

  const summary = asObject(responseJson.summary_metrics);
  const aggregate = asObject(summary.aggregate);
  const tokenMetrics = asArray(responseJson.token_metrics);
  const generatedTokenCount =
    Number(summary.generated_token_count || asArray(responseJson.generated_token_ids).length || 0);
  const promptTokenCount = Number(responseJson.prompt_token_count || summary.prompt_token_count || 0);
  const totalLatencyMs = Number(responseJson.total_latency_ms ?? callMeta?.elapsedMs ?? 0);
  const stageSamples = tokenMetrics.flatMap((tokenStep) => asArray(tokenStep.stage_metrics));
  const stageRows = aggregateStageMetrics(stageSamples, summary.per_stage);
  const nodeRows = aggregateNodeMetrics(stageSamples, podToNodeMap);
  const tokenRows = summarizeTokenRows(tokenMetrics);
  const catalogRows = buildMetricCatalog(stageSamples);
  const tokenLatencyP50 = percentile(tokenRows.map((row) => row.latencyMs), 0.5);
  const tokenLatencyP95 = percentile(tokenRows.map((row) => row.latencyMs), 0.95);
  const tokenLatencyP99 = percentile(tokenRows.map((row) => row.latencyMs), 0.99);
  const tokenTransferP50 = percentile(tokenRows.map((row) => row.transferMs), 0.5);
  const tokenTransferP95 = percentile(tokenRows.map((row) => row.transferMs), 0.95);
  const tokenTransferP99 = percentile(tokenRows.map((row) => row.transferMs), 0.99);
  const criticalPath = buildCriticalPathBreakdown(stageSamples, totalLatencyMs);

  const maxProcessCpu = stageRows.reduce((max, row) => Math.max(max, Number(row.maxCpuPct || 0)), 0);
  const maxSystemCpu = stageRows.reduce(
    (max, row) => Math.max(max, Number(row.maxSystemCpuPct || 0)),
    0
  );
  const maxBandwidth = stageRows.reduce(
    (max, row) => Math.max(max, Number(row.maxBandwidthMbps || 0)),
    0
  );
  const maxThreads = stageRows.reduce((max, row) => Math.max(max, Number(row.maxThreads || 0)), 0);

  const cards = [
    { k: "Prompt Tokens", v: String(promptTokenCount) },
    { k: "Generated Tokens", v: String(generatedTokenCount) },
    { k: "Termination", v: safeText(responseJson.termination_reason || "unknown") },
    { k: "Latency (ms)", v: formatNumber(totalLatencyMs, 1) },
    { k: "Tokens/sec", v: formatNumber(responseJson.tokens_per_second, 3) },
    { k: "Samples", v: String(stageSamples.length) },
    { k: "Compute Sum (ms)", v: formatNumber(aggregate.compute_time_ms_sum, 1) },
    { k: "Transfer Sum (ms)", v: formatNumber(aggregate.transfer_time_ms_sum, 1) },
    {
      k: "Transfer/Compute Ratio",
      v: formatNumber(
        Number(aggregate.transfer_time_ms_sum || 0) /
          Math.max(1e-9, Number(aggregate.compute_time_ms_sum || 0)),
        3
      )
    },
    { k: "Payload Sum (MiB)", v: formatNumber(aggregate.payload_b64_mebibytes_sum, 3) },
    { k: "Network Delta (MiB)", v: formatNumber(aggregate.network_delta_mebibytes_sum, 3) },
    { k: "Max Process Mem (MB)", v: formatNumber(aggregate.max_process_memory_mb, 1) },
    { k: "Max Process CPU (%)", v: formatNumber(maxProcessCpu, 1) },
    { k: "Max System CPU (%)", v: formatNumber(maxSystemCpu, 1) },
    { k: "Max Link Mbps", v: formatNumber(maxBandwidth, 2) },
    { k: "Max Threads", v: String(Math.round(maxThreads)) },
    { k: "Nodes Seen", v: String(nodeRows.length) },
    { k: "Token Latency p50 (ms)", v: formatNumber(tokenLatencyP50, 1) },
    { k: "Token Latency p95 (ms)", v: formatNumber(tokenLatencyP95, 1) },
    { k: "Token Latency p99 (ms)", v: formatNumber(tokenLatencyP99, 1) },
    { k: "Transfer Lat p50 (ms)", v: formatNumber(tokenTransferP50, 1) },
    { k: "Transfer Lat p95 (ms)", v: formatNumber(tokenTransferP95, 1) },
    { k: "Transfer Lat p99 (ms)", v: formatNumber(tokenTransferP99, 1) }
  ];

  const stageTableRows = stageRows
    .map(
      (row) => `
      <tr>
        <td>${escapeHtml(row.stageKey)}</td>
        <td>${escapeHtml(row.serviceName)}</td>
        <td>${safeText(row.stageId)}</td>
        <td>${row.samples}</td>
        <td>${formatNumber(row.computeMs, 1)}</td>
        <td>${formatNumber(row.transferMs, 1)}</td>
        <td>${formatNumber(row.maxMemMb, 1)}</td>
        <td>${formatNumber(row.maxCpuPct, 1)}</td>
        <td>${formatNumber(row.maxSystemCpuPct, 1)}</td>
        <td>${formatNumber(row.networkBytes / (1024 * 1024), 3)}</td>
        <td>${formatNumber(row.payloadBytes / (1024 * 1024), 3)}</td>
        <td>${formatNumber(row.maxBandwidthMbps, 2)}</td>
      </tr>
    `
    )
    .join("");

  const tokenTableRows = tokenRows
    .map(
      (row) => `
      <tr>
        <td>${row.tokenIndex}</td>
        <td>${escapeHtml(row.tokenId)}</td>
        <td>${escapeHtml(row.tokenText)}</td>
        <td>${formatNumber(row.latencyMs, 1)}</td>
        <td>${row.stageHopCount}</td>
        <td>${formatNumber(row.computeMs, 1)}</td>
        <td>${formatNumber(row.transferMs, 1)}</td>
        <td>${formatNumber(row.maxMemMb, 1)}</td>
        <td>${formatNumber(row.maxCpu, 1)}</td>
      </tr>
    `
    )
    .join("");

  const nodeTableRows = nodeRows
    .map(
      (row) => `
      <tr>
        <td>${escapeHtml(row.nodeName)}</td>
        <td>${escapeHtml(row.podHostnamesCsv || "-")}</td>
        <td>${escapeHtml(row.servicesCsv)}</td>
        <td>${escapeHtml(row.stagesCsv || "-")}</td>
        <td>${row.samples}</td>
        <td>${formatNumber(row.computeMs, 1)}</td>
        <td>${formatNumber(row.transferMs, 1)}</td>
        <td>${formatNumber(row.maxMemMb, 1)}</td>
        <td>${formatNumber(row.maxCpuPct, 1)}</td>
        <td>${formatNumber(row.maxSystemCpuPct, 1)}</td>
        <td>${formatNumber(row.networkBytes / (1024 * 1024), 3)}</td>
        <td>${formatNumber(row.payloadBytes / (1024 * 1024), 3)}</td>
        <td>${formatNumber(row.maxBandwidthMbps, 2)}</td>
        <td>${Math.round(row.maxThreads)}</td>
        <td>${Math.round(row.ctxSwitches)}</td>
        <td>${Math.round(row.ioReadCountDelta)}</td>
        <td>${Math.round(row.ioWriteCountDelta)}</td>
      </tr>
    `
    )
    .join("");

  const metricCatalogRows = catalogRows
    .map(
      (row) => `
      <tr>
        <td>${escapeHtml(row.key)}</td>
        <td>${row.count}</td>
        <td>${formatNumber(row.min, 4)}</td>
        <td>${formatNumber(row.avg, 4)}</td>
        <td>${formatNumber(row.max, 4)}</td>
      </tr>
    `
    )
    .join("");

  const criticalPathRows = [
    {
      key: "Gateway Serialization",
      ms: criticalPath.gatewaySerializationMs,
      note: "Gateway payload encode/decode and local prep."
    },
    {
      key: "Gateway Wait on Stages",
      ms: criticalPath.gatewayWaitMs,
      note: "Gateway blocked waiting for distributed stage pipeline."
    },
    {
      key: "Stage Compute (subset of wait)",
      ms: criticalPath.stageComputeMs,
      note: "Sum of compute_time_ms from stage-1..stage-4."
    },
    {
      key: "Stage Transfer (subset of wait)",
      ms: criticalPath.stageTransferMs,
      note: "Inter-stage forward transfer_time_ms (stage->stage)."
    },
    {
      key: "Stage Residual (wait - compute - transfer)",
      ms: criticalPath.stageResidualMs,
      note: "Scheduling/serialization/other queue overhead inside wait window."
    },
    {
      key: "Gateway-visible Pipeline (serialize + wait)",
      ms: criticalPath.pipelineVisibleMs,
      note: "Approximate critical path observed at gateway."
    },
    {
      key: "Response Overhead vs Total",
      ms: criticalPath.overheadMs,
      note: "Total latency minus gateway-visible pipeline."
    }
  ]
    .map(
      (row) => `
      <tr>
        <td>${escapeHtml(row.key)}</td>
        <td>${formatNumber(row.ms, 1)}</td>
        <td>${formatDurationCompact(row.ms)}</td>
        <td>${formatNumber((100 * row.ms) / Math.max(1e-9, totalLatencyMs), 2)}%</td>
        <td>${escapeHtml(row.note)}</td>
      </tr>
    `
    )
    .join("");

  invokeRuns.push(
    buildInvokeRunSummary({
      responseJson,
      nodeRows,
      tokenRows,
      criticalPath,
      totalLatencyMs
    })
  );
  if (invokeRuns.length > 120) {
    invokeRuns = invokeRuns.slice(-120);
  }
  persistSessionState();

  els.invokeMetricsDashboard.innerHTML = `
    <div class="invoke-metric-cards">
      ${cards
        .map(
          (card) => `
        <div class="invoke-metric-card">
          <div class="k">${escapeHtml(card.k)}</div>
          <div class="v">${escapeHtml(card.v)}</div>
        </div>
      `
        )
        .join("")}
    </div>

    <div class="invoke-subpanel">
      <h3>Critical Path Decomposition</h3>
      <table class="metrics-table">
        <thead>
          <tr>
            <th>Component</th>
            <th>Latency (ms)</th>
            <th>Latency</th>
            <th>% of total</th>
            <th>Interpretation</th>
          </tr>
        </thead>
        <tbody>${criticalPathRows}</tbody>
      </table>
    </div>

    <div class="invoke-subpanel">
      <h3>Per-Stage Metrics</h3>
      <table class="metrics-table">
        <thead>
          <tr>
            <th>Stage Key</th>
            <th>Service</th>
            <th>ID</th>
            <th>Samples</th>
            <th>Compute (ms)</th>
            <th>Transfer (ms)</th>
            <th>Max Mem (MB)</th>
            <th>Max Proc CPU (%)</th>
            <th>Max Sys CPU (%)</th>
            <th>Net Delta (MiB)</th>
            <th>Payload (MiB)</th>
            <th>Max Mbps</th>
          </tr>
        </thead>
        <tbody>${stageTableRows || "<tr><td colspan='12'>No stage metrics found.</td></tr>"}</tbody>
      </table>
    </div>

    <div class="invoke-subpanel">
      <h3>Per-Node Metrics</h3>
      <table class="metrics-table">
        <thead>
          <tr>
            <th>Node</th>
            <th>Pod Hostnames</th>
            <th>Services</th>
            <th>Stages</th>
            <th>Samples</th>
            <th>Compute (ms)</th>
            <th>Transfer (ms)</th>
            <th>Max Mem (MB)</th>
            <th>Max Proc CPU (%)</th>
            <th>Max Sys CPU (%)</th>
            <th>Net Delta (MiB)</th>
            <th>Payload (MiB)</th>
            <th>Max Mbps</th>
            <th>Max Threads</th>
            <th>Ctx Sw</th>
            <th>IO Read Δ</th>
            <th>IO Write Δ</th>
          </tr>
        </thead>
        <tbody>${nodeTableRows || "<tr><td colspan='17'>No node metrics found.</td></tr>"}</tbody>
      </table>
    </div>

    <div class="invoke-subpanel">
      <h3>Per-Token Evolution</h3>
      <table class="metrics-table">
        <thead>
          <tr>
            <th>Token #</th>
            <th>ID</th>
            <th>Text</th>
            <th>Latency (ms)</th>
            <th>Hops</th>
            <th>Compute (ms)</th>
            <th>Transfer (ms)</th>
            <th>Max Mem (MB)</th>
            <th>Max CPU (%)</th>
          </tr>
        </thead>
        <tbody>${tokenTableRows || "<tr><td colspan='9'>No token metrics found.</td></tr>"}</tbody>
      </table>
    </div>

    <div class="invoke-subpanel">
      <h3>Metric Catalog (all numeric fields observed in stage metrics)</h3>
      <table class="metrics-table">
        <thead>
          <tr>
            <th>Metric Key</th>
            <th>Samples</th>
            <th>Min</th>
            <th>Avg</th>
            <th>Max</th>
          </tr>
        </thead>
        <tbody>${metricCatalogRows || "<tr><td colspan='5'>No metric catalog data.</td></tr>"}</tbody>
      </table>
    </div>
    ${renderInvokeRunEvolution()}
  `;
}

async function loadTopology() {
  const namespace = getNamespace();
  try {
    const data = await fetchJson(`/api/topology?namespace=${encodeURIComponent(namespace)}`);
    els.clusterMeta.textContent = `namespace=${data.namespace} | updated=${new Date(
      data.generatedAt
    ).toLocaleString()}`;
    renderOverview(data);
    renderPipeline(data);
    renderWorkloads(data);
    podToNodeMap = new Map((data.pods || []).map((pod) => [String(pod.name || ""), String(pod.node || "")]));
  } catch (error) {
    els.clusterMeta.textContent = `Topology error (${namespace}): ${error.message}`;
    podToNodeMap = new Map();
  }
}

async function loadLogs() {
  const namespace = getNamespace();
  const target = els.logTarget.value;
  const source = els.logSource.value;
  const tail = Math.max(20, Math.min(2000, Number.parseInt(els.logTail.value || "300", 10)));
  const previous = els.logPrevious.checked ? "1" : "0";

  try {
    const payload = await fetchJson(
      `/api/logs?namespace=${encodeURIComponent(namespace)}&target=${encodeURIComponent(
        target
      )}&source=${encodeURIComponent(source)}&tail=${tail}&previous=${previous}`
    );
    renderLogs(payload);
  } catch (error) {
    els.logSummary.innerHTML = `<div class="log-pill line-err">Log error: ${error.message}</div>`;
    els.logOutput.textContent = "";
  }
}

async function invokeEndpoint() {
  if (invokeBusy) {
    return;
  }

  const namespace = getNamespace();
  const target = els.endpointTarget.value || "gateway";
  const method = (els.endpointMethod.value || "GET").toUpperCase();
  const path = (els.endpointPath.value || "/health").trim() || "/health";
  const timeoutMs = parseEndpointTimeoutMs(els.endpointTimeout.value, INVOKE_TIMEOUT_DEFAULT_MS);

  let parsedBody = undefined;
  if (method !== "GET" && method !== "HEAD") {
    const rawBody = (els.endpointBody.value || "").trim();
    if (rawBody.length) {
      try {
        parsedBody = JSON.parse(rawBody);
      } catch {
        parsedBody = rawBody;
      }
    }
  }

  const requestPayload = {
    namespace,
    target,
    method,
    path,
    timeoutMs,
    body: parsedBody
  };

  setInvokeBusy(true);

  try {
    const res = await fetch("/api/invoke", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(requestPayload)
    });
    const text = await res.text();
    let payload = null;
    try {
      payload = text ? JSON.parse(text) : null;
    } catch {
      payload = null;
    }

    if (!res.ok) {
      const pieces = [`${res.status} ${res.statusText}`];
      if (payload?.error) {
        pieces.push(payload.error);
      }
      if (payload?.hint) {
        pieces.push(payload.hint);
      }
      if (Array.isArray(payload?.visiblePods) && payload.visiblePods.length) {
        pieces.push(`pods: ${payload.visiblePods.join(", ")}`);
      }
      if (Array.isArray(payload?.attempts) && payload.attempts.length) {
        const preview = payload.attempts
          .slice(0, 8)
          .map((attempt) => `${attempt.candidate?.type || "candidate"}=${attempt.error || attempt.status || "failed"}`)
          .join("; ");
        pieces.push(`attempts: ${preview}`);
      }
      throw new Error(pieces.join(": "));
    }

    els.invokeMeta.innerHTML = `
      <div class="log-pill">status: ${safeText(payload.call?.status)} ${safeText(
      payload.call?.statusText
    )}</div>
      <div class="log-pill">elapsed: ${safeText(payload.call?.elapsedMs)}ms</div>
      <div class="log-pill">target: ${safeText(payload.target)}</div>
      <div class="log-pill">path: ${safeText(payload.path)}</div>
      <div class="log-pill">pod: ${safeText(payload.podName)}</div>
    `;

    renderInvokeMetrics(payload.responseJson, payload.call || {});

    const formatted = payload.responseJson != null
      ? prettyJson(payload.responseJson)
      : payload.responseText || "";
    els.invokeOutput.textContent = formatted;
  } catch (error) {
    els.invokeMeta.innerHTML = `<div class="log-pill line-err">Invoke error: ${escapeHtml(
      error.message
    )}</div>`;
    els.invokeMetricsDashboard.innerHTML = "";
    els.invokeOutput.textContent = "";
  } finally {
    setInvokeBusy(false);
  }
}

function setInvokeBusy(isBusy) {
  invokeBusy = isBusy;
  els.invokeEndpointBtn.disabled = isBusy;
  els.invokeEndpointBtn.textContent = isBusy ? "Running..." : "Run Endpoint";
}

function setChatBusy(isBusy) {
  chatBusy = isBusy;
  els.chatSendBtn.disabled = isBusy;
  els.chatSendBtn.textContent = isBusy ? "Sending..." : "Send";
}

function renderChatMessages() {
  if (!chatMessages.length) {
    els.chatMessages.innerHTML = `<div class="muted">No conversation yet.</div>`;
    updateChatContextInfo();
    return;
  }

  els.chatMessages.innerHTML = chatMessages
    .map((message, index) => {
      const role = (message.role || "user").toLowerCase();
      const content = escapeHtml(message.content || "");
      return `
        <div class="chat-message ${escapeHtml(role)}">
          <div class="meta">#${index + 1} • ${escapeHtml(role)}</div>
          <div>${content.replaceAll("\n", "<br/>")}</div>
        </div>
      `;
    })
    .join("");
  els.chatMessages.scrollTop = els.chatMessages.scrollHeight;
  updateChatContextInfo();
}

function summarizeTurnMetrics(response) {
  const summary = response.summary_metrics || {};
  const aggregate = summary.aggregate || {};
  const stageSamples = asArray(response.token_metrics).flatMap((tokenStep) =>
    asArray(tokenStep.stage_metrics)
  );
  const tokenRows = summarizeTokenRows(response.token_metrics);
  const perNodeRows = aggregateNodeMetrics(stageSamples, podToNodeMap);
  const perNode = {};
  for (const row of perNodeRows) {
    perNode[row.nodeKey] = {
      node: row.nodeKey,
      node_name: row.nodeName,
      pod_hostnames: row.podHostnamesCsv,
      services: row.servicesCsv,
      stages: row.stagesCsv,
      samples: row.samples,
      compute_time_ms_sum: row.computeMs,
      transfer_time_ms_sum: row.transferMs,
      max_process_memory_mb: row.maxMemMb,
      max_cpu_percent: row.maxCpuPct,
      max_system_cpu_percent: row.maxSystemCpuPct,
      network_delta_mib_sum: row.networkBytes / (1024 * 1024),
      payload_mib_sum: row.payloadBytes / (1024 * 1024),
      max_bandwidth_mbps: row.maxBandwidthMbps,
      max_threads: row.maxThreads,
      context_switches: row.ctxSwitches,
      io_read_count_delta_sum: row.ioReadCountDelta,
      io_write_count_delta_sum: row.ioWriteCountDelta
    };
  }

  const totalLatencyMs = Number(response.total_latency_ms || 0);
  const criticalPath = buildCriticalPathBreakdown(stageSamples, totalLatencyMs);

  return {
    turn: chatTurns.length + 1,
    timestamp: new Date().toISOString(),
    prompt_tokens: Number(response.prompt_token_count || 0),
    generated_tokens: Number((response.generated_token_ids || []).length),
    total_latency_ms: totalLatencyMs,
    tokens_per_second: Number(response.tokens_per_second || 0),
    compute_ms_sum: Number(aggregate.compute_time_ms_sum || 0),
    transfer_ms_sum: Number(aggregate.transfer_time_ms_sum || 0),
    max_memory_mb: Number(aggregate.max_process_memory_mb || 0),
    network_mib_sum: Number(aggregate.network_delta_mebibytes_sum || 0),
    payload_mib_sum: Number(aggregate.payload_b64_mebibytes_sum || 0),
    token_latency_p50_ms: percentile(tokenRows.map((row) => row.latencyMs), 0.5),
    token_latency_p95_ms: percentile(tokenRows.map((row) => row.latencyMs), 0.95),
    token_latency_p99_ms: percentile(tokenRows.map((row) => row.latencyMs), 0.99),
    transfer_latency_p50_ms: percentile(tokenRows.map((row) => row.transferMs), 0.5),
    transfer_latency_p95_ms: percentile(tokenRows.map((row) => row.transferMs), 0.95),
    transfer_latency_p99_ms: percentile(tokenRows.map((row) => row.transferMs), 0.99),
    critical_path: criticalPath,
    termination_reason: response.termination_reason || "unknown",
    per_stage: summary.per_stage || {},
    per_node: perNode
  };
}

function buildPerNodeSeries(turns, valueFn) {
  const nodes = new Set();
  for (const turn of turns) {
    for (const nodeKey of Object.keys(asObject(turn.per_node))) {
      nodes.add(nodeKey);
    }
  }

  const sortedNodes = [...nodes].sort((a, b) => String(a).localeCompare(String(b)));
  return sortedNodes.map((nodeKey, idx) => {
    const values = turns.map((turn) => {
      const node = asObject(asObject(turn.per_node)[nodeKey]);
      return Number(valueFn(node) || 0);
    });
    return {
      name: nodeKey,
      values,
      color: CHART_PALETTE[idx % CHART_PALETTE.length]
    };
  });
}

function renderChatMetricsTimeline() {
  if (!chatTurns.length) {
    els.chatMetricsTimeline.innerHTML = `<div class="muted">No turn metrics yet.</div>`;
    return;
  }

  const totalTurns = chatTurns.length;
  const totalLatency = chatTurns.reduce((sum, turn) => sum + Number(turn.total_latency_ms || 0), 0);
  const totalGeneratedTokens = chatTurns.reduce(
    (sum, turn) => sum + Number(turn.generated_tokens || 0),
    0
  );
  const totalCompute = chatTurns.reduce((sum, turn) => sum + Number(turn.compute_ms_sum || 0), 0);
  const totalTransfer = chatTurns.reduce((sum, turn) => sum + Number(turn.transfer_ms_sum || 0), 0);
  const peakMemory = chatTurns.reduce((max, turn) => Math.max(max, Number(turn.max_memory_mb || 0)), 0);
  const totalNetwork = chatTurns.reduce((sum, turn) => sum + Number(turn.network_mib_sum || 0), 0);
  const totalPayload = chatTurns.reduce((sum, turn) => sum + Number(turn.payload_mib_sum || 0), 0);
  const avgLatency = totalLatency / Math.max(1, totalTurns);
  const avgTps =
    totalGeneratedTokens / Math.max(1e-9, totalLatency / 1000.0);
  const avgTokenP95 = chatTurns.reduce(
    (sum, turn) => sum + Number(turn.token_latency_p95_ms || 0),
    0
  ) / Math.max(1, totalTurns);
  const avgTokenP50 = chatTurns.reduce(
    (sum, turn) => sum + Number(turn.token_latency_p50_ms || 0),
    0
  ) / Math.max(1, totalTurns);
  const avgTokenP99 = chatTurns.reduce(
    (sum, turn) => sum + Number(turn.token_latency_p99_ms || 0),
    0
  ) / Math.max(1, totalTurns);
  const avgTransferP95 = chatTurns.reduce(
    (sum, turn) => sum + Number(turn.transfer_latency_p95_ms || 0),
    0
  ) / Math.max(1, totalTurns);
  const avgTransferP50 = chatTurns.reduce(
    (sum, turn) => sum + Number(turn.transfer_latency_p50_ms || 0),
    0
  ) / Math.max(1, totalTurns);
  const avgTransferP99 = chatTurns.reduce(
    (sum, turn) => sum + Number(turn.transfer_latency_p99_ms || 0),
    0
  ) / Math.max(1, totalTurns);
  const criticalGatewayWait = chatTurns.reduce(
    (sum, turn) => sum + Number(asObject(turn.critical_path).gatewayWaitMs || 0),
    0
  );
  const criticalGatewaySerialization = chatTurns.reduce(
    (sum, turn) => sum + Number(asObject(turn.critical_path).gatewaySerializationMs || 0),
    0
  );
  const criticalStageCompute = chatTurns.reduce(
    (sum, turn) => sum + Number(asObject(turn.critical_path).stageComputeMs || 0),
    0
  );
  const criticalStageTransfer = chatTurns.reduce(
    (sum, turn) => sum + Number(asObject(turn.critical_path).stageTransferMs || 0),
    0
  );

  const stageRollup = new Map();
  const nodeRollup = new Map();
  for (const turn of chatTurns) {
    const perStage = asObject(turn.per_stage);
    for (const [stageKey, stageData] of Object.entries(perStage)) {
      const stage = asObject(stageData);
      const curr = stageRollup.get(stageKey) || {
        stageKey,
        serviceName: safeText(stage.service_name),
        stageId: safeText(stage.stage_id),
        samples: 0,
        computeMs: 0,
        transferMs: 0,
        maxMemMb: 0,
        maxCpuPct: 0,
        networkMib: 0,
        payloadMib: 0
      };
      curr.samples += Number(stage.samples || 0);
      curr.computeMs += Number(stage.compute_time_ms_sum || 0);
      curr.transferMs += Number(stage.transfer_time_ms_sum || 0);
      curr.maxMemMb = Math.max(curr.maxMemMb, Number(stage.max_process_memory_mb || 0));
      curr.maxCpuPct = Math.max(curr.maxCpuPct, Number(stage.max_cpu_percent || 0));
      curr.networkMib += Number(stage.network_delta_bytes_sum || 0) / (1024 * 1024);
      curr.payloadMib += Number(stage.payload_b64_bytes_sum || 0) / (1024 * 1024);
      stageRollup.set(stageKey, curr);
    }

    const perNode = asObject(turn.per_node);
    for (const [entryNodeKey, nodeData] of Object.entries(perNode)) {
      const node = asObject(nodeData);
      const logicalNodeKey = String(node.node_name || node.node || entryNodeKey);
      const curr = nodeRollup.get(logicalNodeKey) || {
        nodeKey: logicalNodeKey,
        podHostnames: new Set(),
        services: new Set(),
        stages: new Set(),
        samples: 0,
        computeMs: 0,
        transferMs: 0,
        maxMemMb: 0,
        maxCpuPct: 0,
        maxSystemCpuPct: 0,
        networkMib: 0,
        payloadMib: 0,
        maxBandwidthMbps: 0,
        maxThreads: 0,
        ctxSwitches: 0,
        ioReadCountDelta: 0,
        ioWriteCountDelta: 0
      };

      const podHostnamesCsv = String(node.pod_hostnames || "");
      if (podHostnamesCsv) {
        for (const host of podHostnamesCsv.split(",")) {
          const clean = host.trim();
          if (clean) {
            curr.podHostnames.add(clean);
          }
        }
      }
      const servicesCsv = String(node.services || "");
      const stagesCsv = String(node.stages || "");
      if (servicesCsv) {
        for (const service of servicesCsv.split(",")) {
          const clean = service.trim();
          if (clean) {
            curr.services.add(clean);
          }
        }
      }
      if (stagesCsv) {
        for (const stage of stagesCsv.split(",")) {
          const clean = stage.trim();
          if (clean) {
            curr.stages.add(clean);
          }
        }
      }

      curr.samples += Number(node.samples || 0);
      curr.computeMs += Number(node.compute_time_ms_sum || 0);
      curr.transferMs += Number(node.transfer_time_ms_sum || 0);
      curr.maxMemMb = Math.max(curr.maxMemMb, Number(node.max_process_memory_mb || 0));
      curr.maxCpuPct = Math.max(curr.maxCpuPct, Number(node.max_cpu_percent || 0));
      curr.maxSystemCpuPct = Math.max(
        curr.maxSystemCpuPct,
        Number(node.max_system_cpu_percent || 0)
      );
      curr.networkMib += Number(node.network_delta_mib_sum || 0);
      curr.payloadMib += Number(node.payload_mib_sum || 0);
      curr.maxBandwidthMbps = Math.max(
        curr.maxBandwidthMbps,
        Number(node.max_bandwidth_mbps || 0)
      );
      curr.maxThreads = Math.max(curr.maxThreads, Number(node.max_threads || 0));
      curr.ctxSwitches += Number(node.context_switches || 0);
      curr.ioReadCountDelta += Number(node.io_read_count_delta_sum || 0);
      curr.ioWriteCountDelta += Number(node.io_write_count_delta_sum || 0);

      nodeRollup.set(logicalNodeKey, curr);
    }
  }

  const chatCards = [
    { k: "Turns", v: String(totalTurns) },
    { k: "Generated Tokens", v: String(totalGeneratedTokens) },
    { k: "Avg Latency (ms)", v: formatNumber(avgLatency, 1) },
    { k: "Avg Tok/s", v: formatNumber(avgTps, 3) },
    { k: "Compute Sum (ms)", v: formatNumber(totalCompute, 1) },
    { k: "Transfer Sum (ms)", v: formatNumber(totalTransfer, 1) },
    {
      k: "Transfer/Compute Ratio",
      v: formatNumber(totalTransfer / Math.max(1e-9, totalCompute), 3)
    },
    { k: "Peak Mem (MB)", v: formatNumber(peakMemory, 1) },
    { k: "Network Total (MiB)", v: formatNumber(totalNetwork, 3) },
    { k: "Payload Total (MiB)", v: formatNumber(totalPayload, 3) },
    { k: "Avg Token p50 (ms)", v: formatNumber(avgTokenP50, 1) },
    { k: "Avg Token p95 (ms)", v: formatNumber(avgTokenP95, 1) },
    { k: "Avg Token p99 (ms)", v: formatNumber(avgTokenP99, 1) },
    { k: "Avg Transfer p50 (ms)", v: formatNumber(avgTransferP50, 1) },
    { k: "Avg Transfer p95 (ms)", v: formatNumber(avgTransferP95, 1) },
    { k: "Avg Transfer p99 (ms)", v: formatNumber(avgTransferP99, 1) }
  ];

  const rows = chatTurns
    .map(
      (turn) => `
      <tr>
        <td>${turn.turn}</td>
        <td>${turn.generated_tokens}</td>
        <td>${formatNumber(turn.total_latency_ms, 1)}</td>
        <td>${formatNumber(turn.tokens_per_second, 3)}</td>
        <td>${formatNumber(turn.token_latency_p95_ms, 1)}</td>
        <td>${formatNumber(turn.transfer_latency_p95_ms, 1)}</td>
        <td>${formatNumber(turn.compute_ms_sum, 1)}</td>
        <td>${formatNumber(turn.transfer_ms_sum, 1)}</td>
        <td>${formatNumber(turn.max_memory_mb, 1)}</td>
        <td>${formatNumber(turn.network_mib_sum, 3)}</td>
        <td>${formatNumber(turn.payload_mib_sum, 3)}</td>
        <td>${escapeHtml(turn.termination_reason)}</td>
      </tr>
    `
    )
    .join("");

  const nodeRows = [...nodeRollup.values()]
    .sort((a, b) => String(a.nodeKey).localeCompare(String(b.nodeKey)))
    .map(
      (node) => `
      <tr>
        <td>${escapeHtml(node.nodeKey)}</td>
        <td>${escapeHtml([...node.podHostnames].sort().join(", ") || "-")}</td>
        <td>${escapeHtml([...node.services].sort().join(", "))}</td>
        <td>${escapeHtml([...node.stages].sort().join(", ") || "-")}</td>
        <td>${node.samples}</td>
        <td>${formatNumber(node.computeMs, 1)}</td>
        <td>${formatNumber(node.transferMs, 1)}</td>
        <td>${formatNumber(node.maxMemMb, 1)}</td>
        <td>${formatNumber(node.maxCpuPct, 1)}</td>
        <td>${formatNumber(node.maxSystemCpuPct, 1)}</td>
        <td>${formatNumber(node.networkMib, 3)}</td>
        <td>${formatNumber(node.payloadMib, 3)}</td>
        <td>${formatNumber(node.maxBandwidthMbps, 2)}</td>
        <td>${Math.round(node.maxThreads)}</td>
        <td>${Math.round(node.ctxSwitches)}</td>
        <td>${Math.round(node.ioReadCountDelta)}</td>
        <td>${Math.round(node.ioWriteCountDelta)}</td>
      </tr>
    `
    )
    .join("");

  const stageRows = [...stageRollup.values()]
    .sort((a, b) => String(a.stageKey).localeCompare(String(b.stageKey)))
    .map(
      (stage) => `
      <tr>
        <td>${escapeHtml(stage.stageKey)}</td>
        <td>${escapeHtml(stage.serviceName)}</td>
        <td>${escapeHtml(stage.stageId)}</td>
        <td>${stage.samples}</td>
        <td>${formatNumber(stage.computeMs, 1)}</td>
        <td>${formatNumber(stage.transferMs, 1)}</td>
        <td>${formatNumber(stage.maxMemMb, 1)}</td>
        <td>${formatNumber(stage.maxCpuPct, 1)}</td>
        <td>${formatNumber(stage.networkMib, 3)}</td>
        <td>${formatNumber(stage.payloadMib, 3)}</td>
      </tr>
    `
    )
    .join("");

  const nodeMemorySeries = buildPerNodeSeries(chatTurns, (node) => node.max_process_memory_mb);
  const nodeNetworkSeries = buildPerNodeSeries(chatTurns, (node) => node.network_delta_mib_sum);
  const nodeTransferSeries = buildPerNodeSeries(chatTurns, (node) => node.transfer_time_ms_sum);
  const nodeComputeSeries = buildPerNodeSeries(chatTurns, (node) => node.compute_time_ms_sum);

  const criticalPathSeries = [
    { name: "Gateway Wait", values: chatTurns.map((turn) => Number(asObject(turn.critical_path).gatewayWaitMs || 0)) },
    {
      name: "Gateway Serialization",
      values: chatTurns.map((turn) => Number(asObject(turn.critical_path).gatewaySerializationMs || 0))
    },
    { name: "Stage Compute", values: chatTurns.map((turn) => Number(asObject(turn.critical_path).stageComputeMs || 0)) },
    { name: "Stage Transfer", values: chatTurns.map((turn) => Number(asObject(turn.critical_path).stageTransferMs || 0)) }
  ];

  els.chatMetricsTimeline.innerHTML = `
    <div class="invoke-metric-cards">
      ${chatCards
        .map(
          (card) => `
        <div class="invoke-metric-card">
          <div class="k">${escapeHtml(card.k)}</div>
          <div class="v">${escapeHtml(card.v)}</div>
        </div>
      `
        )
        .join("")}
    </div>
    <div class="invoke-subpanel">
      <h3>Critical Path Evolution Across Turns</h3>
      <div class="chart-grid">
        <div class="chart-card">
          <h4>Gateway Wait vs Serialization vs Stage Components</h4>
          ${buildLineChartSvg(criticalPathSeries, { yLabel: "ms" })}
        </div>
        <div class="chart-card">
          <h4>Critical Path Totals (session)</h4>
          <table class="metrics-table">
            <thead>
              <tr>
                <th>Component</th>
                <th>Total (ms)</th>
                <th>Avg/turn (ms)</th>
              </tr>
            </thead>
            <tbody>
              <tr><td>Gateway Wait</td><td>${formatNumber(criticalGatewayWait, 1)}</td><td>${formatNumber(criticalGatewayWait / Math.max(1, totalTurns), 1)}</td></tr>
              <tr><td>Gateway Serialization</td><td>${formatNumber(criticalGatewaySerialization, 1)}</td><td>${formatNumber(criticalGatewaySerialization / Math.max(1, totalTurns), 1)}</td></tr>
              <tr><td>Stage Compute</td><td>${formatNumber(criticalStageCompute, 1)}</td><td>${formatNumber(criticalStageCompute / Math.max(1, totalTurns), 1)}</td></tr>
              <tr><td>Stage Transfer</td><td>${formatNumber(criticalStageTransfer, 1)}</td><td>${formatNumber(criticalStageTransfer / Math.max(1, totalTurns), 1)}</td></tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>
    <div class="invoke-subpanel">
      <h3>Per-Node Time-Series Across Turns</h3>
      <div class="chart-grid">
        <div class="chart-card">
          <h4>Node Max Memory (MB)</h4>
          ${buildLineChartSvg(nodeMemorySeries, { yLabel: "MB" })}
        </div>
        <div class="chart-card">
          <h4>Node Network Delta (MiB)</h4>
          ${buildLineChartSvg(nodeNetworkSeries, { yLabel: "MiB" })}
        </div>
        <div class="chart-card">
          <h4>Node Transfer Time (ms)</h4>
          ${buildLineChartSvg(nodeTransferSeries, { yLabel: "ms" })}
        </div>
        <div class="chart-card">
          <h4>Node Compute Time (ms)</h4>
          ${buildLineChartSvg(nodeComputeSeries, { yLabel: "ms" })}
        </div>
      </div>
    </div>
    <div class="invoke-subpanel">
      <h3>Per-Stage Rollup Across Chat Session</h3>
      <table class="metrics-table">
        <thead>
          <tr>
            <th>Stage Key</th>
            <th>Service</th>
            <th>ID</th>
            <th>Samples</th>
            <th>Compute (ms)</th>
            <th>Transfer (ms)</th>
            <th>Max Mem (MB)</th>
            <th>Max CPU (%)</th>
            <th>Net Delta (MiB)</th>
            <th>Payload (MiB)</th>
          </tr>
        </thead>
        <tbody>${stageRows || "<tr><td colspan='10'>No per-stage data.</td></tr>"}</tbody>
      </table>
    </div>
    <div class="invoke-subpanel">
      <h3>Per-Node Rollup Across Chat Session</h3>
      <table class="metrics-table">
        <thead>
          <tr>
            <th>Node</th>
            <th>Pod Hostnames</th>
            <th>Services</th>
            <th>Stages</th>
            <th>Samples</th>
            <th>Compute (ms)</th>
            <th>Transfer (ms)</th>
            <th>Max Mem (MB)</th>
            <th>Max CPU (%)</th>
            <th>Max Sys CPU (%)</th>
            <th>Net Delta (MiB)</th>
            <th>Payload (MiB)</th>
            <th>Max Mbps</th>
            <th>Max Threads</th>
            <th>Ctx Sw</th>
            <th>IO Read Δ</th>
            <th>IO Write Δ</th>
          </tr>
        </thead>
        <tbody>${nodeRows || "<tr><td colspan='17'>No per-node data.</td></tr>"}</tbody>
      </table>
    </div>
    <div class="invoke-subpanel">
      <h3>Per-Turn Evolution</h3>
    <table class="metrics-table">
      <thead>
        <tr>
          <th>Turn</th>
          <th>Gen Tokens</th>
          <th>Latency (ms)</th>
          <th>Tok/s</th>
          <th>Tok p95 (ms)</th>
          <th>Xfer p95 (ms)</th>
          <th>Compute Sum (ms)</th>
          <th>Transfer Sum (ms)</th>
          <th>Max Mem (MB)</th>
          <th>Net Delta (MiB)</th>
          <th>Payload (MiB)</th>
          <th>Stop</th>
        </tr>
      </thead>
      <tbody>${rows}</tbody>
    </table>
    </div>
  `;
}

async function sendChatTurn() {
  if (chatBusy) {
    return;
  }

  const userText = (els.chatInput.value || "").trim();
  if (!userText) {
    els.chatMeta.innerHTML = `<div class="log-pill line-warn">Enter a message first.</div>`;
    return;
  }

  const namespace = getNamespace();
  const maxNewTokens = Math.max(
    1,
    Math.min(512, Number.parseInt(els.chatMaxTokens.value || "64", 10))
  );
  const minNewTokens = Math.max(
    1,
    Math.min(maxNewTokens, Number.parseInt(els.chatMinTokens.value || "8", 10))
  );
  const temperature = Math.max(
    0,
    Math.min(2, Number.parseFloat(els.chatTemperature.value || "0.2"))
  );
  const timeoutMs = parseEndpointTimeoutMs(
    els.chatTimeout?.value,
    INVOKE_TIMEOUT_LONG_DEFAULT_MS
  );

  const conversation = [...chatMessages, { role: "user", content: userText }];
  setChatBusy(true);
  els.chatMeta.innerHTML = "";

  try {
    const invokePayload = {
      namespace,
      target: "gateway",
      method: "POST",
      path: "/chat",
      timeoutMs,
      body: {
        messages: conversation,
        max_new_tokens: maxNewTokens,
        min_new_tokens: minNewTokens,
        temperature
      }
    };

    const res = await fetch("/api/invoke", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(invokePayload)
    });
    const text = await res.text();
    let payload = null;
    try {
      payload = text ? JSON.parse(text) : null;
    } catch {
      payload = null;
    }

    if (!res.ok || !payload?.responseJson) {
      const pieces = [`${res.status} ${res.statusText}`];
      if (payload?.error) {
        pieces.push(payload.error);
      }
      if (payload?.hint) {
        pieces.push(payload.hint);
      }
      throw new Error(pieces.join(": "));
    }

    const response = payload.responseJson;
    const assistantText = String(response.assistant_message || response.generated_text || "").trim();

    chatMessages = [...conversation, { role: "assistant", content: assistantText || "(empty response)" }];
    chatTurns.push(summarizeTurnMetrics(response));
    els.chatInput.value = "";

    renderChatMessages();
    renderChatMetricsTimeline();
    persistSessionState();

    els.chatMeta.innerHTML = `
      <div class="log-pill">turn: ${chatTurns.length}</div>
      <div class="log-pill">latency: ${formatNumber(response.total_latency_ms, 1)}ms</div>
      <div class="log-pill">tokens/s: ${formatNumber(response.tokens_per_second, 3)}</div>
      <div class="log-pill">termination: ${escapeHtml(response.termination_reason || "unknown")}</div>
    `;
  } catch (error) {
    els.chatMeta.innerHTML = `<div class="log-pill line-err">Chat error: ${escapeHtml(
      error.message
    )}</div>`;
  } finally {
    setChatBusy(false);
  }
}

function clearChatSession() {
  chatMessages = [];
  chatTurns = [];
  renderChatMessages();
  renderChatMetricsTimeline();
  persistSessionState();
  els.chatMeta.innerHTML = `<div class="log-pill">Chat session cleared.</div>`;
}

function exportChatMetrics() {
  const payload = {
    exported_at: new Date().toISOString(),
    namespace: getNamespace(),
    conversation: chatMessages,
    turns: chatTurns,
    endpoint_runs: invokeRuns
  };
  const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `dli-chat-metrics-${Date.now()}.json`;
  a.click();
  URL.revokeObjectURL(url);
}

function updateAutoRefresh() {
  if (refreshTimer) {
    clearInterval(refreshTimer);
    refreshTimer = null;
  }
  if (!els.autoRefresh.checked) {
    return;
  }

  const seconds = Math.max(2, Number.parseInt(els.refreshSeconds.value || "5", 10));
  refreshTimer = setInterval(() => {
    loadTopology();
    loadLogs();
  }, seconds * 1000);
}

els.refreshTopologyBtn.addEventListener("click", () => {
  loadTopology();
});

els.refreshLogsBtn.addEventListener("click", () => {
  loadLogs();
});

els.autoRefresh.addEventListener("change", updateAutoRefresh);
els.refreshSeconds.addEventListener("change", updateAutoRefresh);
els.logTarget.addEventListener("change", loadLogs);
els.logSource.addEventListener("change", loadLogs);
els.namespaceInput.addEventListener("change", () => {
  loadTopology();
  loadLogs();
  loadEndpointCatalog().catch((error) => {
    els.invokeMeta.innerHTML = `<div class="log-pill line-err">Catalog error: ${escapeHtml(
      error.message
    )}</div>`;
  });
});
els.endpointPreset.addEventListener("change", () => {
  applyEndpointPreset(getSelectedPreset());
});
els.endpointMethod.addEventListener("change", () => {
  syncGenerationControlsEnabledState();
  syncGenerationFieldsFromBody();
});
els.endpointPath.addEventListener("input", () => {
  syncGenerationControlsEnabledState();
  syncGenerationFieldsFromBody();
});
els.endpointBody.addEventListener("input", () => {
  syncGenerationFieldsFromBody();
});
els.endpointPrompt.addEventListener("input", syncBodyFromGenerationFields);
els.endpointMaxTokens.addEventListener("input", syncBodyFromGenerationFields);
els.endpointMinTokens.addEventListener("input", syncBodyFromGenerationFields);
els.endpointTemperature.addEventListener("input", syncBodyFromGenerationFields);
els.invokeEndpointBtn.addEventListener("click", invokeEndpoint);
els.chatSendBtn.addEventListener("click", sendChatTurn);
els.chatClearBtn.addEventListener("click", clearChatSession);
els.chatExportBtn.addEventListener("click", exportChatMetrics);
els.chatInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    sendChatTurn();
  }
});

const initialNamespace = new URLSearchParams(window.location.search).get("namespace");
if (initialNamespace) {
  els.namespaceInput.value = initialNamespace;
}

restoreSessionState();
if (invokeRuns.length) {
  els.invokeMetricsDashboard.innerHTML = renderInvokeRunEvolution();
}
loadTopology();
loadLogs();
loadEndpointCatalog().catch((error) => {
  els.invokeMeta.innerHTML = `<div class="log-pill line-err">Catalog error: ${escapeHtml(
    error.message
  )}</div>`;
});
renderChatMessages();
renderChatMetricsTimeline();
updateChatContextInfo();
syncGenerationControlsEnabledState();
updateAutoRefresh();
