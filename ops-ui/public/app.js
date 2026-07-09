"use strict";

const els = {
  clusterMeta: document.getElementById("clusterMeta"),
  overviewCards: document.getElementById("overviewCards"),
  pipeline: document.getElementById("pipeline"),
  workloadsTable: document.getElementById("workloadsTable"),
  runtimePanel: document.getElementById("runtimePanel"),
  namespaceInput: document.getElementById("namespaceInput"),
  runtimeVariant: document.getElementById("runtimeVariant"),
  refreshSeconds: document.getElementById("refreshSeconds"),
  autoRefresh: document.getElementById("autoRefresh"),
  refreshTopologyBtn: document.getElementById("refreshTopologyBtn"),
  architectureGuide: document.getElementById("architectureGuide"),
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
  endpointLabel: document.getElementById("endpointLabel"),
  endpointPrompt: document.getElementById("endpointPrompt"),
  endpointMaxTokens: document.getElementById("endpointMaxTokens"),
  endpointMinTokens: document.getElementById("endpointMinTokens"),
  endpointTemperature: document.getElementById("endpointTemperature"),
  endpointTopK: document.getElementById("endpointTopK"),
  endpointTopP: document.getElementById("endpointTopP"),
  endpointSeed: document.getElementById("endpointSeed"),
  endpointPrecision: document.getElementById("endpointPrecision"),
  featureTransportJson: document.getElementById("featureTransportJson"),
  featureTransportBinary: document.getElementById("featureTransportBinary"),
  featurePrecisionFp32: document.getElementById("featurePrecisionFp32"),
  featurePrecisionFp16: document.getElementById("featurePrecisionFp16"),
  featurePrecisionBf16: document.getElementById("featurePrecisionBf16"),
  featurePrecisionInt8: document.getElementById("featurePrecisionInt8"),
  featureRebalanceBaseline: document.getElementById("featureRebalanceBaseline"),
  featureRebalanceLatency: document.getElementById("featureRebalanceLatency"),
  featureKvCache: document.getElementById("featureKvCache"),
  featureForwardDedupe: document.getElementById("featureForwardDedupe"),
  featureTopologyAware: document.getElementById("featureTopologyAware"),
  featurePersistentSessions: document.getElementById("featurePersistentSessions"),
  featureBackpressure: document.getElementById("featureBackpressure"),
  featureBackpressureQueue: document.getElementById("featureBackpressureQueue"),
  featureFlagsResetBtn: document.getElementById("featureFlagsResetBtn"),
  endpointBody: document.getElementById("endpointBody"),
  invokeEndpointBtn: document.getElementById("invokeEndpointBtn"),
  invokeClearRunsBtn: document.getElementById("invokeClearRunsBtn"),
  invokeMeta: document.getElementById("invokeMeta"),
  endpointReceivedText: document.getElementById("endpointReceivedText"),
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
let featureModuleCatalog = [];
let latestInvokeFeatureFlags = null;
let latestInvokeEnabledModules = [];
let latestChatFeatureFlags = null;
let latestChatEnabledModules = [];
let runtimeSnapshot = null;
let runtimeVariantCatalog = [
  {
    id: "python",
    label: "PyTorch",
    description: "Python/FastAPI gateway and PyTorch stage pods",
    supportsChat: true,
    supportsFeatureFlags: true
  },
  {
    id: "native",
    label: "Native C++",
    description: "C++ gateway/stage pods using llama.cpp and DLI2 binary frames",
    supportsChat: false,
    supportsFeatureFlags: false
  }
];

const SESSION_STORAGE_KEY = "dli_ops_ui_session_v2";
const HISTORY_POST_BATCH_LIMIT = 1;
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
const ALERT_THRESHOLDS = {
  transferP95Ms: 9000,
  transferComputeRatio: 2.4,
  processCpuPct: 85,
  memoryMb: 2400
};

const FEATURE_FLAG_DEFAULTS = Object.freeze({
  transport_mode: "json_base64",
  activation_precision: "fp32",
  kv_cache_enabled: false,
  forward_dedupe_enabled: false,
  rebalance_profile: "baseline",
  topology_aware_routing: false,
  persistent_sessions_enabled: false,
  backpressure_enabled: false,
  backpressure_queue_size: 0
});

const BASELINE_WORKFLOW_STEPS = [
  "Gateway tokenizes prompt/messages and prepares the stage-1 request.",
  "Stage-1 runs its partition and forwards activations to stage-2.",
  "Stage-2 and stage-3 repeat compute + forward to the next stage.",
  "Stage-4 computes logits and returns next-token to gateway.",
  "Gateway appends token and loops until stop condition."
];

const MODULE_FALLBACK_CATALOG = [
  {
    key: "binary_transport",
    title: "Binary Activation Transport",
    summary:
      "Replace JSON+base64 activations with binary octet-stream envelopes between stages."
  },
  {
    key: "activation_precision",
    title: "Activation Precision Reduction",
    summary:
      "Send FP16/BF16/INT8 activations plus scale metadata, then restore before compute."
  },
  {
    key: "kv_cache",
    title: "Stage Transformer KV Cache",
    summary:
      "Run prefill once, then decode one token at a time with per-stage transformer KV state."
  },
  {
    key: "forward_dedupe_cache",
    title: "Forward Dedupe Cache",
    summary:
      "Optional retry/idempotency cache for duplicate forward requests."
  },
  {
    key: "rebalance",
    title: "Partition Rebalance Profile",
    summary:
      "Switch from baseline partition to a rebalance profile marker for staged experiments."
  },
  {
    key: "topology_aware",
    title: "Topology-Aware Routing",
    summary:
      "Allow topology-aware next-hop override policy for cross-node forwarding."
  },
  {
    key: "persistent_sessions_backpressure",
    title: "Persistent Sessions + Backpressure",
    summary:
      "Reuse HTTP sessions and enforce bounded single-flight/queue behavior."
  }
];

const NATIVE_MODULE_CATALOG = [
  {
    key: "llama_cpp_native",
    title: "Native llama.cpp Runtime",
    summary:
      "C++ gateway and stage pods execute GGUF partitions with llama.cpp-backed native code."
  },
  {
    key: "dli2_binary_frames",
    title: "DLI2 Binary Frames",
    summary:
      "The gateway and stages communicate through /forward-binary using binary DLI2 tensor frames."
  },
  {
    key: "gguf_partition_shards",
    title: "GGUF Partition Shards",
    summary:
      "Each native stage loads its partition-*.dli.gguf shard while the gateway owns tokenizer/model metadata."
  }
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

function getRuntimeVariant() {
  const value = String(els.runtimeVariant?.value || "python").trim().toLowerCase();
  return value === "native" ? "native" : "python";
}

function getRuntimeVariantInfo() {
  const selected = getRuntimeVariant();
  return (
    runtimeVariantCatalog.find((item) => item.id === selected) ||
    runtimeVariantCatalog.find((item) => item.id === "python") ||
    { id: "python", label: "PyTorch", supportsChat: true, supportsFeatureFlags: true }
  );
}

function runtimeQuery() {
  return `variant=${encodeURIComponent(getRuntimeVariant())}`;
}

function supportsFeatureFlags() {
  return getRuntimeVariantInfo().supportsFeatureFlags !== false;
}

function supportsChatEndpoint() {
  return getRuntimeVariantInfo().supportsChat !== false;
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

function floorNumber(value, min, fallback) {
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) {
    return fallback;
  }
  return Math.max(min, parsed);
}

function parseJsonOrNull(text) {
  try {
    return JSON.parse(text);
  } catch {
    return null;
  }
}

function toBool(value) {
  return value === true || String(value).toLowerCase() === "true";
}

function deriveEnabledModules(featureFlags, explicitModules) {
  const flags = asObject(featureFlags);
  const explicit = new Set(
    asArray(explicitModules).map((item) => String(item || "").trim()).filter(Boolean)
  );
  const derived = [
    ["binary_transport", flags.transport_mode === "binary_octet_stream"],
    ["activation_precision", String(flags.activation_precision || "fp32") !== "fp32"],
    ["kv_cache", toBool(flags.kv_cache_enabled)],
    ["forward_dedupe_cache", toBool(flags.forward_dedupe_enabled)],
    ["rebalance", String(flags.rebalance_profile || "baseline") !== "baseline"],
    ["topology_aware", toBool(flags.topology_aware_routing)],
    [
      "persistent_sessions_backpressure",
      toBool(flags.persistent_sessions_enabled) || toBool(flags.backpressure_enabled)
    ]
  ];

  for (const [key, enabled] of derived) {
    if (enabled) {
      explicit.add(key);
    }
  }

  return [...explicit].sort((a, b) => a.localeCompare(b));
}

function getFeatureFlagProfileChips(featureFlags) {
  const flags = asObject(featureFlags);
  if (!Object.keys(flags).length) {
    return [];
  }
  return [
    `transport=${safeText(flags.transport_mode || "json_base64")}`,
    `precision=${safeText(flags.activation_precision || "fp32")}`,
    `kv_cache=${toBool(flags.kv_cache_enabled) ? "on" : "off"}`,
    `dedupe=${toBool(flags.forward_dedupe_enabled) ? "on" : "off"}`,
    `rebalance=${safeText(flags.rebalance_profile || "baseline")}`,
    `topology=${toBool(flags.topology_aware_routing) ? "on" : "off"}`,
    `sessions=${toBool(flags.persistent_sessions_enabled) ? "on" : "off"}`,
    `backpressure=${toBool(flags.backpressure_enabled) ? "on" : "off"}`,
    `queue=${safeText(flags.backpressure_queue_size ?? "-")}`
  ];
}

function buildHistoryOutputPayload(responseJson) {
  const response = asObject(responseJson);
  const generatedText = String(response.generated_text || "");
  const assistantMessage = String(response.assistant_message || "");
  const terminationReason = String(response.termination_reason || "");
  return {
    generatedText,
    assistantMessage,
    terminationReason
  };
}

function extractReceivedText(responseJson) {
  const response = asObject(responseJson);
  const assistant = String(response.assistant_message || "").trim();
  const generated = String(response.generated_text || "").trim();
  const text = assistant || generated;
  if (text) {
    return text;
  }
  if (response.next_token_text != null) {
    return String(response.next_token_text);
  }
  return "";
}

function renderEndpointReceivedText(responseJson) {
  if (!els.endpointReceivedText) {
    return;
  }
  const text = extractReceivedText(responseJson);
  if (!text) {
    els.endpointReceivedText.classList.add("muted");
    els.endpointReceivedText.textContent = "No generated text field was returned by this endpoint.";
    return;
  }
  els.endpointReceivedText.classList.remove("muted");
  els.endpointReceivedText.textContent = text;
}

function clearEndpointReceivedText(message = "Run a text generation endpoint to see the returned text.") {
  if (!els.endpointReceivedText) {
    return;
  }
  els.endpointReceivedText.classList.add("muted");
  els.endpointReceivedText.textContent = message;
}

function buildTopologyHash() {
  const parts = [...podToNodeMap.entries()]
    .map(([pod, node]) => `${pod}:${node}`)
    .sort((a, b) => a.localeCompare(b));
  return parts.join("|");
}

function parseGenerateConfigFromBody(bodyValue) {
  const parsed = asObject(bodyValue);
  return {
    promptChars: String(parsed.prompt || "").length,
    maxNewTokens: Number(parsed.max_new_tokens || 0),
    minNewTokens: Number(parsed.min_new_tokens || 0),
    temperature: Number(parsed.temperature || 0),
    topK: Number(parsed.top_k || 0),
    topP: parsed.top_p != null ? Number(parsed.top_p) : 1,
    seed: parsed.seed != null && parsed.seed !== "" ? Number(parsed.seed) : null
  };
}

// Raw per-token/per-step traces and the redundant session-evolution snapshot are
// huge (~1.5 MB+ per run) and unused by the History page, which renders from
// metrics/config/output and the dashboard summaries. Persisting them inflates
// each entry past the server's POST body limit, causing the save to fail and the
// run to silently go missing from History. Strip them before persisting.
const HISTORY_HEAVY_RESPONSE_FIELDS = [
  "token_metrics",
  "steps",
  "prompt_token_ids",
  "generated_token_ids"
];
const HISTORY_HEAVY_DASHBOARD_FIELDS = ["responseJson", "sessionRunEvolution"];

function slimHistoryResponseJson(responseJsonValue) {
  const response = asObject(responseJsonValue);
  if (!Object.keys(response).length) {
    return undefined;
  }
  const slim = { ...response };
  for (const field of HISTORY_HEAVY_RESPONSE_FIELDS) {
    delete slim[field];
  }
  return slim;
}

function slimHistoryEntry(entry) {
  const slimmed = { ...entry };
  const dashboard = asObject(entry.dashboard);
  if (Object.keys(dashboard).length) {
    const slimDashboard = { ...dashboard };
    for (const field of HISTORY_HEAVY_DASHBOARD_FIELDS) {
      delete slimDashboard[field];
    }
    const slimResponse = slimHistoryResponseJson(dashboard.responseJson);
    if (slimResponse) {
      slimDashboard.responseJson = slimResponse;
    }
    slimmed.dashboard = slimDashboard;
  }
  return slimmed;
}

async function postHistoryEntries(entries) {
  const rows = asArray(entries).filter((item) => item && typeof item === "object");
  if (!rows.length) {
    return;
  }
  const batched = rows.slice(0, HISTORY_POST_BATCH_LIMIT).map(slimHistoryEntry);
  try {
    const res = await fetch("/api/history", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ entries: batched })
    });
    if (!res.ok) {
      const detail = await res.text().catch(() => "");
      console.error(`History persistence failed: ${res.status} ${res.statusText} ${detail}`);
    }
  } catch (error) {
    console.error("History persistence request failed:", error);
  }
}

function makeAlertRibbons(alerts) {
  if (!alerts.length) {
    return "";
  }
  return `
    <div class="invoke-subpanel">
      <h3>Alerts</h3>
      <div class="alert-ribbons">
        ${alerts
          .map(
            (item) => `
            <div class="alert-ribbon ${escapeHtml(item.level || "info")}">${escapeHtml(item.message)}</div>
          `
          )
          .join("")}
      </div>
    </div>
  `;
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

function getCheckedRadioValue(name, fallback) {
  const selected = document.querySelector(`input[name="${name}"]:checked`);
  return selected ? String(selected.value) : String(fallback);
}

function setCheckedRadioValue(name, value, fallback) {
  const next = String(value == null ? fallback : value);
  const wanted = document.querySelector(`input[name="${name}"][value="${next}"]`);
  if (wanted) {
    wanted.checked = true;
    return;
  }
  const fallbackInput = document.querySelector(
    `input[name="${name}"][value="${String(fallback)}"]`
  );
  if (fallbackInput) {
    fallbackInput.checked = true;
  }
}

function normalizeFeatureFlags(rawFlags) {
  const flags = asObject(rawFlags);
  const transportMode =
    String(flags.transport_mode || FEATURE_FLAG_DEFAULTS.transport_mode) ===
    "binary_octet_stream"
      ? "binary_octet_stream"
      : "json_base64";
  const precisionCandidates = new Set(["fp32", "fp16", "bf16", "int8"]);
  const activationPrecision = String(
    flags.activation_precision || FEATURE_FLAG_DEFAULTS.activation_precision
  );
  const precisionMode = precisionCandidates.has(activationPrecision)
    ? activationPrecision
    : FEATURE_FLAG_DEFAULTS.activation_precision;
  const rebalanceProfile =
    String(flags.rebalance_profile || FEATURE_FLAG_DEFAULTS.rebalance_profile) ===
    "latency_balanced_v1"
      ? "latency_balanced_v1"
      : "baseline";
  const queueSize = Math.round(
    clampNumber(
      flags.backpressure_queue_size,
      0,
      256,
      FEATURE_FLAG_DEFAULTS.backpressure_queue_size
    )
  );
  return {
    transport_mode: transportMode,
    activation_precision: precisionMode,
    kv_cache_enabled: toBool(flags.kv_cache_enabled),
    forward_dedupe_enabled: toBool(flags.forward_dedupe_enabled),
    rebalance_profile: rebalanceProfile,
    topology_aware_routing: toBool(flags.topology_aware_routing),
    persistent_sessions_enabled: toBool(flags.persistent_sessions_enabled),
    backpressure_enabled: toBool(flags.backpressure_enabled),
    backpressure_queue_size: queueSize
  };
}

function isBaselineFeatureFlags(flags) {
  const normalized = normalizeFeatureFlags(flags);
  return (
    normalized.transport_mode === FEATURE_FLAG_DEFAULTS.transport_mode &&
    normalized.activation_precision === FEATURE_FLAG_DEFAULTS.activation_precision &&
    normalized.kv_cache_enabled === FEATURE_FLAG_DEFAULTS.kv_cache_enabled &&
    normalized.forward_dedupe_enabled === FEATURE_FLAG_DEFAULTS.forward_dedupe_enabled &&
    normalized.rebalance_profile === FEATURE_FLAG_DEFAULTS.rebalance_profile &&
    normalized.topology_aware_routing === FEATURE_FLAG_DEFAULTS.topology_aware_routing &&
    normalized.persistent_sessions_enabled ===
      FEATURE_FLAG_DEFAULTS.persistent_sessions_enabled &&
    normalized.backpressure_enabled === FEATURE_FLAG_DEFAULTS.backpressure_enabled &&
    normalized.backpressure_queue_size === FEATURE_FLAG_DEFAULTS.backpressure_queue_size
  );
}

function readFeatureFlagsFromControls() {

  if (getRuntimeVariant() === "native") {
    const activationPrecision = getCheckedRadioValue(
      "feature-precision",
      "fp32"
    );

    const rebalanceProfile = getCheckedRadioValue(
      "feature-rebalance",
      "baseline"
    );

    return normalizeFeatureFlags({
      transport_mode: "binary_octet_stream",
      activation_precision: activationPrecision,
      kv_cache_enabled: true,
      forward_dedupe_enabled: false,
      rebalance_profile: rebalanceProfile,
      topology_aware_routing: false,
      persistent_sessions_enabled: false,
      backpressure_enabled: false,
      backpressure_queue_size: 0
    });
  }

  const backpressureEnabled = toBool(els.featureBackpressure?.checked);
  const queueSize = Math.round(
    clampNumber(els.featureBackpressureQueue?.value, 0, 256, 0)
  );
  return normalizeFeatureFlags({
    transport_mode: getCheckedRadioValue(
      "feature-transport",
      FEATURE_FLAG_DEFAULTS.transport_mode
    ),
    activation_precision: getCheckedRadioValue(
      "feature-precision",
      FEATURE_FLAG_DEFAULTS.activation_precision
    ),
    kv_cache_enabled: toBool(els.featureKvCache?.checked),
    forward_dedupe_enabled: toBool(els.featureForwardDedupe?.checked),
    rebalance_profile: getCheckedRadioValue(
      "feature-rebalance",
      FEATURE_FLAG_DEFAULTS.rebalance_profile
    ),
    topology_aware_routing: toBool(els.featureTopologyAware?.checked),
    persistent_sessions_enabled: toBool(els.featurePersistentSessions?.checked),
    backpressure_enabled: backpressureEnabled,
    backpressure_queue_size: backpressureEnabled ? queueSize : 0
  });
}

function applyFeatureFlagControls(flags) {
  const next = normalizeFeatureFlags(flags);
  setCheckedRadioValue(
    "feature-transport",
    next.transport_mode,
    FEATURE_FLAG_DEFAULTS.transport_mode
  );
  setCheckedRadioValue(
    "feature-precision",
    next.activation_precision,
    FEATURE_FLAG_DEFAULTS.activation_precision
  );
  setCheckedRadioValue(
    "feature-rebalance",
    next.rebalance_profile,
    FEATURE_FLAG_DEFAULTS.rebalance_profile
  );
  if (els.featureKvCache) {
    els.featureKvCache.checked = toBool(next.kv_cache_enabled);
  }
  if (els.featureForwardDedupe) {
    els.featureForwardDedupe.checked = toBool(next.forward_dedupe_enabled);
  }
  if (els.featureTopologyAware) {
    els.featureTopologyAware.checked = toBool(next.topology_aware_routing);
  }
  if (els.featurePersistentSessions) {
    els.featurePersistentSessions.checked = toBool(next.persistent_sessions_enabled);
  }
  if (els.featureBackpressure) {
    els.featureBackpressure.checked = toBool(next.backpressure_enabled);
  }
  if (els.featureBackpressureQueue) {
    els.featureBackpressureQueue.value = String(next.backpressure_queue_size);
  }
  syncFeatureFlagControlState();
}

function syncFeatureFlagControlState() {
  if (!els.featureBackpressureQueue) {
    return;
  }
  const enabled = toBool(els.featureBackpressure?.checked);
  els.featureBackpressureQueue.disabled = !enabled;
}

function syncRuntimeUiState() {
  const featurePanel = document.querySelector(".feature-flags-panel");
  const featureControls = [
    els.featureTransportJson,
    els.featureTransportBinary,
    els.featurePrecisionFp32,
    els.featurePrecisionFp16,
    els.featurePrecisionBf16,
    els.featurePrecisionInt8,
    els.featureRebalanceBaseline,
    els.featureRebalanceLatency,
    els.featureKvCache,
    els.featureForwardDedupe,
    els.featureTopologyAware,
    els.featurePersistentSessions,
    els.featureBackpressure,
    els.featureBackpressureQueue,
    els.featureFlagsResetBtn
  ].filter(Boolean);
  const featureFlagsEnabled = supportsFeatureFlags();
  for (const control of featureControls) {
    control.disabled = !featureFlagsEnabled;
  }
  if (featurePanel) {
    featurePanel.hidden = !featureFlagsEnabled;
    featurePanel.classList.toggle("is-disabled", !featureFlagsEnabled);
  }
  if (featureFlagsEnabled) {
    syncFeatureFlagControlState();
  }
}

function syncFeatureFlagControlsFromBody() {
  const parsed = parseJsonOrNull(els.endpointBody.value || "");
  if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
    return;
  }
  applyFeatureFlagControls(asObject(parsed.feature_flags));
}

function syncBodyFromFeatureFlagControls() {
  if (!supportsFeatureFlags()) {
    return;
  }

  const method = String(els.endpointMethod.value || "").toUpperCase();
  if (method === "GET" || method === "HEAD") {
    return;
  }

  const parsed = parseJsonOrNull(els.endpointBody.value || "");
  if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
    return;
  }

  const nextBody = { ...parsed };
  const flags = readFeatureFlagsFromControls();

  if (isBaselineFeatureFlags(flags)) {
    delete nextBody.feature_flags;
  } else {
    nextBody.feature_flags = flags;
  }
  els.endpointBody.value = prettyJson(nextBody);
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
      floorNumber(parsed.max_new_tokens, 1, 24)
    );
  }
  if (parsed.min_new_tokens != null) {
    els.endpointMinTokens.value = String(
      floorNumber(parsed.min_new_tokens, 1, 8)
    );
  }
  if (parsed.temperature != null) {
    els.endpointTemperature.value = String(
      clampNumber(parsed.temperature, 0, 2, 0.2)
    );
  }
  if (els.endpointTopK && parsed.top_k != null) {
    els.endpointTopK.value = String(floorNumber(parsed.top_k, 0, 0));
  }
  if (els.endpointTopP && parsed.top_p != null) {
    els.endpointTopP.value = String(clampNumber(parsed.top_p, 0, 1, 1));
  }
  if (els.endpointSeed) {
    els.endpointSeed.value =
      parsed.seed != null && parsed.seed !== "" ? String(Math.round(Number(parsed.seed))) : "";
  }
  if (els.endpointPrecision) {
    els.endpointPrecision.value =
      String(parsed.activation_precision || "fp32") === "fp16" ? "fp16" : "fp32";
  }

  applyFeatureFlagControls(asObject(parsed.feature_flags));
}

function syncBodyFromGenerationFields() {
  if (!isGenerationEndpoint()) {
    return;
  }

  const current = parseJsonOrNull(els.endpointBody.value || "");
  if (!current || typeof current !== "object" || Array.isArray(current)) {
    return;
  }

  const maxNewTokens = floorNumber(els.endpointMaxTokens.value, 1, 24);
  const minNewTokens = floorNumber(els.endpointMinTokens.value, 1, 8);
  const temperature = clampNumber(els.endpointTemperature.value, 0, 2, 0.2);

  els.endpointMinTokens.value = String(minNewTokens);

  const nextBody = {
    ...current,
    prompt: String(els.endpointPrompt.value || ""),
    max_new_tokens: Math.round(maxNewTokens),
    min_new_tokens: Math.round(minNewTokens),
    temperature
  };

  // top_k (0 = off) and top_p (>=1 = off) are sampling controls the gateway
  // forwards to the terminal stage. Include them only when active so baseline
  // greedy bodies stay clean; the gateway treats an absent top_p as disabled.
  const topK = els.endpointTopK ? floorNumber(els.endpointTopK.value, 0, 0) : 0;
  if (topK > 0) {
    nextBody.top_k = Math.round(topK);
  } else {
    delete nextBody.top_k;
  }

  const topP = els.endpointTopP ? clampNumber(els.endpointTopP.value, 0, 1, 1) : 1;
  if (topP > 0 && topP < 1) {
    nextBody.top_p = topP;
  } else {
    delete nextBody.top_p;
  }

  // seed: blank field => omit (rolling random). A value (incl. -1) => send it;
  // >= 0 is reproducible, -1 forces a fresh random stream each run.
  const seedRaw = els.endpointSeed ? String(els.endpointSeed.value).trim() : "";
  if (seedRaw !== "" && Number.isFinite(Number(seedRaw))) {
    nextBody.seed = Math.round(Number(seedRaw));
  } else {
    delete nextBody.seed;
  }

  // The native gateway reads activation_precision from the top level of the
  // body. Drop it when fp32 to keep baseline bodies clean.
  const precision = els.endpointPrecision
    ? String(els.endpointPrecision.value || "fp32")
    : "fp32";
  if (precision === "fp16") {
    nextBody.activation_precision = "fp16";
  } else {
    delete nextBody.activation_precision;
  }

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
  const normalizedBody =
    bodyValue && typeof bodyValue === "object" && !Array.isArray(bodyValue)
      ? { ...bodyValue }
      : bodyValue;
  if (
    normalizedBody &&
    typeof normalizedBody === "object" &&
    !Array.isArray(normalizedBody) &&
    !supportsFeatureFlags()
  ) {
    delete normalizedBody.feature_flags;
  }
  els.endpointBody.value = prettyJson(normalizedBody);
  syncGenerationControlsEnabledState();
  syncGenerationFieldsFromBody();
}

async function loadEndpointCatalog() {
  const namespace = getNamespace();
  const catalog = await fetchJson(
    `/api/endpoint-catalog?namespace=${encodeURIComponent(namespace)}&${runtimeQuery()}`
  );
  endpointCatalog = catalog;
  if (Array.isArray(catalog.runtimeVariants) && catalog.runtimeVariants.length) {
    runtimeVariantCatalog = catalog.runtimeVariants;
  }
  if (catalog.runtimeVariant?.id && els.runtimeVariant) {
    els.runtimeVariant.value = catalog.runtimeVariant.id;
  }

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
  syncRuntimeUiState();
}

function getModuleCatalogRows() {
  if (featureModuleCatalog.length) {
    return featureModuleCatalog;
  }
  if (getRuntimeVariant() === "native") {
    return NATIVE_MODULE_CATALOG;
  }
  return MODULE_FALLBACK_CATALOG;
}

function renderArchitectureGuide() {
  if (!els.architectureGuide) {
    return;
  }

  const isNativeRuntime = getRuntimeVariant() === "native";
  const runtimeInfo = getRuntimeVariantInfo();
  const invokeEnabled = deriveEnabledModules(
    latestInvokeFeatureFlags,
    latestInvokeEnabledModules
  );
  const chatEnabled = deriveEnabledModules(latestChatFeatureFlags, latestChatEnabledModules);
  const invokeSet = new Set(invokeEnabled);
  const chatSet = new Set(chatEnabled);

  const baselineChips = isNativeRuntime
    ? [
        "C++ gateway",
        "llama.cpp GGUF",
        "DLI2 binary frames",
        "/forward-binary",
        "native stage shards"
      ]
    : [
        "json_base64 transport",
        "fp32 activations",
        "no stage cache",
        "baseline partition",
        "static next-hop",
        "default HTTP requests"
      ];
  const workflowSteps = isNativeRuntime
    ? [
        "Gateway tokenizes the prompt and prepares a DLI2 binary tensor frame.",
        "Native stage-1 executes embedding/layers from its GGUF shard.",
        "Gateway routes the returned frame through native stage-2 and stage-3.",
        "Native stage-4 executes terminal norm/lm-head work and returns next-token metadata.",
        "Gateway detokenizes generated IDs and reports native aggregate/step metrics."
      ]
    : BASELINE_WORKFLOW_STEPS;

  const moduleRows = getModuleCatalogRows();
  const moduleCards = moduleRows
    .map((moduleRow) => {
      const key = String(moduleRow.key || "");
      const title = String(moduleRow.title || key);
      const summary = String(moduleRow.summary || "");
      const invokeOn = invokeSet.has(key);
      const chatOn = chatSet.has(key);
      return `
        <div class="module-card">
          <div class="module-title-row">
            <h3>${escapeHtml(title)}</h3>
            <span class="module-key">${escapeHtml(key)}</span>
          </div>
          <div class="status-badges">
            <span class="status-badge ${invokeOn ? "on" : "off"}">Endpoint: ${invokeOn ? "on" : "off"}</span>
            <span class="status-badge ${chatOn ? "on" : "off"}">Chat: ${chatOn ? "on" : "off"}</span>
          </div>
          <p>${escapeHtml(summary)}</p>
        </div>
      `;
    })
    .join("");

  const invokeProfile = getFeatureFlagProfileChips(latestInvokeFeatureFlags);
  const chatProfile = getFeatureFlagProfileChips(latestChatFeatureFlags);

  els.architectureGuide.innerHTML = `
    <div class="architecture-overview">
      <div class="architecture-card">
        <h3>${isNativeRuntime ? "Native C++ Runtime" : "Baseline Architecture"}</h3>
        <p>${escapeHtml(runtimeInfo.description || "Current runtime target.")}</p>
        <ol class="arch-flow">
          ${workflowSteps.map((step) => `<li>${escapeHtml(step)}</li>`).join("")}
        </ol>
        <div class="profile-row">
          ${baselineChips.map((chip) => `<span class="profile-chip">${escapeHtml(chip)}</span>`).join("")}
        </div>
      </div>
      <div class="architecture-card">
        <h3>Current Activation Profile</h3>
        <p>Live profile extracted from the latest endpoint/chat responses.</p>
        <div class="profile-row">
          <span class="status-badge ${invokeEnabled.length ? "on" : "off"}">
            Endpoint modules: ${invokeEnabled.length ? invokeEnabled.join(", ") : "baseline-only"}
          </span>
        </div>
        <div class="profile-row">
          ${
            invokeProfile.length
              ? invokeProfile.map((chip) => `<span class="profile-chip">${escapeHtml(chip)}</span>`).join("")
              : `<span class="profile-chip">No endpoint profile yet</span>`
          }
        </div>
        <div class="profile-row">
          <span class="status-badge ${chatEnabled.length ? "on" : "off"}">
            Chat modules: ${chatEnabled.length ? chatEnabled.join(", ") : "baseline-only"}
          </span>
        </div>
        <div class="profile-row">
          ${
            chatProfile.length
              ? chatProfile.map((chip) => `<span class="profile-chip">${escapeHtml(chip)}</span>`).join("")
              : `<span class="profile-chip">No chat profile yet</span>`
          }
        </div>
      </div>
    </div>
    <div class="module-grid">
      ${moduleCards}
    </div>
  `;
}

async function loadFeatureModuleCatalog() {
  if (!supportsFeatureFlags()) {
    featureModuleCatalog = [];
    renderArchitectureGuide();
    return;
  }

  const namespace = getNamespace();
  try {
    const res = await fetch("/api/invoke", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        namespace,
        variant: getRuntimeVariant(),
        target: "gateway",
        method: "GET",
        path: "/config",
        timeoutMs: 10000
      })
    });
    if (!res.ok) {
      renderArchitectureGuide();
      return;
    }
    const payload = await res.json();
    const moduleRows = asArray(payload?.responseJson?.feature_modules).filter(
      (item) => item && typeof item === "object" && item.key
    );
    if (moduleRows.length) {
      featureModuleCatalog = moduleRows;
    }
  } catch {
    // keep fallback catalog
  }
  renderArchitectureGuide();
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
  if (!els.pipeline) {
    return;
  }

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

function renderRuntimePanel() {
  if (!els.runtimePanel) {
    return;
  }
  const snapshot = asObject(runtimeSnapshot);
  if (!Object.keys(snapshot).length) {
    els.runtimePanel.innerHTML = `<div class="muted">Runtime telemetry not available yet.</div>`;
    return;
  }
  const byPath = asObject(snapshot.byPath);
  const cards = [
    { k: "In Flight", v: String(snapshot.inFlight || 0) },
    { k: "Queue Depth", v: String(snapshot.queueDepth || 0) },
    { k: "Mode", v: safeText(snapshot.queueMode || "-") },
    { k: "Total Invokes", v: String(snapshot.totalInvokes || 0) },
    { k: "Successful", v: String(snapshot.successfulInvokes || 0) },
    { k: "Failed", v: String(snapshot.failedInvokes || 0) },
    { k: "Dropped (busy)", v: String(snapshot.droppedInFlight || 0) },
    { k: "Timeout Failures", v: String(snapshot.timeoutFailures || 0) }
  ];

  const pathRows = Object.entries(byPath)
    .sort((a, b) => a[0].localeCompare(b[0]))
    .map(([pathKey, row]) => {
      const p = asObject(row);
      return `<li><strong>${escapeHtml(pathKey)}</strong>: total=${safeText(p.total || 0)} success=${safeText(
        p.success || 0
      )} failed=${safeText(p.failed || 0)} dropped=${safeText(p.dropped || 0)} timeout=${safeText(p.timeout || 0)}</li>`;
    })
    .join("");

  const inFlightRows = asArray(snapshot.inFlightRequests)
    .map(
      (item) => `
      <li>${escapeHtml(item.method)} ${escapeHtml(item.path)} runtime=${escapeHtml(
        item.variant || "-"
      )} target=${escapeHtml(item.target)} inFlight=${formatNumber(
        item.inFlightMs,
        0
      )}ms</li>
    `
    )
    .join("");

  const timeoutRows = asArray(snapshot.lastTimeoutCauses)
    .slice(0, 8)
    .map(
      (item) =>
        `<li>${escapeHtml(item.at || "-")} • ${escapeHtml(item.cause || "timeout")}</li>`
    )
    .join("");

  els.runtimePanel.innerHTML = `
    <div class="runtime-cards">
      ${cards
        .map(
          (card) => `
          <div class="runtime-card">
            <div class="k">${escapeHtml(card.k)}</div>
            <div class="v">${escapeHtml(card.v)}</div>
          </div>
        `
        )
        .join("")}
    </div>
    <div class="chart-grid">
      <div class="runtime-list">
        <strong>In-Flight Requests</strong>
        <ul>${inFlightRows || "<li>none</li>"}</ul>
      </div>
      <div class="runtime-list">
        <strong>Recent Timeout Causes</strong>
        <ul>${timeoutRows || "<li>none</li>"}</ul>
      </div>
      <div class="runtime-list">
        <strong>Path Counters</strong>
        <ul>${pathRows || "<li>none</li>"}</ul>
      </div>
    </div>
  `;
}

async function loadRuntime() {
  if (!els.runtimePanel) {
    return;
  }
  const namespace = getNamespace();
  try {
    const payload = await fetchJson(
      `/api/runtime?namespace=${encodeURIComponent(namespace)}&${runtimeQuery()}`
    );
    runtimeSnapshot = asObject(payload.runtime);
    renderRuntimePanel();
  } catch (error) {
    els.runtimePanel.innerHTML = `<div class="log-pill line-err">Runtime error: ${escapeHtml(
      error.message
    )}</div>`;
  }
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
    const transferMs = Number(sample?.rpc_wall_time_ms ?? sample?.transfer_time_ms ?? 0);
    if (stageId === 0) {
      gatewaySerializationMs += computeMs;
      gatewayWaitMs += transferMs;
    } else {
      stageComputeMs += computeMs;
      stageTransferMs += transferMs;
    }
  }

  if (gatewayWaitMs <= 0 && stageComputeMs + stageTransferMs > 0) {
    gatewayWaitMs = stageComputeMs + stageTransferMs;
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

function buildCriticalPathSankey(criticalPath, totalLatencyMs) {
  const total = Math.max(1e-9, Number(totalLatencyMs || 0));
  const rows = [
    {
      key: "Gateway Serialize",
      css: "gateway-serialize",
      value: Number(criticalPath.gatewaySerializationMs || 0)
    },
    {
      key: "Gateway Wait",
      css: "gateway-wait",
      value: Number(criticalPath.gatewayWaitMs || 0)
    },
    {
      key: "Stage Compute",
      css: "stage-compute",
      value: Number(criticalPath.stageComputeMs || 0)
    },
    {
      key: "Stage RPC Wall",
      css: "stage-transfer",
      value: Number(criticalPath.stageTransferMs || 0)
    },
    {
      key: "Residual",
      css: "residual",
      value: Number(criticalPath.stageResidualMs || 0)
    }
  ];

  const segments = rows
    .filter((row) => row.value > 0)
    .map((row) => {
      const pct = Math.max(0.25, (100 * row.value) / total);
      return `<div class="sankey-segment ${row.css}" style="width:${pct}%"></div>`;
    })
    .join("");
  const legend = rows
    .map((row) => {
      const pct = (100 * row.value) / total;
      return `
        <div class="sankey-legend-row">
          <span class="sankey-legend-swatch ${row.css}"></span>
          <span>${escapeHtml(row.key)}</span>
          <span>${formatNumber(row.value, 1)} ms (${formatNumber(pct, 2)}%)</span>
        </div>
      `;
    })
    .join("");

  return `
    <div class="sankey">
      <div class="sankey-track">${segments || `<div class="sankey-segment gateway-wait" style="width:100%"></div>`}</div>
      <div class="sankey-legend">${legend}</div>
    </div>
  `;
}

function buildStageTokenHeatmap(tokenMetrics) {
  const stageMap = new Map();
  const tokens = asArray(tokenMetrics);
  for (const tokenStep of tokens) {
    const tokenIndex = Number(tokenStep.token_index || 0);
    for (const sample of asArray(tokenStep.stage_metrics)) {
      const stageKey = stageLabelFromMetric(sample);
      const stage = stageMap.get(stageKey) || {
        stageKey,
        service: String(sample.service_name || stageKey),
        stageId: Number(sample.stage_id || 0),
        byToken: new Map()
      };
      const latencyValue =
        Number(sample.compute_time_ms || 0) +
        Number(sample.rpc_wall_time_ms ?? sample.transfer_time_ms ?? 0);
      stage.byToken.set(tokenIndex, (stage.byToken.get(tokenIndex) || 0) + latencyValue);
      stageMap.set(stageKey, stage);
    }
  }

  const stages = [...stageMap.values()].sort((a, b) => a.stageId - b.stageId);
  if (!stages.length || !tokens.length) {
    return `<div class="muted">No token/stage samples for heatmap.</div>`;
  }
  const tokenIndices = tokens.map((step) => Number(step.token_index || 0));
  const maxLatency = Math.max(
    1,
    ...stages.flatMap((stage) => tokenIndices.map((idx) => Number(stage.byToken.get(idx) || 0)))
  );
  const headerCells = tokenIndices.map((idx) => `<th>T${idx}</th>`).join("");
  const rows = stages
    .map((stage) => {
      const cells = tokenIndices
        .map((idx) => {
          const value = Number(stage.byToken.get(idx) || 0);
          const alpha = Math.min(0.95, 0.1 + 0.85 * (value / maxLatency));
          const color = `rgba(93,178,255,${alpha})`;
          return `<td class="heatmap-cell" style="background:${color}">${formatNumber(value, 1)}</td>`;
        })
        .join("");
      return `<tr><td>${escapeHtml(stage.stageKey)}</td>${cells}</tr>`;
    })
    .join("");

  return `
    <div class="heatmap-wrap">
      <table class="heatmap-table">
        <thead><tr><th>Stage</th>${headerCells}</tr></thead>
        <tbody>${rows}</tbody>
      </table>
    </div>
  `;
}

function buildInvokeAlerts({
  transferP95Ms,
  transferComputeRatio,
  maxProcessCpu,
  maxMemoryMb,
  maxCgroupPercent = 0,
  maxSessionCount = 0,
  runtime
}) {
  const alerts = [];

  if (Number(transferP95Ms || 0) > ALERT_THRESHOLDS.transferP95Ms) {
    alerts.push({
      level: "warn",
      message: `RPC wall p95 is high (${formatNumber(transferP95Ms, 1)} ms). Blocking stage waits dominate token latency.`
    });
  }

  if (Number(maxCgroupPercent || 0) >= 85) {
    alerts.push({
      level: "warn",
      message: `Container memory pressure: cgroup memory reached ${formatNumber(maxCgroupPercent, 1)}%.`
    });
  }

  if (Number(maxSessionCount || 0) > 4) {
    alerts.push({
      level: "warn",
      message: `Native stage retained ${Math.round(maxSessionCount)} sessions. Check request cleanup/session pruning.`
    });
  }

  if (Number(transferComputeRatio || 0) > ALERT_THRESHOLDS.transferComputeRatio) {
    alerts.push({
      level: "warn",
      message: `True communication/compute ratio ${formatNumber(
        transferComputeRatio,
        3
      )} is above target (${ALERT_THRESHOLDS.transferComputeRatio}).`
    });
  }

  if (Number(maxProcessCpu || 0) >= ALERT_THRESHOLDS.processCpuPct) {
    alerts.push({
      level: "err",
      message: `Node CPU saturation detected (max process CPU ${formatNumber(maxProcessCpu, 1)}%).`
    });
  }

  if (Number(maxMemoryMb || 0) >= ALERT_THRESHOLDS.memoryMb) {
    alerts.push({
      level: "warn",
      message: `Memory cliff risk: peak process memory ${formatNumber(maxMemoryMb, 1)} MB.`
    });
  }

  const inFlight = Number(asObject(runtime).inFlight || 0);
  const dropped = Number(asObject(runtime).droppedInFlight || 0);
  if (inFlight > 0) {
    alerts.push({
      level: "info",
      message: `Queue state: ${inFlight} request(s) currently in flight (drop-on-busy mode).`
    });
  }
  if (dropped > 0) {
    alerts.push({
      level: "warn",
      message: `Dropped requests due to busy pipeline: ${dropped}.`
    });
  }

  return alerts;
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
    runtimeVariant: getRuntimeVariant(),
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
  if (parsed.runtimeVariant && els.runtimeVariant) {
    const nextVariant = String(parsed.runtimeVariant).toLowerCase() === "native" ? "native" : "python";
    els.runtimeVariant.value = nextVariant;
  }

  const lastInvoke = invokeRuns.length ? asObject(invokeRuns[invokeRuns.length - 1]) : {};
  latestInvokeFeatureFlags = asObject(lastInvoke.feature_flags);
  latestInvokeEnabledModules = deriveEnabledModules(
    latestInvokeFeatureFlags,
    lastInvoke.enabled_modules
  );

  const lastTurn = chatTurns.length ? asObject(chatTurns[chatTurns.length - 1]) : {};
  latestChatFeatureFlags = asObject(lastTurn.feature_flags);
  latestChatEnabledModules = deriveEnabledModules(
    latestChatFeatureFlags,
    lastTurn.enabled_modules
  );
}

function updateChatContextInfo() {
  const totalMessages = chatMessages.length;
  const turns = chatTurns.length;
  if (els.chatContextInfo) {
    const pathLabel = supportsChatEndpoint() ? "/chat" : "/generate";
    const modeText = supportsChatEndpoint()
      ? `${totalMessages} messages, ${turns} completed turns (sent on each ${pathLabel} request).`
      : `${totalMessages} messages, ${turns} completed turns (latest user message is sent as ${pathLabel} prompt).`;
    els.chatContextInfo.textContent = `Context: ${modeText}`;
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

  const transportPayloadBytes = toFiniteNumber(m.transport_payload_bytes);
  if (transportPayloadBytes != null) {
    return transportPayloadBytes;
  }

  const requestBodyBytes = toFiniteNumber(m.request_body_bytes);
  const responseBodyBytes = toFiniteNumber(m.response_body_bytes);
  if (requestBodyBytes != null || responseBodyBytes != null) {
    return Number(requestBodyBytes || 0) + Number(responseBodyBytes || 0);
  }

  const tensorWireBytes = toFiniteNumber(m.tensor_wire_bytes);
  if (tensorWireBytes != null) {
    return tensorWireBytes;
  }

  const inputTensorBytes = toFiniteNumber(m.input_tensor_bytes);
  const outputTensorBytes = toFiniteNumber(m.output_tensor_bytes);
  if (inputTensorBytes != null || outputTensorBytes != null) {
    return Number(inputTensorBytes || 0) + Number(outputTensorBytes || 0);
  }

  const tensorBytesIn = toFiniteNumber(m.tensor_bytes_in);
  const tensorBytesOut = toFiniteNumber(m.tensor_bytes_out);
  if (tensorBytesIn != null || tensorBytesOut != null) {
    return Number(tensorBytesIn || 0) + Number(tensorBytesOut || 0);
  }

  const keys = [
    "outbound_payload_bytes",
    "inbound_payload_bytes",
    "request_wire_bytes",
    "response_wire_bytes",
    "stage1_request_payload_bytes",
    "stage1_response_payload_bytes",
    "request_payload_bytes",
    "response_payload_bytes",
    "outbound_payload_b64_bytes",
    "inbound_payload_b64_bytes"
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

function parseNativeStageMetadata(rawValue) {
  if (!rawValue) {
    return {};
  }
  if (typeof rawValue === "object" && !Array.isArray(rawValue)) {
    return rawValue;
  }
  if (typeof rawValue !== "string") {
    return {};
  }
  return parseJsonOrNull(rawValue) || {};
}

function nativeStepToStageMetric(step) {
  const s = asObject(step);
  const metadata = parseNativeStageMetadata(s.stage_metadata || s.stage_metadata_json);
  const metrics = asObject(metadata.metrics);
  const stageId = Number(metadata.stage_id ?? s.stage_id ?? 0);
  const serviceName = String(metadata.service || metadata.service_name || `native-stage-${stageId}`);

  const inputTensorBytes = Number(metrics.input_tensor_bytes || 0);
  const outputTensorBytes = Number(metrics.output_tensor_bytes ?? s.stage_tensor_bytes ?? 0);

  const requestBodyBytes = Number(s.request_body_bytes || 0);
  const responseBodyBytes = Number(s.response_body_bytes || 0);
  const explicitTransportPayloadBytes = Number(s.transport_payload_bytes || 0);
  const transportPayloadBytes =
    explicitTransportPayloadBytes > 0
      ? explicitTransportPayloadBytes
      : requestBodyBytes + responseBodyBytes > 0
      ? requestBodyBytes + responseBodyBytes
      : inputTensorBytes + outputTensorBytes;

  const stageHttpElapsedMs = Number(s.stage_http_elapsed_ms || 0);
  const rpcWallMs = Number(metrics.rpc_wall_time_ms || stageHttpElapsedMs || 0);
  const trueCommMs = Number(metrics.true_comm_ms || s.estimated_true_comm_ms || 0);

  const cgroupCurrentMb = Number(metrics.memory_cgroup_current_mb || 0);
  const cgroupLimitMb = Number(metrics.memory_cgroup_limit_mb || 0);
  const cgroupPercent = Number(metrics.memory_cgroup_percent || 0);
  const sessionCount = Number(metrics.session_count || 0);
  const sessionKvCacheBytes = Number(metrics.session_kv_cache_bytes || 0);
  const modelFileSizeMb = Number(metrics.model_file_size_mb || 0);

  return {
    timestamp_ms: 0,
    hostname: serviceName,
    service_name: serviceName,
    stage_id: stageId,
    token_index: Number(metadata.token_index ?? s.token_index ?? 0),
    generation_mode: String(metadata.generation_mode || s.generation_mode || ""),
    partition_id: String(s.partition_id || ""),
    backend: String(metadata.backend || metrics.backend || ""),
    status: String(metadata.status || metrics.status || ""),
    stage_url: String(s.stage_url || ""),
    stage_http_status: Number(s.stage_http_status || 0),
    stage_http_reason: String(s.stage_http_reason || ""),
    stage_http_elapsed_ms: stageHttpElapsedMs,

    compute_time_ms: Number(metrics.compute_time_ms || 0),
    rpc_wall_time_ms: rpcWallMs,
    transfer_time_ms: rpcWallMs,
    true_comm_ms: trueCommMs,
    estimated_true_comm_ms: Number(s.estimated_true_comm_ms || 0),

    input_tensor_bytes: inputTensorBytes,
    output_tensor_bytes: outputTensorBytes,

    request_body_bytes: requestBodyBytes,
    response_body_bytes: responseBodyBytes,
    transport_payload_bytes: transportPayloadBytes,
    transport_payload_mib: transportPayloadBytes / (1024 * 1024),

    request_wire_bytes: requestBodyBytes,
    response_wire_bytes: responseBodyBytes,
    tensor_wire_bytes: transportPayloadBytes,

    process_memory_mb: Number(metrics.memory_rss_mb || 0),
    memory_rss_mb: Number(metrics.memory_rss_mb || 0),
    memory_cgroup_current_mb: cgroupCurrentMb,
    memory_cgroup_limit_mb: cgroupLimitMb,
    memory_cgroup_percent: cgroupPercent,

    model_load_ms: Number(metrics.model_load_ms || 0),
    model_file_size_mb: modelFileSizeMb,

    kv_cache_bytes: Number(metrics.kv_cache_bytes || 0),
    kv_cache_seq_before: Number(metrics.kv_cache_seq_before || 0),
    kv_cache_seq_after: Number(metrics.kv_cache_seq_after || 0),
    kv_cache_valid: Boolean(metrics.kv_cache_valid),

    session_count: sessionCount,
    session_kv_cache_bytes: sessionKvCacheBytes,
    session_kv_cache_mib: sessionKvCacheBytes / (1024 * 1024),

    // Per-request resource usage sampled from /proc on the native stage.
    process_cpu_percent: Number(metrics.process_cpu_percent || 0),
    system_cpu_percent: Number(metrics.system_cpu_percent || 0),
    process_threads: Number(metrics.process_threads || 0),
    process_context_switches: {
      voluntary: Number(asObject(metrics.process_context_switches).voluntary || 0),
      involuntary: Number(asObject(metrics.process_context_switches).involuntary || 0)
    },
    process_io_delta: {
      read_bytes: Number(asObject(metrics.process_io_delta).read_bytes || 0),
      write_bytes: Number(asObject(metrics.process_io_delta).write_bytes || 0),
      read_count: Number(asObject(metrics.process_io_delta).read_count || 0),
      write_count: Number(asObject(metrics.process_io_delta).write_count || 0)
    },

    // Compute breakdown + ggml thread count (native CPU optimization metrics).
    matmul_ms: Number(metrics.matmul_ms || 0),
    attention_ms: Number(metrics.attention_ms || 0),
    ggml_threads: Number(metrics.ggml_threads || 0),

    // Native stages don't expose per-process net counters, so net delta is the
    // real socket bytes moved this hop (request + response) and bandwidth is
    // those bytes over the RPC wall time.
    network_delta: {
      bytes_sent: requestBodyBytes,
      bytes_recv: responseBodyBytes,
      bytes_total: transportPayloadBytes,
      bandwidth_mbps_total:
        rpcWallMs > 0 ? (transportPayloadBytes * 8) / (rpcWallMs * 1000) : 0
    },

    native_stage_metadata: metadata
  };
}

function normalizeNativeResponseMetrics(responseJson, callMeta = {}) {
  const response = asObject(responseJson);
  const steps = asArray(response.steps);
  const aggregateMetrics = asObject(response.aggregate_metrics);

  if (!steps.length && !Object.keys(aggregateMetrics).length) {
    return responseJson;
  }

  const stageSamples = steps.map(nativeStepToStageMetric);
  const byToken = new Map();
  for (const sample of stageSamples) {
    const tokenIndex = Number(sample.token_index || 0);
    const row = byToken.get(tokenIndex) || {
      token_index: tokenIndex,
      stage_metrics: []
    };
    row.stage_metrics.push(sample);
    byToken.set(tokenIndex, row);
  }

  const generatedTokenIds = asArray(response.generated_token_ids);
  const tokenMetrics = [...byToken.values()]
    .sort((a, b) => Number(a.token_index || 0) - Number(b.token_index || 0))
    .map((row) => {
      const samples = asArray(row.stage_metrics);
      const latencyMs = samples.reduce(
        (sum, sample) =>
          sum +
          Number(sample.compute_time_ms || 0) +
          Number(sample.rpc_wall_time_ms || 0),
        0
      );
      const tokenIndex = Number(row.token_index || 0);
      return {
        token_index: tokenIndex,
        token_id: generatedTokenIds[tokenIndex] ?? "",
        token_text: generatedTokenIds[tokenIndex] == null ? "" : `<tok_${generatedTokenIds[tokenIndex]}>`,
        latency_ms: latencyMs,
        stage_metrics: samples
      };
    });

  const computeSum = Number(
    aggregateMetrics.compute_ms ||
      stageSamples.reduce((sum, sample) => sum + Number(sample.compute_time_ms || 0), 0)
  );
  const rpcWallSum = Number(
    aggregateMetrics.rpc_wall_ms ||
      stageSamples.reduce((sum, sample) => sum + Number(sample.rpc_wall_time_ms || 0), 0)
  );
  const trueCommSum = Number(
    aggregateMetrics.true_comm_ms ||
      stageSamples.reduce((sum, sample) => sum + Number(sample.true_comm_ms || 0), 0)
  );
  // Raw activation/tensor data volume moved between stages (excludes transport
  // framing/encoding overhead).
  const tensorDataBytes =
    Number(aggregateMetrics.tensor_bytes_in || 0) +
    Number(aggregateMetrics.tensor_bytes_out || 0);
  // Bytes actually serialized onto the wire in both directions (request +
  // response bodies). The native gateway sums these as transport_payload_bytes;
  // on the co-located/native path there is no separate OS-level NIC counter, so
  // this is the real network-transfer figure.
  const wireBytes = Number(
    aggregateMetrics.transport_payload_bytes ||
      aggregateMetrics.payload_bytes_sum ||
      tensorDataBytes ||
      stageSamples.reduce((sum, sample) => sum + metricPayloadBytes(sample), 0)
  );
  const payloadBytes = tensorDataBytes || wireBytes;
  const maxMemoryMb = Number(
    aggregateMetrics.memory_rss_mb ||
      stageSamples.reduce((max, sample) => Math.max(max, Number(sample.process_memory_mb || 0)), 0)
  );
  const maxCgroupMemMb = Number(
    aggregateMetrics.memory_cgroup_current_mb ||
      stageSamples.reduce(
        (max, sample) => Math.max(max, Number(sample.memory_cgroup_current_mb || 0)),
        0
      )
  );
  const maxCgroupLimitMb = Number(
    aggregateMetrics.memory_cgroup_limit_mb ||
      stageSamples.reduce(
        (max, sample) => Math.max(max, Number(sample.memory_cgroup_limit_mb || 0)),
        0
      )
  );
  const maxCgroupPercent = Number(
    aggregateMetrics.memory_cgroup_percent ||
      stageSamples.reduce(
        (max, sample) => Math.max(max, Number(sample.memory_cgroup_percent || 0)),
        0
      )
  );
  const maxSessionCount = Number(
    aggregateMetrics.session_count ||
      stageSamples.reduce((max, sample) => Math.max(max, Number(sample.session_count || 0)), 0)
  );
  const maxSessionKvCacheBytes = Number(
    aggregateMetrics.session_kv_cache_bytes ||
      stageSamples.reduce(
        (max, sample) => Math.max(max, Number(sample.session_kv_cache_bytes || 0)),
        0
      )
  );
  const maxModelFileSizeMb = Number(
    aggregateMetrics.model_file_size_mb ||
      stageSamples.reduce(
        (max, sample) => Math.max(max, Number(sample.model_file_size_mb || 0)),
        0
      )
  );

  return {
    ...response,
    prompt_token_count: Number(response.prompt_token_count || asArray(response.prompt_token_ids).length || 0),
    generated_token_count: Number(response.generated_token_count || generatedTokenIds.length || 0),
    summary_metrics: {
      generated_token_count: Number(response.generated_token_count || generatedTokenIds.length || 0),
      prompt_token_count: Number(response.prompt_token_count || asArray(response.prompt_token_ids).length || 0),
      feature_flags: {},
      enabled_modules: ["llama_cpp_native", "dli2_binary_frames", "gguf_partition_shards"],
      aggregate: {
        compute_time_ms_sum: computeSum,
        rpc_wall_time_ms_sum: rpcWallSum,
        transfer_time_ms_sum: rpcWallSum,
        true_comm_ms_sum: trueCommSum,
        rpc_compute_ratio: rpcWallSum / Math.max(1e-9, computeSum),
        true_comm_compute_ratio: trueCommSum / Math.max(1e-9, computeSum),
        payload_bytes_sum: payloadBytes,
        payload_mebibytes_sum: payloadBytes / (1024 * 1024),
        network_delta_bytes_sum: wireBytes,
        network_delta_mebibytes_sum: wireBytes / (1024 * 1024),
        max_process_memory_mb: maxMemoryMb,
        max_cgroup_memory_mb: maxCgroupMemMb,
        max_cgroup_memory_limit_mb: maxCgroupLimitMb,
        max_cgroup_memory_percent: maxCgroupPercent,
        max_session_count: maxSessionCount,
        max_session_kv_cache_bytes: maxSessionKvCacheBytes,
        max_model_file_size_mb: maxModelFileSizeMb
      },
      per_stage: {}
    },
    token_metrics: tokenMetrics
  };
}

function normalizeResponseMetrics(responseJson, callMeta = {}) {
  const response = asObject(responseJson);
  if (response.summary_metrics && response.token_metrics) {
    return responseJson;
  }
  if (
    String(response.runtime || "").includes("cpp-native") ||
    response.aggregate_metrics ||
    response.steps
  ) {
    return normalizeNativeResponseMetrics(responseJson, callMeta);
  }
  return responseJson;
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
      transferMs: Number(s.rpc_wall_time_ms_sum ?? s.transfer_time_ms_sum ?? 0),
      trueCommMs: Number(s.true_comm_ms_sum || 0),
      payloadBytes: Number(s.payload_bytes_sum ?? s.payload_b64_bytes_sum ?? 0),
      networkBytes: Number(s.network_delta_bytes_sum || 0),
      maxMemMb: Number(s.max_process_memory_mb || 0),
      maxCgroupMemMb: Number(s.max_cgroup_memory_mb || s.memory_cgroup_current_mb || 0),
      maxCgroupLimitMb: Number(s.max_cgroup_memory_limit_mb || s.memory_cgroup_limit_mb || 0),
      maxCgroupPercent: Number(s.max_cgroup_memory_percent || s.memory_cgroup_percent || 0),
      maxSessionCount: Number(s.max_session_count || s.session_count || 0),
      maxSessionKvCacheBytes: Number(s.max_session_kv_cache_bytes || s.session_kv_cache_bytes || 0),
      maxModelFileSizeMb: Number(s.max_model_file_size_mb || s.model_file_size_mb || 0),
      maxCpuPct: Number(s.max_cpu_percent || 0),
      maxSystemCpuPct: 0,
      maxBandwidthMbps: 0,
      maxThreads: 0,
      ctxSwitches: 0,
      ioReadCountDelta: 0,
      ioWriteCountDelta: 0,
      matmulMs: 0,
      attentionMs: 0,
      ggmlThreads: 0
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
      trueCommMs: 0,
      payloadBytes: 0,
      networkBytes: 0,
      maxMemMb: 0,
      maxCgroupMemMb: 0,
      maxCgroupLimitMb: 0,
      maxCgroupPercent: 0,
      maxSessionCount: 0,
      maxSessionKvCacheBytes: 0,
      maxModelFileSizeMb: 0,
      maxCpuPct: 0,
      maxSystemCpuPct: 0,
      maxBandwidthMbps: 0,
      maxThreads: 0,
      ctxSwitches: 0,
      ioReadCountDelta: 0,
      ioWriteCountDelta: 0,
      matmulMs: 0,
      attentionMs: 0,
      ggmlThreads: 0
    };

    curr.samples += 1;
    curr.computeMs += Number(sample.compute_time_ms || 0);
    curr.transferMs += Number(sample.rpc_wall_time_ms ?? sample.transfer_time_ms ?? 0);
    curr.trueCommMs += Number(sample.true_comm_ms || 0);
    curr.payloadBytes += metricPayloadBytes(sample);
    curr.networkBytes += metricNetworkDeltaBytes(sample);
    curr.maxMemMb = Math.max(curr.maxMemMb, Number(sample.process_memory_mb || 0));
    curr.maxCgroupMemMb = Math.max(
      curr.maxCgroupMemMb,
      Number(sample.memory_cgroup_current_mb || 0)
    );
    curr.maxCgroupLimitMb = Math.max(
      curr.maxCgroupLimitMb,
      Number(sample.memory_cgroup_limit_mb || 0)
    );
    curr.maxCgroupPercent = Math.max(
      curr.maxCgroupPercent,
      Number(sample.memory_cgroup_percent || 0)
    );
    curr.maxSessionCount = Math.max(
      curr.maxSessionCount,
      Number(sample.session_count || 0)
    );
    curr.maxSessionKvCacheBytes = Math.max(
      curr.maxSessionKvCacheBytes,
      Number(sample.session_kv_cache_bytes || 0)
    );
    curr.maxModelFileSizeMb = Math.max(
      curr.maxModelFileSizeMb,
      Number(sample.model_file_size_mb || 0)
    );
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
      Number(sample.estimated_link_mbps ?? sample.network_delta?.bandwidth_mbps_total ?? 0)
    );
    curr.maxThreads = Math.max(curr.maxThreads, Number(sample.process_threads || 0));
    curr.ctxSwitches += Number(sample.process_context_switches?.voluntary || 0);
    curr.ctxSwitches += Number(sample.process_context_switches?.involuntary || 0);
    curr.ioReadCountDelta += Number(sample.process_io_delta?.read_count || 0);
    curr.ioWriteCountDelta += Number(sample.process_io_delta?.write_count || 0);
    curr.matmulMs += Number(sample.matmul_ms || 0);
    curr.attentionMs += Number(sample.attention_ms || 0);
    curr.ggmlThreads = Math.max(curr.ggmlThreads, Number(sample.ggml_threads || 0));
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
      trueCommMs: 0,
      payloadBytes: 0,
      networkBytes: 0,
      maxMemMb: 0,
      maxCgroupMemMb: 0,
      maxCgroupLimitMb: 0,
      maxCgroupPercent: 0,
      maxSessionCount: 0,
      maxSessionKvCacheBytes: 0,
      maxModelFileSizeMb: 0,
      maxCpuPct: 0,
      maxSystemCpuPct: 0,
      maxBandwidthMbps: 0,
      maxThreads: 0,
      ctxSwitches: 0,
      ioReadCountDelta: 0,
      ioWriteCountDelta: 0,
      matmulMs: 0,
      attentionMs: 0,
      ggmlThreads: 0
    };

    curr.podHostnames.add(podHostname);
    curr.services.add(String(sample?.service_name || "unknown"));
    if (sample?.stage_id != null) {
      curr.stages.add(String(sample.stage_id));
    }
    curr.samples += 1;
    curr.computeMs += Number(sample.compute_time_ms || 0);
    curr.transferMs += Number(sample.rpc_wall_time_ms ?? sample.transfer_time_ms ?? 0);
    curr.trueCommMs += Number(sample.true_comm_ms || 0);
    curr.payloadBytes += metricPayloadBytes(sample);
    curr.networkBytes += metricNetworkDeltaBytes(sample);
    curr.maxMemMb = Math.max(curr.maxMemMb, Number(sample.process_memory_mb || 0));
    curr.maxCgroupMemMb = Math.max(
      curr.maxCgroupMemMb,
      Number(sample.memory_cgroup_current_mb || 0)
    );
    curr.maxCgroupLimitMb = Math.max(
      curr.maxCgroupLimitMb,
      Number(sample.memory_cgroup_limit_mb || 0)
    );
    curr.maxCgroupPercent = Math.max(
      curr.maxCgroupPercent,
      Number(sample.memory_cgroup_percent || 0)
    );
    curr.maxSessionCount = Math.max(
      curr.maxSessionCount,
      Number(sample.session_count || 0)
    );
    curr.maxSessionKvCacheBytes = Math.max(
      curr.maxSessionKvCacheBytes,
      Number(sample.session_kv_cache_bytes || 0)
    );
    curr.maxModelFileSizeMb = Math.max(
      curr.maxModelFileSizeMb,
      Number(sample.model_file_size_mb || 0)
    );
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
    curr.matmulMs += Number(sample.matmul_ms || 0);
    curr.attentionMs += Number(sample.attention_ms || 0);
    curr.ggmlThreads = Math.max(curr.ggmlThreads, Number(sample.ggml_threads || 0));

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
    let trueCommMs = 0;
    let maxMemMb = 0;
    let maxCgroupMemMb = 0;
    let maxCgroupPercent = 0;
    let maxSessionCount = 0;
    let maxSessionKvCacheBytes = 0;
    let maxCpu = 0;

    for (const sample of samples) {
      computeMs += Number(sample.compute_time_ms || 0);
      transferMs += Number(sample.rpc_wall_time_ms ?? sample.transfer_time_ms ?? 0);
      trueCommMs += Number(sample.true_comm_ms || 0);
      maxMemMb = Math.max(maxMemMb, Number(sample.process_memory_mb || 0));
      maxCgroupMemMb = Math.max(maxCgroupMemMb, Number(sample.memory_cgroup_current_mb || 0));
      maxCgroupPercent = Math.max(maxCgroupPercent, Number(sample.memory_cgroup_percent || 0));
      maxSessionCount = Math.max(maxSessionCount, Number(sample.session_count || 0));
      maxSessionKvCacheBytes = Math.max(
        maxSessionKvCacheBytes,
        Number(sample.session_kv_cache_bytes || 0)
      );
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
      trueCommMs,
      maxMemMb,
      maxCgroupMemMb,
      maxCgroupPercent,
      maxSessionCount,
      maxSessionKvCacheBytes,
      maxCpu
    };
  });
}

function buildInvokeRunSummary({
  responseJson,
  nodeRows,
  stageRows,
  tokenRows,
  criticalPath,
  totalLatencyMs,
  requestConfig
}) {
  const summary = asObject(responseJson?.summary_metrics);
  const featureFlags = asObject(summary.feature_flags);
  const enabledModules = deriveEnabledModules(featureFlags, summary.enabled_modules);
  const generatedTokens = Number(asArray(responseJson?.generated_token_ids).length || 0);
  const computeSumMs = tokenRows.reduce((sum, row) => sum + Number(row.computeMs || 0), 0);
  const transferSumMs = tokenRows.reduce((sum, row) => sum + Number(row.transferMs || 0), 0);
  const trueCommSumMs = tokenRows.reduce((sum, row) => sum + Number(row.trueCommMs || 0), 0);
  const maxMemoryMb = tokenRows.reduce((max, row) => Math.max(max, Number(row.maxMemMb || 0)), 0);
  const maxCgroupMemMb = tokenRows.reduce(
    (max, row) => Math.max(max, Number(row.maxCgroupMemMb || 0)),
    0
  );
  const maxCgroupPercent = tokenRows.reduce(
    (max, row) => Math.max(max, Number(row.maxCgroupPercent || 0)),
    0
  );
  const maxSessionCount = tokenRows.reduce(
    (max, row) => Math.max(max, Number(row.maxSessionCount || 0)),
    0
  );
  const maxSessionKvCacheBytes = tokenRows.reduce(
    (max, row) => Math.max(max, Number(row.maxSessionKvCacheBytes || 0)),
    0
  );
  const tokenLatencies = tokenRows.map((row) => Number(row.latencyMs || 0));
  const transferLatencies = tokenRows.map((row) => Number(row.transferMs || 0));
  const aggregate = asObject(summary.aggregate);
  const rpcComputeRatio = Number(
    aggregate.rpc_compute_ratio ?? transferSumMs / Math.max(1e-9, computeSumMs)
  );
  const trueCommComputeRatio = Number(
    aggregate.true_comm_compute_ratio ?? trueCommSumMs / Math.max(1e-9, computeSumMs)
  );
  const matmulSumMs = stageRows.reduce((sum, row) => sum + Number(row.matmulMs || 0), 0);
  const attentionSumMs = stageRows.reduce((sum, row) => sum + Number(row.attentionMs || 0), 0);
  const ggmlThreads = stageRows.reduce((max, row) => Math.max(max, Number(row.ggmlThreads || 0)), 0);
  return {
    timestamp: new Date().toISOString(),
    totalLatencyMs: Number(totalLatencyMs || 0),
    tokensPerSecond: Number(responseJson?.tokens_per_second || 0),
    generatedTokens,
    computeSumMs,
    transferSumMs,
    trueCommSumMs,
    matmulSumMs,
    attentionSumMs,
    ggmlThreads,
    transferComputeRatio: trueCommComputeRatio,
    rpcComputeRatio,
    trueCommComputeRatio,
    maxMemoryMb,
    maxCgroupMemMb,
    maxCgroupPercent,
    maxSessionCount,
    maxSessionKvCacheBytes,
    p50TokenLatencyMs: percentile(tokenLatencies, 0.5),
    p95TokenLatencyMs: percentile(tokenLatencies, 0.95),
    p99TokenLatencyMs: percentile(tokenLatencies, 0.99),
    p50TransferLatencyMs: percentile(transferLatencies, 0.5),
    p95TransferLatencyMs: percentile(transferLatencies, 0.95),
    p99TransferLatencyMs: percentile(transferLatencies, 0.99),
    criticalPath,
    feature_flags: featureFlags,
    enabled_modules: enabledModules,
    isBaseline: enabledModules.length === 0,
    promptTokens: Number(responseJson?.prompt_token_count || summary.prompt_token_count || 0),
    config: {
      promptChars: Number(requestConfig?.promptChars || 0),
      maxNewTokens: Number(requestConfig?.maxNewTokens || 0),
      minNewTokens: Number(requestConfig?.minNewTokens || 0),
      temperature: Number(requestConfig?.temperature || 0),
      topK: Number(requestConfig?.topK || 0),
      topP: requestConfig?.topP != null ? Number(requestConfig.topP) : 1,
      seed: requestConfig?.seed != null ? Number(requestConfig.seed) : null,
      timeoutMs: Number(requestConfig?.timeoutMs || 0),
      method: String(requestConfig?.method || "POST"),
      path: String(requestConfig?.path || "/generate"),
      target: String(requestConfig?.target || "gateway"),
      variant: String(requestConfig?.variant || getRuntimeVariant())
    },
    payloadMib: Number(aggregate.payload_mebibytes_sum ?? aggregate.payload_b64_mebibytes_sum ?? 0),
    networkMib: Number(aggregate.network_delta_mebibytes_sum || 0),
    perStage: stageRows.map((row) => ({
      stageKey: row.stageKey,
      service: row.serviceName,
      stageId: row.stageId,
      samples: row.samples,
      computeMs: row.computeMs,
      transferMs: row.transferMs,
      trueCommMs: row.trueCommMs,
      maxMemMb: row.maxMemMb,
      maxCgroupMemMb: row.maxCgroupMemMb,
      maxCgroupLimitMb: row.maxCgroupLimitMb,
      maxCgroupPercent: row.maxCgroupPercent,
      maxSessionCount: row.maxSessionCount,
      maxSessionKvCacheBytes: row.maxSessionKvCacheBytes,
      maxModelFileSizeMb: row.maxModelFileSizeMb,
      maxCpuPct: row.maxCpuPct,
      maxSystemCpuPct: row.maxSystemCpuPct,
      networkMib: row.networkBytes / (1024 * 1024),
      payloadMib: row.payloadBytes / (1024 * 1024),
      maxMbps: row.maxBandwidthMbps
    })),
    perNode: nodeRows.map((row) => ({
      node: row.nodeName,
      computeMs: row.computeMs,
      transferMs: row.transferMs,
      maxMemMb: row.maxMemMb,
      maxCgroupMemMb: row.maxCgroupMemMb,
      maxCgroupLimitMb: row.maxCgroupLimitMb,
      maxCgroupPercent: row.maxCgroupPercent,
      maxSessionCount: row.maxSessionCount,
      maxSessionKvCacheBytes: row.maxSessionKvCacheBytes,
      maxModelFileSizeMb: row.maxModelFileSizeMb,
      maxCpuPct: row.maxCpuPct,
      maxSystemCpuPct: row.maxSystemCpuPct,
      maxMbps: row.maxBandwidthMbps,
      networkDeltaMib: row.networkBytes / (1024 * 1024),
      payloadMib: row.payloadBytes / (1024 * 1024),
      podHostnames: row.podHostnamesCsv,
      services: row.servicesCsv
    }))
  };
}

function updateLatestModuleState(responseJson, source) {
  responseJson = normalizeResponseMetrics(responseJson);
  const summary = asObject(asObject(responseJson).summary_metrics);
  const featureFlags = asObject(summary.feature_flags);
  const enabledModules = deriveEnabledModules(featureFlags, summary.enabled_modules);

  if (source === "chat") {
    latestChatFeatureFlags = featureFlags;
    latestChatEnabledModules = enabledModules;
  } else {
    latestInvokeFeatureFlags = featureFlags;
    latestInvokeEnabledModules = enabledModules;
  }
}

function renderInvokeRunEvolution() {
  if (!invokeRuns.length) {
    return `<div class="muted">No endpoint run history in this browser session yet.</div>`;
  }
  const runs = invokeRuns.slice(-40);
  const latencySeries = [{ name: "Latency ms", values: runs.map((r) => Number(r.totalLatencyMs || 0)) }];
  const transferSeries = [{ name: "RPC wall ms", values: runs.map((r) => Number(r.transferSumMs || 0)) }];
  const trueCommSeries = [{ name: "True comm ms", values: runs.map((r) => Number(r.trueCommSumMs || 0)) }];
  const computeSeries = [{ name: "Compute ms", values: runs.map((r) => Number(r.computeSumMs || 0)) }];
  const tpsSeries = [{ name: "Tokens/sec", values: runs.map((r) => Number(r.tokensPerSecond || 0)) }];
  const ratioSeries = [
    { name: "RPC/Compute", values: runs.map((r) => Number(r.rpcComputeRatio || 0)) },
    { name: "True Comm/Compute", values: runs.map((r) => Number(r.trueCommComputeRatio || 0)) }
  ];
  const tokenPercentileSeries = [
    { name: "p50", values: runs.map((r) => Number(r.p50TokenLatencyMs || 0)) },
    { name: "p95", values: runs.map((r) => Number(r.p95TokenLatencyMs || 0)) },
    { name: "p99", values: runs.map((r) => Number(r.p99TokenLatencyMs || 0)) }
  ];
  const transferPercentileSeries = [
    { name: "p50", values: runs.map((r) => Number(r.p50TransferLatencyMs || 0)) },
    { name: "p95", values: runs.map((r) => Number(r.p95TransferLatencyMs || 0)) },
    { name: "p99", values: runs.map((r) => Number(r.p99TransferLatencyMs || 0)) }
  ];

  return `
    <div class="invoke-subpanel">
      <h3>Endpoint Run Evolution (session, last ${runs.length})</h3>
      <div class="chart-grid">
        <div class="chart-card">
          <h4>Latency per Run (ms)</h4>
          ${buildLineChartSvg(latencySeries, { yLabel: "Latency ms" })}
        </div>
        <div class="chart-card">
          <h4>RPC + True Comm vs Compute per Run</h4>
          ${buildLineChartSvg(
            [
              { name: "RPC ms", values: transferSeries[0].values },
              { name: "True comm ms", values: trueCommSeries[0].values },
              { name: "Compute ms", values: computeSeries[0].values }
            ],
            { yLabel: "ms" }
          )}
        </div>
        <div class="chart-card">
          <h4>Throughput per Run</h4>
          ${buildLineChartSvg(tpsSeries, { yLabel: "tokens/sec" })}
        </div>
        <div class="chart-card">
          <h4>RPC/Compute + True Comm/Compute</h4>
          ${buildLineChartSvg(ratioSeries, { yLabel: "ratio" })}
        </div>
        <div class="chart-card">
          <h4>Token Latency p50/p95/p99</h4>
          ${buildLineChartSvg(tokenPercentileSeries, { yLabel: "ms" })}
        </div>
        <div class="chart-card">
          <h4>RPC Wall p50/p95/p99</h4>
          ${buildLineChartSvg(transferPercentileSeries, { yLabel: "ms" })}
        </div>
      </div>
    </div>
  `;
}

function renderInvokeMetrics(responseJson, callMeta, requestConfig = {}) {
  responseJson = normalizeResponseMetrics(responseJson, callMeta);
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
  const rpcComputeRatio = Number(
    aggregate.rpc_compute_ratio ??
      Number(aggregate.transfer_time_ms_sum || 0) / Math.max(1e-9, Number(aggregate.compute_time_ms_sum || 0))
  );
  const trueCommComputeRatio = Number(aggregate.true_comm_compute_ratio ?? aggregate.comm_compute_ratio ?? 0);
  const transferComputeRatio = trueCommComputeRatio;
  const networkBytesSum = Number(aggregate.network_delta_bytes_sum || 0);
  const payloadBytesSum = Number(aggregate.payload_bytes_sum ?? aggregate.payload_b64_bytes_sum ?? 0);
  const bytesPerGeneratedToken = networkBytesSum / Math.max(1, generatedTokenCount);
  const payloadBytesPerGeneratedToken = payloadBytesSum / Math.max(1, generatedTokenCount);
  const baselineRef = [...invokeRuns]
    .reverse()
    .find((run) => run && run.isBaseline && Number(run.maxMemoryMb || 0) > 0);
  const baselineSavedMb = baselineRef ? Number(baselineRef.maxMemoryMb || 0) - Number(aggregate.max_process_memory_mb || 0) : 0;
  const mibPerGiBSaved =
    baselineSavedMb > 0
      ? Number(aggregate.network_delta_mebibytes_sum || 0) / (baselineSavedMb / 1024)
      : null;

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
  const maxCgroupMemMb = stageRows.reduce(
    (max, row) => Math.max(max, Number(row.maxCgroupMemMb || 0)),
    0
  );
  const maxCgroupLimitMb = stageRows.reduce(
    (max, row) => Math.max(max, Number(row.maxCgroupLimitMb || 0)),
    0
  );
  const maxCgroupPercent = stageRows.reduce(
    (max, row) => Math.max(max, Number(row.maxCgroupPercent || 0)),
    0
  );
  const maxSessionCount = stageRows.reduce(
    (max, row) => Math.max(max, Number(row.maxSessionCount || 0)),
    0
  );
  const maxSessionKvCacheBytes = stageRows.reduce(
    (max, row) => Math.max(max, Number(row.maxSessionKvCacheBytes || 0)),
    0
  );
  const maxModelFileSizeMb = stageRows.reduce(
    (max, row) => Math.max(max, Number(row.maxModelFileSizeMb || 0)),
    0
  );
  const matmulMsSum = stageRows.reduce((sum, row) => sum + Number(row.matmulMs || 0), 0);
  const attentionMsSum = stageRows.reduce((sum, row) => sum + Number(row.attentionMs || 0), 0);
  const ggmlThreads = stageRows.reduce((max, row) => Math.max(max, Number(row.ggmlThreads || 0)), 0);

  const cards = [
    { k: "Prompt Tokens", v: String(promptTokenCount) },
    { k: "Generated Tokens", v: String(generatedTokenCount) },
    { k: "Termination", v: safeText(responseJson.termination_reason || "unknown") },
    { k: "Latency (ms)", v: formatNumber(totalLatencyMs, 1) },
    { k: "Tokens/sec", v: formatNumber(responseJson.tokens_per_second, 3) },
    { k: "Samples", v: String(stageSamples.length) },
    { k: "Compute Sum (ms)", v: formatNumber(aggregate.compute_time_ms_sum, 1) },
    { k: "RPC Wall Sum (ms)", v: formatNumber(aggregate.rpc_wall_time_ms_sum ?? aggregate.transfer_time_ms_sum, 1) },
    { k: "True Comm Sum (ms)", v: formatNumber(aggregate.true_comm_ms_sum, 1) },
    {
      k: "RPC/Compute Ratio",
      v: formatNumber(rpcComputeRatio, 3)
    },
    {
      k: "True Comm/Compute Ratio",
      v: formatNumber(trueCommComputeRatio, 3)
    },
    { k: "Payload Sum (MiB)", v: formatNumber(aggregate.payload_mebibytes_sum ?? aggregate.payload_b64_mebibytes_sum, 3) },
    { k: "Network Delta (MiB)", v: formatNumber(aggregate.network_delta_mebibytes_sum, 3) },
    { k: "Max Process Mem (MB)", v: formatNumber(aggregate.max_process_memory_mb, 1) },
    { k: "Max Cgroup Mem (MB)", v: formatNumber(maxCgroupMemMb, 1) },
    { k: "Cgroup Mem Limit (MB)", v: formatNumber(maxCgroupLimitMb, 1) },
    { k: "Max Cgroup Mem (%)", v: formatNumber(maxCgroupPercent, 1) },
    { k: "Max Session Count", v: String(Math.round(maxSessionCount)) },
    { k: "Max Session KV Cache (MiB)", v: formatNumber(maxSessionKvCacheBytes / (1024 * 1024), 3) },
    { k: "Model File Size (MB)", v: formatNumber(maxModelFileSizeMb, 1) },
    { k: "Max Process CPU (%)", v: formatNumber(maxProcessCpu, 1) },
    { k: "Max System CPU (%)", v: formatNumber(maxSystemCpu, 1) },
    { k: "Max Link Mbps", v: formatNumber(maxBandwidth, 2) },
    { k: "Max Threads", v: String(Math.round(maxThreads)) },
    { k: "GGML Threads", v: String(Math.round(ggmlThreads)) },
    { k: "Matmul Sum (ms)", v: formatNumber(matmulMsSum, 1) },
    { k: "Attention Sum (ms)", v: formatNumber(attentionMsSum, 1) },
    { k: "Nodes Seen", v: String(nodeRows.length) },
    { k: "Token Latency p50 (ms)", v: formatNumber(tokenLatencyP50, 1) },
    { k: "Token Latency p95 (ms)", v: formatNumber(tokenLatencyP95, 1) },
    { k: "Token Latency p99 (ms)", v: formatNumber(tokenLatencyP99, 1) },
    { k: "RPC Wall p50 (ms)", v: formatNumber(tokenTransferP50, 1) },
    { k: "RPC Wall p95 (ms)", v: formatNumber(tokenTransferP95, 1) },
    { k: "RPC Wall p99 (ms)", v: formatNumber(tokenTransferP99, 1) }
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
        <td>${formatNumber(row.matmulMs, 1)}</td>
        <td>${formatNumber(row.attentionMs, 1)}</td>
        <td>${formatNumber(row.transferMs, 1)}</td>
        <td>${formatNumber(row.maxMemMb, 1)}</td>
        <td>${formatNumber(row.maxCgroupMemMb, 1)}</td>
        <td>${formatNumber(row.maxCgroupLimitMb, 1)}</td>
        <td>${formatNumber(row.maxCgroupPercent, 1)}</td>
        <td>${Math.round(row.maxSessionCount || 0)}</td>
        <td>${formatNumber(row.maxSessionKvCacheBytes / (1024 * 1024), 3)}</td>
        <td>${formatNumber(row.maxCpuPct, 1)}</td>
        <td>${formatNumber(row.maxSystemCpuPct, 1)}</td>
        <td>${formatNumber(row.networkBytes / (1024 * 1024), 3)}</td>
        <td>${formatNumber(row.payloadBytes / (1024 * 1024), 3)}</td>
        <td>${formatNumber(row.maxBandwidthMbps, 2)}</td>
        <td>${Math.round(row.ggmlThreads || 0)}</td>
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

  const criticalPathItems = [
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
      key: "Stage RPC Wall (subset of wait)",
      ms: criticalPath.stageTransferMs,
      note: "Inter-stage blocking RPC wall time (stage->stage)."
    },
    {
      key: "Stage Residual (wait - compute - RPC wall)",
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
  ];
  const criticalPathRows = criticalPathItems
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

  const efficiencyKpis = [
    {
      key: "Network bytes / generated token",
      value: `${formatNumber(bytesPerGeneratedToken, 1)} B/token`,
      note: "Lower is better for communication cost."
    },
    {
      key: "Payload bytes / generated token",
      value: `${formatNumber(payloadBytesPerGeneratedToken, 1)} B/token`,
      note: "Serialization payload pressure per emitted token."
    },
    {
      key: "MiB transferred / GiB memory saved vs baseline",
      value:
        mibPerGiBSaved == null
          ? "n/a"
          : `${formatNumber(mibPerGiBSaved, 3)} MiB/GiB`,
      note:
        mibPerGiBSaved == null
          ? "Needs a baseline run and positive memory reduction."
          : "Lower means better memory-vs-network tradeoff."
    },
    {
      key: "True communication/compute ratio",
      value: formatNumber(trueCommComputeRatio, 3),
      note: "Serialization, wire residual, and unpack/decode time divided by compute time."
    },
    {
      key: "RPC/compute ratio",
      value: formatNumber(rpcComputeRatio, 3),
      note: "Blocking RPC wall time divided by compute time."
    }
  ];
  const efficiencyKpiRows = efficiencyKpis
    .map(
      (row) => `
      <tr>
        <td>${escapeHtml(row.key)}</td>
        <td>${escapeHtml(row.value)}</td>
        <td>${escapeHtml(row.note)}</td>
      </tr>
    `
    )
    .join("");

  const runSummary = buildInvokeRunSummary({
    responseJson,
    nodeRows,
    stageRows,
    tokenRows,
    criticalPath,
    totalLatencyMs,
    requestConfig
  });
  invokeRuns.push(runSummary);
  if (invokeRuns.length > 120) {
    invokeRuns = invokeRuns.slice(-120);
  }
  persistSessionState();

  const alerts = buildInvokeAlerts({
    transferP95Ms: tokenTransferP95,
    transferComputeRatio,
    maxProcessCpu,
    maxMemoryMb: Number(aggregate.max_process_memory_mb || 0),
    maxCgroupPercent,
    maxSessionCount,
    runtime: runtimeSnapshot
  });
  const criticalPathSankey = buildCriticalPathSankey(criticalPath, totalLatencyMs);
  const stageHeatmap = buildStageTokenHeatmap(tokenMetrics);

  postHistoryEntries([
    {
      type: "endpoint",
      label: els.endpointLabel ? String(els.endpointLabel.value || "").trim() : "",
      createdAt: runSummary.timestamp,
      namespace: getNamespace(),
      target: runSummary.config?.target || requestConfig.target || "gateway",
      method: runSummary.config?.method || requestConfig.method || "POST",
      path: runSummary.config?.path || requestConfig.path || "/generate",
      config: {
        promptChars: runSummary.config?.promptChars || 0,
        promptTokens: runSummary.promptTokens || 0,
        maxNewTokens: runSummary.config?.maxNewTokens || 0,
        minNewTokens: runSummary.config?.minNewTokens || 0,
        temperature: runSummary.config?.temperature || 0,
        topK: runSummary.config?.topK || 0,
        topP: runSummary.config?.topP != null ? runSummary.config.topP : 1,
        seed: runSummary.config?.seed != null ? runSummary.config.seed : null,
        timeoutMs: runSummary.config?.timeoutMs || 0,
        topologyHash: buildTopologyHash()
      },
      profile: {
        baseline: runSummary.isBaseline,
        enabledModules: runSummary.enabled_modules,
        featureFlags: runSummary.feature_flags,
        moduleVariants: getFeatureFlagProfileChips(runSummary.feature_flags)
      },
      metrics: {
        generatedTokens: runSummary.generatedTokens,
        latencyMs: runSummary.totalLatencyMs,
        tokensPerSecond: runSummary.tokensPerSecond,
        computeMs: runSummary.computeSumMs,
        transferMs: runSummary.transferSumMs,
        trueCommMs: runSummary.trueCommSumMs,
        matmulMs: runSummary.matmulSumMs,
        attentionMs: runSummary.attentionSumMs,
        ggmlThreads: runSummary.ggmlThreads,
        transferComputeRatio: runSummary.transferComputeRatio,
        rpcComputeRatio: runSummary.rpcComputeRatio,
        trueCommComputeRatio: runSummary.trueCommComputeRatio,
        payloadMib: runSummary.payloadMib,
        networkMib: runSummary.networkMib,
        maxMemoryMb: runSummary.maxMemoryMb,
        tokenP50Ms: runSummary.p50TokenLatencyMs,
        tokenP95Ms: runSummary.p95TokenLatencyMs,
        tokenP99Ms: runSummary.p99TokenLatencyMs,
        transferP50Ms: runSummary.p50TransferLatencyMs,
        transferP95Ms: runSummary.p95TransferLatencyMs,
        transferP99Ms: runSummary.p99TransferLatencyMs,
        criticalPath: runSummary.criticalPath,
        perStage: runSummary.perStage,
        perNode: runSummary.perNode
      },
      output: buildHistoryOutputPayload(responseJson),
      dashboard: {
        kind: "endpoint",
        cards,
        alerts,
        efficiencyKpis,
        criticalPath,
        criticalPathItems,
        stageRows,
        nodeRows,
        tokenRows,
        metricCatalogRows: catalogRows,
        runSummary,
        sessionRunEvolution: invokeRuns.slice(-40),
        requestConfig,
        callMeta,
        responseJson
      }
    }
  ]);

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
    ${makeAlertRibbons(alerts)}

    <div class="invoke-subpanel">
      <h3>Efficiency KPI</h3>
      <table class="metrics-table">
        <thead>
          <tr>
            <th>KPI</th>
            <th>Value</th>
            <th>Interpretation</th>
          </tr>
        </thead>
        <tbody>${efficiencyKpiRows}</tbody>
      </table>
    </div>

    <div class="invoke-subpanel">
      <h3>Critical-path Sankey (gateway wait vs stage compute/RPC wall)</h3>
      ${criticalPathSankey}
    </div>

    <div class="invoke-subpanel">
      <h3>Stage-by-Token Heatmap (latency ms)</h3>
      ${stageHeatmap}
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
            <th>Matmul (ms)</th>
            <th>Attention (ms)</th>
            <th>RPC Wall (ms)</th>
            <th>Max RSS Mem (MB)</th>
            <th>Cgroup Mem (MB)</th>
            <th>Cgroup Limit (MB)</th>
            <th>Cgroup %</th>
            <th>Sessions</th>
            <th>Session KV (MiB)</th>
            <th>Max Proc CPU (%)</th>
            <th>Max Sys CPU (%)</th>
            <th>Net Delta (MiB)</th>
            <th>Transport Payload (MiB)</th>
            <th>Max Mbps</th>
            <th>GGML Threads</th>
          </tr>
        </thead>
        <tbody>${stageTableRows || "<tr><td colspan='20'>No stage metrics found.</td></tr>"}</tbody>
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
            <th>RPC Wall (ms)</th>
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
            <th>RPC Wall (ms)</th>
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
    const data = await fetchJson(
      `/api/topology?namespace=${encodeURIComponent(namespace)}&${runtimeQuery()}`
    );
    if (Array.isArray(data.runtimeVariants) && data.runtimeVariants.length) {
      runtimeVariantCatalog = data.runtimeVariants;
    }
    const runtimeLabel = data.runtimeVariant?.label || getRuntimeVariantInfo().label || getRuntimeVariant();
    els.clusterMeta.textContent = `runtime=${runtimeLabel} | namespace=${data.namespace} | updated=${new Date(
      data.generatedAt
    ).toLocaleString()}`;
    renderOverview(data);
    renderWorkloads(data);
    const podNodeEntries = [];
    for (const pod of data.pods || []) {
      if (pod?.name) {
        podNodeEntries.push([String(pod.name), String(pod.node || "")]);
      }
      if (pod?.app) {
        podNodeEntries.push([String(pod.app), String(pod.node || "")]);
      }
    }
    podToNodeMap = new Map(podNodeEntries);
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
      )}&source=${encodeURIComponent(source)}&tail=${tail}&previous=${previous}&${runtimeQuery()}`
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
    if (parsedBody && typeof parsedBody === "object" && !Array.isArray(parsedBody)) {
      if (supportsFeatureFlags()) {
        const flags = readFeatureFlagsFromControls();
        if (isBaselineFeatureFlags(flags)) {
          delete parsedBody.feature_flags;
        } else {
          parsedBody.feature_flags = flags;
        }
      } else {
        delete parsedBody.feature_flags;
      }
      els.endpointBody.value = prettyJson(parsedBody);
    }
  }

  const requestPayload = {
    namespace,
    variant: getRuntimeVariant(),
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
      <div class="log-pill">runtime: ${safeText(payload.variant || getRuntimeVariant())}</div>
      <div class="log-pill">target: ${safeText(payload.target)}</div>
      <div class="log-pill">path: ${safeText(payload.path)}</div>
      <div class="log-pill">pod: ${safeText(payload.podName)}</div>
    `;

    const responseJson = normalizeResponseMetrics(payload.responseJson, payload.call || {});
    renderEndpointReceivedText(responseJson);
    renderInvokeMetrics(responseJson, payload.call || {}, {
      ...parseGenerateConfigFromBody(parsedBody),
      timeoutMs,
      method,
      path,
      target,
      variant: getRuntimeVariant()
    });
    updateLatestModuleState(responseJson, "invoke");
    renderArchitectureGuide();

    const formatted = responseJson != null
      ? prettyJson(responseJson)
      : payload.responseText || "";
    els.invokeOutput.textContent = formatted;
  } catch (error) {
    els.invokeMeta.innerHTML = `<div class="log-pill line-err">Invoke error: ${escapeHtml(
      error.message
    )}</div>`;
    clearEndpointReceivedText("No received text because the endpoint call failed.");
    els.invokeMetricsDashboard.innerHTML = "";
    els.invokeOutput.textContent = "";
  } finally {
    setInvokeBusy(false);
    loadRuntime();
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

function summarizeTurnMetrics(response, requestConfig = {}) {
  response = normalizeResponseMetrics(response);
  const summary = response.summary_metrics || {};
  const aggregate = summary.aggregate || {};
  const featureFlags = asObject(summary.feature_flags);
  const enabledModules = deriveEnabledModules(featureFlags, summary.enabled_modules);
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
      rpc_wall_time_ms_sum: row.transferMs,
      true_comm_ms_sum: row.trueCommMs,
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
    transfer_ms_sum: Number(aggregate.rpc_wall_time_ms_sum ?? aggregate.transfer_time_ms_sum ?? 0),
    rpc_wall_ms_sum: Number(aggregate.rpc_wall_time_ms_sum ?? aggregate.transfer_time_ms_sum ?? 0),
    true_comm_ms_sum: Number(aggregate.true_comm_ms_sum || 0),
    max_memory_mb: Number(aggregate.max_process_memory_mb || 0),
    network_mib_sum: Number(aggregate.network_delta_mebibytes_sum || 0),
    payload_mib_sum: Number(aggregate.payload_mebibytes_sum ?? aggregate.payload_b64_mebibytes_sum ?? 0),
    token_latency_p50_ms: percentile(tokenRows.map((row) => row.latencyMs), 0.5),
    token_latency_p95_ms: percentile(tokenRows.map((row) => row.latencyMs), 0.95),
    token_latency_p99_ms: percentile(tokenRows.map((row) => row.latencyMs), 0.99),
    transfer_latency_p50_ms: percentile(tokenRows.map((row) => row.transferMs), 0.5),
    transfer_latency_p95_ms: percentile(tokenRows.map((row) => row.transferMs), 0.95),
    transfer_latency_p99_ms: percentile(tokenRows.map((row) => row.transferMs), 0.99),
    transfer_compute_ratio: Number(
      aggregate.true_comm_compute_ratio ?? aggregate.comm_compute_ratio ?? 0
    ),
    rpc_compute_ratio: Number(
      aggregate.rpc_compute_ratio ??
        Number(aggregate.transfer_time_ms_sum || 0) /
          Math.max(1e-9, Number(aggregate.compute_time_ms_sum || 0))
    ),
    true_comm_compute_ratio: Number(
      aggregate.true_comm_compute_ratio ?? aggregate.comm_compute_ratio ?? 0
    ),
    config: {
      promptChars: Number(requestConfig.promptChars || 0),
      maxNewTokens: Number(requestConfig.maxNewTokens || 0),
      minNewTokens: Number(requestConfig.minNewTokens || 0),
      temperature: Number(requestConfig.temperature || 0),
      topK: Number(requestConfig.topK || 0),
      topP: requestConfig.topP != null ? Number(requestConfig.topP) : 1,
      seed: requestConfig.seed != null ? Number(requestConfig.seed) : null,
      timeoutMs: Number(requestConfig.timeoutMs || 0),
      method: String(requestConfig.method || "POST"),
      path: String(requestConfig.path || "/chat"),
      target: String(requestConfig.target || "gateway")
    },
    is_baseline: enabledModules.length === 0,
    feature_flags: featureFlags,
    enabled_modules: enabledModules,
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
  const totalTrueComm = chatTurns.reduce((sum, turn) => sum + Number(turn.true_comm_ms_sum || 0), 0);
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
        trueCommMs: 0,
        maxMemMb: 0,
        maxCpuPct: 0,
        networkMib: 0,
        payloadMib: 0
      };
      curr.samples += Number(stage.samples || 0);
      curr.computeMs += Number(stage.compute_time_ms_sum || 0);
      curr.transferMs += Number(stage.rpc_wall_time_ms_sum ?? stage.transfer_time_ms_sum ?? 0);
      curr.trueCommMs += Number(stage.true_comm_ms_sum || 0);
      curr.maxMemMb = Math.max(curr.maxMemMb, Number(stage.max_process_memory_mb || 0));
      curr.maxCpuPct = Math.max(curr.maxCpuPct, Number(stage.max_cpu_percent || 0));
      curr.networkMib += Number(stage.network_delta_bytes_sum || 0) / (1024 * 1024);
      curr.payloadMib +=
        Number(stage.payload_bytes_sum ?? stage.payload_b64_bytes_sum ?? 0) / (1024 * 1024);
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
        trueCommMs: 0,
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
      curr.transferMs += Number(node.rpc_wall_time_ms_sum ?? node.transfer_time_ms_sum ?? 0);
      curr.trueCommMs += Number(node.true_comm_ms_sum || 0);
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
    { k: "RPC Wall Sum (ms)", v: formatNumber(totalTransfer, 1) },
    { k: "True Comm Sum (ms)", v: formatNumber(totalTrueComm, 1) },
    {
      k: "RPC/Compute Ratio",
      v: formatNumber(totalTransfer / Math.max(1e-9, totalCompute), 3)
    },
    {
      k: "True Comm/Compute Ratio",
      v: formatNumber(totalTrueComm / Math.max(1e-9, totalCompute), 3)
    },
    { k: "Peak Mem (MB)", v: formatNumber(peakMemory, 1) },
    { k: "Network Total (MiB)", v: formatNumber(totalNetwork, 3) },
    { k: "Payload Total (MiB)", v: formatNumber(totalPayload, 3) },
    { k: "Avg Token p50 (ms)", v: formatNumber(avgTokenP50, 1) },
    { k: "Avg Token p95 (ms)", v: formatNumber(avgTokenP95, 1) },
    { k: "Avg Token p99 (ms)", v: formatNumber(avgTokenP99, 1) },
    { k: "Avg RPC Wall p50 (ms)", v: formatNumber(avgTransferP50, 1) },
    { k: "Avg RPC Wall p95 (ms)", v: formatNumber(avgTransferP95, 1) },
    { k: "Avg RPC Wall p99 (ms)", v: formatNumber(avgTransferP99, 1) }
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
  const nodeTransferSeries = buildPerNodeSeries(
    chatTurns,
    (node) => node.rpc_wall_time_ms_sum ?? node.transfer_time_ms_sum
  );
  const nodeComputeSeries = buildPerNodeSeries(chatTurns, (node) => node.compute_time_ms_sum);
  const turnLatencySeries = [
    { name: "Latency ms", values: chatTurns.map((turn) => Number(turn.total_latency_ms || 0)) }
  ];
  const turnThroughputSeries = [
    { name: "Tokens/sec", values: chatTurns.map((turn) => Number(turn.tokens_per_second || 0)) }
  ];
  const turnRatioSeries = [
    {
      name: "RPC/Compute",
      values: chatTurns.map((turn) => Number(turn.rpc_compute_ratio || 0))
    },
    {
      name: "True Comm/Compute",
      values: chatTurns.map((turn) => Number(turn.true_comm_compute_ratio || 0))
    }
  ];
  const turnTokenPercentileSeries = [
    { name: "p50", values: chatTurns.map((turn) => Number(turn.token_latency_p50_ms || 0)) },
    { name: "p95", values: chatTurns.map((turn) => Number(turn.token_latency_p95_ms || 0)) },
    { name: "p99", values: chatTurns.map((turn) => Number(turn.token_latency_p99_ms || 0)) }
  ];
  const turnTransferPercentileSeries = [
    { name: "p50", values: chatTurns.map((turn) => Number(turn.transfer_latency_p50_ms || 0)) },
    { name: "p95", values: chatTurns.map((turn) => Number(turn.transfer_latency_p95_ms || 0)) },
    { name: "p99", values: chatTurns.map((turn) => Number(turn.transfer_latency_p99_ms || 0)) }
  ];

  const criticalPathSeries = [
    { name: "Gateway Wait", values: chatTurns.map((turn) => Number(asObject(turn.critical_path).gatewayWaitMs || 0)) },
    {
      name: "Gateway Serialization",
      values: chatTurns.map((turn) => Number(asObject(turn.critical_path).gatewaySerializationMs || 0))
    },
    { name: "Stage Compute", values: chatTurns.map((turn) => Number(asObject(turn.critical_path).stageComputeMs || 0)) },
    { name: "Stage RPC Wall", values: chatTurns.map((turn) => Number(asObject(turn.critical_path).stageTransferMs || 0)) }
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
      <h3>Turn Trend Charts</h3>
      <div class="chart-grid">
        <div class="chart-card">
          <h4>Latency per Turn (ms)</h4>
          ${buildLineChartSvg(turnLatencySeries, { yLabel: "ms" })}
        </div>
        <div class="chart-card">
          <h4>Throughput per Turn</h4>
          ${buildLineChartSvg(turnThroughputSeries, { yLabel: "tokens/sec" })}
        </div>
        <div class="chart-card">
          <h4>RPC/Compute + True Comm/Compute</h4>
          ${buildLineChartSvg(turnRatioSeries, { yLabel: "ratio" })}
        </div>
        <div class="chart-card">
          <h4>Token Latency p50/p95/p99</h4>
          ${buildLineChartSvg(turnTokenPercentileSeries, { yLabel: "ms" })}
        </div>
        <div class="chart-card">
          <h4>RPC Wall p50/p95/p99</h4>
          ${buildLineChartSvg(turnTransferPercentileSeries, { yLabel: "ms" })}
        </div>
      </div>
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
              <tr><td>Stage RPC Wall</td><td>${formatNumber(criticalStageTransfer, 1)}</td><td>${formatNumber(criticalStageTransfer / Math.max(1, totalTurns), 1)}</td></tr>
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
          <h4>Node RPC Wall Time (ms)</h4>
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
            <th>RPC Wall (ms)</th>
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
            <th>RPC Wall (ms)</th>
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
          <th>RPC Wall Sum (ms)</th>
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
  const featureFlags = readFeatureFlagsFromControls();
  const chatPath = supportsChatEndpoint() ? "/chat" : "/generate";

  const conversation = [...chatMessages, { role: "user", content: userText }];
  const requestConfig = {
    promptChars: userText.length,
    maxNewTokens,
    minNewTokens,
    temperature,
    topK: els.endpointTopK ? floorNumber(els.endpointTopK.value, 0, 0) : 0,
    topP: els.endpointTopP ? clampNumber(els.endpointTopP.value, 0, 1, 1) : 1,
    seed:
      els.endpointSeed && String(els.endpointSeed.value).trim() !== "" &&
      Number.isFinite(Number(els.endpointSeed.value))
        ? Math.round(Number(els.endpointSeed.value))
        : null,
    timeoutMs,
    method: "POST",
    path: chatPath,
    target: "gateway",
    variant: getRuntimeVariant()
  };
  setChatBusy(true);
  els.chatMeta.innerHTML = "";

  // The native gateway reads activation_precision from the top level of the
  // body (not feature_flags). Drive it from the always-visible precision select
  // so it works for the native variant (where the feature-flags panel is hidden).
  const chatActivationPrecision = els.endpointPrecision
    ? String(els.endpointPrecision.value || "fp32")
    : "fp32";
  const activationPrecisionField =
    chatActivationPrecision === "fp16"
      ? { activation_precision: "fp16" }
      : {};

  // Sampling controls (shared with the endpoint tester). top_k 0 = off,
  // top_p >= 1 = off; only send them when active.
  const chatTopK = els.endpointTopK ? floorNumber(els.endpointTopK.value, 0, 0) : 0;
  const chatTopP = els.endpointTopP ? clampNumber(els.endpointTopP.value, 0, 1, 1) : 1;
  const chatSeedRaw = els.endpointSeed ? String(els.endpointSeed.value).trim() : "";
  const samplingFields = {
    ...(chatTopK > 0 ? { top_k: Math.round(chatTopK) } : {}),
    ...(chatTopP > 0 && chatTopP < 1 ? { top_p: chatTopP } : {}),
    ...(chatSeedRaw !== "" && Number.isFinite(Number(chatSeedRaw))
      ? { seed: Math.round(Number(chatSeedRaw)) }
      : {})
  };

  try {
    const invokePayload = {
      namespace,
      variant: getRuntimeVariant(),
      target: "gateway",
      method: "POST",
      path: chatPath,
      timeoutMs,
      body: supportsChatEndpoint()
        ? {
            messages: conversation,
            max_new_tokens: maxNewTokens,
            min_new_tokens: minNewTokens,
            temperature,
            ...samplingFields,
            ...activationPrecisionField,
            ...(supportsFeatureFlags() ? { feature_flags: featureFlags } : {})
          }
        : {
            prompt: userText,
            max_new_tokens: maxNewTokens,
            min_new_tokens: minNewTokens,
            temperature,
            ...samplingFields,
            ...activationPrecisionField
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

    const response = normalizeResponseMetrics(payload.responseJson, payload.call || {});
    const assistantText = String(response.assistant_message || response.generated_text || "").trim();
    const turnSummary = summarizeTurnMetrics(response, requestConfig);
    chatMessages = [...conversation, { role: "assistant", content: assistantText || "(empty response)" }];
    chatTurns.push(turnSummary);
    updateLatestModuleState(response, "chat");
    els.chatInput.value = "";

    renderChatMessages();
    renderChatMetricsTimeline();
    renderArchitectureGuide();
    persistSessionState();
    postHistoryEntries([
      {
        type: "chat",
        createdAt: turnSummary.timestamp,
        namespace: getNamespace(),
        target: "gateway",
        method: "POST",
        path: chatPath,
        config: {
          promptChars: turnSummary.config?.promptChars || 0,
          promptTokens: turnSummary.prompt_tokens || 0,
          maxNewTokens: turnSummary.config?.maxNewTokens || 0,
          minNewTokens: turnSummary.config?.minNewTokens || 0,
          temperature: turnSummary.config?.temperature || 0,
          topK: turnSummary.config?.topK || 0,
          topP: turnSummary.config?.topP != null ? turnSummary.config.topP : 1,
          seed: turnSummary.config?.seed != null ? turnSummary.config.seed : null,
          timeoutMs: turnSummary.config?.timeoutMs || 0,
          topologyHash: buildTopologyHash()
        },
        profile: {
          baseline: turnSummary.is_baseline,
          enabledModules: turnSummary.enabled_modules,
          featureFlags: turnSummary.feature_flags,
          moduleVariants: getFeatureFlagProfileChips(turnSummary.feature_flags)
        },
        metrics: {
          generatedTokens: turnSummary.generated_tokens,
          latencyMs: turnSummary.total_latency_ms,
          tokensPerSecond: turnSummary.tokens_per_second,
          computeMs: turnSummary.compute_ms_sum,
          transferMs: turnSummary.transfer_ms_sum,
          trueCommMs: turnSummary.true_comm_ms_sum,
          transferComputeRatio: turnSummary.transfer_compute_ratio,
          rpcComputeRatio: turnSummary.rpc_compute_ratio,
          trueCommComputeRatio: turnSummary.true_comm_compute_ratio,
          payloadMib: turnSummary.payload_mib_sum,
          networkMib: turnSummary.network_mib_sum,
          maxMemoryMb: turnSummary.max_memory_mb,
          tokenP50Ms: turnSummary.token_latency_p50_ms,
          tokenP95Ms: turnSummary.token_latency_p95_ms,
          tokenP99Ms: turnSummary.token_latency_p99_ms,
          transferP50Ms: turnSummary.transfer_latency_p50_ms,
          transferP95Ms: turnSummary.transfer_latency_p95_ms,
          transferP99Ms: turnSummary.transfer_latency_p99_ms,
          criticalPath: turnSummary.critical_path,
          perStage: Object.entries(asObject(turnSummary.per_stage)).map(([stageKey, stage]) => ({
            stageKey,
            ...asObject(stage)
          })),
          perNode: Object.values(asObject(turnSummary.per_node)).map((node) => asObject(node))
        },
        output: buildHistoryOutputPayload(response),
        dashboard: {
          kind: "chat",
          turnSummary,
          chatTurns,
          conversation: chatMessages,
          requestConfig,
          callMeta: payload.call || {},
          responseJson: response
        }
      }
    ]);

    els.chatMeta.innerHTML = `
      <div class="log-pill">turn: ${chatTurns.length}</div>
      <div class="log-pill">runtime: ${escapeHtml(getRuntimeVariant())}</div>
      <div class="log-pill">path: ${escapeHtml(chatPath)}</div>
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
    loadRuntime();
  }
}

function clearChatSession() {
  chatMessages = [];
  chatTurns = [];
  latestChatFeatureFlags = null;
  latestChatEnabledModules = [];
  renderChatMessages();
  renderChatMetricsTimeline();
  renderArchitectureGuide();
  persistSessionState();
  els.chatMeta.innerHTML = `<div class="log-pill">Chat session cleared.</div>`;
}

function clearEndpointRunSession() {
  invokeRuns = [];
  latestInvokeFeatureFlags = null;
  latestInvokeEnabledModules = [];
  persistSessionState();
  renderArchitectureGuide();
  els.invokeMeta.innerHTML = `<div class="log-pill">Endpoint session results cleared.</div>`;
  els.invokeMetricsDashboard.innerHTML =
    `<div class="muted">No endpoint run history in this browser session yet.</div>`;
  clearEndpointReceivedText();
  els.invokeOutput.textContent = "";
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
    loadRuntime();
  }, seconds * 1000);
}

function reloadRuntimeScopedData() {
  featureModuleCatalog = [];
  runtimeSnapshot = null;
  latestInvokeFeatureFlags = null;
  latestInvokeEnabledModules = [];
  latestChatFeatureFlags = null;
  latestChatEnabledModules = [];
  syncRuntimeUiState();
  loadTopology();
  loadLogs();
  loadRuntime();
  loadEndpointCatalog().catch((error) => {
    els.invokeMeta.innerHTML = `<div class="log-pill line-err">Catalog error: ${escapeHtml(
      error.message
    )}</div>`;
  });
  loadFeatureModuleCatalog();
  renderArchitectureGuide();
  updateChatContextInfo();
  persistSessionState();
}

els.refreshTopologyBtn.addEventListener("click", () => {
  loadTopology();
  loadRuntime();
});

els.refreshLogsBtn.addEventListener("click", () => {
  loadLogs();
});

els.autoRefresh.addEventListener("change", updateAutoRefresh);
els.refreshSeconds.addEventListener("change", updateAutoRefresh);
if (els.runtimeVariant) {
  els.runtimeVariant.addEventListener("change", reloadRuntimeScopedData);
}
els.logTarget.addEventListener("change", loadLogs);
els.logSource.addEventListener("change", loadLogs);
els.namespaceInput.addEventListener("change", () => {
  loadTopology();
  loadLogs();
  loadRuntime();
  loadEndpointCatalog().catch((error) => {
    els.invokeMeta.innerHTML = `<div class="log-pill line-err">Catalog error: ${escapeHtml(
      error.message
    )}</div>`;
  });
  loadFeatureModuleCatalog();
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
if (els.endpointTopK) {
  els.endpointTopK.addEventListener("input", syncBodyFromGenerationFields);
}
if (els.endpointTopP) {
  els.endpointTopP.addEventListener("input", syncBodyFromGenerationFields);
}
if (els.endpointSeed) {
  els.endpointSeed.addEventListener("input", syncBodyFromGenerationFields);
}
if (els.endpointPrecision) {
  els.endpointPrecision.addEventListener("change", syncBodyFromGenerationFields);
}
[
  els.featureTransportJson,
  els.featureTransportBinary,
  els.featurePrecisionFp32,
  els.featurePrecisionFp16,
  els.featurePrecisionBf16,
  els.featurePrecisionInt8,
  els.featureRebalanceBaseline,
  els.featureRebalanceLatency,
  els.featureKvCache,
  els.featureForwardDedupe,
  els.featureTopologyAware,
  els.featurePersistentSessions,
  els.featureBackpressure,
  els.featureBackpressureQueue
]
  .filter(Boolean)
  .forEach((control) => {
    control.addEventListener("input", () => {
      syncFeatureFlagControlState();
      syncBodyFromFeatureFlagControls();
    });
    control.addEventListener("change", () => {
      syncFeatureFlagControlState();
      syncBodyFromFeatureFlagControls();
    });
  });
if (els.featureFlagsResetBtn) {
  els.featureFlagsResetBtn.addEventListener("click", () => {
    applyFeatureFlagControls(FEATURE_FLAG_DEFAULTS);
    syncBodyFromFeatureFlagControls();
  });
}
els.invokeEndpointBtn.addEventListener("click", invokeEndpoint);
if (els.invokeClearRunsBtn) {
  els.invokeClearRunsBtn.addEventListener("click", clearEndpointRunSession);
}
els.chatSendBtn.addEventListener("click", sendChatTurn);
els.chatClearBtn.addEventListener("click", clearChatSession);
els.chatExportBtn.addEventListener("click", exportChatMetrics);
els.chatInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    sendChatTurn();
  }
});

const initialParams = new URLSearchParams(window.location.search);
const initialNamespace = initialParams.get("namespace");
if (initialNamespace) {
  els.namespaceInput.value = initialNamespace;
}

restoreSessionState();
const initialRuntimeVariant = initialParams.get("variant") || initialParams.get("runtime");
if (initialRuntimeVariant && els.runtimeVariant) {
  els.runtimeVariant.value = String(initialRuntimeVariant).toLowerCase() === "native" ? "native" : "python";
}
syncRuntimeUiState();
if (invokeRuns.length) {
  els.invokeMetricsDashboard.innerHTML = renderInvokeRunEvolution();
}
renderArchitectureGuide();
loadTopology();
loadLogs();
loadRuntime();
loadEndpointCatalog().catch((error) => {
  els.invokeMeta.innerHTML = `<div class="log-pill line-err">Catalog error: ${escapeHtml(
    error.message
  )}</div>`;
});
loadFeatureModuleCatalog();
renderChatMessages();
renderChatMetricsTimeline();
updateChatContextInfo();
syncGenerationControlsEnabledState();
updateAutoRefresh();

/* =========================================================================
 * Automated Test Suites — replay §11 cases against /generate, save to History.
 * Reuses renderInvokeMetrics() (same save-to-history path as the manual tester).
 * ========================================================================= */
(function initAutoSuites() {
  const AUTO_PROMPT =
    "Explain how to overcome the challenges introduced by artificial intelligence in university education.";

  const G = {
    temp: 0, topK: 0, topP: 1, seed: "", precision: "fp32", routing: "hub",
    enabled: true, nOverride: "", threads: "", attn: ""
  };
  const row = (o) => ({ maxNew: o.tokens, minNew: o.tokens, ...G, ...o });

  // §11.11 decoding presets, reused across the length sweep.
  const P11 = {
    greedy: { temp: 0, topK: 0, topP: 1, seed: "" },
    conservative: { temp: 0.4, topK: 20, topP: 0.85, seed: 42 },
    balanced: { temp: 0.7, topK: 40, topP: 0.9, seed: 42 },
    diverse: { temp: 1.0, topK: 80, topP: 0.95, seed: 42 }
  };
  const LENS = [256, 512, 1024, 2048, 4096];

  // Built-in suites — one entry per §11.10 matrix row. HTTP-drivable knobs are applied
  // from the /generate body; threads/attn rows only take effect with the kubectl sweep on.
  const SUITES = {
    token_scaling: () => [128, 512, 1024, 2048, 4096].map((t) => row({ label: `11.1-token-${t}`, tokens: t })),
    attention_11_2a: () => ["1", "0"].map((a) => row({ label: `11.2a-attn-${a}`, tokens: 1024, attn: a })),
    threads_11_2b: () => ["1", "2", "3", "4"].map((t) => row({ label: `11.2b-threads-${t}`, tokens: 1024, threads: t })),
    concurrency_11_4: () => [1, 2, 4, 8].map((n) => row({ label: `11.4-conc-N${n}`, tokens: 256, nOverride: String(n) })),
    precision_11_6: () => [
      row({ label: "11.6-fp32", tokens: 512, seed: 42, precision: "fp32" }),
      row({ label: "11.6-fp16", tokens: 512, seed: 42, precision: "fp16" })
    ],
    fp16_sampling_11_6c: () => [
      row({ label: "11.6c-fp16-balanced", tokens: 512, minNew: 8, seed: 42, temp: 0.7, topK: 40, topP: 0.9, precision: "fp16" })
    ],
    temperature_11_7a: () => [0, 0.2, 0.5, 0.7, 1.0, 1.2].map((t) =>
      row({ label: `11.7a-temp-${t}`, tokens: 256, minNew: 8, seed: 42, temp: t, topK: 40, topP: 0.9 })),
    topk_11_7b: () => [0, 20, 40, 80].map((k) =>
      row({ label: `11.7b-topk-${k}`, tokens: 256, minNew: 8, seed: 42, temp: 0.7, topK: k, topP: 0.9 })),
    topp_11_7c: () => [0.8, 0.9, 0.95, 1.0].map((p) =>
      row({ label: `11.7c-topp-${p}`, tokens: 256, minNew: 8, seed: 42, temp: 0.7, topK: 40, topP: p })),
    longctx_11_7d: () => [row({ label: "11.7d-longctx-2048", tokens: 2048, seed: 42, temp: 0.7, topK: 40, topP: 0.9 })],
    correctness_11_8: () => [
      row({ label: "11.8-determinism-a", tokens: 256, minNew: 8, seed: 42, temp: 0.8, topK: 40, topP: 0.9 }),
      row({ label: "11.8-determinism-b", tokens: 256, minNew: 8, seed: 42, temp: 0.8, topK: 40, topP: 0.9 }),
      row({ label: "11.8-diversity-a", tokens: 256, minNew: 8, seed: "", temp: 0.8, topK: 40, topP: 0.9 }),
      row({ label: "11.8-diversity-b", tokens: 256, minNew: 8, seed: "", temp: 0.8, topK: 40, topP: 0.9 }),
      row({ label: "11.8-topk1-greedy", tokens: 256, minNew: 8, seed: 42, temp: 1.0, topK: 1, topP: 1 })
    ],
    endurance_11_9: () => [row({ label: "11.9-endurance-2048", tokens: 2048 })],
    length_x_sampling_11_11: () => {
      const out = [];
      for (const [name, p] of Object.entries(P11)) {
        for (const t of LENS) out.push(row({ label: `11.11-${name}-${t}`, tokens: t, ...p }));
      }
      return out;
    },
    routing_11_13: () => [
      row({ label: "11.13-routing-hub", tokens: 1024, routing: "hub" }),
      row({ label: "11.13-routing-chain", tokens: 1024, routing: "chain" })
    ],
    smoke_all: () => [
      ...SUITES.token_scaling(),
      ...SUITES.precision_11_6(),
      ...SUITES.temperature_11_7a(),
      ...SUITES.routing_11_13()
    ]
  };

  // Dropdown label per suite (order preserved).
  const SUITE_LABELS = {
    token_scaling: "11.1 token scaling",
    attention_11_2a: "11.2a attention (needs kubectl sweep)",
    threads_11_2b: "11.2b threads (needs kubectl sweep)",
    concurrency_11_4: "11.4 concurrency (use concurrency mode)",
    precision_11_6: "11.6 precision fp32/fp16",
    fp16_sampling_11_6c: "11.6c fp16 + sampling",
    temperature_11_7a: "11.7a temperature",
    topk_11_7b: "11.7b top_k",
    topp_11_7c: "11.7c top_p",
    longctx_11_7d: "11.7d long-context quality",
    correctness_11_8: "11.8 determinism / diversity",
    endurance_11_9: "11.9 endurance",
    length_x_sampling_11_11: "11.11 length × sampling (20 cases)",
    routing_11_13: "11.13 routing hub vs chain",
    smoke_all: "smoke (11.1 + 11.6 + 11.7a + 11.13)"
  };

  // Suites whose effect requires the opt-in kubectl env sweep.
  const NEEDS_CLUSTER = new Set(["attention_11_2a", "threads_11_2b"]);

  const $ = (id) => document.getElementById(id);
  const body = $("autoTestBody");
  if (!body) return;
  const attr = (s) =>
    String(s == null ? "" : s).replace(/&/g, "&amp;").replace(/"/g, "&quot;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  const mean = (a) => (a.length ? a.reduce((x, y) => x + y, 0) / a.length : 0);

  let stopFlag = false;

  function opt(v, cur, txt) {
    return `<option value="${attr(v)}"${String(cur) === String(v) ? " selected" : ""}>${attr(txt || v)}</option>`;
  }

  function rowHtml(r) {
    return `<tr>
      <td><input type="checkbox" class="a-on"${r.enabled ? " checked" : ""}></td>
      <td><input type="text" class="a-label auto-label" value="${attr(r.label)}"></td>
      <td><select class="a-routing">${opt("hub", r.routing)}${opt("chain", r.routing)}</select></td>
      <td><input type="number" class="a-nover" min="1" step="1" placeholder="N" value="${attr(r.nOverride)}"></td>
      <td><input type="number" class="a-maxnew" min="1" step="1" value="${attr(r.maxNew)}"></td>
      <td><input type="number" class="a-minnew" min="1" step="1" value="${attr(r.minNew)}"></td>
      <td><input type="number" class="a-temp" min="0" max="2" step="0.1" value="${attr(r.temp)}"></td>
      <td><input type="number" class="a-topk" min="0" step="1" value="${attr(r.topK)}"></td>
      <td><input type="number" class="a-topp" min="0" max="1" step="0.05" value="${attr(r.topP)}"></td>
      <td><input type="text" class="a-seed" placeholder="—" value="${attr(r.seed)}"></td>
      <td><select class="a-prec">${opt("fp32", r.precision)}${opt("fp16", r.precision)}</select></td>
      <td><input type="number" class="a-threads" min="1" max="4" step="1" placeholder="—" value="${attr(r.threads)}"></td>
      <td><input type="text" class="a-attn" placeholder="—" value="${attr(r.attn)}"></td>
      <td class="auto-remove auto-rowops">
        <button type="button" class="a-up" title="Move up">▲</button>
        <button type="button" class="a-down" title="Move down">▼</button>
        <button type="button" class="a-remove danger-button" title="Remove row">✕</button>
      </td>
    </tr>`;
  }

  function renderRows(rows) {
    body.innerHTML = rows.map(rowHtml).join("");
  }

  /* ---- suite catalog: built-ins + user-defined custom suites (localStorage) ---- */
  const CUSTOM_KEY = "opsui.autoCustomSuites.v1";
  function loadCustom() {
    try {
      const o = JSON.parse(localStorage.getItem(CUSTOM_KEY) || "{}");
      return o && typeof o === "object" && !Array.isArray(o) ? o : {};
    } catch {
      return {};
    }
  }
  function saveCustom(o) {
    try {
      localStorage.setItem(CUSTOM_KEY, JSON.stringify(o));
    } catch {
      /* ignore quota */
    }
  }

  function refreshSuiteOptions(selected) {
    const sel = $("autoSuite");
    if (!sel) return;
    const current = selected || sel.value;
    const builtin = Object.keys(SUITE_LABELS)
      .map((k) => `<option value="${attr(k)}">${attr(SUITE_LABELS[k])}</option>`)
      .join("");
    const custom = loadCustom();
    const customKeys = Object.keys(custom).sort();
    const customOpts = customKeys.length
      ? `<optgroup label="Custom suites">${customKeys
          .map((k) => `<option value="custom:${attr(k)}">${attr(k)} (${(custom[k] || []).length})</option>`)
          .join("")}</optgroup>`
      : "";
    sel.innerHTML = `<optgroup label="Built-in (§11)">${builtin}</optgroup>${customOpts}`;
    if (current) sel.value = current;
    if (!sel.value) sel.value = "token_scaling";
  }

  function suiteRows(value) {
    if (value && value.startsWith("custom:")) {
      const rows = loadCustom()[value.slice(7)] || [];
      return rows.map((r) => ({ ...G, ...r })); // re-hydrate against current defaults
    }
    return (SUITES[value] || SUITES.token_scaling)();
  }

  function loadDefaults() {
    const value = ($("autoSuite") && $("autoSuite").value) || "token_scaling";
    renderRows(suiteRows(value));
    // NB: do NOT clear the log here — loading/building a suite should not wipe the
    // previous run's console. The log is only reset when a fresh queue starts running.
    logLine(`Loaded ${body.children.length} case(s) for "${value}".`);
    if (NEEDS_CLUSTER.has(value) && (!$("autoClusterSweep") || !$("autoClusterSweep").checked)) {
      logLine(`Note: this suite sweeps threads/attn — tick "Sweep threads/attn via kubectl" or those columns are ignored.`);
    }
  }

  function saveAsSuite() {
    const rows = readRows().map(({ el, ...rest }) => rest); // drop DOM ref
    if (!rows.length) {
      logLine("Nothing to save — the table is empty.");
      return;
    }
    const name = (window.prompt("Save current table as a new suite.\nName:", "") || "").trim();
    if (!name) return;
    if (SUITE_LABELS[name]) {
      logLine(`"${name}" collides with a built-in key — pick another name.`);
      return;
    }
    const custom = loadCustom();
    const existed = Object.prototype.hasOwnProperty.call(custom, name);
    custom[name] = rows;
    saveCustom(custom);
    refreshSuiteOptions(`custom:${name}`);
    logLine(`${existed ? "Updated" : "Saved"} custom suite "${name}" with ${rows.length} row(s).`);
  }

  function deleteSuite() {
    const value = ($("autoSuite") && $("autoSuite").value) || "";
    if (!value.startsWith("custom:")) {
      logLine("Select a custom suite to delete (built-in §11 suites can't be removed).");
      return;
    }
    const name = value.slice(7);
    const custom = loadCustom();
    delete custom[name];
    saveCustom(custom);
    refreshSuiteOptions("token_scaling");
    loadDefaults();
    logLine(`Deleted custom suite "${name}".`);
  }

  function readRows() {
    return Array.from(body.querySelectorAll("tr")).map((tr) => ({
      el: tr,
      enabled: tr.querySelector(".a-on").checked,
      label: tr.querySelector(".a-label").value.trim() || "auto",
      routing: tr.querySelector(".a-routing").value,
      nOverride: tr.querySelector(".a-nover").value.trim(),
      maxNew: Math.max(1, Number(tr.querySelector(".a-maxnew").value) || 1),
      minNew: Math.max(1, Number(tr.querySelector(".a-minnew").value) || 1),
      temp: Number(tr.querySelector(".a-temp").value) || 0,
      topK: Number(tr.querySelector(".a-topk").value) || 0,
      topP: tr.querySelector(".a-topp").value === "" ? 1 : Number(tr.querySelector(".a-topp").value),
      seed: tr.querySelector(".a-seed").value.trim(),
      precision: tr.querySelector(".a-prec").value,
      threads: tr.querySelector(".a-threads").value.trim(),
      attn: tr.querySelector(".a-attn").value.trim()
    }));
  }

  function buildBody(r) {
    const b = {
      prompt: AUTO_PROMPT,
      max_new_tokens: r.maxNew,
      min_new_tokens: r.minNew,
      temperature: r.temp,
      top_k: r.topK,
      top_p: r.topP
    };
    if (r.seed !== "" && r.seed != null && !Number.isNaN(Number(r.seed))) b.seed = Number(r.seed);
    if (r.precision === "fp16") b.activation_precision = "fp16";
    if (r.routing === "chain") b.native_stage_chaining_enabled = true;
    return b;
  }

  async function invokeOnce(r, label, timeoutMs, ctx = {}) {
    const at = new Date().toISOString();
    const meta = {
      suite: ctx.suite || "",
      mode: ctx.mode || "",
      sweep: ctx.sweep ? "yes" : "no",
      nUsed: ctx.n || 1,
      threads: ctx.threads || "",
      attn: ctx.attn || "",
      pass: ctx.pass || 1,
      caseKey: ctx.caseKey || label,
      caseLabel: ctx.caseLabel || label
    };
    const b = buildBody(r);
    const requestPayload = {
      namespace: getNamespace(),
      variant: getRuntimeVariant(),
      target: "gateway",
      method: "POST",
      path: "/generate",
      timeoutMs,
      body: b
    };
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
      if (!res.ok || !payload || !payload.responseJson) {
        throw new Error(payload?.error || `${res.status} ${res.statusText}`);
      }
      const responseJson = normalizeResponseMetrics(payload.responseJson, payload.call || {});
      // renderInvokeMetrics reads the label from els.endpointLabel at save time; set it
      // synchronously right before the (synchronous) render+save so parallel runs don't race.
      if (els.endpointLabel) els.endpointLabel.value = label;
      renderInvokeMetrics(responseJson, payload.call || {}, {
        ...parseGenerateConfigFromBody(b),
        timeoutMs,
        method: "POST",
        path: "/generate",
        target: "gateway",
        variant: getRuntimeVariant()
      });
      const agg = responseJson.aggregate_metrics || {};
      const out = {
        label,
        ...meta,
        status: "ok",
        latencyMs: Number(payload.call?.elapsedMs || responseJson.total_latency_ms || 0),
        tokPerSec: Number(responseJson.tokens_per_second || 0),
        computeMs: Number(agg.compute_ms || 0),
        rpcMs: Number(agg.rpc_wall_ms || 0),
        commMs: Number(agg.true_comm_ms || 0),
        payloadMib: Number(agg.transport_payload_mebibytes || 0),
        genTok: Number(responseJson.generated_token_count || 0),
        at
      };
      recordResult(out); // intermediate result — persisted + rendered immediately
      return out;
    } catch (e) {
      recordResult({
        label,
        ...meta,
        status: `fail: ${e.message}`,
        latencyMs: 0, tokPerSec: 0, computeMs: 0, rpcMs: 0, commMs: 0, payloadMib: 0, genTok: 0,
        at
      });
      throw e;
    }
  }

  /* ---- live intermediate-results tracker (crash-safe via localStorage) ---- */
  const RESULTS_KEY = "opsui.autoResults.v1";
  let results = [];
  function loadResults() {
    try {
      const raw = JSON.parse(localStorage.getItem(RESULTS_KEY) || "[]");
      results = Array.isArray(raw) ? raw : [];
    } catch {
      results = [];
    }
  }
  function saveResults() {
    try {
      localStorage.setItem(RESULTS_KEY, JSON.stringify(results.slice(-500)));
    } catch {
      /* quota / private mode — keep in-memory only */
    }
  }
  function recordResult(entry) {
    results.push(entry);
    if (results.length > 500) results = results.slice(-500);
    saveResults(); // persist BEFORE the next test starts, so nothing is lost mid-suite
    renderResults();
    renderAggregates(); // keep per-case aggregates live as runs land
  }
  const fmt = (n, d) => (Number.isFinite(n) ? n.toFixed(d) : "–");
  function renderResults() {
    const tb = $("autoResultsBody");
    if (!tb) return;
    tb.innerHTML = results
      .map((e, i) => {
        const ok = e.status === "ok";
        const sweepTag = e.sweep === "yes" ? `sweep t${e.threads || "?"}/a${e.attn || "?"}` : "";
        return `<tr class="${ok ? "auto-row-done" : "auto-row-fail"}">
          <td>${i + 1}</td>
          <td>${attr(e.label)}</td>
          <td>${attr(e.suite || "")}</td>
          <td>${attr(e.mode || "")}${e.sweep === "yes" ? ` <span class="auto-sweep-tag" title="${attr(sweepTag)}">⚙</span>` : ""}</td>
          <td>${attr(ok ? "ok" : e.status)}</td>
          <td>${fmt((e.latencyMs || 0) / 1000, 1)}</td>
          <td>${fmt(e.tokPerSec, 2)}</td>
          <td>${fmt((e.computeMs || 0) / 1000, 1)}</td>
          <td>${fmt((e.rpcMs || 0) / 1000, 1)}</td>
          <td>${fmt((e.commMs || 0) / 1000, 1)}</td>
          <td>${fmt(e.payloadMib, 1)}</td>
          <td>${e.genTok || 0}</td>
          <td>${e.at ? new Date(e.at).toLocaleTimeString() : ""}</td>
        </tr>`;
      })
      .join("");
    const c = $("autoResultsCount");
    if (c) c.textContent = results.length ? `${results.length} result(s)` : "";
  }
  function clearResults() {
    results = [];
    saveResults();
    renderResults();
    renderAggregates();
  }

  /* ---- per-case aggregation (median / min / max / mean over successful runs) ---- */
  const median = (arr) => {
    if (!arr.length) return NaN;
    const s = [...arr].sort((a, b) => a - b);
    const m = Math.floor(s.length / 2);
    return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
  };
  function computeAggregates() {
    const groups = new Map();
    for (const e of results) {
      if (e.status !== "ok") continue; // aggregate only successful runs
      const key = e.caseKey || e.label;
      if (!groups.has(key)) groups.set(key, []);
      groups.get(key).push(e);
    }
    const out = [];
    for (const [key, arr] of groups) {
      const f = arr[0];
      const lat = arr.map((x) => Number(x.latencyMs) || 0);
      const tps = arr.map((x) => Number(x.tokPerSec) || 0);
      out.push({
        caseKey: key,
        caseLabel: f.caseLabel || f.label,
        suite: f.suite || "",
        mode: f.mode || "",
        sweep: f.sweep || "no",
        threads: f.threads || "",
        attn: f.attn || "",
        n: arr.length,
        latencyMedianMs: median(lat),
        latencyMinMs: Math.min(...lat),
        latencyMaxMs: Math.max(...lat),
        latencyMeanMs: mean(lat),
        tokPerSecMedian: median(tps),
        tokPerSecMean: mean(tps),
        computeMeanMs: mean(arr.map((x) => Number(x.computeMs) || 0)),
        rpcMeanMs: mean(arr.map((x) => Number(x.rpcMs) || 0)),
        commMeanMs: mean(arr.map((x) => Number(x.commMs) || 0)),
        payloadMibMean: mean(arr.map((x) => Number(x.payloadMib) || 0))
      });
    }
    // stable-ish ordering: by suite then case label
    out.sort((a, b) => (a.suite + a.caseLabel).localeCompare(b.suite + b.caseLabel));
    return out;
  }
  function renderAggregates() {
    const tb = $("autoAggBody");
    if (!tb) return;
    const aggs = computeAggregates();
    tb.innerHTML = aggs
      .map(
        (a) => `<tr>
          <td>${attr(a.caseLabel)}</td>
          <td>${attr(a.suite)}</td>
          <td>${attr(a.mode)}${a.sweep === "yes" ? ` <span class="auto-sweep-tag" title="sweep t${attr(a.threads)}/a${attr(a.attn)}">⚙</span>` : ""}</td>
          <td>${a.sweep}</td>
          <td>${a.n}</td>
          <td>${fmt(a.latencyMedianMs / 1000, 1)}</td>
          <td>${fmt(a.latencyMinMs / 1000, 1)}</td>
          <td>${fmt(a.latencyMaxMs / 1000, 1)}</td>
          <td>${fmt(a.latencyMeanMs / 1000, 1)}</td>
          <td>${fmt(a.tokPerSecMedian, 2)}</td>
          <td>${fmt(a.computeMeanMs / 1000, 1)}</td>
          <td>${fmt(a.rpcMeanMs / 1000, 1)}</td>
          <td>${fmt(a.commMeanMs / 1000, 1)}</td>
        </tr>`
      )
      .join("");
    const c = $("autoAggCount");
    if (c) c.textContent = aggs.length ? `${aggs.length} case(s)` : "";
  }

  /* ---- CSV export: individual runs, aggregates, or both in one file ---- */
  const csvEsc = (s) => `"${String(s == null ? "" : s).replace(/"/g, '""')}"`;
  function download(name, text) {
    const blob = new Blob([text], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = name;
    a.click();
    URL.revokeObjectURL(url);
  }
  function runsCsvText() {
    const head = [
      "idx", "label", "case", "suite", "mode", "sweep", "pass", "n_used", "threads", "attn",
      "status", "latency_ms", "tok_per_sec", "compute_ms", "rpc_wall_ms", "true_comm_ms",
      "payload_mib", "generated_tokens", "timestamp"
    ];
    return [head.join(",")]
      .concat(
        results.map((e, i) =>
          [i + 1, csvEsc(e.label), csvEsc(e.caseLabel || ""), csvEsc(e.suite || ""), csvEsc(e.mode || ""),
           csvEsc(e.sweep || "no"), e.pass || 1, e.nUsed || 1, csvEsc(e.threads || ""), csvEsc(e.attn || ""),
           csvEsc(e.status), e.latencyMs || 0, e.tokPerSec || 0, e.computeMs || 0, e.rpcMs || 0, e.commMs || 0,
           e.payloadMib || 0, e.genTok || 0, csvEsc(e.at || "")].join(",")
        )
      )
      .join("\n");
  }
  function aggCsvText() {
    const head = [
      "case", "suite", "mode", "sweep", "threads", "attn", "n",
      "latency_median_ms", "latency_min_ms", "latency_max_ms", "latency_mean_ms",
      "tok_per_sec_median", "tok_per_sec_mean", "compute_mean_ms", "rpc_wall_mean_ms",
      "true_comm_mean_ms", "payload_mib_mean"
    ];
    return [head.join(",")]
      .concat(
        computeAggregates().map((a) =>
          [csvEsc(a.caseLabel), csvEsc(a.suite), csvEsc(a.mode), csvEsc(a.sweep), csvEsc(a.threads),
           csvEsc(a.attn), a.n, a.latencyMedianMs.toFixed(1), a.latencyMinMs.toFixed(1),
           a.latencyMaxMs.toFixed(1), a.latencyMeanMs.toFixed(1), a.tokPerSecMedian.toFixed(3),
           a.tokPerSecMean.toFixed(3), a.computeMeanMs.toFixed(1), a.rpcMeanMs.toFixed(1),
           a.commMeanMs.toFixed(1), a.payloadMibMean.toFixed(3)].join(",")
        )
      )
      .join("\n");
  }
  function exportResultsCsv() {
    if (!results.length) return;
    download(`auto-runs-${Date.now()}.csv`, runsCsvText());
  }
  function exportAggregatesCsv() {
    if (!results.length) return;
    download(`auto-aggregates-${Date.now()}.csv`, aggCsvText());
  }
  function exportAllCsv() {
    if (!results.length) return;
    // one file: individual runs, a blank line, then the aggregates section
    const text = `# INDIVIDUAL RUNS\n${runsCsvText()}\n\n# AGGREGATES (per case, over successful runs)\n${aggCsvText()}\n`;
    download(`auto-runs+aggregates-${Date.now()}.csv`, text);
  }

  const LOG_MAX_LINES = 800;
  function logLine(msg) {
    const el = $("autoLog");
    el.textContent += `${new Date().toLocaleTimeString()}  ${msg}\n`;
    const lines = el.textContent.split("\n");
    if (lines.length > LOG_MAX_LINES) el.textContent = lines.slice(-LOG_MAX_LINES).join("\n");
    el.scrollTop = el.scrollHeight;
  }
  // Run/Load/Add stay enabled while a queue runs so you can build & enqueue more suites.
  function setBusy(b) {
    $("autoRunBtn").textContent = b ? "Queue / running…" : "Run suite";
    $("autoStopBtn").disabled = !b;
    $("autoProgress").innerHTML = b ? `<div class="log-pill">queue processing…</div>` : "";
  }

  // Opt-in: patch the configmap (DLI_GGML_THREADS / DLI_GGML_ATTENTION), rollout-restart
  // the 4 stage deployments, and wait for readiness. Governor is host sysfs and stays
  // manual (§11.2c). Only fires when the sweep checkbox is on AND a row sets a value that
  // differs from what we last applied — so consecutive same-value rows don't re-patch.
  let appliedThreads = null;
  let appliedAttn = null;
  async function applyClusterConfig(threads, attn) {
    const payload = { namespace: getNamespace() };
    if (threads !== "" && threads != null) payload.threads = String(threads);
    if (attn !== "" && attn != null) payload.attention = String(attn);
    if (payload.threads == null && payload.attention == null) return;
    logLine(`  ↳ cluster: patch ${JSON.stringify({ threads: payload.threads, attn: payload.attention })} + rollout restart (waiting)…`);
    const res = await fetch("/api/cluster-config", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });
    const text = await res.text();
    let data = null;
    try { data = text ? JSON.parse(text) : null; } catch { data = null; }
    if (!res.ok || !data || data.ok !== true) {
      throw new Error(`cluster-config failed: ${data?.error || `${res.status} ${res.statusText}`}`);
    }
    if (payload.threads != null) appliedThreads = payload.threads;
    if (payload.attention != null) appliedAttn = payload.attention;
    logLine(`  ↳ cluster ready (verified ${JSON.stringify(data.verified || {})}).`);
  }

  /* ---- suite queue: enqueue snapshots and run them back-to-back ---- */
  let queue = [];
  let processing = false;
  let abortQueue = false;
  let currentJob = null;

  function snapshotJob() {
    const rows = readRows().filter((r) => r.enabled).map(({ el, ...rest }) => rest); // strip DOM ref
    return {
      id: `${Date.now()}-${Math.random().toString(36).slice(2, 6)}`,
      suite: ($("autoSuite") && $("autoSuite").value) || "",
      mode: $("autoMode").value,
      globalN: Math.max(1, Number($("autoN").value) || 1),
      R: Math.max(1, Math.floor(Number($("autoRepeat").value) || 1)),
      delay: Math.max(0, Number($("autoDelay").value) || 0),
      timeoutMs: Math.max(0, Number($("autoTimeout").value) || 0),
      clusterSweep: !!($("autoClusterSweep") && $("autoClusterSweep").checked),
      rows
    };
  }

  function renderQueue() {
    const tb = $("autoQueueBody");
    if (tb) {
      const cur = currentJob
        ? `<tr class="auto-row-active"><td>▶</td><td>${attr(currentJob.suite)}</td><td>${attr(currentJob.mode)}${currentJob.R > 1 ? ` ×${currentJob.R}` : ""}</td><td>${currentJob.rows.length}</td><td>${currentJob.clusterSweep ? "yes" : "no"}</td><td>running</td></tr>`
        : "";
      tb.innerHTML =
        cur +
        queue
          .map(
            (j, i) => `<tr>
              <td>${i + 1}</td>
              <td>${attr(j.suite)}</td>
              <td>${attr(j.mode)}${j.R > 1 ? ` ×${j.R}` : ""}</td>
              <td>${j.rows.length}</td>
              <td>${j.clusterSweep ? "yes" : "no"}</td>
              <td><button type="button" class="a-dequeue danger-button" data-id="${attr(j.id)}" title="Remove from queue">✕</button></td>
            </tr>`
          )
          .join("");
    }
    const c = $("autoQueueCount");
    if (c) c.textContent = queue.length ? `${queue.length} queued` : processing ? "running…" : "empty";
  }

  function enqueue() {
    const job = snapshotJob();
    if (!job.rows.length) {
      logLine("Nothing to queue — no enabled rows.");
      return;
    }
    queue.push(job);
    renderQueue();
    logLine(
      `Queued "${job.suite}" [${job.mode}${job.R > 1 ? ` ×${job.R}` : ""}] — ${job.rows.length} case(s). ${processing ? `will run after current (${queue.length} in queue).` : "starting…"}`
    );
    if (!processing) processQueue();
  }

  async function processQueue() {
    if (processing) return;
    processing = true;
    abortQueue = false;
    stopFlag = false;
    setBusy(true);
    logLine("──────── queue run started ────────");
    while (queue.length && !abortQueue) {
      currentJob = queue.shift();
      renderQueue();
      stopFlag = false; // a fresh job starts un-stopped (Stop clears the whole queue instead)
      await runJob(currentJob);
      currentJob = null;
      renderQueue();
      const gap = Math.max(0, Number($("autoSuiteGap").value) || 0);
      if (queue.length && !abortQueue && gap) {
        logLine(`… ${gap}ms before next suite (${queue.length} left) …`);
        await sleep(gap);
      }
    }
    processing = false;
    currentJob = null;
    renderQueue();
    setBusy(false);
    logLine(abortQueue ? "Queue stopped." : "Queue drained — all suites done.");
    if (typeof loadRuntime === "function") loadRuntime();
  }

  async function runJob(job) {
    const { mode, suite, globalN, R, delay, timeoutMs, clusterSweep, rows } = job;
    const total = rows.length * R;
    logLine(
      `▶ Suite "${suite}" [${mode}${R > 1 ? ` × ${R} passes = ${total}` : ""}]${mode === "simple" ? "" : `, N=${globalN}`}, gap ${delay}ms${clusterSweep ? ", cluster env sweep ON" : ""}.`
    );
    appliedThreads = null;
    appliedAttn = null;
    let i = 0;
    for (let pass = 1; pass <= R && !stopFlag; pass++) {
      if (R > 1) logLine(`— pass ${pass}/${R} —`);
      for (const r of rows) {
        if (stopFlag) {
          logLine("Stopped by user.");
          break;
        }
        i++;
        const N = r.nOverride !== "" && Number(r.nOverride) > 0 ? Math.floor(Number(r.nOverride)) : globalN;
        const passTag = R > 1 ? ` p${pass}/${R}` : "";
        try {
          if (clusterSweep && (r.threads !== "" || r.attn !== "")) {
            const needThreads = r.threads !== "" && r.threads !== appliedThreads;
            const needAttn = r.attn !== "" && r.attn !== appliedAttn;
            if (needThreads || needAttn) {
              await applyClusterConfig(needThreads ? r.threads : "", needAttn ? r.attn : "");
            }
          }
          const sweepDesc = clusterSweep ? `t${appliedThreads || ""}a${appliedAttn || ""}` : "nosweep";
          const ctx = {
            mode,
            suite,
            sweep: clusterSweep,
            n: mode === "simple" ? 1 : N,
            threads: clusterSweep ? appliedThreads || "" : "",
            attn: clusterSweep ? appliedAttn || "" : "",
            pass,
            caseKey: `${suite}::${r.label}::${mode}::${sweepDesc}`,
            caseLabel: r.label
          };
          if (mode === "simple") {
            const out = await invokeOnce(r, `${r.label}${passTag}`, timeoutMs, ctx);
            logLine(`[${i}/${total}] ${r.label}${passTag} → ${out.latencyMs} ms, ${out.tokPerSec.toFixed(2)} tok/s`);
          } else if (mode === "batch") {
            const lat = [];
            for (let c = 1; c <= N && !stopFlag; c++) {
              const out = await invokeOnce(r, `${r.label}${passTag} b${c}/${N}`, timeoutMs, ctx);
              lat.push(out.latencyMs);
              logLine(`[${i}/${total}] ${r.label}${passTag} b${c}/${N} → ${out.latencyMs} ms`);
            }
            logLine(`[${i}/${total}] ${r.label}${passTag} batch mean ${mean(lat).toFixed(0)} ms over ${lat.length} run(s)`);
          } else {
            const settled = await Promise.allSettled(
              Array.from({ length: N }, (_, c) => invokeOnce(r, `${r.label}${passTag} c${c + 1}/${N}`, timeoutMs, ctx))
            );
            const ok = settled.filter((s) => s.status === "fulfilled");
            const lat = ok.map((s) => s.value.latencyMs);
            logLine(
              `[${i}/${total}] ${r.label}${passTag} concurrency N=${N}: ${ok.length} ok / ${settled.length - ok.length} dropped-or-failed, mean ${lat.length ? mean(lat).toFixed(0) : "–"} ms`
            );
          }
        } catch (e) {
          logLine(`[${i}/${total}] ${r.label}${passTag} ✗ ${e.message}`);
        }
        if (i < total && !stopFlag && delay) await sleep(delay);
      }
    }
    renderAggregates();
    logLine(`✔ Suite "${suite}" done (${i}/${total} run(s)).`);
  }

  function clearQueue() {
    const n = queue.length;
    queue = [];
    renderQueue();
    if (n) logLine(`Cleared ${n} queued suite(s) (current one keeps running).`);
  }

  $("autoLoadDefaultsBtn").addEventListener("click", loadDefaults);
  $("autoSuite").addEventListener("change", loadDefaults);
  $("autoAddRowBtn").addEventListener("click", () => {
    body.insertAdjacentHTML("beforeend", rowHtml(row({ label: "custom", tokens: 256 })));
  });
  body.addEventListener("click", (e) => {
    const btn = e.target.closest("button");
    if (!btn) return;
    const tr = btn.closest("tr");
    if (!tr) return;
    if (btn.classList.contains("a-remove")) {
      tr.remove();
    } else if (btn.classList.contains("a-up")) {
      const prev = tr.previousElementSibling;
      if (prev) tr.parentNode.insertBefore(tr, prev); // run order follows DOM order
    } else if (btn.classList.contains("a-down")) {
      const next = tr.nextElementSibling;
      if (next) tr.parentNode.insertBefore(next, tr);
    }
  });
  $("autoRunBtn").addEventListener("click", enqueue); // Run = enqueue current table; press again to schedule more
  $("autoStopBtn").addEventListener("click", () => {
    stopFlag = true; // halt the current suite after the in-flight request
    abortQueue = true; // and don't start any more queued suites
    const cleared = queue.length;
    queue = [];
    renderQueue();
    logLine(`Stop requested — halting current suite${cleared ? ` and clearing ${cleared} queued` : ""} (finishing in-flight request)…`);
  });
  if ($("autoClearQueueBtn")) $("autoClearQueueBtn").addEventListener("click", clearQueue);
  if ($("autoQueueBody")) {
    $("autoQueueBody").addEventListener("click", (e) => {
      const btn = e.target.closest(".a-dequeue");
      if (!btn) return;
      const id = btn.getAttribute("data-id");
      queue = queue.filter((j) => j.id !== id);
      renderQueue();
    });
  }
  if ($("autoClearResultsBtn")) $("autoClearResultsBtn").addEventListener("click", clearResults);
  if ($("autoExportResultsBtn")) $("autoExportResultsBtn").addEventListener("click", exportResultsCsv);
  if ($("autoExportAggBtn")) $("autoExportAggBtn").addEventListener("click", exportAggregatesCsv);
  if ($("autoExportAllBtn")) $("autoExportAllBtn").addEventListener("click", exportAllCsv);
  if ($("autoSaveSuiteBtn")) $("autoSaveSuiteBtn").addEventListener("click", saveAsSuite);
  if ($("autoDeleteSuiteBtn")) $("autoDeleteSuiteBtn").addEventListener("click", deleteSuite);

  loadResults();
  renderResults(); // restore any intermediate results from a previous / interrupted run
  renderAggregates(); // restore per-case aggregates too
  renderQueue();
  refreshSuiteOptions("token_scaling"); // build the dropdown (built-ins + saved custom suites)
  loadDefaults();
})();
