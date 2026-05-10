#!/usr/bin/env node
"use strict";

const fs = require("node:fs");
const fsp = require("node:fs/promises");
const http = require("node:http");
const path = require("node:path");
const { promisify } = require("node:util");
const { execFile, spawn } = require("node:child_process");

const execFileAsync = promisify(execFile);

const PORT = Number.parseInt(process.env.OPS_UI_PORT || "4070", 10);
const NAMESPACE = process.env.OPS_UI_NAMESPACE || "inference";
const KUBECTL_TIMEOUT_MS = Number.parseInt(process.env.OPS_UI_KUBECTL_TIMEOUT_MS || "15000", 10);
const INVOKE_TIMEOUT_MIN_MS = 1000;
const INVOKE_TIMEOUT_DEFAULT_MS = 15000;
const INVOKE_TIMEOUT_LONG_DEFAULT_MS = 300000;
const INVOKE_TIMEOUT_MAX_MS = 600000;
const PUBLIC_DIR = path.join(__dirname, "public");
const DATA_DIR = path.join(__dirname, "data");
const HISTORY_FILE = path.join(DATA_DIR, "metrics-history.ndjson");
const HISTORY_MAX_ENTRIES = clampNumber(process.env.OPS_UI_HISTORY_MAX_ENTRIES || "4000", 200, 20000, 4000);
const IN_FLIGHT_INVOKES = new Map();
const historyEntries = [];
let historyLoaded = false;
let nextHistorySeq = 1;
let historyWriteChain = Promise.resolve();
const runtimeStats = {
  startedAt: new Date().toISOString(),
  totalInvokes: 0,
  successfulInvokes: 0,
  failedInvokes: 0,
  droppedInFlight: 0,
  timeoutFailures: 0,
  lastTimeoutCauses: [],
  byPath: {}
};

const LOG_TARGETS = {
  gateway: {
    deployment: "inference-gateway-deployment",
    appLabel: "inference-gateway",
    appContainer: "inference-gateway",
    service: "inference-gateway"
  },
  "stage-1": {
    deployment: "inference-stage-1-deployment",
    appLabel: "inference-stage-1",
    appContainer: "inference-stage-1",
    service: "inference-stage-1"
  },
  "stage-2": {
    deployment: "inference-stage-2-deployment",
    appLabel: "inference-stage-2",
    appContainer: "inference-stage-2",
    service: "inference-stage-2"
  },
  "stage-3": {
    deployment: "inference-stage-3-deployment",
    appLabel: "inference-stage-3",
    appContainer: "inference-stage-3",
    service: "inference-stage-3"
  },
  "stage-4": {
    deployment: "inference-stage-4-deployment",
    appLabel: "inference-stage-4",
    appContainer: "inference-stage-4",
    service: "inference-stage-4"
  }
};

const MIME = {
  ".html": "text/html; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".js": "application/javascript; charset=utf-8",
  ".json": "application/json; charset=utf-8"
};
const NAMESPACE_NAME_RE = /^[a-z0-9]([-a-z0-9]*[a-z0-9])?$/;
const HTTP_METHOD_RE = /^(GET|POST|PUT|PATCH|DELETE|HEAD)$/i;

function clampNumber(value, min, max, fallback) {
  const parsed = Number.parseInt(value, 10);
  if (Number.isNaN(parsed)) {
    return fallback;
  }
  return Math.max(min, Math.min(max, parsed));
}

function buildLabelSelector(matchLabels) {
  const entries = Object.entries(matchLabels || {});
  if (!entries.length) {
    return "";
  }
  return entries.map(([k, v]) => `${k}=${v}`).join(",");
}

function sortPodsByCreationTime(items) {
  return [...items].sort((a, b) => {
    const aTs = new Date(a.metadata?.creationTimestamp || 0).getTime();
    const bTs = new Date(b.metadata?.creationTimestamp || 0).getTime();
    return aTs - bTs;
  });
}

function normalizePathForHttp(rawPath) {
  let pathValue = String(rawPath || "/health").trim();
  if (!pathValue.startsWith("/")) {
    pathValue = `/${pathValue}`;
  }
  return pathValue;
}

async function readJsonBody(req, { maxBytes = 1024 * 1024 } = {}) {
  return await new Promise((resolve, reject) => {
    const chunks = [];
    let size = 0;

    req.on("data", (chunk) => {
      size += chunk.length;
      if (size > maxBytes) {
        reject(new Error(`Request body too large (>${maxBytes} bytes).`));
        req.destroy();
        return;
      }
      chunks.push(chunk);
    });

    req.on("end", () => {
      if (!chunks.length) {
        resolve({});
        return;
      }
      try {
        const raw = Buffer.concat(chunks).toString("utf-8");
        resolve(JSON.parse(raw));
      } catch (error) {
        reject(new Error(`Invalid JSON body: ${String(error.message || error)}`));
      }
    });

    req.on("error", reject);
  });
}

async function resolvePodsForTarget(namespace, targetConfig) {
  let deployment = null;
  let selector = `app=${targetConfig.appLabel}`;
  let deploymentMissing = false;
  let matchedBy = "selector";

  try {
    deployment = await runKubectl(
      ["-n", namespace, "get", "deploy", targetConfig.deployment, "-o", "json"],
      { parseJson: true }
    );
    selector = buildLabelSelector(deployment.spec?.selector?.matchLabels) || selector;
  } catch (deploymentError) {
    const depErrText = `${String(deploymentError.message || deploymentError)} ${String(
      deploymentError.stderr || ""
    )}`;
    if (/deployments?.*not found/i.test(depErrText)) {
      deploymentMissing = true;
    } else {
      throw deploymentError;
    }
  }

  const selectedPods = await runKubectl(
    [
      "-n",
      namespace,
      "get",
      "pods",
      "-l",
      selector,
      "--sort-by=.metadata.creationTimestamp",
      "-o",
      "json"
    ],
    { parseJson: true }
  );

  let podItems = sortPodsByCreationTime(selectedPods.items || []);
  if (!podItems.length) {
    const allPods = await runKubectl(["-n", namespace, "get", "pods", "-o", "json"], {
      parseJson: true
    });
    const fallbackPrefix = `${targetConfig.deployment}-`;
    podItems = sortPodsByCreationTime(
      (allPods.items || []).filter((pod) => {
        const podName = pod.metadata?.name || "";
        const appLabel = pod.metadata?.labels?.app || "";
        return (
          podName.startsWith(fallbackPrefix) ||
          podName.includes(targetConfig.appLabel) ||
          appLabel === targetConfig.appLabel
        );
      })
    );
    matchedBy = "name-fallback";
  }

  return {
    deployment,
    deploymentMissing,
    selector,
    matchedBy,
    podItems
  };
}

async function buildHttpCandidates(namespace, targetConfig, pod) {
  const candidates = [];
  const podIp = pod?.status?.podIP;
  const podName = pod?.metadata?.name || null;

  if (podIp) {
    candidates.push({
      type: "podIP",
      baseUrl: `http://${podIp}:8000`,
      podName
    });
  }

  if (targetConfig.service) {
    try {
      const svc = await runKubectl(
        ["-n", namespace, "get", "svc", targetConfig.service, "-o", "json"],
        { parseJson: true }
      );
      const port = (svc.spec?.ports || []).find((p) => p.port === 8000)?.port
        || (svc.spec?.ports || [])[0]?.port
        || 8000;
      const clusterIp = svc.spec?.clusterIP;
      if (clusterIp && clusterIp !== "None") {
        candidates.push({
          type: "serviceClusterIP",
          baseUrl: `http://${clusterIp}:${port}`,
          service: targetConfig.service
        });
      }
      candidates.push({
        type: "serviceDNS",
        baseUrl: `http://${targetConfig.service}.${namespace}.svc.cluster.local:${port}`,
        service: targetConfig.service
      });
    } catch {
      // keep podIP candidates only
    }
  }

  return candidates;
}

async function runKubectl(args, { parseJson = false } = {}) {
  const { stdout } = await execFileAsync("kubectl", args, {
    timeout: KUBECTL_TIMEOUT_MS,
    maxBuffer: 8 * 1024 * 1024
  });

  if (!parseJson) {
    return stdout;
  }

  return JSON.parse(stdout);
}

function normalizeNode(node) {
  const labels = node.metadata?.labels || {};
  const roles = Object.keys(labels)
    .filter((key) => key.startsWith("node-role.kubernetes.io/"))
    .map((key) => key.replace("node-role.kubernetes.io/", ""))
    .filter(Boolean);

  const readyCondition = (node.status?.conditions || []).find((c) => c.type === "Ready");

  return {
    name: node.metadata?.name,
    roles: roles.length ? roles : ["worker"],
    internalIp: (node.status?.addresses || []).find((a) => a.type === "InternalIP")?.address || null,
    ready: readyCondition?.status === "True",
    labels
  };
}

function normalizeDeployment(dep) {
  const desired = dep.spec?.replicas || 0;
  const ready = dep.status?.readyReplicas || 0;
  const updated = dep.status?.updatedReplicas || 0;
  const available = dep.status?.availableReplicas || 0;

  return {
    name: dep.metadata?.name,
    desired,
    ready,
    updated,
    available,
    status: ready >= desired && desired > 0 ? "healthy" : ready > 0 ? "degraded" : "down"
  };
}

function normalizePod(pod) {
  const containerStatuses = pod.status?.containerStatuses || [];
  const restartCount = containerStatuses.reduce((acc, c) => acc + (c.restartCount || 0), 0);
  const readyContainers = containerStatuses.filter((c) => c.ready).length;

  return {
    name: pod.metadata?.name,
    app: pod.metadata?.labels?.app || null,
    phase: pod.status?.phase || "Unknown",
    node: pod.spec?.nodeName || null,
    podIp: pod.status?.podIP || null,
    restarts: restartCount,
    readyContainers,
    totalContainers: containerStatuses.length
  };
}

function normalizeService(svc) {
  return {
    name: svc.metadata?.name,
    type: svc.spec?.type || "ClusterIP",
    clusterIp: svc.spec?.clusterIP || null,
    ports: (svc.spec?.ports || []).map((p) => ({
      port: p.port,
      nodePort: p.nodePort || null,
      targetPort: p.targetPort
    }))
  };
}

function buildOrchestration(deployments, pods) {
  const depByName = new Map(deployments.map((d) => [d.name, d]));
  const podByApp = new Map(pods.map((p) => [p.app, p]));

  const gatewayDeployment = depByName.get("inference-gateway-deployment");
  const gatewayPod = podByApp.get("inference-gateway");

  const stages = [1, 2, 3, 4].map((id) => {
    const app = `inference-stage-${id}`;
    const deploymentName = `${app}-deployment`;
    const deployment = depByName.get(deploymentName) || null;
    const pod = podByApp.get(app) || null;
    return {
      id,
      app,
      deploymentName,
      deploymentStatus: deployment?.status || "missing",
      readyReplicas: deployment?.ready || 0,
      desiredReplicas: deployment?.desired || 0,
      podName: pod?.name || null,
      podPhase: pod?.phase || "Unknown",
      node: pod?.node || null
    };
  });

  return {
    gateway: {
      deploymentName: "inference-gateway-deployment",
      deploymentStatus: gatewayDeployment?.status || "missing",
      readyReplicas: gatewayDeployment?.ready || 0,
      desiredReplicas: gatewayDeployment?.desired || 0,
      podName: gatewayPod?.name || null,
      podPhase: gatewayPod?.phase || "Unknown",
      node: gatewayPod?.node || null
    },
    stages
  };
}

function analyzeLogs(rawText) {
  const lines = rawText.split(/\r?\n/).filter(Boolean);
  const summary = {
    totalLines: lines.length,
    errorLines: 0,
    warningLines: 0,
    infoLines: 0,
    hints: []
  };

  const hintRules = [
    {
      regex: /Temporary failure in name resolution|Name or service not known|no such host/i,
      hint: "DNS lookup failure from pod. Check CoreDNS and node DNS."
    },
    {
      regex: /ImagePullBackOff|ErrImagePull|manifest.*unknown|no matching manifest/i,
      hint: "Container image cannot be pulled for this node architecture or tag."
    },
    {
      regex: /DiskPressure|No space left on device|Evicted/i,
      hint: "Node disk pressure/storage issue detected."
    },
    {
      regex: /FailedScheduling|untolerated taint|didn't match Pod's node affinity\/selector/i,
      hint: "Scheduling rule mismatch: taints/tolerations or node selectors."
    },
    {
      regex: /connection refused|timed out|TLS handshake timeout/i,
      hint: "Network/API connectivity failure between components."
    },
    {
      regex: /TypeError: cannot unpack non-iterable NoneType object|position_embeddings/i,
      hint: "Model execution/API mismatch in Transformers layer invocation."
    },
    {
      regex: /500 Server Error|HTTPError|Bad Gateway/i,
      hint: "Upstream stage failed and propagated HTTP error."
    }
  ];

  const hints = new Set();

  for (const line of lines) {
    if (/\b(ERROR|Error|Traceback|Exception|CRITICAL)\b/.test(line)) {
      summary.errorLines += 1;
    } else if (/\b(WARN|Warning)\b/.test(line)) {
      summary.warningLines += 1;
    } else if (/\bINFO\b/.test(line)) {
      summary.infoLines += 1;
    }

    for (const rule of hintRules) {
      if (rule.regex.test(line)) {
        hints.add(rule.hint);
      }
    }
  }

  summary.hints = [...hints];
  return summary;
}

function updateRuntimePathCounter(pathValue, field, delta = 1) {
  const key = String(pathValue || "unknown");
  const row = runtimeStats.byPath[key] || {
    total: 0,
    success: 0,
    failed: 0,
    dropped: 0,
    timeout: 0
  };
  row[field] = Math.max(0, Number(row[field] || 0) + delta);
  runtimeStats.byPath[key] = row;
}

function pushTimeoutCause(cause) {
  const clean = String(cause || "timeout");
  runtimeStats.lastTimeoutCauses.unshift({
    at: new Date().toISOString(),
    cause: clean
  });
  if (runtimeStats.lastTimeoutCauses.length > 60) {
    runtimeStats.lastTimeoutCauses = runtimeStats.lastTimeoutCauses.slice(0, 60);
  }
}

function runtimeSnapshot() {
  return {
    startedAt: runtimeStats.startedAt,
    totalInvokes: runtimeStats.totalInvokes,
    successfulInvokes: runtimeStats.successfulInvokes,
    failedInvokes: runtimeStats.failedInvokes,
    droppedInFlight: runtimeStats.droppedInFlight,
    timeoutFailures: runtimeStats.timeoutFailures,
    inFlight: IN_FLIGHT_INVOKES.size,
    queueDepth: 0,
    queueMode: "drop-on-busy",
    byPath: runtimeStats.byPath,
    lastTimeoutCauses: runtimeStats.lastTimeoutCauses,
    inFlightRequests: [...IN_FLIGHT_INVOKES.values()].map((item) => ({
      namespace: item.namespace,
      target: item.target,
      method: item.method,
      path: item.path,
      startedAt: new Date(item.startedAt).toISOString(),
      inFlightMs: Math.max(0, Date.now() - item.startedAt)
    }))
  };
}

function parseJsonSafe(raw) {
  try {
    return JSON.parse(raw);
  } catch {
    return null;
  }
}

function toFiniteNumber(value, fallback = 0) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function roundNumber(value, digits = 4) {
  const n = toFiniteNumber(value, 0);
  const factor = 10 ** digits;
  return Math.round(n * factor) / factor;
}

function toSafeIsoDate(rawValue, fallbackIso) {
  if (!rawValue) {
    return fallbackIso;
  }
  const parsed = new Date(rawValue);
  const timestamp = parsed.getTime();
  if (!Number.isFinite(timestamp)) {
    return fallbackIso;
  }
  return parsed.toISOString();
}

function sanitizeHistoryText(rawValue, maxLen = 24000) {
  const text = String(rawValue == null ? "" : rawValue);
  if (!text) {
    return "";
  }
  return text.length > maxLen ? text.slice(0, maxLen) : text;
}

function normalizeHistoryFeatureFlags(rawFlags) {
  const input =
    rawFlags && typeof rawFlags === "object" && !Array.isArray(rawFlags) ? rawFlags : {};
  const result = {};
  const keys = [
    "transport_mode",
    "activation_precision",
    "kv_cache_enabled",
    "forward_dedupe_enabled",
    "rebalance_profile",
    "topology_aware_routing",
    "persistent_sessions_enabled",
    "backpressure_enabled",
    "backpressure_queue_size"
  ];
  for (const key of keys) {
    if (!(key in input)) {
      continue;
    }
    const value = input[key];
    if (typeof value === "string" || typeof value === "number" || typeof value === "boolean") {
      result[key] = value;
    }
  }
  return result;
}

function deriveHistoryModuleVariants(featureFlags, rawVariants) {
  const variants = new Set();
  for (const variant of Array.isArray(rawVariants) ? rawVariants : []) {
    const text = String(variant || "").trim();
    if (text) {
      variants.add(text);
    }
  }

  const flags = normalizeHistoryFeatureFlags(featureFlags);
  const pushPair = (key, value) => {
    variants.add(`${key}=${String(value)}`);
  };

  if (Object.prototype.hasOwnProperty.call(flags, "transport_mode")) {
    pushPair("transport", flags.transport_mode);
  }
  if (Object.prototype.hasOwnProperty.call(flags, "activation_precision")) {
    pushPair("precision", flags.activation_precision);
  }
  if (Object.prototype.hasOwnProperty.call(flags, "rebalance_profile")) {
    pushPair("rebalance", flags.rebalance_profile);
  }
  if (Object.prototype.hasOwnProperty.call(flags, "kv_cache_enabled")) {
    pushPair("kv_cache", flags.kv_cache_enabled ? "on" : "off");
  }
  if (Object.prototype.hasOwnProperty.call(flags, "forward_dedupe_enabled")) {
    pushPair("dedupe", flags.forward_dedupe_enabled ? "on" : "off");
  }
  if (Object.prototype.hasOwnProperty.call(flags, "topology_aware_routing")) {
    pushPair("topology", flags.topology_aware_routing ? "on" : "off");
  }
  if (Object.prototype.hasOwnProperty.call(flags, "persistent_sessions_enabled")) {
    pushPair("sessions", flags.persistent_sessions_enabled ? "on" : "off");
  }
  if (Object.prototype.hasOwnProperty.call(flags, "backpressure_enabled")) {
    pushPair("backpressure", flags.backpressure_enabled ? "on" : "off");
  }
  if (Object.prototype.hasOwnProperty.call(flags, "backpressure_queue_size")) {
    pushPair("queue", flags.backpressure_queue_size);
  }

  return [...variants].slice(0, 64);
}

function normalizeHistoryStageRows(rows) {
  return (Array.isArray(rows) ? rows : [])
    .slice(0, 32)
    .map((item) => {
      const row = item && typeof item === "object" && !Array.isArray(item) ? item : {};
      return {
        stageKey: row.stageKey ? String(row.stageKey) : "",
        service: row.service ? String(row.service) : "",
        stageId: Math.round(toFiniteNumber(row.stageId, 0)),
        samples: Math.round(toFiniteNumber(row.samples, 0)),
        computeMs: roundNumber(row.computeMs, 3),
        transferMs: roundNumber(row.transferMs, 3),
        maxMemMb: roundNumber(row.maxMemMb, 3),
        maxCpuPct: roundNumber(row.maxCpuPct, 3),
        maxSystemCpuPct: roundNumber(row.maxSystemCpuPct, 3),
        networkMib: roundNumber(row.networkMib, 4),
        payloadMib: roundNumber(row.payloadMib, 4),
        maxMbps: roundNumber(row.maxMbps, 4)
      };
    })
    .filter((row) => row.stageKey || row.service || row.samples > 0);
}

function normalizeHistoryNodeRows(rows) {
  return (Array.isArray(rows) ? rows : [])
    .slice(0, 64)
    .map((item) => {
      const row = item && typeof item === "object" && !Array.isArray(item) ? item : {};
      return {
        node: row.node ? String(row.node) : "",
        podHostnames: row.podHostnames ? String(row.podHostnames) : "",
        services: row.services ? String(row.services) : "",
        computeMs: roundNumber(row.computeMs, 3),
        transferMs: roundNumber(row.transferMs, 3),
        maxMemMb: roundNumber(row.maxMemMb, 3),
        maxCpuPct: roundNumber(row.maxCpuPct, 3),
        maxSystemCpuPct: roundNumber(row.maxSystemCpuPct, 3),
        maxMbps: roundNumber(row.maxMbps, 4),
        networkDeltaMib: roundNumber(row.networkDeltaMib, 4),
        payloadMib: roundNumber(row.payloadMib, 4)
      };
    })
    .filter((row) => row.node || row.services || row.podHostnames);
}

function normalizeHistoryEntry(input) {
  if (!input || typeof input !== "object" || Array.isArray(input)) {
    return null;
  }

  const nowIso = new Date().toISOString();
  const type = String(input.type || "").toLowerCase();
  if (type !== "endpoint" && type !== "chat") {
    return null;
  }

  const createdAt = toSafeIsoDate(input.createdAt, nowIso);
  const namespace = String(input.namespace || NAMESPACE);
  const target = String(input.target || "gateway");
  const method = String(input.method || "POST").toUpperCase();
  const pathValue = normalizePathForHttp(input.path || (type === "chat" ? "/chat" : "/generate"));
  const config = input.config && typeof input.config === "object" && !Array.isArray(input.config)
    ? input.config
    : {};
  const metrics = input.metrics && typeof input.metrics === "object" && !Array.isArray(input.metrics)
    ? input.metrics
    : {};
  const profile = input.profile && typeof input.profile === "object" && !Array.isArray(input.profile)
    ? input.profile
    : {};
  const output = input.output && typeof input.output === "object" && !Array.isArray(input.output)
    ? input.output
    : {};
  const normalizedFeatureFlags = normalizeHistoryFeatureFlags(profile.featureFlags);
  const moduleVariants = deriveHistoryModuleVariants(
    normalizedFeatureFlags,
    profile.moduleVariants
  );
  const generatedText = sanitizeHistoryText(output.generatedText, 24000);
  const assistantMessage = sanitizeHistoryText(output.assistantMessage, 24000);
  const terminationReason = sanitizeHistoryText(output.terminationReason, 256);
  const outputPreview = sanitizeHistoryText(
    generatedText || assistantMessage || output.preview,
    320
  );

  const record = {
    id: Number(nextHistorySeq++),
    type,
    createdAt,
    namespace,
    target,
    method,
    path: pathValue,
    config: {
      promptChars: Math.max(0, Math.round(toFiniteNumber(config.promptChars, 0))),
      promptTokens: Math.max(0, Math.round(toFiniteNumber(config.promptTokens, 0))),
      maxNewTokens: Math.max(0, Math.round(toFiniteNumber(config.maxNewTokens, 0))),
      minNewTokens: Math.max(0, Math.round(toFiniteNumber(config.minNewTokens, 0))),
      temperature: roundNumber(config.temperature, 4),
      timeoutMs: Math.max(0, Math.round(toFiniteNumber(config.timeoutMs, 0))),
      modelName: config.modelName ? String(config.modelName) : "",
      topologyHash: config.topologyHash ? String(config.topologyHash) : ""
    },
    profile: {
      baseline: Boolean(profile.baseline),
      enabledModules: Array.isArray(profile.enabledModules)
        ? profile.enabledModules.map((item) => String(item)).slice(0, 32)
        : [],
      featureFlags: normalizedFeatureFlags,
      moduleVariants
    },
    metrics: {
      generatedTokens: Math.max(0, Math.round(toFiniteNumber(metrics.generatedTokens, 0))),
      latencyMs: roundNumber(metrics.latencyMs, 3),
      tokensPerSecond: roundNumber(metrics.tokensPerSecond, 6),
      computeMs: roundNumber(metrics.computeMs, 3),
      transferMs: roundNumber(metrics.transferMs, 3),
      transferComputeRatio: roundNumber(metrics.transferComputeRatio, 6),
      rpcComputeRatio: roundNumber(
        metrics.rpcComputeRatio ?? metrics.transferComputeRatio,
        6
      ),
      trueCommComputeRatio: roundNumber(
        metrics.trueCommComputeRatio ?? metrics.transferComputeRatio,
        6
      ),
      payloadMib: roundNumber(metrics.payloadMib, 6),
      networkMib: roundNumber(metrics.networkMib, 6),
      maxMemoryMb: roundNumber(metrics.maxMemoryMb, 3),
      tokenP50Ms: roundNumber(metrics.tokenP50Ms, 3),
      tokenP95Ms: roundNumber(metrics.tokenP95Ms, 3),
      tokenP99Ms: roundNumber(metrics.tokenP99Ms, 3),
      transferP50Ms: roundNumber(metrics.transferP50Ms, 3),
      transferP95Ms: roundNumber(metrics.transferP95Ms, 3),
      transferP99Ms: roundNumber(metrics.transferP99Ms, 3),
      criticalPath: metrics.criticalPath && typeof metrics.criticalPath === "object" && !Array.isArray(metrics.criticalPath)
        ? metrics.criticalPath
        : {},
      perStage: normalizeHistoryStageRows(metrics.perStage),
      perNode: normalizeHistoryNodeRows(metrics.perNode)
    },
    output: {
      generatedText,
      assistantMessage,
      terminationReason,
      preview: outputPreview
    }
  };

  return record;
}

async function ensureHistoryLoaded() {
  if (historyLoaded) {
    return;
  }
  historyLoaded = true;
  await fsp.mkdir(DATA_DIR, { recursive: true });
  let raw = "";
  try {
    raw = await fsp.readFile(HISTORY_FILE, "utf-8");
  } catch (error) {
    if (error.code !== "ENOENT") {
      throw error;
    }
    raw = "";
  }

  const lines = raw
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean);
  for (const line of lines) {
    const parsed = parseJsonSafe(line);
    if (!parsed || typeof parsed !== "object") {
      continue;
    }
    historyEntries.push(parsed);
    const parsedId = Number(parsed.id || 0);
    if (Number.isFinite(parsedId) && parsedId >= nextHistorySeq) {
      nextHistorySeq = parsedId + 1;
    }
  }
  if (historyEntries.length > HISTORY_MAX_ENTRIES) {
    historyEntries.splice(0, historyEntries.length - HISTORY_MAX_ENTRIES);
  }
}

async function compactHistoryFile() {
  const serialized = historyEntries.map((entry) => JSON.stringify(entry)).join("\n");
  const payload = serialized ? `${serialized}\n` : "";
  await fsp.writeFile(HISTORY_FILE, payload, "utf-8");
}

async function appendHistoryRecords(records) {
  if (!records.length) {
    return [];
  }

  await ensureHistoryLoaded();
  const normalized = records.map(normalizeHistoryEntry).filter(Boolean);
  if (!normalized.length) {
    return [];
  }

  historyWriteChain = historyWriteChain.then(async () => {
    await fsp.mkdir(DATA_DIR, { recursive: true });
    const chunk = normalized.map((entry) => JSON.stringify(entry)).join("\n");
    await fsp.appendFile(HISTORY_FILE, `${chunk}\n`, "utf-8");
    historyEntries.push(...normalized);
    if (historyEntries.length > HISTORY_MAX_ENTRIES) {
      historyEntries.splice(0, historyEntries.length - HISTORY_MAX_ENTRIES);
      await compactHistoryFile();
    }
  });
  await historyWriteChain;
  return normalized;
}

async function deleteHistoryRecords({ ids = [], all = false } = {}) {
  await ensureHistoryLoaded();

  const normalizedIds = new Set(
    (Array.isArray(ids) ? ids : [])
      .map((id) => Number.parseInt(String(id), 10))
      .filter((id) => Number.isFinite(id) && id > 0)
  );

  if (!all && normalizedIds.size === 0) {
    return {
      deleted: 0,
      remaining: historyEntries.length,
      totalStored: historyEntries.length
    };
  }

  historyWriteChain = historyWriteChain.then(async () => {
    if (all) {
      historyEntries.splice(0, historyEntries.length);
    } else {
      for (let i = historyEntries.length - 1; i >= 0; i -= 1) {
        const entryId = Number.parseInt(String(historyEntries[i]?.id || 0), 10);
        if (normalizedIds.has(entryId)) {
          historyEntries.splice(i, 1);
        }
      }
    }
    await compactHistoryFile();
  });

  const before = historyEntries.length;
  await historyWriteChain;
  const after = historyEntries.length;

  return {
    deleted: Math.max(0, before - after),
    remaining: after,
    totalStored: after
  };
}

function sendJson(res, statusCode, payload) {
  const body = JSON.stringify(payload);
  res.writeHead(statusCode, {
    "Content-Type": "application/json; charset=utf-8",
    "Cache-Control": "no-store"
  });
  res.end(body);
}

function sendText(res, statusCode, payload) {
  res.writeHead(statusCode, {
    "Content-Type": "text/plain; charset=utf-8",
    "Cache-Control": "no-store"
  });
  res.end(payload);
}

function resolveNamespace(query) {
  const fromQuery = (query?.get("namespace") || "").trim();
  const namespace = fromQuery || NAMESPACE;
  return namespace;
}

function isValidNamespaceName(namespace) {
  return NAMESPACE_NAME_RE.test(namespace);
}

async function getAvailableNamespacesSafe() {
  try {
    const nsJson = await runKubectl(["get", "namespaces", "-o", "json"], { parseJson: true });
    return (nsJson.items || [])
      .map((item) => item.metadata?.name)
      .filter(Boolean)
      .sort();
  } catch {
    return [];
  }
}

async function handleTopology(req, res, query) {
  const namespace = resolveNamespace(query);
  if (!isValidNamespaceName(namespace)) {
    sendJson(res, 400, {
      error: `Invalid namespace '${namespace}'.`,
      namespace
    });
    return;
  }

  try {
    const [nodesJson, deploymentsJson, podsJson, servicesJson] = await Promise.all([
      runKubectl(["get", "nodes", "-o", "json"], { parseJson: true }),
      runKubectl(["-n", namespace, "get", "deploy", "-o", "json"], { parseJson: true }),
      runKubectl(["-n", namespace, "get", "pods", "-o", "json"], { parseJson: true }),
      runKubectl(["-n", namespace, "get", "svc", "-o", "json"], { parseJson: true })
    ]);

    const nodes = (nodesJson.items || []).map(normalizeNode);
    const deployments = (deploymentsJson.items || []).map(normalizeDeployment);
    const pods = (podsJson.items || []).map(normalizePod);
    const services = (servicesJson.items || []).map(normalizeService);
    const orchestration = buildOrchestration(deployments, pods);

    sendJson(res, 200, {
      generatedAt: new Date().toISOString(),
      namespace,
      nodes,
      deployments,
      pods,
      services,
      orchestration
    });
  } catch (error) {
    const details = String(error.message || error);
    const stderr = error.stderr ? String(error.stderr) : "";
    const hasMissingNamespace =
      /namespaces?\s+["'][^"']+["']\s+not found/i.test(stderr) ||
      /namespaces?\s+["'][^"']+["']\s+not found/i.test(details);
    const availableNamespaces = hasMissingNamespace ? await getAvailableNamespacesSafe() : [];

    sendJson(res, 500, {
      error: "Failed to collect topology data",
      ...(hasMissingNamespace
        ? {
            hint: `Namespace '${namespace}' is missing in the current kubectl context.`,
            availableNamespaces
          }
        : {}),
      namespace,
      details
    });
  }
}

async function handleLogs(req, res, query) {
  const namespace = resolveNamespace(query);
  if (!isValidNamespaceName(namespace)) {
    sendJson(res, 400, {
      error: `Invalid namespace '${namespace}'.`,
      namespace
    });
    return;
  }

  const target = query.get("target") || "gateway";
  const source = query.get("source") || "app";
  const previous = query.get("previous") === "1";
  const tail = clampNumber(query.get("tail"), 20, 2000, 300);

  const targetConfig = LOG_TARGETS[target];
  if (!targetConfig) {
    sendJson(res, 400, { error: `Unknown target '${target}'.` });
    return;
  }

  const containerName = source === "init" ? "fetch-stage-partition" : targetConfig.appContainer;

  if (source === "init") {
    if (!target.startsWith("stage-")) {
      sendJson(res, 400, { error: "Init logs are available only for stage targets." });
      return;
    }
  } else if (source !== "app") {
    sendJson(res, 400, { error: `Unknown source '${source}'.` });
    return;
  }

  try {
    const {
      deployment,
      deploymentMissing,
      selector,
      matchedBy,
      podItems
    } = await resolvePodsForTarget(namespace, targetConfig);

    if (!podItems.length) {
      const allPods = await runKubectl(
        ["-n", namespace, "get", "pods", "-o", "json"],
        { parseJson: true }
      );
      const podNames = (allPods.items || []).map((pod) => pod.metadata?.name).filter(Boolean);
      sendJson(res, 404, {
        error: "No pod found for log target",
        target,
        selector,
        namespace,
        matchedBy,
        deployment: deployment
          ? {
              name: deployment.metadata?.name || targetConfig.deployment,
              desiredReplicas: deployment.spec?.replicas || 0,
              readyReplicas: deployment.status?.readyReplicas || 0,
              availableReplicas: deployment.status?.availableReplicas || 0
            }
          : {
              name: targetConfig.deployment,
              missing: true
            },
        hint:
          deploymentMissing
            ? `Deployment '${targetConfig.deployment}' is missing and no pod matched '${selector}'.`
            : (deployment.spec?.replicas || 0) === 0
            ? "Deployment is scaled to 0 replicas."
            : "No pod currently exists for this deployment selector. Check scheduling/image pull events with: kubectl -n <ns> describe deploy/<name> and kubectl -n <ns> get events --sort-by=.lastTimestamp",
        visiblePods: podNames
      });
      return;
    }

    const podName = podItems[podItems.length - 1].metadata?.name;
    const args = ["-n", namespace, "logs", podName, "-c", containerName, "--tail", String(tail)];
    if (previous) {
      args.push("--previous");
    }

    const raw = await runKubectl(args);
    const summary = analyzeLogs(raw);
    sendJson(res, 200, {
      generatedAt: new Date().toISOString(),
      namespace,
      podName,
      selector,
      deploymentMissing,
      matchedBy,
      target,
      source,
      previous,
      tail,
      summary,
      log: raw
    });
  } catch (error) {
    const stderr = error.stderr ? String(error.stderr) : "";
    const details = String(error.message || error);
    const hasMissingDeployment =
      /deployments?.*not found/i.test(stderr) ||
      /deployments?.*not found/i.test(details);
    const hasMissingNamespace =
      /namespaces?\s+["'][^"']+["']\s+not found/i.test(stderr) ||
      /namespaces?\s+["'][^"']+["']\s+not found/i.test(details);
    const availableNamespaces = hasMissingNamespace ? await getAvailableNamespacesSafe() : [];

    const payload = {
      error: "Failed to fetch logs",
      namespace,
      details,
      stderr
    };
    if (hasMissingNamespace) {
      payload.hint = `Namespace '${namespace}' is missing in the current kubectl context. Use the namespace field in UI or set OPS_UI_NAMESPACE.`;
      payload.availableNamespaces = availableNamespaces;
    } else if (hasMissingDeployment) {
      payload.hint = `Deployment '${targetConfig.deployment}' was not found in namespace '${namespace}'.`;
    }
    sendJson(res, 500, payload);
  }
}

function buildInvokeHeaders(rawHeaders) {
  if (!rawHeaders || typeof rawHeaders !== "object" || Array.isArray(rawHeaders)) {
    return {};
  }

  const headers = {};
  for (const [key, value] of Object.entries(rawHeaders)) {
    if (!key) {
      continue;
    }
    headers[String(key)] = String(value);
  }
  return headers;
}

function isLongInferencePath(method, pathValue) {
  return method === "POST" && /^\/(generate|chat)(\/|$)/i.test(pathValue);
}

function parseInvokeTimeoutMs(rawValue, fallbackMs) {
  const parsed = Number.parseInt(rawValue, 10);
  if (Number.isNaN(parsed)) {
    return fallbackMs;
  }
  if (parsed === 0) {
    return 0;
  }
  return Math.max(INVOKE_TIMEOUT_MIN_MS, Math.min(INVOKE_TIMEOUT_MAX_MS, parsed));
}

function buildInvokeLockKey(namespace, target, method, pathValue) {
  return `${namespace}::${target}::${method}::${pathValue}`;
}

async function tryHttpCall({ url, method, headers, body, timeoutMs }) {
  const controller = new AbortController();
  const timeoutId = timeoutMs > 0 ? setTimeout(() => controller.abort(), timeoutMs) : null;
  const startedAt = Date.now();

  try {
    const response = await fetch(url, {
      method,
      headers,
      body,
      signal: controller.signal
    });
    const responseText = await response.text();
    let responseJson = null;
    try {
      responseJson = responseText ? JSON.parse(responseText) : null;
    } catch {
      responseJson = null;
    }

    return {
      ok: true,
      status: response.status,
      statusText: response.statusText,
      elapsedMs: Date.now() - startedAt,
      responseHeaders: Object.fromEntries(response.headers.entries()),
      responseText,
      responseJson
    };
  } catch (error) {
    return {
      ok: false,
      elapsedMs: Date.now() - startedAt,
      error: String(error.message || error)
    };
  } finally {
    if (timeoutId != null) {
      clearTimeout(timeoutId);
    }
  }
}

function extractPortForwardLocalPort(outputText) {
  const text = String(outputText || "");
  const match = text.match(/Forwarding from (?:127\.0\.0\.1|\[::1\]):(\d+)\s*->\s*8000/i);
  if (!match) {
    return null;
  }
  const parsed = Number.parseInt(match[1], 10);
  return Number.isFinite(parsed) ? parsed : null;
}

async function stopChildProcess(proc) {
  if (!proc || proc.exitCode != null || proc.killed) {
    return;
  }

  await new Promise((resolve) => {
    let done = false;
    const finish = () => {
      if (done) {
        return;
      }
      done = true;
      resolve();
    };

    const forceTimer = setTimeout(() => {
      if (proc.exitCode == null && !proc.killed) {
        proc.kill("SIGKILL");
      }
    }, 1000);

    proc.once("exit", () => {
      clearTimeout(forceTimer);
      finish();
    });

    try {
      proc.kill("SIGTERM");
    } catch {
      clearTimeout(forceTimer);
      finish();
    }
  });
}

async function tryPortForwardCall({
  namespace,
  podName,
  method,
  headers,
  body,
  pathValue,
  timeoutMs
}) {
  let processHandle = null;
  const startupTimeoutMs = timeoutMs === 0 ? 20000 : Math.max(2500, Math.min(20000, timeoutMs));
  const startedAt = Date.now();

  try {
    processHandle = spawn(
      "kubectl",
      [
        "-n",
        namespace,
        "port-forward",
        `pod/${podName}`,
        ":8000",
        "--address",
        "127.0.0.1"
      ],
      {
        stdio: ["ignore", "pipe", "pipe"]
      }
    );

    const localPort = await new Promise((resolve, reject) => {
      let settled = false;
      let outputBuffer = "";

      const finishResolve = (value) => {
        if (settled) {
          return;
        }
        settled = true;
        resolve(value);
      };
      const finishReject = (error) => {
        if (settled) {
          return;
        }
        settled = true;
        reject(error);
      };

      const startupTimer = setTimeout(() => {
        finishReject(new Error("kubectl port-forward startup timeout"));
      }, startupTimeoutMs);

      const checkChunk = (chunk) => {
        outputBuffer += String(chunk || "");
        const maybePort = extractPortForwardLocalPort(outputBuffer);
        if (maybePort != null) {
          clearTimeout(startupTimer);
          finishResolve(maybePort);
        }
      };

      processHandle.stdout.on("data", checkChunk);
      processHandle.stderr.on("data", checkChunk);

      processHandle.once("error", (error) => {
        clearTimeout(startupTimer);
        finishReject(
          new Error(`kubectl port-forward failed: ${String(error?.message || error)}`)
        );
      });

      processHandle.once("exit", (code, signal) => {
        if (settled) {
          return;
        }
        clearTimeout(startupTimer);
        finishReject(
          new Error(
            `kubectl port-forward exited before ready (code=${code}, signal=${signal || "none"})`
          )
        );
      });
    });

    const remainingMs = timeoutMs === 0
      ? 0
      : Math.max(INVOKE_TIMEOUT_MIN_MS, timeoutMs - (Date.now() - startedAt));
    const forwardedCall = await tryHttpCall({
      url: `http://127.0.0.1:${localPort}${pathValue}`,
      method,
      headers,
      body,
      timeoutMs: remainingMs
    });

    return {
      ...forwardedCall,
      candidate: {
        type: "kubectlPortForward",
        podName,
        localPort,
        baseUrl: `http://127.0.0.1:${localPort}`
      }
    };
  } finally {
    await stopChildProcess(processHandle);
  }
}

async function handleInvoke(req, res, query) {
  if ((req.method || "GET").toUpperCase() !== "POST") {
    sendJson(res, 405, { error: "Use POST for /api/invoke." });
    return;
  }

  let payload = {};
  try {
    payload = await readJsonBody(req);
  } catch (error) {
    sendJson(res, 400, { error: String(error.message || error) });
    return;
  }

  const namespace = resolveNamespace(new URLSearchParams({
    namespace: String(payload.namespace || query.get("namespace") || "")
  }));

  if (!isValidNamespaceName(namespace)) {
    sendJson(res, 400, {
      error: `Invalid namespace '${namespace}'.`,
      namespace
    });
    return;
  }

  const target = String(payload.target || "gateway");
  const targetConfig = LOG_TARGETS[target];
  if (!targetConfig) {
    sendJson(res, 400, { error: `Unknown target '${target}'.` });
    return;
  }

  const method = String(payload.method || "GET").toUpperCase();
  if (!HTTP_METHOD_RE.test(method)) {
    sendJson(res, 400, { error: `Unsupported HTTP method '${method}'.` });
    return;
  }

  const pathValue = normalizePathForHttp(payload.path || "/health");
  const longInferenceCall = isLongInferencePath(method, pathValue);
  const recommendedDefaultTimeoutMs = longInferenceCall
    ? INVOKE_TIMEOUT_LONG_DEFAULT_MS
    : INVOKE_TIMEOUT_DEFAULT_MS;
  const timeoutMs = parseInvokeTimeoutMs(payload.timeoutMs, recommendedDefaultTimeoutMs);
  const headers = buildInvokeHeaders(payload.headers);
  runtimeStats.totalInvokes += 1;
  updateRuntimePathCounter(pathValue, "total");
  let runtimeTerminalMarked = false;
  const markInvokeSuccess = () => {
    if (runtimeTerminalMarked) {
      return;
    }
    runtimeTerminalMarked = true;
    runtimeStats.successfulInvokes += 1;
    updateRuntimePathCounter(pathValue, "success");
  };
  const markInvokeFailure = () => {
    if (runtimeTerminalMarked) {
      return;
    }
    runtimeTerminalMarked = true;
    runtimeStats.failedInvokes += 1;
    updateRuntimePathCounter(pathValue, "failed");
  };
  const markInvokeDropped = () => {
    if (runtimeTerminalMarked) {
      return;
    }
    runtimeTerminalMarked = true;
    runtimeStats.droppedInFlight += 1;
    updateRuntimePathCounter(pathValue, "dropped");
  };

  let lockKey = null;
  if (longInferenceCall) {
    lockKey = buildInvokeLockKey(namespace, target, method, pathValue);
    const existingInvoke = IN_FLIGHT_INVOKES.get(lockKey);
    if (existingInvoke) {
      const inFlightMs = Math.max(0, Date.now() - existingInvoke.startedAt);
      markInvokeDropped();
      sendJson(res, 429, {
        error: "Request discarded because another request is already in progress",
        hint: "Wait for the active /generate or /chat request to finish, then retry.",
        namespace,
        target,
        method,
        path: pathValue,
        inFlightMs
      });
      return;
    }
    IN_FLIGHT_INVOKES.set(lockKey, {
      startedAt: Date.now(),
      namespace,
      target,
      method,
      path: pathValue
    });
  }

  let body = null;
  if (method !== "GET" && method !== "HEAD" && payload.body !== undefined) {
    if (typeof payload.body === "string") {
      body = payload.body;
    } else {
      body = JSON.stringify(payload.body);
      if (!Object.keys(headers).some((h) => h.toLowerCase() === "content-type")) {
        headers["Content-Type"] = "application/json";
      }
    }
  }

  try {
    let resolved;
    try {
      resolved = await resolvePodsForTarget(namespace, targetConfig);
    } catch (error) {
      const details = String(error.message || error);
      const stderr = error.stderr ? String(error.stderr) : "";
      markInvokeFailure();
      sendJson(res, 500, {
        error: "Failed to resolve target pods",
        namespace,
        target,
        details,
        stderr
      });
      return;
    }

    const {
      deployment,
      deploymentMissing,
      selector,
      matchedBy,
      podItems
    } = resolved;

  if (!podItems.length) {
    const allPods = await runKubectl(["-n", namespace, "get", "pods", "-o", "json"], {
      parseJson: true
    });
    const podNames = (allPods.items || []).map((pod) => pod.metadata?.name).filter(Boolean);
    markInvokeFailure();
    sendJson(res, 404, {
      error: "No pod found for target",
      namespace,
      target,
      selector,
      matchedBy,
      deployment: deployment
        ? {
            name: deployment.metadata?.name || targetConfig.deployment,
            desiredReplicas: deployment.spec?.replicas || 0,
            readyReplicas: deployment.status?.readyReplicas || 0,
            availableReplicas: deployment.status?.availableReplicas || 0
          }
        : {
            name: targetConfig.deployment,
            missing: true
          },
      hint: deploymentMissing
        ? `Deployment '${targetConfig.deployment}' is missing and no pod matched '${selector}'.`
        : "No pod matched selector for this target.",
      visiblePods: podNames
    });
    return;
  }

  const selectedPod = podItems[podItems.length - 1];
  const candidates = await buildHttpCandidates(namespace, targetConfig, selectedPod);
  if (!candidates.length) {
    markInvokeFailure();
    sendJson(res, 500, {
      error: "No HTTP candidates available for target",
      namespace,
      target
    });
    return;
  }

  const attempts = [];
  let successful = null;
  const candidatesToTry = longInferenceCall ? candidates.slice(0, 1) : candidates;

  for (const candidate of candidatesToTry) {
    const fullUrl = `${candidate.baseUrl}${pathValue}`;
    const result = await tryHttpCall({
      url: fullUrl,
      method,
      headers,
      body,
      timeoutMs
    });
    attempts.push({
      candidate,
      fullUrl,
      ok: result.ok,
      status: result.status || null,
      elapsedMs: result.elapsedMs,
      error: result.error || null
    });

    if (result.ok) {
      successful = {
        ...result,
        fullUrl,
        candidate
      };
      break;
    }
  }

  if (!successful) {
    try {
      const fallbackViaPortForward = await tryPortForwardCall({
        namespace,
        podName: selectedPod.metadata?.name || "",
        method,
        headers,
        body,
        pathValue,
        timeoutMs
      });

      if (fallbackViaPortForward?.ok) {
        markInvokeSuccess();
        sendJson(res, 200, {
          generatedAt: new Date().toISOString(),
          namespace,
          target,
          method,
          path: pathValue,
          podName: selectedPod.metadata?.name || null,
          podIp: selectedPod.status?.podIP || null,
          selector,
          matchedBy,
          deploymentMissing,
          call: {
            fullUrl: `http://127.0.0.1:${fallbackViaPortForward.candidate.localPort}${pathValue}`,
            candidate: fallbackViaPortForward.candidate,
            elapsedMs: fallbackViaPortForward.elapsedMs,
            status: fallbackViaPortForward.status,
            statusText: fallbackViaPortForward.statusText
          },
          responseHeaders: fallbackViaPortForward.responseHeaders,
          responseText: fallbackViaPortForward.responseText,
          responseJson: fallbackViaPortForward.responseJson,
          attempts: [
            ...attempts,
            {
              candidate: fallbackViaPortForward.candidate,
              fullUrl: `http://127.0.0.1:${fallbackViaPortForward.candidate.localPort}${pathValue}`,
              ok: true,
              status: fallbackViaPortForward.status || null,
              elapsedMs: fallbackViaPortForward.elapsedMs,
              error: null
            }
          ]
        });
        return;
      }

      attempts.push({
        candidate: fallbackViaPortForward?.candidate || { type: "kubectlPortForward" },
        fullUrl: fallbackViaPortForward?.candidate?.baseUrl
          ? `${fallbackViaPortForward.candidate.baseUrl}${pathValue}`
          : null,
        ok: false,
        status: fallbackViaPortForward?.status || null,
        elapsedMs: fallbackViaPortForward?.elapsedMs || 0,
        error: fallbackViaPortForward?.error || "kubectl port-forward fallback failed"
      });
    } catch (portForwardError) {
      attempts.push({
        candidate: {
          type: "kubectlPortForward",
          podName: selectedPod.metadata?.name || null
        },
        fullUrl: null,
        ok: false,
        status: null,
        elapsedMs: 0,
        error: String(portForwardError?.message || portForwardError)
      });
    }

    const abortedAttempts = attempts.filter((attempt) =>
      /aborted|timeout/i.test(String(attempt.error || ""))
    ).length;
    if (abortedAttempts > 0) {
      runtimeStats.timeoutFailures += abortedAttempts;
      updateRuntimePathCounter(pathValue, "timeout", abortedAttempts);
      for (const attempt of attempts) {
        if (/aborted|timeout/i.test(String(attempt.error || ""))) {
          pushTimeoutCause(`${attempt.candidate?.type || "candidate"}: ${attempt.error}`);
        }
      }
    }
    const timeoutHint =
      abortedAttempts > 0
        ? `Request timed out for ${abortedAttempts}/${attempts.length} candidate(s). Increase timeoutMs for /generate or /chat, or set timeout to 0 for no timeout.`
        : undefined;

    markInvokeFailure();
    sendJson(res, 502, {
      error: "All endpoint call attempts failed",
      hint: timeoutHint,
      namespace,
      target,
      path: pathValue,
      method,
      selector,
      matchedBy,
      podName: selectedPod.metadata?.name || null,
      attempts
    });
    return;
  }

    markInvokeSuccess();
    sendJson(res, 200, {
      generatedAt: new Date().toISOString(),
      namespace,
      target,
      method,
      path: pathValue,
      podName: selectedPod.metadata?.name || null,
      podIp: selectedPod.status?.podIP || null,
      selector,
      matchedBy,
      deploymentMissing,
      call: {
        fullUrl: successful.fullUrl,
        candidate: successful.candidate,
        elapsedMs: successful.elapsedMs,
        status: successful.status,
        statusText: successful.statusText
      },
      responseHeaders: successful.responseHeaders,
      responseText: successful.responseText,
      responseJson: successful.responseJson,
      attempts
    });
  } finally {
    if (lockKey) {
      IN_FLIGHT_INVOKES.delete(lockKey);
    }
  }
}

function handleEndpointCatalog(req, res, query) {
  const namespace = resolveNamespace(query);
  sendJson(res, 200, {
    namespace,
    targets: Object.keys(LOG_TARGETS),
    presets: [
      {
        id: "gateway-health",
        label: "Gateway Health",
        target: "gateway",
        method: "GET",
        path: "/health",
        source: "gateway"
      },
      {
        id: "gateway-config",
        label: "Gateway Config",
        target: "gateway",
        method: "GET",
        path: "/config",
        source: "gateway"
      },
      {
        id: "gateway-generate",
        label: "Gateway Generate",
        target: "gateway",
        method: "POST",
        path: "/generate",
        timeoutMs: INVOKE_TIMEOUT_LONG_DEFAULT_MS,
        source: "gateway",
        body: {
          prompt: "Write one short sentence about distributed inference.",
          max_new_tokens: 24,
          min_new_tokens: 8,
          temperature: 0.2
        }
      },
      {
        id: "gateway-chat",
        label: "Gateway Chat",
        target: "gateway",
        method: "POST",
        path: "/chat",
        timeoutMs: INVOKE_TIMEOUT_LONG_DEFAULT_MS,
        source: "gateway",
        body: {
          messages: [
            { role: "user", content: "Write one short sentence about distributed inference." }
          ],
          max_new_tokens: 48,
          min_new_tokens: 8,
          temperature: 0.2
        }
      },
      {
        id: "stage-health",
        label: "Stage Health",
        target: "stage-1",
        method: "GET",
        path: "/health",
        source: "stage"
      }
    ]
  });
}

function handleRuntime(req, res, query) {
  const namespace = resolveNamespace(query);
  sendJson(res, 200, {
    namespace,
    generatedAt: new Date().toISOString(),
    runtime: runtimeSnapshot()
  });
}

async function handleHistory(req, res, query) {
  const method = (req.method || "GET").toUpperCase();

  if (method === "GET") {
    const typeFilter = String(query.get("type") || "").trim().toLowerCase();
    const namespaceFilter = String(query.get("namespace") || "").trim();
    const limit = clampNumber(query.get("limit"), 1, 5000, 300);

    await ensureHistoryLoaded();

    let rows = historyEntries;

    if (typeFilter === "endpoint" || typeFilter === "chat") {
      rows = rows.filter((row) => String(row.type || "") === typeFilter);
    }

    if (namespaceFilter) {
      rows = rows.filter((row) => String(row.namespace || "") === namespaceFilter);
    }

    const selected = rows.slice(-limit).reverse();

    sendJson(res, 200, {
      generatedAt: new Date().toISOString(),
      totalStored: historyEntries.length,
      returned: selected.length,
      entries: selected
    });
    return;
  }

  if (method === "DELETE") {
    let payload = {};

    try {
      payload = await readJsonBody(req, { maxBytes: 64 * 1024 });
    } catch (error) {
      sendJson(res, 400, {
        ok: false,
        error: String(error.message || error)
      });
      return;
    }

    const result = await deleteHistoryRecords({
      ids: payload.ids,
      all: payload.all === true
    });

    sendJson(res, 200, {
      ok: true,
      generatedAt: new Date().toISOString(),
      ...result
    });
    return;
  }

  if (method !== "POST") {
    sendJson(res, 405, { error: "Use GET, POST, or DELETE for /api/history." });
    return;
  }

  let payload = {};

  try {
    payload = await readJsonBody(req, { maxBytes: 4 * 1024 * 1024 });
  } catch (error) {
    sendJson(res, 400, { error: String(error.message || error) });
    return;
  }

  const rawEntries = Array.isArray(payload.entries)
    ? payload.entries
    : payload.entry
    ? [payload.entry]
    : [];

  if (!rawEntries.length) {
    sendJson(res, 400, { error: "Missing history entry payload." });
    return;
  }

  if (rawEntries.length > 50) {
    sendJson(res, 400, { error: "Too many entries in one request (max 50)." });
    return;
  }

  const saved = await appendHistoryRecords(rawEntries);

  sendJson(res, 200, {
    saved: saved.length,
    ids: saved.map((entry) => entry.id)
  });
}

async function serveStatic(req, res, reqPath) {
  const cleanPath = reqPath === "/" ? "/index.html" : reqPath;
  const normalized = path.normalize(cleanPath).replace(/^(\.\.[/\\])+/, "");
  const fullPath = path.join(PUBLIC_DIR, normalized);

  if (!fullPath.startsWith(PUBLIC_DIR)) {
    sendText(res, 403, "Forbidden");
    return;
  }

  try {
    const content = await fsp.readFile(fullPath);
    const ext = path.extname(fullPath).toLowerCase();
    res.writeHead(200, {
      "Content-Type": MIME[ext] || "application/octet-stream",
      "Cache-Control": "no-store"
    });
    res.end(content);
  } catch (error) {
    if (error.code === "ENOENT") {
      sendText(res, 404, "Not found");
      return;
    }
    sendText(res, 500, "Server error");
  }
}

const server = http.createServer(async (req, res) => {
  try {
    if (!req.url) {
      sendText(res, 400, "Bad request");
      return;
    }

    const url = new URL(req.url, `http://${req.headers.host || "localhost"}`);

    if (url.pathname === "/api/health") {
      sendJson(res, 200, {
        status: "ok",
        namespace: NAMESPACE,
        kubectlTimeoutMs: KUBECTL_TIMEOUT_MS
      });
      return;
    }

    if (url.pathname === "/api/topology") {
      await handleTopology(req, res, url.searchParams);
      return;
    }

    if (url.pathname === "/api/endpoint-catalog") {
      handleEndpointCatalog(req, res, url.searchParams);
      return;
    }

    if (url.pathname === "/api/logs") {
      await handleLogs(req, res, url.searchParams);
      return;
    }

    if (url.pathname === "/api/invoke") {
      await handleInvoke(req, res, url.searchParams);
      return;
    }

    if (url.pathname === "/api/runtime") {
      handleRuntime(req, res, url.searchParams);
      return;
    }

    if (url.pathname === "/api/history") {
      await handleHistory(req, res, url.searchParams);
      return;
    }

    await serveStatic(req, res, url.pathname);
  } catch (error) {
    sendJson(res, 500, {
      error: "Unhandled server error",
      details: String(error.message || error)
    });
  }
});

server.listen(PORT, () => {
  // eslint-disable-next-line no-console
  console.log(
    `[ops-ui] listening on http://0.0.0.0:${PORT} (namespace=${NAMESPACE}, kubectl-timeout=${KUBECTL_TIMEOUT_MS}ms)`
  );
});

process.on("SIGINT", () => {
  // eslint-disable-next-line no-console
  console.log("\n[ops-ui] shutting down");
  server.close(() => process.exit(0));
});
