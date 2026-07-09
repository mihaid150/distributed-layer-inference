# DLI Ops UI

Small JavaScript UI for:

- cluster topology (gateway + stages)
- orchestration health (nodes/pods/deployments)
- stage/gateway logs with lightweight interpretation hints
- endpoint interaction (health/config/generate/custom calls) via UI
- chat against distributed gateway (`/chat`) with per-turn metric evolution
- runtime switch between the PyTorch track and the native C++/llama.cpp track

## Run

From repo root:

```bash
cd ops-ui
npm start
```

Open:

- `http://localhost:4070`

## Requirements

- `kubectl` installed on the machine running the UI
- active kube context with access to the target cluster
- namespace defaults to `inference`

Environment variables:

- `OPS_UI_PORT` (default `4070`)
- `OPS_UI_NAMESPACE` (default `inference`)
- `OPS_UI_RUNTIME_VARIANT` (default `python`; set `native` for C++/llama.cpp pods)
- `OPS_UI_KUBECTL_TIMEOUT_MS` (default `15000`)
- `OPS_UI_NATIVE_MODELS_FILE` (default `native-models.json`; repo-root paths like `ops-ui/native-models.json` also work)
- `OPS_UI_NATIVE_TOOLS_IMAGE` (default `mipeda150/distributed-layer-inference-native-tools:latest`)
- `OPS_UI_NATIVE_CONFIGMAP` (default `dli-native-config`)
- `HF_NATIVE_MODEL_ARTIFACT_REPO` optional template used when catalog entries omit `artifactRepo`; supports `{slug}`

## Native Model Selection

The native runtime uses restart-based model activation. In the Native C++ runtime,
the **Native Model Controller** panel can:

1. list configured GGUF models from `ops-ui/native-models.json`;
2. start a Kubernetes Job that downloads a full GGUF, generates a matching
   `stage_map.yaml`, creates `partition-N.dli.gguf` shards, validates them, and
   uploads the artifacts to Hugging Face;
3. activate a prepared catalog model, or activate the output of a completed prep
   Job by patching `dli-native-config` and restarting the native gateway/stage
   deployments.

Setup:

```bash
cp ops-ui/native-models.example.json ops-ui/native-models.json
```

Edit the catalog entries with real Hugging Face repos/files. The prep Job reads
`HF_TOKEN` and `HF_UPLOAD_TOKEN` from the `hf-hub` secret when available.
The prep container runs the compiled C++ `dli-native-model-controller` binary
from the native tools image; it does not run a Python controller.
Artifact paths are model-scoped, for example:

```text
<model-slug>/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
<model-slug>/partition-1.dli.gguf
<model-slug>/partition-2.dli.gguf
```

## Notes

- Logs are pulled via `kubectl logs` on demand.
- For init container logs, use `source=init` in the UI.
- Endpoint calls are executed server-side from `ops-ui/server.js` and proxied back to UI.
- The runtime switch changes Kubernetes targets from `inference-*` to `inference-native-*`.
- Native C++ gateway responses are normalized in the browser so aggregate and per-step native metrics render in the same dashboard as PyTorch responses.
- Chat metrics timeline can be exported as JSON for evaluation datasets.
- If running on master, use `KUBECONFIG=/etc/rancher/k3s/k3s.yaml npm start`.
- This folder is excluded from Docker build context for stage/gateway images.
