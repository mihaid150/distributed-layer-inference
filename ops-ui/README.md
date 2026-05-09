# DLI Ops UI

Small JavaScript UI for:

- cluster topology (gateway + stages)
- orchestration health (nodes/pods/deployments)
- stage/gateway logs with lightweight interpretation hints
- endpoint interaction (health/config/generate/custom calls) via UI
- chat against distributed gateway (`/chat`) with per-turn metric evolution

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
- `OPS_UI_KUBECTL_TIMEOUT_MS` (default `15000`)

## Notes

- Logs are pulled via `kubectl logs` on demand.
- For init container logs, use `source=init` in the UI.
- Endpoint calls are executed server-side from `ops-ui/server.js` and proxied back to UI.
- Chat metrics timeline can be exported as JSON for evaluation datasets.
- If running on master, use `KUBECONFIG=/etc/rancher/k3s/k3s.yaml npm start`.
- This folder is excluded from Docker build context for stage/gateway images.
