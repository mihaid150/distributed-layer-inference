#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ -d "${SCRIPT_DIR}/k8s" ]]; then
  K8S_DIR="${SCRIPT_DIR}/k8s"
elif [[ -d "${REPO_ROOT}/k8s" ]]; then
  K8S_DIR="${REPO_ROOT}/k8s"
else
  echo "[deploy_k3s] Could not find k8s directory." >&2
  echo "[deploy_k3s] Checked: ${SCRIPT_DIR}/k8s and ${REPO_ROOT}/k8s" >&2
  exit 1
fi

APPLY_RETRIES="${APPLY_RETRIES:-10}"
APPLY_RETRY_SLEEP_SECONDS="${APPLY_RETRY_SLEEP_SECONDS:-3}"
API_WAIT_RETRIES="${API_WAIT_RETRIES:-45}"
API_WAIT_SLEEP_SECONDS="${API_WAIT_SLEEP_SECONDS:-2}"
KUBECTL_REQUEST_TIMEOUT="${KUBECTL_REQUEST_TIMEOUT:-20s}"

# `--validate=false` avoids OpenAPI download failures when API server is under load.
KUBECTL_VALIDATE_FLAG="${KUBECTL_VALIDATE_FLAG:---validate=false}"

kubectl_safe() {
  kubectl --request-timeout="${KUBECTL_REQUEST_TIMEOUT}" "$@"
}

required_manifests=(
  "namespace.yaml"
  "configmap.yaml"
  "inference-stage-1-service.yaml"
  "inference-stage-2-service.yaml"
  "inference-stage-3-service.yaml"
  "inference-stage-4-service.yaml"
  "inference-gateway-service.yaml"
  "inference-stage-1-deployment.yaml"
  "inference-stage-2-deployment.yaml"
  "inference-stage-3-deployment.yaml"
  "inference-stage-4-deployment.yaml"
  "inference-gateway-deployment.yaml"
)

validate_manifest_set() {
  local missing=0
  local manifest

  for manifest in "${required_manifests[@]}"; do
    if [[ ! -f "${K8S_DIR}/${manifest}" ]]; then
      echo "[deploy_k3s] Missing required file: ${K8S_DIR}/${manifest}" >&2
      missing=1
    fi
  done

  if [[ "${missing}" -ne 0 ]]; then
    echo "[deploy_k3s] Deployment aborted due to missing manifests." >&2
    return 1
  fi
}

wait_for_api_server() {
  local i
  for ((i = 1; i <= API_WAIT_RETRIES; i++)); do
    if kubectl_safe version >/dev/null 2>&1; then
      return 0
    fi
    echo "[deploy_k3s] waiting for Kubernetes API (${i}/${API_WAIT_RETRIES})..."
    sleep "${API_WAIT_SLEEP_SECONDS}"
  done

  echo "[deploy_k3s] Kubernetes API did not become ready in time." >&2
  return 1
}

apply_with_retry() {
  local manifest_file="$1"
  local i

  for ((i = 1; i <= APPLY_RETRIES; i++)); do
    if kubectl_safe apply "${KUBECTL_VALIDATE_FLAG}" -f "${manifest_file}"; then
      return 0
    fi
    echo "[deploy_k3s] apply failed for ${manifest_file} (${i}/${APPLY_RETRIES}), retrying..."
    sleep "${APPLY_RETRY_SLEEP_SECONDS}"
  done

  echo "[deploy_k3s] failed to apply ${manifest_file} after ${APPLY_RETRIES} attempts." >&2
  return 1
}

wait_for_api_server
validate_manifest_set

apply_with_retry "${K8S_DIR}/namespace.yaml"
apply_with_retry "${K8S_DIR}/configmap.yaml"

apply_with_retry "${K8S_DIR}/inference-stage-1-service.yaml"
apply_with_retry "${K8S_DIR}/inference-stage-2-service.yaml"
apply_with_retry "${K8S_DIR}/inference-stage-3-service.yaml"
apply_with_retry "${K8S_DIR}/inference-stage-4-service.yaml"
apply_with_retry "${K8S_DIR}/inference-gateway-service.yaml"

apply_with_retry "${K8S_DIR}/inference-stage-1-deployment.yaml"
apply_with_retry "${K8S_DIR}/inference-stage-2-deployment.yaml"
apply_with_retry "${K8S_DIR}/inference-stage-3-deployment.yaml"
apply_with_retry "${K8S_DIR}/inference-stage-4-deployment.yaml"
apply_with_retry "${K8S_DIR}/inference-gateway-deployment.yaml"

kubectl_safe get pods -n inference -o wide || true
