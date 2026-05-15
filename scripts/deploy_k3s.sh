cat > ~/dli/deploy_k3s.sh <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
K8S_DIR="${ROOT_DIR}/k8s"

KUBECONFIG_PATH="${KUBECONFIG:-/etc/rancher/k3s/k3s.yaml}"
export KUBECONFIG="${KUBECONFIG_PATH}"

usage() {
  cat <<USAGE
Usage:
  ./deploy_k3s.sh <profile> [action]

Profiles:
  python    Deploy YAMLs from k8s/python
  native    Deploy YAMLs from k8s/native

Actions:
  apply     Apply manifests. Default.
  delete    Delete manifests.
  diff      Show kubectl diff.

Examples:
  ./deploy_k3s.sh native
  ./deploy_k3s.sh native apply
  ./deploy_k3s.sh python apply
  ./deploy_k3s.sh native delete
  ./deploy_k3s.sh native diff
USAGE
}

PROFILE="${1:-}"
ACTION="${2:-apply}"

if [[ -z "${PROFILE}" || "${PROFILE}" == "-h" || "${PROFILE}" == "--help" ]]; then
  usage
  exit 0
fi

case "${PROFILE}" in
  python|pytorch)
    PROFILE="python"
    ;;
  native|cpp|llama|gguf)
    PROFILE="native"
    ;;
  *)
    echo "Unknown profile: ${PROFILE}" >&2
    usage
    exit 1
    ;;
esac

case "${ACTION}" in
  apply|delete|diff)
    ;;
  *)
    echo "Unknown action: ${ACTION}" >&2
    usage
    exit 1
    ;;
esac

PROFILE_DIR="${K8S_DIR}/${PROFILE}"

if [[ ! -d "${PROFILE_DIR}" ]]; then
  echo "Profile directory not found: ${PROFILE_DIR}" >&2
  echo "Available profiles:" >&2
  find "${K8S_DIR}" -maxdepth 1 -mindepth 1 -type d -printf "  %f\n" | sort >&2
  exit 1
fi

echo "DLI K3s deploy"
echo "--------------"
echo "Root       : ${ROOT_DIR}"
echo "Kubeconfig : ${KUBECONFIG}"
echo "Profile    : ${PROFILE}"
echo "Directory  : ${PROFILE_DIR}"
echo "Action     : ${ACTION}"
echo

kubectl version --client=true >/dev/null

ordered_files=()

for name in \
  namespace.yaml \
  persistent-volumes.yaml \
  configmap.yaml \
  services.yaml \
  deployments.yaml
do
  if [[ -f "${PROFILE_DIR}/${name}" ]]; then
    ordered_files+=("${PROFILE_DIR}/${name}")
  fi
done

# Also include any additional YAML files not already included.
while IFS= read -r file; do
  skip=0
  for existing in "${ordered_files[@]}"; do
    if [[ "${file}" == "${existing}" ]]; then
      skip=1
      break
    fi
  done
  if [[ "${skip}" -eq 0 ]]; then
    ordered_files+=("${file}")
  fi
done < <(find "${PROFILE_DIR}" -maxdepth 1 -type f \( -name "*.yaml" -o -name "*.yml" \) | sort)

if [[ "${#ordered_files[@]}" -eq 0 ]]; then
  echo "No YAML files found in ${PROFILE_DIR}" >&2
  exit 1
fi

echo "Manifest order:"
for file in "${ordered_files[@]}"; do
  echo "  - ${file#${ROOT_DIR}/}"
done
echo

if [[ "${ACTION}" == "apply" ]]; then
  for file in "${ordered_files[@]}"; do
    echo "Applying ${file#${ROOT_DIR}/}"
    kubectl apply -f "${file}"
  done

  echo
  echo "Current inference resources:"
  kubectl -n inference get pods -o wide || true
  kubectl -n inference get svc || true
  kubectl -n inference get deploy || true
fi

if [[ "${ACTION}" == "delete" ]]; then
  for (( idx=${#ordered_files[@]}-1 ; idx>=0 ; idx-- )); do
    file="${ordered_files[$idx]}"
    echo "Deleting ${file#${ROOT_DIR}/}"
    kubectl delete -f "${file}" --ignore-not-found=true
  done
fi

if [[ "${ACTION}" == "diff" ]]; then
  for file in "${ordered_files[@]}"; do
    echo "Diff ${file#${ROOT_DIR}/}"
    kubectl diff -f "${file}" || true
  done
fi
EOF

chmod +x ~/dli/deploy_k3s.sh