#!/usr/bin/env bash
# Build locally and optionally push Docker Hub images for distributed-layer-inference.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}"

# Defaults can be overridden from the shell:
#   DOCKERHUB_NAMESPACE=mipeda150 BUILDER=mybuilder scripts/build_images.sh
DOCKERHUB_NAMESPACE="${DOCKERHUB_NAMESPACE:-mipeda150}"
BUILDER="${BUILDER:-multiarch-insecure}"
PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.org/simple}"
PIP_TRUSTED_HOST="${PIP_TRUSTED_HOST:-pypi.org}"
INSECURE_FLAG="${INSECURE_FLAG:-}"
NATIVE_BUILD_IMAGE="${NATIVE_BUILD_IMAGE:-}"
NATIVE_RUNTIME_BASE_ARG="${NATIVE_RUNTIME_BASE_ARG:-RUNTIME_IMAGE}"
NATIVE_CMAKE_BUILD_TYPE="${NATIVE_CMAKE_BUILD_TYPE:-Release}"
NATIVE_GGML_NATIVE="${NATIVE_GGML_NATIVE:-OFF}"
NATIVE_GGML_CPU_ARM_ARCH="${NATIVE_GGML_CPU_ARM_ARCH:-armv8-a}"
NATIVE_GGML_CPU_ALL_VARIANTS="${NATIVE_GGML_CPU_ALL_VARIANTS:-OFF}"
ALLOW_SELF_BASE="${ALLOW_SELF_BASE:-}"
ENABLE_INLINE_CACHE="${ENABLE_INLINE_CACHE:-1}"
ENABLE_REGISTRY_CACHE="${ENABLE_REGISTRY_CACHE:-1}"
CACHE_TAG_PREFIX="${CACHE_TAG_PREFIX:-buildcache}"
SELF_BASE_DEFAULT_MODE="${SELF_BASE_DEFAULT_MODE:-2}"

prompt() {
  local message="$1"
  local default="${2:-}"
  local value

  if [[ -n "${default}" ]]; then
    read -rp "${message} (default: ${default}): " value
    printf '%s\n' "${value:-${default}}"
  else
    read -rp "${message}: " value
    printf '%s\n' "${value}"
  fi
}

lower() {
  tr '[:upper:]' '[:lower:]'
}

read_docker_hub_tags() {
  local image_repo="$1"
  local hub_url="https://registry.hub.docker.com/v2/repositories/${image_repo}/tags?page_size=100"
  local page_json page_tags next_url

  HUB_TAGS=()

  while [[ -n "${hub_url}" ]]; do
    page_json="$(curl --connect-timeout 5 --max-time 15 -fsSL "${hub_url}" 2>/dev/null || true)"
    [[ -n "${page_json}" ]] || break

    if command -v jq >/dev/null 2>&1; then
      mapfile -t page_tags < <(printf '%s' "${page_json}" | jq -r '.results[].name' 2>/dev/null || true)
      next_url="$(printf '%s' "${page_json}" | jq -r '.next // empty' 2>/dev/null || true)"
    elif command -v python3 >/dev/null 2>&1; then
      mapfile -t page_tags < <(printf '%s' "${page_json}" | python3 -c 'import json,sys; data=json.load(sys.stdin); [print(i.get("name")) for i in data.get("results", []) if i.get("name")]' 2>/dev/null || true)
      next_url="$(printf '%s' "${page_json}" | python3 -c 'import json,sys; data=json.load(sys.stdin); print(data.get("next") or "")' 2>/dev/null || true)"
    else
      break
    fi

    if [[ ${#page_tags[@]} -gt 0 ]]; then
      HUB_TAGS+=("${page_tags[@]}")
    fi

    hub_url="${next_url}"
  done
}

detect_platforms() {
  local image="$1"
  docker buildx imagetools inspect "${image}" 2>/dev/null \
    | awk -F: '/Platform:/{gsub(/ /,"",$2); print $2}' \
    || true
}

ask_cleanup() {
  local clean

  read -rp "Cleanup local images and dangling layers? [y/N]: " clean
  if [[ "${clean,,}" != "y" ]]; then
    return 0
  fi

  echo "Removing local image references..."
  for tag in "${LOCAL_TAGS[@]:-}"; do
    docker image rm -f "${tag}" 2>/dev/null || true
  done
  for tag in "${PUSH_TAGS[@]:-}"; do
    docker image rm -f "${tag}" 2>/dev/null || true
  done
  if [[ -n "${MANIFEST_TAG:-}" ]]; then
    docker image rm -f "${MANIFEST_TAG}" 2>/dev/null || true
  fi

  echo "Pruning dangling images..."
  docker image prune -f >/dev/null || true
  echo "Cleanup complete."
}

cat <<EOF
Distributed Layer Inference image builder
-----------------------------------------
Services:
  gateway-python -> legacy Python/FastAPI gateway runtime
  stage-python   -> legacy Python/PyTorch stage runtime
  native-gateway -> native C++ GGUF gateway runtime from docker/Dockerfile.native
  native-stage   -> native C++ GGUF stage runtime from docker/Dockerfile.native
  native-tools   -> native C++ GGUF manifest/shard tools from docker/Dockerfile.native
EOF

SERVICE="$(prompt "Service to build (gateway-python | stage-python | native-gateway | native-stage | native-tools)" | lower)"
if [[ -z "${SERVICE}" ]]; then
  echo "Service is required." >&2
  exit 1
fi

STAGE_RUNTIME=""
DOCKER_BUILD_TARGET=""
IMAGE_KIND="python"

case "${SERVICE}" in
  gateway|gateway-python|gw|master|inference-gateway|dli-gateway)
    SERVICE="gateway-python"
    IMAGE_KIND="python"
    DOCKERFILE="docker/Dockerfile.gateway-python"
    DOCKER_BUILD_TARGET=""
    DEFAULT_IMAGE_REPO="${DOCKERHUB_NAMESPACE}/distributed-layer-inference-gateway-python"
    ;;

  stage|stage-python|stage-pytorch|stage-legacy|client|worker|inference-stage|dli-stage)
    SERVICE="stage-python"
    IMAGE_KIND="python"
    STAGE_RUNTIME="python-pytorch-legacy"
    DOCKERFILE="docker/Dockerfile.stage-python"
    DOCKER_BUILD_TARGET=""
    DEFAULT_IMAGE_REPO="${DOCKERHUB_NAMESPACE}/distributed-layer-inference-stage-python"
    ;;

  native-gateway|gateway-cpp|gateway-native|cpp-gateway|dli-gateway-cpp)
    SERVICE="native-gateway"
    IMAGE_KIND="native"
    DOCKERFILE="docker/Dockerfile.native"
    DOCKER_BUILD_TARGET="gateway-runtime"
    DEFAULT_IMAGE_REPO="${DOCKERHUB_NAMESPACE}/distributed-layer-inference-native-gateway"
    ;;

  native-stage|stage-cpp|stage-native|cpp-stage|dli-stage-cpp)
    SERVICE="native-stage"
    IMAGE_KIND="native"
    STAGE_RUNTIME="cpp-native-gguf"
    DOCKERFILE="docker/Dockerfile.native"
    DOCKER_BUILD_TARGET="stage-runtime"
    DEFAULT_IMAGE_REPO="${DOCKERHUB_NAMESPACE}/distributed-layer-inference-native-stage"
    ;;

  native-tools|tools-cpp|cpp-tools|dli-tools-cpp)
    SERVICE="native-tools"
    IMAGE_KIND="native"
    DOCKERFILE="docker/Dockerfile.native"
    DOCKER_BUILD_TARGET="tools-runtime"
    DEFAULT_IMAGE_REPO="${DOCKERHUB_NAMESPACE}/distributed-layer-inference-native-tools"
    ;;

  *)
    echo "Unknown service: ${SERVICE}" >&2
    exit 1
    ;;
esac

if [[ ! -f "${DOCKERFILE}" ]]; then
  echo "Dockerfile not found at ${DOCKERFILE}. Run this from the repo root." >&2
  exit 1
fi

if [[ "${IMAGE_KIND}" == "native" ]]; then
  DOCKERFILE_BASE_DEFAULT="$(
    awk -F= -v arg="${NATIVE_RUNTIME_BASE_ARG}" '
      $0 ~ "^ARG[[:space:]]+" arg "=" {print $2; exit}
    ' "${DOCKERFILE}" | tr -d '[:space:]'
  )"
  DOCKERFILE_BASE_DEFAULT="${DOCKERFILE_BASE_DEFAULT:-debian:bookworm-slim}"
else
  DOCKERFILE_BASE_DEFAULT="$(awk -F= '/^ARG[[:space:]]+BASE_IMAGE=/{print $2; exit}' "${DOCKERFILE}" | tr -d '[:space:]')"
  DOCKERFILE_BASE_DEFAULT="${DOCKERFILE_BASE_DEFAULT:-python:3.11-slim}"
fi

IMAGE_REPO="$(prompt "Image repo" "${DEFAULT_IMAGE_REPO}")"
if [[ -z "${IMAGE_REPO}" ]]; then
  echo "Image repo is required." >&2
  exit 1
fi

echo "Listing existing tags for ${IMAGE_REPO} on Docker Hub..."
declare -a HUB_TAGS
read_docker_hub_tags "${IMAGE_REPO}"

if [[ ${#HUB_TAGS[@]} -gt 0 ]]; then
  echo "Available tags:"
  idx=1
  for tag in "${HUB_TAGS[@]}"; do
    echo "  [${idx}] ${tag}"
    idx=$((idx + 1))
  done
else
  echo "  (unable to list tags from Docker Hub for ${IMAGE_REPO})"
fi

read -rp "Base image tag number or tag (blank = Dockerfile default ${DOCKERFILE_BASE_DEFAULT}): " BASE_INPUT
BASE_IMAGE=""
if [[ -n "${BASE_INPUT}" ]]; then
  if [[ "${BASE_INPUT}" =~ ^[0-9]+$ ]] && [[ ${#HUB_TAGS[@]} -gt 0 ]]; then
    selected_index="${BASE_INPUT}"
    if (( selected_index >= 1 && selected_index <= ${#HUB_TAGS[@]} )); then
      BASE_IMAGE="${IMAGE_REPO}:${HUB_TAGS[$((selected_index - 1))]}"
    else
      echo "Index out of range; using raw value as image/tag."
      BASE_IMAGE="${BASE_INPUT}"
    fi
  elif [[ "${BASE_INPUT}" == *":"* || "${BASE_INPUT}" == */* ]]; then
    BASE_IMAGE="${BASE_INPUT}"
  else
    BASE_IMAGE="${IMAGE_REPO}:${BASE_INPUT}"
  fi
fi

NEW_TAG="$(prompt "New tag (example: v2026-05-05-1830)")"
if [[ -z "${NEW_TAG}" ]]; then
  echo "A new tag is required." >&2
  exit 1
fi

echo "Select build target:"
echo "  [1] both amd64 + arm64"
echo "  [2] amd64 only"
echo "  [3] arm64 only"
ARCH_CHOICE="$(prompt "Choice (1/2/3)" "1")"

declare -a PLATFORMS
declare -a ARCH_SUFFIXES
case "${ARCH_CHOICE}" in
  2)
    PLATFORMS=("linux/amd64")
    ARCH_SUFFIXES=("amd64")
    ;;
  3)
    PLATFORMS=("linux/arm64")
    ARCH_SUFFIXES=("arm64")
    ;;
  *)
    PLATFORMS=("linux/amd64" "linux/arm64")
    ARCH_SUFFIXES=("amd64" "arm64")
    ;;
esac

declare -A BASE_BY_ARCH
declare -A CACHE_FROM_BY_ARCH
declare -A REGISTRY_CACHE_REF_BY_ARCH
if [[ -n "${BASE_IMAGE}" ]]; then
  for arch in "${ARCH_SUFFIXES[@]}"; do
    BASE_BY_ARCH["${arch}"]="${BASE_IMAGE}"
  done

  if [[ "${BASE_IMAGE}" == *"-amd64" ]]; then
    BASE_BY_ARCH["arm64"]=""
  elif [[ "${BASE_IMAGE}" == *"-arm64" ]]; then
    BASE_BY_ARCH["amd64"]=""
  fi

  SUPPORTED_PLATFORMS="$(detect_platforms "${BASE_IMAGE}")"
  if [[ -n "${SUPPORTED_PLATFORMS}" ]]; then
    for arch in "${ARCH_SUFFIXES[@]}"; do
      platform="linux/${arch}"
      if ! grep -q "^${platform}$" <<<"${SUPPORTED_PLATFORMS}"; then
        BASE_BY_ARCH["${arch}"]=""
        echo "Warning: base image ${BASE_IMAGE} does not advertise ${platform}; ${arch} will use Dockerfile default base."
      fi
    done
  else
    echo "Warning: unable to inspect base image platforms; using it for selected architectures."
  fi

  if [[ "${BASE_IMAGE}" == "${IMAGE_REPO}:"* || "${BASE_IMAGE}" == "${IMAGE_REPO}@"* ]]; then
    if [[ "${ALLOW_SELF_BASE}" == "1" ]]; then
      echo "ALLOW_SELF_BASE=1 set; using same-repo image as BASE_IMAGE."
    else
      echo "Selected base image is from the same target repo."
      echo "  [1] use it as BASE_IMAGE"
      echo "  [2] use it only as --cache-from to avoid recursive layer growth"
      SELF_BASE_MODE="$(prompt "Choice (1/2)" "${SELF_BASE_DEFAULT_MODE}")"
      if [[ "${SELF_BASE_MODE}" == "2" ]]; then
        for arch in "${ARCH_SUFFIXES[@]}"; do
          CACHE_FROM_BY_ARCH["${arch}"]="${BASE_BY_ARCH[${arch}]:-}"
          BASE_BY_ARCH["${arch}"]=""
        done
        BASE_IMAGE=""
      else
        echo "Warning: self-base mode [1] often invalidates dependency-layer cache and may reinstall requirements each build."
      fi
    fi
  fi
fi

if [[ "${ENABLE_REGISTRY_CACHE}" == "1" ]]; then
  for arch in "${ARCH_SUFFIXES[@]}"; do
    REGISTRY_CACHE_REF_BY_ARCH["${arch}"]="${IMAGE_REPO}:${CACHE_TAG_PREFIX}-${SERVICE}-${arch}"
  done
fi

MANIFEST_TAG="${IMAGE_REPO}:${NEW_TAG}"
LOCAL_TAGS=()
PUSH_TAGS=()
for arch in "${ARCH_SUFFIXES[@]}"; do
  LOCAL_TAGS+=("${IMAGE_REPO}:${arch}-local")
  PUSH_TAGS+=("${IMAGE_REPO}:${NEW_TAG}-${arch}")
done

cat <<EOF

Summary
-------
Service        : ${SERVICE}
Image kind     : ${IMAGE_KIND}
Builder        : ${BUILDER}
Image repo     : ${IMAGE_REPO}
Tag            : ${NEW_TAG}
Platforms      : ${PLATFORMS[*]}
Base image     : ${BASE_IMAGE:-<dockerfile-default:${DOCKERFILE_BASE_DEFAULT}>}
Dockerfile     : ${DOCKERFILE}
Docker target  : ${DOCKER_BUILD_TARGET:-<default>}
Build context  : ${REPO_ROOT}
EOF

if [[ "${SERVICE}" == *stage* ]]; then
  echo "Stage runtime  : ${STAGE_RUNTIME}"
fi

case "${SERVICE}" in
  stage-python)
    echo "Model source   : PyTorch .pt partitions from Hugging Face initContainer"
    ;;
  native-stage)
    echo "Model source   : DLI GGUF partition-N.dli.gguf shards from Hugging Face/initContainer"
    ;;
  native-gateway)
    echo "Model source   : full GGUF model for tokenizer, mounted/downloaded at runtime"
    ;;
  native-tools)
    echo "Model source   : offline shard generation/validation image"
    ;;
esac

if [[ "${IMAGE_KIND}" == "native" ]]; then
  echo "Native base arg        : ${NATIVE_RUNTIME_BASE_ARG}"
  echo "Native CMake type      : ${NATIVE_CMAKE_BUILD_TYPE}"
  echo "Native GGML_NATIVE     : ${NATIVE_GGML_NATIVE}"
  echo "Native ARM arch        : ${NATIVE_GGML_CPU_ARM_ARCH}"
  echo "Native CPU variants    : ${NATIVE_GGML_CPU_ALL_VARIANTS}"

  if [[ -n "${NATIVE_BUILD_IMAGE}" ]]; then
    echo "Native build image override: ${NATIVE_BUILD_IMAGE}"
  fi
fi

for arch in "${ARCH_SUFFIXES[@]}"; do
  if [[ -n "${CACHE_FROM_BY_ARCH[${arch}]:-}" ]]; then
    echo "Cache ${arch}    : ${CACHE_FROM_BY_ARCH[${arch}]}"
  fi
  if [[ -n "${REGISTRY_CACHE_REF_BY_ARCH[${arch}]:-}" ]]; then
    echo "Build cache ${arch}: ${REGISTRY_CACHE_REF_BY_ARCH[${arch}]}"
  fi
done

read -rp "Ready to build local image(s) for ${PLATFORMS[*]}? [y/N]: " CONTINUE_BUILD
if [[ "${CONTINUE_BUILD,,}" != "y" ]]; then
  echo "Aborting."
  exit 0
fi

if ! docker buildx inspect "${BUILDER}" >/dev/null 2>&1; then
  echo "Warning: buildx builder '${BUILDER}' was not found or is not available." >&2
  echo "Docker will fail unless this builder exists. Override with BUILDER=<name>." >&2
fi

for i in "${!PLATFORMS[@]}"; do
  platform="${PLATFORMS[$i]}"
  arch="${ARCH_SUFFIXES[$i]}"
  local_tag="${LOCAL_TAGS[$i]}"
  registry_cache_to_added=0

  docker image rm -f "${local_tag}" 2>/dev/null || true

CMD=(docker buildx build
  --builder "${BUILDER}"
  --platform "${platform}"
)

if [[ -n "${DOCKER_BUILD_TARGET}" ]]; then
  CMD+=(--target "${DOCKER_BUILD_TARGET}")
fi

if [[ "${IMAGE_KIND}" == "python" ]]; then
  CMD+=(
    --build-arg "PIP_INDEX_URL=${PIP_INDEX_URL}"
    --build-arg "PIP_TRUSTED_HOST=${PIP_TRUSTED_HOST}"
  )
fi

if [[ "${IMAGE_KIND}" == "native" ]]; then
  CMD+=(
    --build-arg "CMAKE_BUILD_TYPE=${NATIVE_CMAKE_BUILD_TYPE}"
    --build-arg "GGML_NATIVE=${NATIVE_GGML_NATIVE}"
    --build-arg "GGML_CPU_ARM_ARCH=${NATIVE_GGML_CPU_ARM_ARCH}"
    --build-arg "GGML_CPU_ALL_VARIANTS=${NATIVE_GGML_CPU_ALL_VARIANTS}"
  )

  if [[ -n "${NATIVE_BUILD_IMAGE}" ]]; then
    CMD+=(--build-arg "BUILD_IMAGE=${NATIVE_BUILD_IMAGE}")
  fi
fi

CMD+=(
  -t "${local_tag}"
  -f "${DOCKERFILE}"
  .
  --load
)

  if [[ "${ENABLE_INLINE_CACHE}" == "1" ]]; then
    CMD+=(--build-arg "BUILDKIT_INLINE_CACHE=1")
  fi

  base_for_arch="${BASE_BY_ARCH[${arch}]:-}"
if [[ -n "${base_for_arch}" ]]; then
  if [[ "${IMAGE_KIND}" == "native" ]]; then
    CMD+=(--build-arg "${NATIVE_RUNTIME_BASE_ARG}=${base_for_arch}")
  else
    CMD+=(--build-arg "BASE_IMAGE=${base_for_arch}")
  fi
fi

  cache_for_arch="${CACHE_FROM_BY_ARCH[${arch}]:-}"
  if [[ -n "${cache_for_arch}" ]]; then
    if docker buildx imagetools inspect "${cache_for_arch}" >/dev/null 2>&1; then
      CMD+=(--cache-from "type=registry,ref=${cache_for_arch}")
    else
      echo "Info: cache source ${cache_for_arch} not found; skipping."
    fi
  fi

  registry_cache_ref="${REGISTRY_CACHE_REF_BY_ARCH[${arch}]:-}"
  if [[ -n "${registry_cache_ref}" ]]; then
    if docker buildx imagetools inspect "${registry_cache_ref}" >/dev/null 2>&1; then
      CMD+=(--cache-from "type=registry,ref=${registry_cache_ref}")
    else
      echo "Info: build cache ${registry_cache_ref} not found yet; first build will populate it."
    fi
    if [[ "${registry_cache_to_added}" -eq 0 ]]; then
      CMD+=(--cache-to "type=registry,ref=${registry_cache_ref},mode=max,ignore-error=true")
      registry_cache_to_added=1
    fi
  fi

  echo "Running: ${CMD[*]}"
  "${CMD[@]}"
done

echo "Tagging local image(s) for push..."
for i in "${!LOCAL_TAGS[@]}"; do
  docker tag "${LOCAL_TAGS[$i]}" "${PUSH_TAGS[$i]}"
done

read -rp "Push arch image(s) to ${IMAGE_REPO}? [y/N]: " CONTINUE_PUSH
if [[ "${CONTINUE_PUSH,,}" != "y" ]]; then
  echo "Skipping push."
  ask_cleanup
  exit 0
fi

for tag in "${PUSH_TAGS[@]}"; do
  echo "Pushing ${tag}..."
  docker push "${tag}"
done

echo "Creating and pushing manifest ${MANIFEST_TAG}..."
IMAGETOOLS_CMD=(docker buildx imagetools create --tag "${MANIFEST_TAG}")
if [[ -n "${INSECURE_FLAG}" ]]; then
  IMAGETOOLS_CMD+=("${INSECURE_FLAG}")
fi
IMAGETOOLS_CMD+=("${PUSH_TAGS[@]}")

if "${IMAGETOOLS_CMD[@]}"; then
  echo "Manifest published via docker buildx imagetools."
else
  echo "docker buildx imagetools failed; retrying with docker manifest workflow."
  docker manifest rm "${MANIFEST_TAG}" >/dev/null 2>&1 || true

  MANIFEST_CREATE_CMD=(docker manifest create "${MANIFEST_TAG}")
  if [[ -n "${INSECURE_FLAG}" ]]; then
    MANIFEST_CREATE_CMD+=("${INSECURE_FLAG}")
  fi
  MANIFEST_CREATE_CMD+=("${PUSH_TAGS[@]}")

  if ! "${MANIFEST_CREATE_CMD[@]}"; then
    echo "Manifest may already exist; retrying with --amend."
    MANIFEST_AMEND_CMD=(docker manifest create --amend "${MANIFEST_TAG}")
    if [[ -n "${INSECURE_FLAG}" ]]; then
      MANIFEST_AMEND_CMD+=("${INSECURE_FLAG}")
    fi
    MANIFEST_AMEND_CMD+=("${PUSH_TAGS[@]}")
    "${MANIFEST_AMEND_CMD[@]}"
  fi

  MANIFEST_PUSH_CMD=(docker manifest push "${MANIFEST_TAG}")
  if [[ -n "${INSECURE_FLAG}" ]]; then
    MANIFEST_PUSH_CMD+=("${INSECURE_FLAG}")
  fi
  "${MANIFEST_PUSH_CMD[@]}"
fi

echo "Inspecting final manifest:"
docker buildx imagetools inspect "${MANIFEST_TAG}" || docker manifest inspect "${MANIFEST_TAG}" || true

echo "Latest tags for ${IMAGE_REPO} on Docker Hub:"
if command -v jq >/dev/null 2>&1; then
  curl --connect-timeout 5 --max-time 15 -fsSL "https://registry.hub.docker.com/v2/repositories/${IMAGE_REPO}/tags?page_size=20" | jq || true
else
  curl --connect-timeout 5 --max-time 15 -fsSL "https://registry.hub.docker.com/v2/repositories/${IMAGE_REPO}/tags?page_size=20" || true
fi

ask_cleanup
