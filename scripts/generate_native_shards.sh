#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:?usage: scripts/generate_native_shards.sh <stage_map.yaml> <model.gguf> <output-dir>}"
MODEL_PATH="${2:?usage: scripts/generate_native_shards.sh <stage_map.yaml> <model.gguf> <output-dir>}"
OUTPUT_DIR="${3:?usage: scripts/generate_native_shards.sh <stage_map.yaml> <model.gguf> <output-dir>}"

BUILD_DIR="${BUILD_DIR:-/tmp/dli-native-build}"
PARTITION_DIR="${OUTPUT_DIR}/manifests"
SHARD_DIR="${OUTPUT_DIR}/shards"

rm -rf "${PARTITION_DIR}" "${SHARD_DIR}"
mkdir -p "${PARTITION_DIR}" "${SHARD_DIR}"

"${BUILD_DIR}/bin/dli-partition-manifest" \
  --config "${CONFIG_PATH}" \
  --model "${MODEL_PATH}" \
  --output-dir "${PARTITION_DIR}" \
  | jq

"${BUILD_DIR}/bin/dli-gguf-shard-writer" \
  --model "${MODEL_PATH}" \
  --manifest-dir "${PARTITION_DIR}" \
  --output-dir "${SHARD_DIR}"

"${BUILD_DIR}/bin/dli-gguf-shard-validate" \
  --config "${CONFIG_PATH}" \
  --manifest-dir "${PARTITION_DIR}" \
  --shard-dir "${SHARD_DIR}"

echo "Generated shards in ${SHARD_DIR}"