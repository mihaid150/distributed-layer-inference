#!/usr/bin/env bash
# Compatibility wrapper: build_images.sh now handles build, push, manifest creation, and cleanup.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${SCRIPT_DIR}/build_images.sh"
