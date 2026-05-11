"""
Compatibility package for the legacy Python/PyTorch stage runtime.

New stage runtime work should live under ``dli.stage_runtimes``.  This package
keeps the old ``dli.inference_stage`` import path stable for existing gateway,
tests, Docker images, and Kubernetes manifests.
"""

