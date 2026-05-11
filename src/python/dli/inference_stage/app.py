"""Compatibility ASGI entrypoint for the legacy Python/PyTorch stage app."""

from dli.stage_runtimes.python_legacy.app import app

__all__ = ["app"]

