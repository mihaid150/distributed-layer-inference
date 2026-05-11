# DLI Stage Runtimes

This directory holds non-Python runtime workspaces for inference stages.

- `python-pytorch-legacy/` documents the legacy stage runtime, whose importable
  code lives in `src/dli/stage_runtimes/python_legacy`.
- `cpp/` is the planned native C++/ggml stage runtime workspace.

The Python gateway and binary `/forward-binary` protocol remain the compatibility
contract between runtime implementations.
