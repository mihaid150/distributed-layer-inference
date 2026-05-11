# Python/PyTorch Legacy Stage Runtime

The legacy runtime source is packaged at:

`src/dli/stage_runtimes/python_legacy`

The old `dli.inference_stage` package remains as a compatibility shim for tests,
gateway imports, and older Docker/Kubernetes references.

The legacy stage image is built from `docker/Dockerfile.stage` and runs:

`dli.stage_runtimes.python_legacy.app:app`
