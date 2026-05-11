# Native C++ DLI Stage Runtime

This is the workspace for the planned C++ stage server. It is intentionally a
skeleton: no native stage server is implemented or compiled yet.

The target runtime is not `llama-server`. The planned server is a custom DLI
stage process that preserves the existing gateway contract:

- `GET /health`
- `GET /config`
- `POST /forward-binary`

The server will eventually load DLI GGUF stage shards and execute partial Llama
layers with ggml/llama.cpp internals:

- stage 1: token ids -> embedding + local layers -> hidden states
- stage 2/3: hidden states -> local layers -> hidden states
- stage 4: hidden states -> local layers + norm + lm_head -> next token id

Suggested layout:

- `include/dli_stage/`: public C++ headers for protocol, tensor views, server,
  model runtime, and metrics.
- `src/`: native stage server implementation.
- `tests/`: protocol and runtime tests.
- `third_party/`: pinned external sources or build metadata for llama.cpp,
  ggml, cpp-httplib/libcurl, and JSON support.
- `cmake/`: toolchain and dependency helpers.

The placeholder image target is `docker/Dockerfile.stage-cpp`. It is not
deployable until a native `/forward-binary` server is implemented.
