# DLI2 Native Tensor ABI

Status: initial native correctness contract.

This ABI defines the tensor payloads exchanged between the native gateway and native partition nodes.

---

## Payload classes

### Token ids

Used as input to the source partition, meaning the partition with:

```text
owns_embedding=true