import unittest

import torch
from torch import nn

from dli.common.feature_flags import FeatureFlags
from dli.common.schemas import StageForwardRequest, StageForwardResponse, TokenStepMetric
from dli.feature_modules.activation_transport_binary import BinaryTransportModule
from dli.feature_modules.kv_cache_stage import StageForwardCache
from dli.inference_gateway.generation_loop import GenerationLoop
from dli.inference_gateway.stage_client import StageClient
from dli.inference_stage.model_partition_loader import ModelPartition
from dli.inference_stage.stage_executor import StageExecutor


class PluralCacheLayer(nn.Module):
    def forward(self, hidden_states, past_key_values=None, use_cache=False):
        return hidden_states


class SingularCacheLayer(nn.Module):
    def forward(self, hidden_states, past_key_value=None, use_cache=False):
        return hidden_states


class NoCacheLayer(nn.Module):
    def forward(self, hidden_states):
        return hidden_states


def _executor_for(layer: nn.Module) -> StageExecutor:
    partition = ModelPartition(
        embedding=None,
        layers=[layer],
        norm=None,
        lm_head=None,
        metadata={},
    )
    return StageExecutor(partition=partition, device="cpu", stage_id=1)


class TransportMetricsTests(unittest.TestCase):
    def test_binary_v2_tensor_roundtrip_supported_dtypes(self):
        tensors = [
            torch.arange(6, dtype=torch.int64).reshape(2, 3),
            torch.arange(6, dtype=torch.float32).reshape(2, 3),
            torch.arange(6, dtype=torch.float16).reshape(2, 3),
            torch.arange(6, dtype=torch.bfloat16).reshape(2, 3),
            torch.arange(6, dtype=torch.int8).reshape(2, 3),
        ]

        for tensor in tensors:
            with self.subTest(dtype=str(tensor.dtype)):
                blob, metadata = BinaryTransportModule.encode_tensor_blob(tensor)
                restored = BinaryTransportModule.decode_tensor_blob(blob, metadata)
                self.assertEqual(restored.dtype, tensor.dtype)
                self.assertEqual(list(restored.shape), list(tensor.shape))
                self.assertTrue(torch.equal(restored, tensor))

    def test_binary_v2_request_response_envelopes_are_raw_header_format(self):
        request = {
            "request_id": "r1",
            "token_index": 0,
            "tensor_b64": "",
            "tensor_dtype": "torch.float32",
            "tensor_shape": [1, 2],
            "metadata": {},
            "metrics": [],
            "transport": {},
            "feature_flags": FeatureFlags().model_dump(),
        }
        tensor = torch.ones((1, 2), dtype=torch.float32)
        blob, metadata = BinaryTransportModule.encode_tensor_blob(tensor)

        packed = BinaryTransportModule.pack_request(
            request_dict=request,
            tensor_blob=blob,
            tensor_metadata=metadata,
        )

        self.assertTrue(packed.startswith(BinaryTransportModule.MAGIC))
        unpacked = BinaryTransportModule.unpack_request(packed)
        self.assertEqual(unpacked["wire_version"], BinaryTransportModule.WIRE_VERSION)
        self.assertEqual(unpacked["request"]["request_id"], "r1")

        restored = BinaryTransportModule.decode_tensor_blob(
            unpacked["tensor_blob"],
            unpacked["tensor"],
        )
        self.assertTrue(torch.equal(restored, tensor))

        response = {
            "request_id": "r1",
            "token_index": 0,
            "metrics": [],
            "server_wall_ms": 3.5,
        }
        response_packed = BinaryTransportModule.pack_response(response)
        self.assertTrue(response_packed.startswith(BinaryTransportModule.MAGIC))
        self.assertEqual(
            BinaryTransportModule.unpack_response(response_packed)["server_wall_ms"],
            3.5,
        )

    def test_forward_dedupe_flag_is_separate_from_kv_cache_flag(self):
        flags = FeatureFlags(kv_cache_enabled=True, forward_dedupe_enabled=False)
        self.assertFalse(StageForwardCache.is_enabled(flags))

        flags = FeatureFlags(kv_cache_enabled=True, forward_dedupe_enabled=True)
        self.assertTrue(StageForwardCache.is_enabled(flags))

    def test_stage_executor_detects_cache_signature_variants(self):
        self.assertEqual(_executor_for(PluralCacheLayer()).cache_arg_name, "past_key_values")
        self.assertEqual(_executor_for(SingularCacheLayer()).cache_arg_name, "past_key_value")

        executor = _executor_for(NoCacheLayer())
        self.assertIsNone(executor.cache_arg_name)
        with self.assertRaisesRegex(RuntimeError, "past_key_values"):
            executor.ensure_kv_cache_supported()

    def test_gateway_summary_exposes_rpc_and_true_comm_ratios(self):
        loop = GenerationLoop.__new__(GenerationLoop)
        token_metrics = [
            TokenStepMetric(
                token_index=0,
                token_id=1,
                token_text="x",
                latency_ms=20.0,
                stage_metrics=[
                    {
                        "service_name": "stage",
                        "stage_id": 1,
                        "compute_time_ms": 10.0,
                        "transfer_time_ms": 30.0,
                        "rpc_wall_time_ms": 30.0,
                        "true_comm_ms": 5.0,
                        "outbound_payload_bytes": 100,
                        "network_delta": {"bytes_total": 200},
                    }
                ],
            )
        ]

        summary = loop._build_summary_metrics(
            token_metrics=token_metrics,
            total_latency_ms=20.0,
            prompt_token_count=2,
            feature_flags=FeatureFlags().model_dump(),
        )
        aggregate = summary["aggregate"]
        self.assertEqual(aggregate["rpc_compute_ratio"], 3.0)
        self.assertEqual(aggregate["true_comm_compute_ratio"], 0.5)
        self.assertEqual(aggregate["comm_compute_ratio"], 0.5)

    def test_binary_stage_client_does_not_preencode_base64(self):
        flags = FeatureFlags(transport_mode="binary_octet_stream")
        request = StageForwardRequest(
            request_id="r1",
            token_index=0,
            tensor_b64="",
            tensor_dtype="torch.int64",
            tensor_shape=[1, 2],
            metadata={},
            metrics=[],
            transport={},
            feature_flags=flags,
        )
        response_model = StageForwardResponse(
            request_id="r1",
            token_index=0,
            next_token_id=7,
            metrics=[],
            server_wall_ms=1.0,
        )

        class FakeResponse:
            status_code = 200
            content = BinaryTransportModule.pack_response(response_model.model_dump())

            def raise_for_status(self):
                return None

        client = StageClient("http://stage-1/forward")
        client.codec.encode_tensor = lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("base64 encode called")
        )
        client._post = lambda **_kwargs: FakeResponse()

        result = client.forward_with_transport(request, input_tensor=torch.tensor([[1, 2]]))

        self.assertEqual(result.response.next_token_id, 7)
        self.assertEqual(result.transport["encoding"], "binary_octet_stream")
        self.assertGreaterEqual(result.transport["true_comm_ms"], 0.0)


if __name__ == "__main__":
    unittest.main()
