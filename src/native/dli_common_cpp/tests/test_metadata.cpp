#include "dli/common/metadata.hpp"
#include "dli/common/tensor.hpp"

#include <cassert>
#include <iostream>
#include <string>

using dli::common::infer_sequence_length_from_shape;
using dli::common::parse_request_metadata;

void test_parse_decode_metadata() {
    const std::string metadata =
        R"({"request_id":"demo-request-001","token_index":7,"generation_mode":"decode","tensor":{"dtype":"float16","shape":[1,1,4],"byte_order":"little"},"feature_flags":{"kv_cache_enabled":true},"sampling":{"temperature":0.0,"top_k":null,"top_p":null}})";

    const auto parsed = parse_request_metadata(metadata);

    assert(parsed.request_id == "demo-request-001");
    assert(parsed.token_index == 7);
    assert(parsed.generation_mode == "decode");

    assert(parsed.tensor.dtype == "float16");
    assert(parsed.tensor.byte_order == "little");
    assert(parsed.tensor.shape.size() == 3);
    assert(parsed.tensor.shape[0] == 1);
    assert(parsed.tensor.shape[1] == 1);
    assert(parsed.tensor.shape[2] == 4);

    assert(parsed.kv_cache_enabled);
    assert(parsed.has_temperature);
    assert(parsed.temperature == 0.0);

    assert(!parsed.has_top_k);
    assert(!parsed.has_top_p);

    assert(parsed.stage_input_token_count == 1);
}

void test_parse_prefill_metadata() {
    const std::string metadata =
        R"({"request_id":"prefill-001","token_index":0,"generation_mode":"prefill","tensor":{"dtype":"int64","shape":[1,24],"byte_order":"little"},"feature_flags":{"kv_cache_enabled":true},"sampling":{"temperature":0.2,"top_k":40,"top_p":0.95}})";

    const auto parsed = parse_request_metadata(metadata);

    assert(parsed.request_id == "prefill-001");
    assert(parsed.token_index == 0);
    assert(parsed.generation_mode == "prefill");

    assert(parsed.tensor.dtype == "int64");
    assert(parsed.tensor.shape.size() == 2);
    assert(parsed.tensor.shape[0] == 1);
    assert(parsed.tensor.shape[1] == 24);

    assert(parsed.kv_cache_enabled);
    assert(parsed.has_temperature);
    assert(parsed.temperature == 0.2);

    assert(parsed.has_top_k);
    assert(parsed.top_k == 40);

    assert(parsed.has_top_p);
    assert(parsed.top_p == 0.95);

    assert(parsed.stage_input_token_count == 24);
}

void test_sequence_length_inference() {
    assert(infer_sequence_length_from_shape({}) == 0);
    assert(infer_sequence_length_from_shape({8}) == 8);
    assert(infer_sequence_length_from_shape({1, 24}) == 24);
    assert(infer_sequence_length_from_shape({1, 1, 2048}) == 1);
    assert(infer_sequence_length_from_shape({1, 32, 2048}) == 32);
}

int main() {
    test_parse_decode_metadata();
    test_parse_prefill_metadata();
    test_sequence_length_inference();

    std::cout << "test_metadata: OK\n";
    return 0;
}