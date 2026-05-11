#include "dli/common/protocol.hpp"

#include <cassert>
#include <cstdint>
#include <iostream>

using dli::common::DliFrame;
using dli::common::ProtocolError;
using dli::common::decode_frame;
using dli::common::encode_frame;

void test_roundtrip_with_tensor() {
    DliFrame frame;
    frame.metadata_json = R"({"request_id":"abc","token_index":1,"tensor":{"dtype":"float16","shape":[1,1,4]}})";
    frame.tensor_bytes = {0, 1, 2, 3, 4, 5, 250, 251, 252, 253};

    const auto encoded = encode_frame(frame);
    const auto decoded = decode_frame(encoded);

    assert(decoded.metadata_json == frame.metadata_json);
    assert(decoded.tensor_bytes == frame.tensor_bytes);
}

void test_roundtrip_without_tensor() {
    DliFrame frame;
    frame.metadata_json = R"({"type":"health-like","ok":true})";
    frame.tensor_bytes = {};

    const auto encoded = encode_frame(frame);
    const auto decoded = decode_frame(encoded);

    assert(decoded.metadata_json == frame.metadata_json);
    assert(decoded.tensor_bytes.empty());
}

void test_invalid_magic_is_rejected() {
    DliFrame frame;
    frame.metadata_json = R"({"ok":true})";
    frame.tensor_bytes = {1, 2, 3};

    auto encoded = encode_frame(frame);
    encoded[0] = static_cast<std::uint8_t>('X');

    bool rejected = false;
    try {
        (void)decode_frame(encoded);
    } catch (const ProtocolError&) {
        rejected = true;
    }

    assert(rejected);
}

void test_truncated_frame_is_rejected() {
    DliFrame frame;
    frame.metadata_json = R"({"ok":true})";
    frame.tensor_bytes = {1, 2, 3, 4, 5};

    auto encoded = encode_frame(frame);
    encoded.pop_back();

    bool rejected = false;
    try {
        (void)decode_frame(encoded);
    } catch (const ProtocolError&) {
        rejected = true;
    }

    if (!rejected) {
    std::cerr << "expected ProtocolError was not thrown\n";
    std::exit(1);
}
}

int main() {
    test_roundtrip_with_tensor();
    test_roundtrip_without_tensor();
    test_invalid_magic_is_rejected();
    test_truncated_frame_is_rejected();

    std::cout << "test_protocol: OK\n";
    return 0;
}