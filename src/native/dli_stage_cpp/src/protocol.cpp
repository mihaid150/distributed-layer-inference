#include "dli_stage/protocol.hpp"

#include <cstring>
#include <limits>
#include <stdexcept>

namespace dli_stage {

namespace {

void append_u32_le(std::vector<std::uint8_t>& out, std::uint32_t value) {
    out.push_back(static_cast<std::uint8_t>(value & 0xffu));
    out.push_back(static_cast<std::uint8_t>((value >> 8u) & 0xffu));
    out.push_back(static_cast<std::uint8_t>((value >> 16u) & 0xffu));
    out.push_back(static_cast<std::uint8_t>((value >> 24u) & 0xffu));
}

void append_u64_le(std::vector<std::uint8_t>& out, std::uint64_t value) {
    for (int shift = 0; shift < 64; shift += 8) {
        out.push_back(static_cast<std::uint8_t>((value >> static_cast<unsigned>(shift)) & 0xffu));
    }
}

std::uint32_t read_u32_le(const std::uint8_t* data) {
    return static_cast<std::uint32_t>(data[0])
        | (static_cast<std::uint32_t>(data[1]) << 8u)
        | (static_cast<std::uint32_t>(data[2]) << 16u)
        | (static_cast<std::uint32_t>(data[3]) << 24u);
}

std::uint64_t read_u64_le(const std::uint8_t* data) {
    std::uint64_t value = 0;
    for (int i = 0; i < 8; ++i) {
        value |= static_cast<std::uint64_t>(data[i]) << static_cast<unsigned>(i * 8);
    }
    return value;
}

bool magic_matches(const std::uint8_t* data) {
    return std::memcmp(data, DLI2_MAGIC, 4) == 0;
}

} // namespace

ProtocolError::ProtocolError(const std::string& message)
    : std::runtime_error(message) {}

std::vector<std::uint8_t> encode_frame(const DliFrame& frame) {
    if (frame.metadata_json.size() > DLI2_MAX_METADATA_BYTES) {
        throw ProtocolError("DLI2 metadata JSON exceeds maximum allowed size");
    }

    if (frame.tensor_bytes.size() > DLI2_MAX_TENSOR_BYTES) {
        throw ProtocolError("DLI2 tensor payload exceeds maximum allowed size");
    }

    const auto metadata_len = static_cast<std::uint32_t>(frame.metadata_json.size());
    const auto tensor_len = static_cast<std::uint64_t>(frame.tensor_bytes.size());

    if (static_cast<std::uint64_t>(DLI2_HEADER_BYTES) + metadata_len + tensor_len >
        static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
        throw ProtocolError("DLI2 frame is too large for this platform");
    }

    std::vector<std::uint8_t> out;
    out.reserve(DLI2_HEADER_BYTES + frame.metadata_json.size() + frame.tensor_bytes.size());

    out.insert(out.end(), DLI2_MAGIC, DLI2_MAGIC + 4);
    append_u32_le(out, DLI2_VERSION);
    append_u32_le(out, metadata_len);
    append_u64_le(out, tensor_len);

    const auto* metadata_begin = reinterpret_cast<const std::uint8_t*>(frame.metadata_json.data());
    out.insert(out.end(), metadata_begin, metadata_begin + frame.metadata_json.size());
    out.insert(out.end(), frame.tensor_bytes.begin(), frame.tensor_bytes.end());

    return out;
}

DliFrame decode_frame(const std::vector<std::uint8_t>& bytes) {
    return decode_frame(bytes.data(), bytes.size());
}

DliFrame decode_frame(const std::uint8_t* data, std::size_t size) {
    if (data == nullptr && size > 0) {
        throw ProtocolError("DLI2 decode received null data pointer");
    }

    if (size < DLI2_HEADER_BYTES) {
        throw ProtocolError("DLI2 frame is smaller than fixed header");
    }

    if (!magic_matches(data)) {
        throw ProtocolError("DLI2 frame has invalid magic");
    }

    const std::uint32_t version = read_u32_le(data + 4);
    if (version != DLI2_VERSION) {
        throw ProtocolError("DLI2 frame has unsupported version");
    }

    const std::uint32_t metadata_len = read_u32_le(data + 8);
    const std::uint64_t tensor_len = read_u64_le(data + 12);

    if (metadata_len > DLI2_MAX_METADATA_BYTES) {
        throw ProtocolError("DLI2 metadata length exceeds maximum allowed size");
    }

    if (tensor_len > DLI2_MAX_TENSOR_BYTES) {
        throw ProtocolError("DLI2 tensor length exceeds maximum allowed size");
    }

    const std::uint64_t expected_size =
        static_cast<std::uint64_t>(DLI2_HEADER_BYTES) + metadata_len + tensor_len;

    if (expected_size != static_cast<std::uint64_t>(size)) {
        throw ProtocolError("DLI2 frame size does not match metadata/tensor lengths");
    }

    const std::uint8_t* metadata_start = data + DLI2_HEADER_BYTES;
    const std::uint8_t* tensor_start = metadata_start + metadata_len;

    DliFrame frame;
    frame.metadata_json.assign(
        reinterpret_cast<const char*>(metadata_start),
        reinterpret_cast<const char*>(metadata_start + metadata_len)
    );
    frame.tensor_bytes.assign(tensor_start, tensor_start + tensor_len);

    return frame;
}

} // namespace dli_stage