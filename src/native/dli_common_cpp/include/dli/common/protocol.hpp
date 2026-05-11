#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace dli::common {

constexpr char DLI2_MAGIC[4] = {'D', 'L', 'I', '2'};
constexpr std::uint32_t DLI2_VERSION = 2;
constexpr std::size_t DLI2_HEADER_BYTES = 20;

constexpr std::uint32_t DLI2_MAX_METADATA_BYTES = 16u * 1024u * 1024u; // 16 MiB
constexpr std::uint64_t DLI2_MAX_TENSOR_BYTES = 8ull * 1024ull * 1024ull * 1024ull; // 8 GiB

struct DliFrame {
    std::string metadata_json;
    std::vector<std::uint8_t> tensor_bytes;
};

class ProtocolError : public std::runtime_error {
public:
    explicit ProtocolError(const std::string& message);
};

std::vector<std::uint8_t> encode_frame(const DliFrame& frame);

DliFrame decode_frame(const std::vector<std::uint8_t>& bytes);

DliFrame decode_frame(const std::uint8_t* data, std::size_t size);

} // namespace dli::common