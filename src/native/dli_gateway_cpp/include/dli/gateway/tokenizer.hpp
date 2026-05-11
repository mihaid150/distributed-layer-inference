#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace dli::gateway {

struct TokenizedPrompt {
    std::string prompt;
    std::vector<std::int64_t> token_ids;
};

class TokenizerStub {
public:
    TokenizedPrompt tokenize(const std::string& prompt) const;
};

std::vector<std::uint8_t> int64_tokens_to_little_endian_bytes(
    const std::vector<std::int64_t>& token_ids
);

} // namespace dli::gateway