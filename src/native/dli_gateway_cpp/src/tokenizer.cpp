#include "dli/gateway/tokenizer.hpp"

#include <cstdint>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace dli::gateway {

namespace {

void append_i64_le(std::vector<std::uint8_t>& out, std::int64_t value) {
    const auto unsigned_value = static_cast<std::uint64_t>(value);

    for (int shift = 0; shift < 64; shift += 8) {
        out.push_back(
            static_cast<std::uint8_t>(
                (unsigned_value >> static_cast<unsigned>(shift)) & 0xffu
            )
        );
    }
}

} // namespace

TokenizedPrompt TokenizerStub::tokenize(const std::string& prompt) const {
    TokenizedPrompt result;
    result.prompt = prompt;

    std::istringstream stream(prompt);
    std::string word;

    // Fake BOS token.
    result.token_ids.push_back(1);

    std::int64_t next_id = 100;

    while (stream >> word) {
        result.token_ids.push_back(next_id++);
    }

    // Ensure non-empty prompt representation.
    if (result.token_ids.size() == 1) {
        result.token_ids.push_back(100);
    }

    return result;
}

std::string TokenizerStub::backend_name() const {
    return "stub";
}

std::unique_ptr<Tokenizer> make_stub_tokenizer() {
    return std::make_unique<TokenizerStub>();
}

std::vector<std::uint8_t> int64_tokens_to_little_endian_bytes(
    const std::vector<std::int64_t>& token_ids
) {
    std::vector<std::uint8_t> out;
    out.reserve(token_ids.size() * sizeof(std::int64_t));

    for (const std::int64_t token_id : token_ids) {
        append_i64_le(out, token_id);
    }

    return out;
}

} // namespace dli::gateway