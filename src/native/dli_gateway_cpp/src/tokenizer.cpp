#include "dli/gateway/tokenizer.hpp"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
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

void validate_model_path(const std::string& model_path) {
    if (model_path.empty()) {
        throw std::runtime_error("LlamaTokenizer requires a non-empty model path");
    }

    const std::filesystem::path path(model_path);

    if (!std::filesystem::exists(path)) {
        throw std::runtime_error("LlamaTokenizer model path does not exist: " + model_path);
    }

    if (!std::filesystem::is_regular_file(path)) {
        throw std::runtime_error("LlamaTokenizer model path is not a regular file: " + model_path);
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

LlamaTokenizer::LlamaTokenizer(std::string model_path)
    : model_path_(std::move(model_path)) {
    validate_model_path(model_path_);
}

TokenizedPrompt LlamaTokenizer::tokenize(const std::string& prompt) const {
    // Skeleton phase:
    // The model path is validated, but real llama.cpp tokenization is not used yet.
    return fallback_.tokenize(prompt);
}

std::string LlamaTokenizer::backend_name() const {
    return "llama-tokenizer-skeleton";
}

const std::string& LlamaTokenizer::model_path() const {
    return model_path_;
}

std::unique_ptr<Tokenizer> make_stub_tokenizer() {
    return std::make_unique<TokenizerStub>();
}

std::unique_ptr<Tokenizer> make_tokenizer_for_model_path(
    const std::string& model_path
) {
    if (model_path.empty()) {
        return make_stub_tokenizer();
    }

    return std::make_unique<LlamaTokenizer>(model_path);
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