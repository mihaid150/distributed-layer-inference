#include "dli/gateway/tokenizer.hpp"

#include "llama.h"

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

std::vector<llama_token> tokenize_with_vocab(
    const llama_vocab* vocab,
    const std::string& prompt
) {
    if (vocab == nullptr) {
        throw std::runtime_error("cannot tokenize with null llama vocab");
    }

    const bool add_special = true;
    const bool parse_special = false;

    int token_count = llama_tokenize(
        vocab,
        prompt.c_str(),
        static_cast<int32_t>(prompt.size()),
        nullptr,
        0,
        add_special,
        parse_special
    );

    if (token_count == 0) {
        return {};
    }

    if (token_count < 0) {
        token_count = -token_count;
    }

    std::vector<llama_token> tokens(static_cast<std::size_t>(token_count));

    const int actual_count = llama_tokenize(
        vocab,
        prompt.c_str(),
        static_cast<int32_t>(prompt.size()),
        tokens.data(),
        static_cast<int32_t>(tokens.size()),
        add_special,
        parse_special
    );

    if (actual_count < 0) {
        throw std::runtime_error("llama_tokenize reported insufficient token buffer unexpectedly");
    }

    tokens.resize(static_cast<std::size_t>(actual_count));
    return tokens;
}

std::string llama_token_to_string(
    const llama_vocab* vocab,
    llama_token token
) {
    if (vocab == nullptr) {
        throw std::runtime_error("cannot detokenize with null llama vocab");
    }

    const bool special = true;
    int required = llama_token_to_piece(
        vocab,
        token,
        nullptr,
        0,
        0,
        special
    );

    if (required == 0) {
        return "";
    }

    if (required < 0) {
        required = -required;
    }

    std::string piece(static_cast<std::size_t>(required), '\0');

    const int actual = llama_token_to_piece(
        vocab,
        token,
        piece.data(),
        static_cast<int32_t>(piece.size()),
        0,
        special
    );

    if (actual < 0) {
        throw std::runtime_error("llama_token_to_piece reported insufficient buffer unexpectedly");
    }

    piece.resize(static_cast<std::size_t>(actual));
    return piece;
}

} // namespace

TokenizedPrompt TokenizerStub::tokenize(const std::string& prompt) const {
    TokenizedPrompt result;
    result.prompt = prompt;

    std::istringstream stream(prompt);
    std::string word;

    result.token_ids.push_back(1);

    std::int64_t next_id = 100;

    while (stream >> word) {
        result.token_ids.push_back(next_id++);
    }

    if (result.token_ids.size() == 1) {
        result.token_ids.push_back(100);
    }

    return result;
}

std::string TokenizerStub::detokenize(const std::vector<int>& token_ids) const {
    std::ostringstream out;

    for (std::size_t i = 0; i < token_ids.size(); ++i) {
        if (i > 0) {
            out << " ";
        }
        out << "<tok_" << token_ids[i] << ">";
    }

    return out.str();
}

std::string TokenizerStub::backend_name() const {
    return "stub";
}

LlamaTokenizer::LlamaTokenizer(std::string model_path)
    : model_path_(std::move(model_path)) {
    validate_model_path(model_path_);

    llama_model_params model_params = llama_model_default_params();

    model_params.vocab_only = true;
    model_params.n_gpu_layers = 0;

    model_ = llama_model_load_from_file(model_path_.c_str(), model_params);
    if (model_ == nullptr) {
        throw std::runtime_error("failed to load GGUF model for tokenizer: " + model_path_);
    }

    vocab_ = llama_model_get_vocab(model_);
    if (vocab_ == nullptr) {
        llama_model_free(model_);
        model_ = nullptr;
        throw std::runtime_error("failed to get llama vocab from model: " + model_path_);
    }
}

LlamaTokenizer::~LlamaTokenizer() {
    if (model_ != nullptr) {
        llama_model_free(model_);
        model_ = nullptr;
        vocab_ = nullptr;
    }
}

TokenizedPrompt LlamaTokenizer::tokenize(const std::string& prompt) const {
    TokenizedPrompt result;
    result.prompt = prompt;

    const std::vector<llama_token> llama_tokens =
        tokenize_with_vocab(vocab_, prompt);

    result.token_ids.reserve(llama_tokens.size());

    for (const llama_token token : llama_tokens) {
        result.token_ids.push_back(static_cast<std::int64_t>(token));
    }

    if (result.token_ids.empty()) {
        throw std::runtime_error("llama tokenizer returned no tokens");
    }

    return result;
}

std::string LlamaTokenizer::detokenize(const std::vector<int>& token_ids) const {
    std::string out;

    for (const int token_id : token_ids) {
        out += llama_token_to_string(vocab_, static_cast<llama_token>(token_id));
    }

    return out;
}

std::string LlamaTokenizer::backend_name() const {
    return "llama.cpp";
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