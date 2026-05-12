#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace dli::gateway {

struct TokenizedPrompt {
    std::string prompt;
    std::vector<std::int64_t> token_ids;
};

class Tokenizer {
public:
    virtual ~Tokenizer() = default;

    virtual TokenizedPrompt tokenize(const std::string& prompt) const = 0;

    virtual std::string backend_name() const = 0;
};

class TokenizerStub final : public Tokenizer {
public:
    TokenizedPrompt tokenize(const std::string& prompt) const override;

    std::string backend_name() const override;
};

class LlamaTokenizer final : public Tokenizer {
public:
    explicit LlamaTokenizer(std::string model_path);

    TokenizedPrompt tokenize(const std::string& prompt) const override;

    std::string backend_name() const override;

    const std::string& model_path() const;

private:
    std::string model_path_;
    TokenizerStub fallback_;
};

std::unique_ptr<Tokenizer> make_stub_tokenizer();

std::unique_ptr<Tokenizer> make_tokenizer_for_model_path(
    const std::string& model_path
);

std::vector<std::uint8_t> int64_tokens_to_little_endian_bytes(
    const std::vector<std::int64_t>& token_ids
);

} // namespace dli::gateway