#include "dli_stage/metadata.hpp"

#include <optional>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>

namespace dli_stage {

namespace {

std::string unescape_json_string(const std::string& value) {
    std::string out;
    out.reserve(value.size());

    for (std::size_t i = 0; i < value.size(); ++i) {
        const char c = value[i];

        if (c != '\\' || i + 1 >= value.size()) {
            out.push_back(c);
            continue;
        }

        const char escaped = value[++i];
        switch (escaped) {
            case '"':
                out.push_back('"');
                break;
            case '\\':
                out.push_back('\\');
                break;
            case '/':
                out.push_back('/');
                break;
            case 'b':
                out.push_back('\b');
                break;
            case 'f':
                out.push_back('\f');
                break;
            case 'n':
                out.push_back('\n');
                break;
            case 'r':
                out.push_back('\r');
                break;
            case 't':
                out.push_back('\t');
                break;
            default:
                // Keep unsupported escape sequences readable rather than failing.
                out.push_back('\\');
                out.push_back(escaped);
                break;
        }
    }

    return out;
}

std::optional<std::string> extract_string_field(
    const std::string& json,
    const std::string& key
) {
    const std::regex pattern(
        "\"" + key + R"dli("\s*:\s*"((?:\\.|[^"\\])*)")dli"
    );

    std::smatch match;
    if (!std::regex_search(json, match, pattern)) {
        return std::nullopt;
    }

    return unescape_json_string(match[1].str());
}

std::optional<int> extract_int_field(
    const std::string& json,
    const std::string& key
) {
    const std::regex pattern(
        "\"" + key + R"dli("\s*:\s*(-?\d+))dli"
    );

    std::smatch match;
    if (!std::regex_search(json, match, pattern)) {
        return std::nullopt;
    }

    try {
        return std::stoi(match[1].str());
    } catch (const std::exception&) {
        throw std::runtime_error("invalid integer metadata field: " + key);
    }
}

std::optional<double> extract_double_field(
    const std::string& json,
    const std::string& key
) {
    const std::regex pattern(
        "\"" + key + R"dli("\s*:\s*(-?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?))dli"
    );

    std::smatch match;
    if (!std::regex_search(json, match, pattern)) {
        return std::nullopt;
    }

    try {
        return std::stod(match[1].str());
    } catch (const std::exception&) {
        throw std::runtime_error("invalid floating-point metadata field: " + key);
    }
}

std::optional<bool> extract_bool_field(
    const std::string& json,
    const std::string& key
) {
    const std::regex pattern(
        "\"" + key + R"dli("\s*:\s*(true|false))dli"
    );

    std::smatch match;
    if (!std::regex_search(json, match, pattern)) {
        return std::nullopt;
    }

    return match[1].str() == "true";
}

std::vector<std::int64_t> extract_shape(const std::string& json) {
    const std::regex shape_pattern(R"dli("shape"\s*:\s*\[([^\]]*)\])dli");

    std::smatch shape_match;
    if (!std::regex_search(json, shape_match, shape_pattern)) {
        return {};
    }

    const std::string shape_body = shape_match[1].str();
    const std::regex integer_pattern(R"(-?\d+)");

    std::vector<std::int64_t> shape;

    auto begin = std::sregex_iterator(
        shape_body.begin(),
        shape_body.end(),
        integer_pattern
    );
    auto end = std::sregex_iterator();

    for (auto it = begin; it != end; ++it) {
        try {
            shape.push_back(std::stoll((*it).str()));
        } catch (const std::exception&) {
            throw std::runtime_error("invalid tensor shape dimension");
        }
    }

    return shape;
}

} // namespace

int infer_sequence_length_from_shape(const std::vector<std::int64_t>& shape) {
    if (shape.empty()) {
        return 0;
    }

    // token ids: [batch, seq]
    // hidden states: [batch, seq, hidden]
    if (shape.size() >= 2) {
        const auto seq = shape[1];
        return seq > 0 ? static_cast<int>(seq) : 0;
    }

    // fallback: [seq]
    const auto seq = shape[0];
    return seq > 0 ? static_cast<int>(seq) : 0;
}

ParsedRequestMetadata parse_request_metadata(const std::string& metadata_json) {
    ParsedRequestMetadata parsed;

    if (const auto value = extract_string_field(metadata_json, "request_id")) {
        parsed.request_id = *value;
    }

    if (const auto value = extract_int_field(metadata_json, "token_index")) {
        parsed.token_index = *value;
    }

    if (const auto value = extract_string_field(metadata_json, "generation_mode")) {
        parsed.generation_mode = *value;
    }

    if (const auto value = extract_string_field(metadata_json, "dtype")) {
        parsed.tensor.dtype = *value;
    }

    if (const auto value = extract_string_field(metadata_json, "byte_order")) {
        parsed.tensor.byte_order = *value;
    }

    parsed.tensor.shape = extract_shape(metadata_json);
    parsed.stage_input_token_count = infer_sequence_length_from_shape(parsed.tensor.shape);

    if (const auto value = extract_bool_field(metadata_json, "kv_cache_enabled")) {
        parsed.kv_cache_enabled = *value;
    }

    if (const auto value = extract_double_field(metadata_json, "temperature")) {
        parsed.has_temperature = true;
        parsed.temperature = *value;
    }

    if (const auto value = extract_int_field(metadata_json, "top_k")) {
        parsed.has_top_k = true;
        parsed.top_k = *value;
    }

    if (const auto value = extract_double_field(metadata_json, "top_p")) {
        parsed.has_top_p = true;
        parsed.top_p = *value;
    }

    return parsed;
}

} // namespace dli_stage