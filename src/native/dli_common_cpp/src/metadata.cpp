#include "dli/common/metadata.hpp"

#include <optional>
#include <regex>
#include <stdexcept>
#include <string>

namespace dli::common {

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
    const std::regex shape_pattern(R"dli("(?:shape|sh)"\s*:\s*\[([^\]]*)\])dli");
    const std::regex integer_pattern(R"dli(-?\d+)dli");

    std::smatch shape_match;
    if (!std::regex_search(json, shape_match, shape_pattern)) {
        return {};
    }

    const std::string shape_body = shape_match[1].str();

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



std::optional<std::string> extract_string_alias(
    const std::string& json,
    const std::string& long_key,
    const std::string& short_key
) {
    if (const auto value = extract_string_field(json, long_key)) {
        return value;
    }
    return extract_string_field(json, short_key);
}

std::optional<int> extract_int_alias(
    const std::string& json,
    const std::string& long_key,
    const std::string& short_key
) {
    if (const auto value = extract_int_field(json, long_key)) {
        return value;
    }
    return extract_int_field(json, short_key);
}

std::optional<bool> extract_bool_alias(
    const std::string& json,
    const std::string& long_key,
    const std::string& short_key
) {
    if (const auto value = extract_bool_field(json, long_key)) {
        return value;
    }
    return extract_bool_field(json, short_key);
}

} // namespace

ParsedRequestMetadata parse_request_metadata(const std::string& metadata_json) {
    ParsedRequestMetadata parsed;

    if (const auto value = extract_string_alias(metadata_json, "request_id", "rid")) {
        parsed.request_id = *value;
    }

    if (const auto value = extract_int_alias(metadata_json, "token_index", "ti")) {
        parsed.token_index = *value;
    }

    if (const auto value = extract_int_alias(metadata_json, "next_token_id", "nt")) {
        parsed.has_next_token_id = true;
        parsed.next_token_id = *value;
    }

    if (const auto value = extract_string_alias(metadata_json, "generation_mode", "gm")) {
        parsed.generation_mode = *value;
    }

    if (const auto value = extract_string_alias(metadata_json, "activation_precision", "ap")) {
    parsed.activation_precision = *value;
}

    if (const auto value = extract_string_alias(metadata_json, "transport_mode", "tm")) {
        parsed.transport_mode = *value;
    }

    if (const auto value = extract_string_field(metadata_json, "rebalance_profile")) {
        parsed.rebalance_profile = *value;
    }

    if (const auto value = extract_bool_alias(metadata_json, "persistent_sessions_enabled", "ps")) {
        parsed.persistent_sessions_enabled = *value;
    }

    if (const auto value = extract_bool_field(metadata_json, "topology_aware_routing")) {
        parsed.topology_aware_routing = *value;
    }

    if (const auto value = extract_string_field(metadata_json, "metadata_level")) {
        parsed.metadata_level = *value;
    } else if (const auto value = extract_string_field(metadata_json, "ml")) {
        parsed.metadata_level = *value;
    }

    if (const auto value = extract_bool_alias(metadata_json, "native_stage_chaining_enabled", "nsc")) {
        parsed.native_stage_chaining_enabled = *value;
    }

    if (const auto value = extract_string_alias(metadata_json, "dtype", "dt")) {
        parsed.tensor.dtype = *value;
    }

    if (const auto value = extract_string_alias(metadata_json, "byte_order", "bo")) {
        parsed.tensor.byte_order = *value;
    }

    parsed.tensor.shape = extract_shape(metadata_json);
    parsed.stage_input_token_count = infer_sequence_length_from_shape(parsed.tensor.shape);

    if (const auto value = extract_bool_alias(metadata_json, "kv_cache_enabled", "kv")) {
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

    if (const auto value = extract_double_field(metadata_json, "seed")) {
        parsed.has_seed = true;
        parsed.seed = static_cast<long long>(*value);
    }

    return parsed;
}

} // namespace dli::common