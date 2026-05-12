#pragma once

#include <cctype>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace dli::tools {

struct PartitionManifest {
    std::string partition_id;
    int stage_id = 0;
    std::string service_name;
    std::string physical_node;
    std::string partition_file;
    std::string next_stage_url;

    bool owns_embedding = false;
    bool owns_norm = false;
    bool owns_lm_head = false;
    std::vector<int> layers;

    std::vector<std::string> tensor_names;
};

inline std::string read_text_file(const std::filesystem::path& path) {
    std::ifstream input(path);
    if (!input.is_open()) {
        throw std::runtime_error("failed to open file: " + path.string());
    }

    std::ostringstream out;
    out << input.rdbuf();
    return out.str();
}

inline void skip_ws(const std::string& text, std::size_t& pos) {
    while (pos < text.size() && std::isspace(static_cast<unsigned char>(text[pos])) != 0) {
        ++pos;
    }
}

inline std::size_t find_key(const std::string& json, const std::string& key) {
    const std::string quoted_key = "\"" + key + "\"";
    const std::size_t key_pos = json.find(quoted_key);
    if (key_pos == std::string::npos) {
        throw std::runtime_error("manifest missing key: " + key);
    }

    const std::size_t colon = json.find(':', key_pos + quoted_key.size());
    if (colon == std::string::npos) {
        throw std::runtime_error("manifest key has no value: " + key);
    }

    std::size_t value_pos = colon + 1;
    skip_ws(json, value_pos);
    return value_pos;
}

inline std::string parse_json_string_at(const std::string& json, std::size_t& pos) {
    skip_ws(json, pos);

    if (pos >= json.size() || json[pos] != '"') {
        throw std::runtime_error("expected JSON string");
    }

    ++pos;
    std::string value;

    while (pos < json.size()) {
        const char c = json[pos++];

        if (c == '"') {
            return value;
        }

        if (c == '\\') {
            if (pos >= json.size()) {
                throw std::runtime_error("unterminated JSON escape sequence");
            }

            const char esc = json[pos++];
            switch (esc) {
                case '"': value.push_back('"'); break;
                case '\\': value.push_back('\\'); break;
                case '/': value.push_back('/'); break;
                case 'b': value.push_back('\b'); break;
                case 'f': value.push_back('\f'); break;
                case 'n': value.push_back('\n'); break;
                case 'r': value.push_back('\r'); break;
                case 't': value.push_back('\t'); break;
                default:
                    throw std::runtime_error("unsupported JSON escape sequence");
            }

            continue;
        }

        value.push_back(c);
    }

    throw std::runtime_error("unterminated JSON string");
}

inline std::string parse_json_string_value(const std::string& json, const std::string& key) {
    std::size_t pos = find_key(json, key);
    return parse_json_string_at(json, pos);
}

inline int parse_json_int_value(const std::string& json, const std::string& key) {
    std::size_t pos = find_key(json, key);
    skip_ws(json, pos);

    const std::size_t begin = pos;

    if (pos < json.size() && json[pos] == '-') {
        ++pos;
    }

    while (pos < json.size() && std::isdigit(static_cast<unsigned char>(json[pos])) != 0) {
        ++pos;
    }

    if (begin == pos) {
        throw std::runtime_error("expected integer value for key: " + key);
    }

    return std::stoi(json.substr(begin, pos - begin));
}

inline bool parse_json_bool_value(const std::string& json, const std::string& key) {
    std::size_t pos = find_key(json, key);
    skip_ws(json, pos);

    if (json.compare(pos, 4, "true") == 0) {
        return true;
    }

    if (json.compare(pos, 5, "false") == 0) {
        return false;
    }

    throw std::runtime_error("expected boolean value for key: " + key);
}

inline std::vector<std::string> parse_json_string_array_value(
    const std::string& json,
    const std::string& key
) {
    std::size_t pos = find_key(json, key);
    skip_ws(json, pos);

    if (pos >= json.size() || json[pos] != '[') {
        throw std::runtime_error("expected string array for key: " + key);
    }

    ++pos;

    std::vector<std::string> values;

    while (true) {
        skip_ws(json, pos);

        if (pos >= json.size()) {
            throw std::runtime_error("unterminated string array for key: " + key);
        }

        if (json[pos] == ']') {
            ++pos;
            return values;
        }

        values.push_back(parse_json_string_at(json, pos));

        skip_ws(json, pos);

        if (pos < json.size() && json[pos] == ',') {
            ++pos;
            continue;
        }

        if (pos < json.size() && json[pos] == ']') {
            ++pos;
            return values;
        }

        throw std::runtime_error("expected ',' or ']' in string array for key: " + key);
    }
}

inline std::vector<int> parse_json_int_array_value(
    const std::string& json,
    const std::string& key
) {
    std::size_t pos = find_key(json, key);
    skip_ws(json, pos);

    if (pos >= json.size() || json[pos] != '[') {
        throw std::runtime_error("expected integer array for key: " + key);
    }

    ++pos;

    std::vector<int> values;

    while (true) {
        skip_ws(json, pos);

        if (pos >= json.size()) {
            throw std::runtime_error("unterminated integer array for key: " + key);
        }

        if (json[pos] == ']') {
            ++pos;
            return values;
        }

        const std::size_t begin = pos;

        if (pos < json.size() && json[pos] == '-') {
            ++pos;
        }

        while (pos < json.size() && std::isdigit(static_cast<unsigned char>(json[pos])) != 0) {
            ++pos;
        }

        if (begin == pos) {
            throw std::runtime_error("expected integer item in array for key: " + key);
        }

        values.push_back(std::stoi(json.substr(begin, pos - begin)));

        skip_ws(json, pos);

        if (pos < json.size() && json[pos] == ',') {
            ++pos;
            continue;
        }

        if (pos < json.size() && json[pos] == ']') {
            ++pos;
            return values;
        }

        throw std::runtime_error("expected ',' or ']' in integer array for key: " + key);
    }
}

inline PartitionManifest parse_partition_manifest_text(const std::string& json) {
    PartitionManifest manifest;

    manifest.partition_id = parse_json_string_value(json, "partition_id");
    manifest.stage_id = parse_json_int_value(json, "stage_id");
    manifest.service_name = parse_json_string_value(json, "service_name");
    manifest.physical_node = parse_json_string_value(json, "physical_node");
    manifest.partition_file = parse_json_string_value(json, "partition_file");
    manifest.next_stage_url = parse_json_string_value(json, "next_stage_url");

    manifest.owns_embedding = parse_json_bool_value(json, "embedding");
    manifest.owns_norm = parse_json_bool_value(json, "norm");
    manifest.owns_lm_head = parse_json_bool_value(json, "lm_head");
    manifest.layers = parse_json_int_array_value(json, "layers");

    manifest.tensor_names = parse_json_string_array_value(json, "tensor_names");

    if (manifest.partition_id.empty()) {
        throw std::runtime_error("manifest partition_id is empty");
    }

    if (manifest.stage_id <= 0) {
        throw std::runtime_error("manifest stage_id must be positive");
    }

    return manifest;
}

inline PartitionManifest load_partition_manifest(const std::filesystem::path& path) {
    return parse_partition_manifest_text(read_text_file(path));
}

} // namespace dli::tools
