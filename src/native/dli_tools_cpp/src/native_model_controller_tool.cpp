#include "dli/common/json_escape.hpp"

#include "gguf.h"

#include <algorithm>
#include <cerrno>
#include <cctype>
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

namespace {

struct CliOptions {
    std::string command;
    std::string model_id;
    std::string model_name;
    std::string model_slug;
    std::string local_model_path;
    std::string source_repo;
    std::string source_revision = "main";
    std::string source_file;
    std::string artifact_repo;
    std::string artifact_revision = "main";
    bool artifact_private = false;
    std::string full_gguf_file;
    int num_layers = 0;
    int num_stages = 4;
    std::string physical_nodes;
    std::string gateway_service_name = "inference-native-gateway";
    std::string stage_service_prefix = "inference-native-stage";
    std::string container_model_root = "/app/models";
    std::string work_dir = "/work";
    bool no_upload = false;
    std::string partition_manifest_bin = "/usr/local/bin/dli-partition-manifest";
    std::string shard_writer_bin = "/usr/local/bin/dli-gguf-shard-writer";
    std::string shard_validate_bin = "/usr/local/bin/dli-gguf-shard-validate";
};

struct GgufMetadata {
    std::string architecture;
    std::string name;
    int block_count = -1;
};

struct CommandResult {
    int exit_code = 0;
    std::string output;
};

void print_usage(const char* program_name) {
    std::cerr
        << "Usage:\n"
        << "  " << program_name << " prepare --model-id <id> --source-repo <repo> --source-file <file> --artifact-repo <repo>\n\n"
        << "Options:\n"
        << "  --model-id <id>                 Catalog model id.\n"
        << "  --model-name <name>             Model name to write into stage_map.yaml.\n"
        << "  --model-slug <slug>             Artifact subdirectory slug. Default: slugified model id.\n"
        << "  --local-model-path <path>       Use an already-mounted full GGUF instead of downloading.\n"
        << "  --source-repo <repo>            Hugging Face source repo for full GGUF.\n"
        << "  --source-revision <rev>         Source revision. Default: main.\n"
        << "  --source-file <path>            Source GGUF path in source repo.\n"
        << "  --artifact-repo <repo>          Hugging Face repo receiving prepared artifacts.\n"
        << "  --artifact-revision <rev>       Artifact revision/branch. Default: main.\n"
        << "  --artifact-private              Create artifact repo as private when possible.\n"
        << "  --full-gguf-file <name>         Full GGUF filename in artifact slug directory.\n"
        << "  --num-layers <n>                Override layer count from GGUF metadata.\n"
        << "  --num-stages <n>                Number of native stages. Default: 4.\n"
        << "  --physical-nodes <csv>          Node names written into stage_map.yaml.\n"
        << "  --gateway-service-name <name>   Gateway service name. Default: inference-native-gateway.\n"
        << "  --stage-service-prefix <prefix> Stage service prefix. Default: inference-native-stage.\n"
        << "  --container-model-root <path>   Inference container model root. Default: /app/models.\n"
        << "  --work-dir <path>               Controller work directory. Default: /work.\n"
        << "  --no-upload                     Prepare locally and skip HF upload.\n"
        << "  --partition-manifest-bin <path> dli-partition-manifest path.\n"
        << "  --shard-writer-bin <path>       dli-gguf-shard-writer path.\n"
        << "  --shard-validate-bin <path>     dli-gguf-shard-validate path.\n";
}

std::string require_value(int& index, int argc, char** argv, const std::string& flag) {
    if (index + 1 >= argc) {
        throw std::runtime_error(flag + " requires a value");
    }
    return argv[++index];
}

CliOptions parse_args(int argc, char** argv) {
    if (argc < 2) {
        print_usage(argv[0]);
        throw std::runtime_error("missing command");
    }

    CliOptions options;
    options.command = argv[1];
    if (options.command == "--help" || options.command == "-h") {
        print_usage(argv[0]);
        std::exit(0);
    }
    if (options.command != "prepare") {
        throw std::runtime_error("unknown command: " + options.command);
    }

    for (int i = 2; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            std::exit(0);
        }
        if (arg == "--model-id") {
            options.model_id = require_value(i, argc, argv, arg);
        } else if (arg == "--model-name") {
            options.model_name = require_value(i, argc, argv, arg);
        } else if (arg == "--model-slug") {
            options.model_slug = require_value(i, argc, argv, arg);
        } else if (arg == "--local-model-path") {
            options.local_model_path = require_value(i, argc, argv, arg);
        } else if (arg == "--source-repo") {
            options.source_repo = require_value(i, argc, argv, arg);
        } else if (arg == "--source-revision") {
            options.source_revision = require_value(i, argc, argv, arg);
        } else if (arg == "--source-file") {
            options.source_file = require_value(i, argc, argv, arg);
        } else if (arg == "--artifact-repo") {
            options.artifact_repo = require_value(i, argc, argv, arg);
        } else if (arg == "--artifact-revision") {
            options.artifact_revision = require_value(i, argc, argv, arg);
        } else if (arg == "--artifact-private") {
            options.artifact_private = true;
        } else if (arg == "--full-gguf-file") {
            options.full_gguf_file = require_value(i, argc, argv, arg);
        } else if (arg == "--num-layers") {
            options.num_layers = std::stoi(require_value(i, argc, argv, arg));
        } else if (arg == "--num-stages") {
            options.num_stages = std::stoi(require_value(i, argc, argv, arg));
        } else if (arg == "--physical-nodes") {
            options.physical_nodes = require_value(i, argc, argv, arg);
        } else if (arg == "--gateway-service-name") {
            options.gateway_service_name = require_value(i, argc, argv, arg);
        } else if (arg == "--stage-service-prefix") {
            options.stage_service_prefix = require_value(i, argc, argv, arg);
        } else if (arg == "--container-model-root") {
            options.container_model_root = require_value(i, argc, argv, arg);
        } else if (arg == "--work-dir") {
            options.work_dir = require_value(i, argc, argv, arg);
        } else if (arg == "--no-upload") {
            options.no_upload = true;
        } else if (arg == "--partition-manifest-bin") {
            options.partition_manifest_bin = require_value(i, argc, argv, arg);
        } else if (arg == "--shard-writer-bin") {
            options.shard_writer_bin = require_value(i, argc, argv, arg);
        } else if (arg == "--shard-validate-bin") {
            options.shard_validate_bin = require_value(i, argc, argv, arg);
        } else {
            throw std::runtime_error("unknown argument: " + arg);
        }
    }

    if (options.model_id.empty()) {
        throw std::runtime_error("--model-id is required");
    }
    if (options.num_stages <= 0) {
        throw std::runtime_error("--num-stages must be greater than zero");
    }
    if (options.local_model_path.empty() && (options.source_repo.empty() || options.source_file.empty())) {
        throw std::runtime_error("--source-repo and --source-file are required unless --local-model-path is set");
    }
    if (!options.no_upload && options.artifact_repo.empty()) {
        throw std::runtime_error("--artifact-repo is required unless --no-upload is set");
    }
    if (options.model_name.empty()) {
        options.model_name = options.model_id;
    }
    if (options.artifact_revision.empty()) {
        options.artifact_revision = "main";
    }
    if (options.source_revision.empty()) {
        options.source_revision = "main";
    }

    return options;
}

std::string trim_copy(const std::string& value) {
    std::size_t begin = 0;
    while (begin < value.size() && std::isspace(static_cast<unsigned char>(value[begin]))) {
        ++begin;
    }

    std::size_t end = value.size();
    while (end > begin && std::isspace(static_cast<unsigned char>(value[end - 1]))) {
        --end;
    }
    return value.substr(begin, end - begin);
}

std::string slugify(std::string value) {
    std::string out;
    bool last_dash = false;
    for (char ch : value) {
        const unsigned char c = static_cast<unsigned char>(ch);
        const char lower = static_cast<char>(std::tolower(c));
        const bool allowed =
            (lower >= 'a' && lower <= 'z') ||
            (lower >= '0' && lower <= '9') ||
            lower == '.' ||
            lower == '_' ||
            lower == '-';
        if (allowed) {
            out.push_back(lower);
            last_dash = lower == '-';
        } else if (!last_dash) {
            out.push_back('-');
            last_dash = true;
        }
    }
    while (!out.empty() && (out.front() == '-' || out.front() == '.' || out.front() == '_')) {
        out.erase(out.begin());
    }
    while (!out.empty() && (out.back() == '-' || out.back() == '.' || out.back() == '_')) {
        out.pop_back();
    }
    return out.empty() ? "model" : out;
}

std::vector<std::vector<int>> split_layers(int num_layers, int num_stages) {
    if (num_layers <= 0) {
        throw std::runtime_error("num_layers must be greater than zero");
    }
    if (num_stages <= 0) {
        throw std::runtime_error("num_stages must be greater than zero");
    }
    if (num_stages > num_layers) {
        throw std::runtime_error("num_stages cannot be greater than num_layers");
    }

    const int base = num_layers / num_stages;
    const int remainder = num_layers % num_stages;
    std::vector<std::vector<int>> result;
    result.reserve(static_cast<std::size_t>(num_stages));
    int cursor = 0;
    for (int index = 0; index < num_stages; ++index) {
        const int size = base + (index < remainder ? 1 : 0);
        std::vector<int> layers;
        layers.reserve(static_cast<std::size_t>(size));
        for (int layer = cursor; layer < cursor + size; ++layer) {
            layers.push_back(layer);
        }
        cursor += size;
        result.push_back(std::move(layers));
    }
    return result;
}

std::vector<std::string> parse_physical_nodes(const std::string& raw_nodes, int num_stages) {
    std::vector<std::string> values;
    std::stringstream stream(raw_nodes);
    std::string item;
    while (std::getline(stream, item, ',')) {
        item = trim_copy(item);
        if (!item.empty()) {
            values.push_back(item);
        }
    }
    while (static_cast<int>(values.size()) < num_stages) {
        values.push_back("dli-worker-" + std::to_string(values.size() + 1));
    }
    values.resize(static_cast<std::size_t>(num_stages));
    return values;
}

std::string yaml_int_list(const std::vector<int>& values, int indent) {
    std::ostringstream out;
    const std::string prefix(static_cast<std::size_t>(indent), ' ');
    for (const int value : values) {
        out << prefix << "- " << value << "\n";
    }
    std::string text = out.str();
    if (!text.empty() && text.back() == '\n') {
        text.pop_back();
    }
    return text;
}

std::string strip_trailing_slashes(std::string value) {
    while (value.size() > 1 && value.back() == '/') {
        value.pop_back();
    }
    return value;
}

std::string generate_stage_map(
    const std::string& model_name,
    const std::string& model_slug,
    const std::string& full_gguf_file,
    int num_layers,
    int num_stages,
    const std::string& gateway_service_name,
    const std::string& stage_service_prefix,
    const std::vector<std::string>& physical_nodes,
    const std::string& container_model_root
) {
    const auto ranges = split_layers(num_layers, num_stages);
    const std::string model_root = strip_trailing_slashes(container_model_root);
    std::ostringstream out;
    out
        << "model_name: " << model_name << "\n"
        << "model_slug: " << model_slug << "\n"
        << "model_path: " << model_root << "/" << model_slug << "/" << full_gguf_file << "\n"
        << "num_layers: " << num_layers << "\n\n"
        << "inference_gateway:\n"
        << "  service_name: " << gateway_service_name << "\n"
        << "  port: 8000\n"
        << "  first_stage_url: http://" << stage_service_prefix << "-1:8000/forward-binary\n\n"
        << "inference_stages:\n";

    for (int index = 0; index < num_stages; ++index) {
        const int stage_id = index + 1;
        out
            << "- stage_id: " << stage_id << "\n"
            << "  partition_id: partition-" << stage_id << "\n"
            << "  service_name: " << stage_service_prefix << "-" << stage_id << "\n"
            << "  physical_node: " << physical_nodes[static_cast<std::size_t>(index)] << "\n"
            << "  partition_file: " << model_root << "/" << model_slug << "/stage_" << stage_id << ".pt\n"
            << "  native_partition_file: " << model_root << "/" << model_slug << "/partition-" << stage_id << ".dli.gguf\n"
            << "  backend: llama\n"
            << "  components:\n"
            << "    embedding: " << (stage_id == 1 ? "true" : "false") << "\n"
            << "    layers:\n"
            << yaml_int_list(ranges[static_cast<std::size_t>(index)], 4) << "\n"
            << "    norm: " << (stage_id == num_stages ? "true" : "false") << "\n"
            << "    lm_head: " << (stage_id == num_stages ? "true" : "false") << "\n"
            << "  next_stage_url: ";
        if (stage_id < num_stages) {
            out << "http://" << stage_service_prefix << "-" << (stage_id + 1) << ":8000/forward-binary";
        } else {
            out << "null";
        }
        out << "\n\n";
    }

    out
        << "topology:\n"
        << "  probe_timeout_seconds: 0.25\n"
        << "  route_candidates:\n";
    for (int stage_id = 1; stage_id < num_stages; ++stage_id) {
        out
            << "    \"" << stage_id << "\":\n"
            << "    - http://" << stage_service_prefix << "-" << (stage_id + 1) << ":8000/forward-binary\n";
    }
    return out.str();
}

std::string json_string(const std::string& value) {
    return "\"" + dli::common::json_escape(value) + "\"";
}

std::string getenv_string(const char* key) {
    const char* value = std::getenv(key);
    return value != nullptr ? value : "";
}

std::vector<char*> to_exec_argv(const std::vector<std::string>& args) {
    std::vector<char*> result;
    result.reserve(args.size() + 1);
    for (const auto& arg : args) {
        result.push_back(const_cast<char*>(arg.c_str()));
    }
    result.push_back(nullptr);
    return result;
}

std::string join_command(const std::vector<std::string>& args) {
    std::ostringstream out;
    for (std::size_t i = 0; i < args.size(); ++i) {
        if (i > 0) {
            out << " ";
        }
        const bool needs_quotes = args[i].find_first_of(" \t\n\"'") != std::string::npos;
        if (!needs_quotes) {
            out << args[i];
        } else {
            out << "'";
            for (char ch : args[i]) {
                if (ch == '\'') {
                    out << "'\\''";
                } else {
                    out << ch;
                }
            }
            out << "'";
        }
    }
    return out.str();
}

CommandResult run_command(
    const std::vector<std::string>& args,
    bool capture_output = false,
    bool allow_failure = false,
    const std::vector<std::string>& masked_args = {}
) {
    if (args.empty()) {
        throw std::runtime_error("empty command");
    }

    std::cerr << "+ " << join_command(masked_args.empty() ? args : masked_args) << "\n";

    int pipe_fds[2] = {-1, -1};
    if (capture_output && ::pipe(pipe_fds) != 0) {
        throw std::runtime_error(std::string("pipe failed: ") + std::strerror(errno));
    }

    const pid_t pid = ::fork();
    if (pid < 0) {
        if (pipe_fds[0] >= 0) {
            ::close(pipe_fds[0]);
        }
        if (pipe_fds[1] >= 0) {
            ::close(pipe_fds[1]);
        }
        throw std::runtime_error(std::string("fork failed: ") + std::strerror(errno));
    }

    if (pid == 0) {
        if (capture_output) {
            ::close(pipe_fds[0]);
            (void)::dup2(pipe_fds[1], STDOUT_FILENO);
            (void)::dup2(pipe_fds[1], STDERR_FILENO);
            ::close(pipe_fds[1]);
        }
        std::vector<char*> exec_args = to_exec_argv(args);
        ::execvp(exec_args[0], exec_args.data());
        std::cerr << "exec failed for " << args[0] << ": " << std::strerror(errno) << "\n";
        _exit(127);
    }

    std::string output;
    if (capture_output) {
        ::close(pipe_fds[1]);
        char buffer[8192];
        while (true) {
            const ssize_t count = ::read(pipe_fds[0], buffer, sizeof(buffer));
            if (count < 0) {
                if (errno == EINTR) {
                    continue;
                }
                ::close(pipe_fds[0]);
                throw std::runtime_error(std::string("read failed: ") + std::strerror(errno));
            }
            if (count == 0) {
                break;
            }
            output.append(buffer, buffer + count);
        }
        ::close(pipe_fds[0]);
    }

    int status = 0;
    while (::waitpid(pid, &status, 0) < 0) {
        if (errno != EINTR) {
            throw std::runtime_error(std::string("waitpid failed: ") + std::strerror(errno));
        }
    }

    int exit_code = 1;
    if (WIFEXITED(status)) {
        exit_code = WEXITSTATUS(status);
    } else if (WIFSIGNALED(status)) {
        exit_code = 128 + WTERMSIG(status);
    }

    if (exit_code != 0 && !allow_failure) {
        std::ostringstream message;
        message << "command failed with exit code " << exit_code << ": " << join_command(masked_args.empty() ? args : masked_args);
        if (!output.empty()) {
            message << "\n" << output;
        }
        throw std::runtime_error(message.str());
    }

    return {exit_code, output};
}

void write_text_file(const std::filesystem::path& path, const std::string& content, mode_t mode = 0644) {
    std::filesystem::create_directories(path.parent_path());
    std::ofstream out(path);
    if (!out.is_open()) {
        throw std::runtime_error("failed to open file for writing: " + path.string());
    }
    out << content;
    out.close();
    if (::chmod(path.c_str(), mode) != 0) {
        throw std::runtime_error("chmod failed for " + path.string() + ": " + std::strerror(errno));
    }
}

int find_key(const gguf_context* ctx, const std::string& key) {
    return gguf_find_key(ctx, key.c_str());
}

std::string string_key_or_empty(const gguf_context* ctx, const std::string& key) {
    const int key_id = find_key(ctx, key);
    if (key_id < 0 || gguf_get_kv_type(ctx, key_id) != GGUF_TYPE_STRING) {
        return "";
    }
    const char* value = gguf_get_val_str(ctx, key_id);
    return value != nullptr ? value : "";
}

int int_key_or_default(const gguf_context* ctx, const std::vector<std::string>& keys, int default_value) {
    for (const auto& key : keys) {
        const int key_id = find_key(ctx, key);
        if (key_id < 0) {
            continue;
        }
        const gguf_type type = gguf_get_kv_type(ctx, key_id);
        if (type == GGUF_TYPE_INT32) {
            return gguf_get_val_i32(ctx, key_id);
        }
        if (type == GGUF_TYPE_UINT32) {
            const std::uint32_t value = gguf_get_val_u32(ctx, key_id);
            if (value > static_cast<std::uint32_t>(std::numeric_limits<int>::max())) {
                throw std::runtime_error("GGUF integer metadata too large: " + key);
            }
            return static_cast<int>(value);
        }
    }
    return default_value;
}

GgufMetadata inspect_gguf(const std::filesystem::path& model_path) {
    gguf_init_params params{};
    params.no_alloc = true;
    params.ctx = nullptr;

    gguf_context* raw_ctx = gguf_init_from_file(model_path.c_str(), params);
    if (raw_ctx == nullptr) {
        throw std::runtime_error("failed to open GGUF file: " + model_path.string());
    }
    std::unique_ptr<gguf_context, decltype(&gguf_free)> ctx(raw_ctx, gguf_free);

    GgufMetadata metadata;
    metadata.architecture = string_key_or_empty(ctx.get(), "general.architecture");
    metadata.name = string_key_or_empty(ctx.get(), "general.name");
    metadata.block_count = int_key_or_default(
        ctx.get(),
        {
            "llama.block_count",
            "mistral.block_count",
            "qwen2.block_count",
            "gptneox.block_count"
        },
        -1
    );
    return metadata;
}

std::filesystem::path download_model(const CliOptions& options, const std::filesystem::path& work_dir) {
    if (!options.local_model_path.empty()) {
        const std::filesystem::path path(options.local_model_path);
        if (!std::filesystem::exists(path)) {
            throw std::runtime_error("local model path does not exist: " + path.string());
        }
        return path;
    }

    const std::filesystem::path local_dir = work_dir / "full-model";
    std::filesystem::create_directories(local_dir);
    const std::filesystem::path output_path = local_dir / std::filesystem::path(options.source_file).filename();
    const std::string url =
        "https://huggingface.co/" +
        options.source_repo +
        "/resolve/" +
        options.source_revision +
        "/" +
        options.source_file;

    const std::string token = !getenv_string("HF_TOKEN").empty()
        ? getenv_string("HF_TOKEN")
        : getenv_string("HF_UPLOAD_TOKEN");
    std::filesystem::path curl_config;
    std::vector<std::string> args = {
        "curl",
        "-L",
        "-f",
        "-sS",
        "--retry",
        "5",
        "--retry-delay",
        "3",
        "--connect-timeout",
        "30",
        "-o",
        output_path.string()
    };
    std::vector<std::string> masked_args = args;
    if (!token.empty()) {
        curl_config = work_dir / "hf-download-curl.conf";
        write_text_file(curl_config, "header = \"Authorization: Bearer " + token + "\"\n", 0600);
        args.push_back("--config");
        args.push_back(curl_config.string());
        masked_args.push_back("--config");
        masked_args.push_back(curl_config.string());
    }
    args.push_back(url);
    masked_args.push_back(url);

    std::cerr << "Downloading " << options.source_repo << "/" << options.source_file
              << "@" << options.source_revision << "\n";
    run_command(args, false, false, masked_args);
    return output_path;
}

std::vector<std::filesystem::path> list_shards(const std::filesystem::path& shard_dir) {
    std::vector<std::filesystem::path> shards;
    for (const auto& entry : std::filesystem::directory_iterator(shard_dir)) {
        if (!entry.is_regular_file()) {
            continue;
        }
        const std::string name = entry.path().filename().string();
        if (name.rfind("partition-", 0) == 0 &&
            name.size() >= std::string(".dli.gguf").size() &&
            name.substr(name.size() - std::string(".dli.gguf").size()) == ".dli.gguf") {
            shards.push_back(entry.path());
        }
    }
    std::sort(shards.begin(), shards.end());
    return shards;
}

void copy_directory_recursive(const std::filesystem::path& source, const std::filesystem::path& destination) {
    std::filesystem::create_directories(destination);
    for (const auto& entry : std::filesystem::recursive_directory_iterator(source)) {
        const auto relative = std::filesystem::relative(entry.path(), source);
        const auto target = destination / relative;
        if (entry.is_directory()) {
            std::filesystem::create_directories(target);
        } else if (entry.is_regular_file()) {
            std::filesystem::create_directories(target.parent_path());
            std::filesystem::copy_file(entry.path(), target, std::filesystem::copy_options::overwrite_existing);
        }
    }
}

std::pair<std::string, std::string> split_repo_id(const std::string& repo_id) {
    const std::size_t slash = repo_id.find('/');
    if (slash == std::string::npos || slash == 0 || slash + 1 >= repo_id.size()) {
        throw std::runtime_error("Hugging Face repo id must be <namespace>/<repo>: " + repo_id);
    }
    return {repo_id.substr(0, slash), repo_id.substr(slash + 1)};
}

void create_repo_if_needed(
    const CliOptions& options,
    const std::filesystem::path& work_dir,
    const std::string& token
) {
    const std::filesystem::path curl_config = work_dir / "hf-upload-curl.conf";
    write_text_file(curl_config, "header = \"Authorization: Bearer " + token + "\"\n", 0600);

    const auto [namespace_name, repo_name] = split_repo_id(options.artifact_repo);
    const std::string body =
        "{\"name\":" + json_string(repo_name) +
        ",\"organization\":" + json_string(namespace_name) +
        ",\"type\":\"model\",\"private\":" +
        (options.artifact_private ? "true" : "false") +
        "}";
    const std::filesystem::path output_path = work_dir / "hf-create-repo-response.json";
    const std::vector<std::string> args = {
        "curl",
        "-sS",
        "-o",
        output_path.string(),
        "-w",
        "%{http_code}",
        "-X",
        "POST",
        "-H",
        "Content-Type: application/json",
        "--config",
        curl_config.string(),
        "--data-binary",
        body,
        "https://huggingface.co/api/repos/create"
    };
    const CommandResult result = run_command(args, true, true);
    const std::string status = trim_copy(result.output);
    if (status == "200" || status == "201" || status == "409") {
        return;
    }
    if (status == "400") {
        std::ifstream in(output_path);
        const std::string body_text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        if (body_text.find("exist") != std::string::npos || body_text.find("already") != std::string::npos) {
            return;
        }
    }
    throw std::runtime_error(
        "failed to create/verify HF repo " + options.artifact_repo +
        "; HTTP status " + status +
        " response file " + output_path.string()
    );
}

void upload_artifacts(
    const CliOptions& options,
    const std::filesystem::path& artifact_dir,
    const std::filesystem::path& work_dir,
    const std::string& model_slug
) {
    if (options.no_upload) {
        std::cerr << "Skipping Hugging Face upload because --no-upload was set.\n";
        return;
    }

    const std::string token = !getenv_string("HF_UPLOAD_TOKEN").empty()
        ? getenv_string("HF_UPLOAD_TOKEN")
        : getenv_string("HF_TOKEN");
    if (token.empty()) {
        throw std::runtime_error("HF_UPLOAD_TOKEN or HF_TOKEN is required for Hugging Face upload");
    }

    create_repo_if_needed(options, work_dir, token);

    const std::filesystem::path askpass = work_dir / "hf-git-askpass.sh";
    write_text_file(
        askpass,
        "#!/bin/sh\n"
        "case \"$1\" in\n"
        "*Username*) printf '%s\\n' 'hf' ;;\n"
        "*) printf '%s\\n' \"${HF_UPLOAD_TOKEN:-$HF_TOKEN}\" ;;\n"
        "esac\n",
        0700
    );
    ::setenv("GIT_ASKPASS", askpass.c_str(), 1);
    ::setenv("GIT_TERMINAL_PROMPT", "0", 1);

    const std::filesystem::path clone_dir = work_dir / "hf-repo";
    std::filesystem::remove_all(clone_dir);
    const std::string remote_url = "https://huggingface.co/" + options.artifact_repo;

    run_command({"git", "clone", remote_url, clone_dir.string()});

    if (options.artifact_revision != "main") {
        run_command({"git", "-C", clone_dir.string(), "fetch", "origin", options.artifact_revision}, false, true);
        const auto checkout = run_command(
            {"git", "-C", clone_dir.string(), "checkout", options.artifact_revision},
            false,
            true
        );
        if (checkout.exit_code != 0) {
            run_command({"git", "-C", clone_dir.string(), "checkout", "-b", options.artifact_revision});
        }
    }

    run_command({"git", "-C", clone_dir.string(), "lfs", "install", "--local"});
    run_command({"git", "-C", clone_dir.string(), "lfs", "track", "*.gguf"});
    run_command({"git", "-C", clone_dir.string(), "config", "user.email", "dli-native-controller@local"});
    run_command({"git", "-C", clone_dir.string(), "config", "user.name", "DLI Native Controller"});

    const std::filesystem::path target_dir = clone_dir / model_slug;
    std::filesystem::remove_all(target_dir);
    copy_directory_recursive(artifact_dir, target_dir);

    run_command({"git", "-C", clone_dir.string(), "add", ".gitattributes", model_slug});
    const CommandResult status = run_command(
        {"git", "-C", clone_dir.string(), "status", "--porcelain"},
        true
    );
    if (trim_copy(status.output).empty()) {
        std::cerr << "No HF artifact changes to commit.\n";
        return;
    }

    run_command({
        "git",
        "-C",
        clone_dir.string(),
        "commit",
        "-m",
        "Prepare DLI native artifacts for " + options.model_id
    });
    run_command({
        "git",
        "-C",
        clone_dir.string(),
        "push",
        "origin",
        "HEAD:" + options.artifact_revision
    });
}

std::string metadata_json(
    const CliOptions& options,
    const std::string& model_slug,
    const std::string& architecture,
    int num_layers,
    const std::string& full_file_name
) {
    std::ostringstream out;
    out
        << "{\n"
        << "  \"model_id\": " << json_string(options.model_id) << ",\n"
        << "  \"model_name\": " << json_string(options.model_name) << ",\n"
        << "  \"model_slug\": " << json_string(model_slug) << ",\n"
        << "  \"architecture\": " << json_string(architecture) << ",\n"
        << "  \"num_layers\": " << num_layers << ",\n"
        << "  \"num_stages\": " << options.num_stages << ",\n"
        << "  \"source_repo\": " << json_string(options.source_repo) << ",\n"
        << "  \"source_revision\": " << json_string(options.source_revision) << ",\n"
        << "  \"source_file\": " << json_string(options.source_file) << ",\n"
        << "  \"artifact_repo\": " << json_string(options.artifact_repo) << ",\n"
        << "  \"artifact_revision\": " << json_string(options.artifact_revision) << ",\n"
        << "  \"full_gguf_file\": " << json_string(model_slug + "/" + full_file_name) << ",\n"
        << "  \"stage_map_file\": " << json_string(model_slug + "/stage_map.yaml") << "\n"
        << "}\n";
    return out.str();
}

std::string activation_json(
    const CliOptions& options,
    const std::string& model_slug,
    const std::string& stage_map_yaml,
    const std::string& metadata,
    const std::string& full_file_name
) {
    std::ostringstream out;
    out
        << "{\n"
        << "  \"modelId\": " << json_string(options.model_id) << ",\n"
        << "  \"modelName\": " << json_string(options.model_name) << ",\n"
        << "  \"modelSlug\": " << json_string(model_slug) << ",\n"
        << "  \"data\": {\n"
        << "    \"DLI_ACTIVE_MODEL_ID\": " << json_string(options.model_id) << ",\n"
        << "    \"DLI_ACTIVE_MODEL_SLUG\": " << json_string(model_slug) << ",\n"
        << "    \"HF_NATIVE_STAGE_REPO\": " << json_string(options.artifact_repo) << ",\n"
        << "    \"HF_NATIVE_STAGE_REVISION\": " << json_string(options.artifact_revision) << ",\n"
        << "    \"HF_NATIVE_FULL_GGUF_REPO\": " << json_string(options.artifact_repo) << ",\n"
        << "    \"HF_NATIVE_FULL_GGUF_REVISION\": " << json_string(options.artifact_revision) << ",\n"
        << "    \"HF_NATIVE_FULL_GGUF_FILE\": " << json_string(model_slug + "/" + full_file_name) << ",\n"
        << "    \"stage_map.yaml\": " << json_string(stage_map_yaml) << "\n"
        << "  },\n"
        << "  \"metadata\": " << metadata
        << "}\n";
    return out.str();
}

void prepare(CliOptions options) {
    const std::filesystem::path work_dir(options.work_dir);
    std::filesystem::create_directories(work_dir);

    const std::string model_slug = options.model_slug.empty()
        ? slugify(options.model_id)
        : options.model_slug;

    const std::filesystem::path full_model_path = download_model(options, work_dir);
    const GgufMetadata info = inspect_gguf(full_model_path);
    if (info.architecture != "llama") {
        throw std::runtime_error(
            "native CPU executor currently supports GGUF general.architecture=llama; got '" +
            info.architecture +
            "'"
        );
    }

    const int num_layers = options.num_layers > 0 ? options.num_layers : info.block_count;
    if (num_layers <= 0) {
        throw std::runtime_error("could not determine model layer count from GGUF metadata");
    }

    const std::string full_file_name = !options.full_gguf_file.empty()
        ? options.full_gguf_file
        : std::filesystem::path(
            !options.source_file.empty()
                ? options.source_file
                : full_model_path.filename().string()
        ).filename().string();

    const auto physical_nodes = parse_physical_nodes(options.physical_nodes, options.num_stages);
    const std::string stage_map_yaml = generate_stage_map(
        options.model_name,
        model_slug,
        full_file_name,
        num_layers,
        options.num_stages,
        options.gateway_service_name,
        options.stage_service_prefix,
        physical_nodes,
        options.container_model_root
    );

    const std::filesystem::path generated_dir = work_dir / "generated";
    const std::filesystem::path manifest_dir = generated_dir / "manifests";
    const std::filesystem::path shard_dir = generated_dir / "shards" / model_slug;
    const std::filesystem::path artifact_dir = generated_dir / "artifacts" / model_slug;
    std::filesystem::create_directories(manifest_dir);
    std::filesystem::create_directories(shard_dir);
    std::filesystem::create_directories(artifact_dir);

    const std::filesystem::path stage_map_path = generated_dir / "stage_map.yaml";
    write_text_file(stage_map_path, stage_map_yaml);

    run_command({
        options.partition_manifest_bin,
        "--config",
        stage_map_path.string(),
        "--model",
        full_model_path.string(),
        "--output-dir",
        manifest_dir.string()
    });
    run_command({
        options.shard_writer_bin,
        "--model",
        full_model_path.string(),
        "--manifest-dir",
        manifest_dir.string(),
        "--output-dir",
        shard_dir.string(),
        "--source-model",
        options.model_name
    });
    run_command({
        options.shard_validate_bin,
        "--config",
        stage_map_path.string(),
        "--manifest-dir",
        manifest_dir.string(),
        "--shard-dir",
        shard_dir.string()
    });

    std::filesystem::copy_file(
        full_model_path,
        artifact_dir / full_file_name,
        std::filesystem::copy_options::overwrite_existing
    );
    for (const auto& shard : list_shards(shard_dir)) {
        std::filesystem::copy_file(
            shard,
            artifact_dir / shard.filename(),
            std::filesystem::copy_options::overwrite_existing
        );
    }
    write_text_file(artifact_dir / "stage_map.yaml", stage_map_yaml);
    const std::string metadata = metadata_json(
        options,
        model_slug,
        info.architecture,
        num_layers,
        full_file_name
    );
    write_text_file(artifact_dir / "native-model.json", metadata);

    upload_artifacts(options, artifact_dir, work_dir, model_slug);

    std::cout << "DLI_ACTIVATION_JSON_BEGIN\n";
    std::cout << activation_json(options, model_slug, stage_map_yaml, metadata, full_file_name);
    std::cout << "DLI_ACTIVATION_JSON_END\n";
}

} // namespace

int main(int argc, char** argv) {
    try {
        CliOptions options = parse_args(argc, argv);
        prepare(std::move(options));
        return 0;
    } catch (const std::exception& exc) {
        std::cerr << "dli-native-model-controller failed: " << exc.what() << "\n";
        return 1;
    }
}
