#include "pipeline.h"
#include "png.h"

#include <cpp-httplib/httplib.h>
#include <nlohmann/json.hpp>

#include <atomic>
#include <cctype>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <vector>

namespace fs = std::filesystem;

struct server_args {
    std::string model;
    std::string fm;
    std::string host       = "127.0.0.1";
    std::string model_name = "sensenova-u1.5";
    std::string output_dir;
    bool allow_output_dir  = false;
    int port    = 8080;
    int ngl     = 999;
    int threads = 0;
};

static void print_usage(const char * prog) {
    fprintf(stderr,
            "usage: %s -m MODEL.gguf --fm FM.gguf [options]\n"
            "\n"
            "options:\n"
            "  -m, --model PATH        language model GGUF (required)\n"
            "      --fm PATH           FM sidecar GGUF (required)\n"
            "      --host ADDR         listen address (default: 127.0.0.1)\n"
            "      --port N            listen port (default: 8080)\n"
            "      --model-name NAME   model id reported by /v1/models (default: sensenova-u1.5)\n"
            "      --output-dir PATH   directory for url responses (default: <tmp>/llama-sensenova-server)\n"
            "      --allow-output-dir  accept output_dir from /v1/infographic/generate\n"
            "  -ngl N                  layers to offload to GPU (default: 999)\n"
            "  -t, --threads N         CPU threads (default: number of cores)\n"
            "\n"
            "endpoints:\n"
            "  POST /v1/chat/completions     OpenAI Chat API, single turn visual understanding\n"
            "  POST /v1/images/generations   OpenAI Images API\n"
            "  POST /v1/images/edits         OpenAI Images API, multipart or JSON with base64 images\n"
            "  POST /v1/interleave/generate  interleaved text and image generation\n"
            "  POST /v1/infographic/generate private protocol used by omni-note\n"
            "  POST /generate                native protocol (legacy)\n"
            "  GET  /images/{file}           files written for url responses\n"
            "  GET  /v1/models, /health, /props\n",
            prog);
}

static bool parse_args(int argc, char ** argv, server_args & a) {
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto next = [&](const char * name) -> const char * {
            if (i + 1 >= argc) {
                fprintf(stderr, "error: missing value for %s\n", name);
                return nullptr;
            }
            return argv[++i];
        };

        if (arg == "-m" || arg == "--model") {
            const char * v = next(arg.c_str()); if (!v) return false; a.model = v;
        } else if (arg == "--fm") {
            const char * v = next(arg.c_str()); if (!v) return false; a.fm = v;
        } else if (arg == "--host") {
            const char * v = next(arg.c_str()); if (!v) return false; a.host = v;
        } else if (arg == "--port") {
            const char * v = next(arg.c_str()); if (!v) return false; a.port = atoi(v);
        } else if (arg == "--model-name") {
            const char * v = next(arg.c_str()); if (!v) return false; a.model_name = v;
        } else if (arg == "--output-dir") {
            const char * v = next(arg.c_str()); if (!v) return false; a.output_dir = v;
        } else if (arg == "--allow-output-dir") {
            a.allow_output_dir = true;
        } else if (arg == "-ngl") {
            const char * v = next(arg.c_str()); if (!v) return false; a.ngl = atoi(v);
        } else if (arg == "-t" || arg == "--threads") {
            const char * v = next(arg.c_str()); if (!v) return false; a.threads = atoi(v);
        } else if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            exit(0);
        } else {
            fprintf(stderr, "error: unknown argument: %s\n", arg.c_str());
            return false;
        }
    }

    if (a.model.empty() || a.fm.empty()) {
        print_usage(argv[0]);
        return false;
    }
    if (a.port <= 0 || a.port > 65535) {
        fprintf(stderr, "error: invalid port %d\n", a.port);
        return false;
    }
    return true;
}

static std::string base64_encode(const uint8_t * data, size_t len) {
    static const char tbl[] =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string out;
    out.reserve((len + 2)/3*4);
    size_t i = 0;
    for (; i + 2 < len; i += 3) {
        const uint32_t v = ((uint32_t) data[i] << 16) | ((uint32_t) data[i + 1] << 8) | data[i + 2];
        out.push_back(tbl[(v >> 18) & 63]);
        out.push_back(tbl[(v >> 12) & 63]);
        out.push_back(tbl[(v >> 6) & 63]);
        out.push_back(tbl[v & 63]);
    }
    if (i < len) {
        const uint32_t b0 = data[i];
        const uint32_t b1 = i + 1 < len ? data[i + 1] : 0;
        const uint32_t v  = (b0 << 16) | (b1 << 8);
        out.push_back(tbl[(v >> 18) & 63]);
        out.push_back(tbl[(v >> 12) & 63]);
        out.push_back(i + 1 < len ? tbl[(v >> 6) & 63] : '=');
        out.push_back('=');
    }
    return out;
}

static int base64_value(char c) {
    if (c >= 'A' && c <= 'Z') return c - 'A';
    if (c >= 'a' && c <= 'z') return c - 'a' + 26;
    if (c >= '0' && c <= '9') return c - '0' + 52;
    if (c == '+') return 62;
    if (c == '/') return 63;
    return -1;
}

static bool base64_decode(const std::string & in, std::vector<uint8_t> & out, std::string & err) {
    out.clear();
    out.reserve(in.size()/4*3);
    uint32_t acc = 0;
    int bits = 0;
    for (char c : in) {
        if (c == '=' || c == '\n' || c == '\r' || c == ' ' || c == '\t') {
            continue;
        }
        const int v = base64_value(c);
        if (v < 0) {
            err = "invalid base64 data";
            return false;
        }
        acc = (acc << 6) | (uint32_t) v;
        bits += 6;
        if (bits >= 8) {
            bits -= 8;
            out.push_back((uint8_t) ((acc >> bits) & 0xff));
        }
    }
    return true;
}

// input images may be a file path, a data URL, or raw base64
static bool make_image_input(const std::string & value, sensenova_image_input & img, std::string & err) {
    if (value.rfind("data:", 0) == 0) {
        const size_t comma = value.find(',');
        if (comma == std::string::npos) {
            err = "invalid data URL";
            return false;
        }
        return base64_decode(value.substr(comma + 1), img.data, err);
    }
    std::error_code ec;
    if (fs::exists(value, ec) && !ec) {
        img.path = value;
        return true;
    }
    return base64_decode(value, img.data, err);
}

static bool set_json_images(const nlohmann::json & j, std::vector<sensenova_image_input> & images, std::string & err) {
    auto append = [&](const nlohmann::json & item) -> bool {
        if (!item.is_string()) {
            err = "image entries must be strings";
            return false;
        }
        sensenova_image_input img;
        if (!make_image_input(item.get<std::string>(), img, err)) {
            return false;
        }
        images.push_back(std::move(img));
        return true;
    };

    for (const char * key : {"image", "images"}) {
        if (!j.contains(key)) {
            continue;
        }
        const nlohmann::json & v = j[key];
        if (v.is_array()) {
            for (const auto & item : v) {
                if (!append(item)) {
                    return false;
                }
            }
        } else if (!append(v)) {
            return false;
        }
    }
    return true;
}

static void json_error_legacy(httplib::Response & res, int status, const std::string & message) {
    nlohmann::json j;
    j["error"] = message;
    res.status = status;
    res.set_content(j.dump(), "application/json");
}

static void json_error_openai(httplib::Response & res, int status, const std::string & message,
                              const char * type = "invalid_request_error") {
    nlohmann::json j;
    j["error"] = {
        {"message", message},
        {"type", type},
        {"code", nullptr},
    };
    res.status = status;
    res.set_content(j.dump(), "application/json");
}

static void private_error(httplib::Response & res, const std::string & message) {
    // omni-note client reads status from the body, so failures stay HTTP 200
    nlohmann::json j = {
        {"status", "error"},
        {"image_url", ""},
        {"image_path", ""},
        {"generation_time", 0.0},
        {"width", 0},
        {"height", 0},
        {"message", message},
    };
    res.set_content(j.dump(), "application/json");
}

static std::string random_hex(size_t n) {
    thread_local std::mt19937_64 rng(
            (uint64_t) std::random_device{}() ^
            (uint64_t) std::chrono::steady_clock::now().time_since_epoch().count());
    static const char digits[] = "0123456789abcdef";
    std::string out;
    out.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        out.push_back(digits[rng() & 15]);
    }
    return out;
}

static int random_seed() {
    thread_local std::mt19937 rng(
            (uint32_t) std::random_device{}() ^
            (uint32_t) std::chrono::steady_clock::now().time_since_epoch().count());
    return (int) (rng() & 0x7fffffff);
}

static bool parse_size(const std::string & value, int & width, int & height, std::string & err) {
    std::string s = value;
    for (char & c : s) {
        c = (char) tolower((unsigned char) c);
    }
    if (s.empty() || s == "auto") {
        width  = 1024;
        height = 1024;
        return true;
    }
    int w = 0;
    int h = 0;
    char tail = 0;
    if (sscanf(s.c_str(), "%dx%d%c", &w, &h, &tail) != 2) {
        err = "size must be WxH, for example 1024x1024";
        return false;
    }
    if (w <= 0 || h <= 0 || w % 32 != 0 || h % 32 != 0) {
        err = "size width and height must be positive multiples of 32";
        return false;
    }
    width  = w;
    height = h;
    return true;
}

static bool write_file(const fs::path & path, const std::vector<uint8_t> & data, std::string & err) {
    std::ofstream f(path, std::ios::binary);
    if (!f) {
        err = "cannot open " + path.string();
        return false;
    }
    f.write(reinterpret_cast<const char *>(data.data()), (std::streamsize) data.size());
    if (!f) {
        err = "failed to write " + path.string();
        return false;
    }
    return true;
}

struct gen_output {
    std::vector<uint8_t> png;
    int width = 0;
    int height = 0;
    std::string think_text;
    int64_t elapsed_ms = 0;
};

static bool generate_png(
        sensenova_model & model,
        std::mutex & mu,
        const sensenova_gen_params & p,
        gen_output & out,
        std::string & err) {
    sensenova_result result;
    const auto t0 = std::chrono::steady_clock::now();
    {
        std::lock_guard<std::mutex> lock(mu);
        if (!sensenova_generate(model, p, result, err)) {
            return false;
        }
    }
    if (!png_encode_rgb(result.rgb.data(), result.width, result.height, out.png, err)) {
        return false;
    }
    out.width      = result.width;
    out.height     = result.height;
    out.think_text = std::move(result.think_text);
    out.elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - t0).count();
    return true;
}

static std::string request_base_url(const httplib::Request & req, const server_args & a) {
    std::string host = req.get_header_value("Host");
    if (host.empty()) {
        host = a.host + ":" + std::to_string(a.port);
    }
    return "http://" + host;
}

// split "<image>" markers into ordered text and image segments
static nlohmann::json interleave_segments(
        const std::string & text,
        const std::vector<gen_output> & images,
        const std::vector<std::string> & locs,
        bool use_paths) {
    static const std::string marker = "<image>";
    nlohmann::json segs = nlohmann::json::array();
    auto image_item = [&](size_t i) {
        nlohmann::json item = {
            {"type", "image"},
            {"index", (int) i},
            {"width", images[i].width},
            {"height", images[i].height},
        };
        if (locs.empty()) {
            item["b64_json"] = base64_encode(images[i].png.data(), images[i].png.size());
        } else if (use_paths) {
            item["image_path"] = locs[i];
        } else {
            item["url"] = locs[i];
        }
        return item;
    };

    size_t pos = 0;
    size_t idx = 0;
    for (;;) {
        const size_t hit = text.find(marker, pos);
        if (hit == std::string::npos) {
            break;
        }
        if (idx < images.size()) {
            if (hit > pos) {
                segs.push_back({{"type", "text"}, {"text", text.substr(pos, hit - pos)}});
            }
            segs.push_back(image_item(idx++));
        }
        pos = hit + marker.size();
    }
    if (pos < text.size()) {
        segs.push_back({{"type", "text"}, {"text", text.substr(pos)}});
    }
    // a marker always comes with an image, but keep every image reachable
    while (idx < images.size()) {
        segs.push_back(image_item(idx++));
    }
    return segs;
}

static void openai_images_response(
        httplib::Response & res,
        const nlohmann::json & data,
        int64_t elapsed_ms,
        const std::string & model_name) {
    nlohmann::json out = {
        {"created", (int64_t) std::time(nullptr)},
        {"data", data},
        {"model", model_name},
    };
    res.set_header("X-Elapsed-Ms", std::to_string(elapsed_ms));
    res.set_content(out.dump(), "application/json");
}

// OpenAI messages for a single user turn, system messages are ignored
static bool parse_chat_request(const nlohmann::json & j, sensenova_qa_params & p, std::string & err) {
    if (!j.contains("messages") || !j["messages"].is_array() || j["messages"].empty()) {
        err = "messages is required";
        return false;
    }

    bool have_user = false;
    for (const auto & m : j["messages"]) {
        if (!m.is_object() || !m.contains("role") || !m["role"].is_string()) {
            err = "each message needs a role";
            return false;
        }
        const std::string role = m["role"].get<std::string>();
        if (role == "system" || role == "developer") {
            continue;
        }
        if (role != "user" || have_user) {
            err = "only a single user message is supported";
            return false;
        }
        have_user = true;

        const nlohmann::json & content = m["content"];
        if (content.is_string()) {
            p.question = content.get<std::string>();
            continue;
        }
        if (!content.is_array()) {
            err = "content must be a string or an array";
            return false;
        }
        for (const auto & part : content) {
            if (!part.is_object() || !part.contains("type") || !part["type"].is_string()) {
                err = "invalid content part";
                return false;
            }
            const std::string type = part["type"].get<std::string>();
            if (type == "text") {
                p.question += part.value("text", std::string());
            } else if (type == "image_url") {
                std::string url;
                if (part.contains("image_url") && part["image_url"].is_object()) {
                    url = part["image_url"].value("url", std::string());
                }
                if (url.empty()) {
                    err = "image_url needs a url";
                    return false;
                }
                if (url.rfind("http://", 0) == 0 || url.rfind("https://", 0) == 0) {
                    err = "remote image URLs are not supported, send a data URL or base64";
                    return false;
                }
                sensenova_image_input img;
                if (!make_image_input(url, img, err)) {
                    return false;
                }
                p.images.push_back(std::move(img));
            } else {
                err = "unsupported content part: " + type;
                return false;
            }
        }
    }

    if (!have_user) {
        err = "a user message is required";
        return false;
    }
    if (p.question.empty()) {
        err = "question text is required";
        return false;
    }
    return true;
}

static httplib::Server * g_server = nullptr;
static std::atomic_flag g_terminating = ATOMIC_FLAG_INIT;

static void on_signal(int) {
    if (g_terminating.test_and_set()) {
        // stop() waits for the running generation, so a second interrupt exits now
        exit(1);
    }
    if (g_server != nullptr) {
        g_server->stop();
    }
}

int main(int argc, char ** argv) {
    server_args a;
    if (!parse_args(argc, argv, a)) {
        return 1;
    }

    if (a.output_dir.empty()) {
        a.output_dir = (fs::temp_directory_path() / "llama-sensenova-server").string();
    }
    std::error_code ec;
    fs::create_directories(a.output_dir, ec);
    if (ec) {
        fprintf(stderr, "error: cannot create output dir %s: %s\n",
                a.output_dir.c_str(), ec.message().c_str());
        return 1;
    }
    const fs::path out_root = fs::absolute(a.output_dir, ec).lexically_normal();

    const auto load_t0 = std::chrono::steady_clock::now();
    std::string err;
    fprintf(stderr, "loading model %s\n", a.model.c_str());
    std::unique_ptr<sensenova_model> model = sensenova_model_load(a.model, a.fm, a.ngl, a.threads, err);
    if (!model) {
        fprintf(stderr, "error: %s\n", err.c_str());
        return 1;
    }
    const double load_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - load_t0).count();
    const int64_t start_epoch = (int64_t) std::time(nullptr);

    httplib::Server svr;
    std::mutex mu;

    svr.set_payload_max_length(64 << 20);
    svr.set_read_timeout(600);
    svr.set_write_timeout(600);

    svr.Get("/health", [&](const httplib::Request &, httplib::Response & res) {
        nlohmann::json j = {
            {"status", "ok"},
            {"model_loaded", true},
            {"load_time", load_seconds},
        };
        res.set_content(j.dump(), "application/json");
    });

    svr.Get("/v1/models", [&](const httplib::Request &, httplib::Response & res) {
        nlohmann::json item = {
            {"id", a.model_name},
            {"object", "model"},
            {"created", start_epoch},
            {"owned_by", "local"},
        };
        nlohmann::json j = {
            {"object", "list"},
            {"data", nlohmann::json::array()},
        };
        j["data"].push_back(item);
        res.set_content(j.dump(), "application/json");
    });

    svr.Get("/props", [&](const httplib::Request &, httplib::Response & res) {
        nlohmann::json j;
        j["model"]      = a.model;
        j["fm"]         = a.fm;
        j["model_name"] = a.model_name;
        j["n_embd"]     = model->n_embd;
        j["output_dir"] = out_root.string();
        j["supports"] = {
            {"think", true},
            {"edits", true},
            {"vqa", true},
            {"interleave", true},
            {"cfg_norm", {"none", "global", "channel"}},
        };
        j["defaults"] = {
            {"width", 1024}, {"height", 1024}, {"steps", 8},
            {"seed", 42}, {"cfg", 1.0}, {"img_cfg", 1.0}, {"shift", 3.0},
            {"cfg_norm", "none"}, {"think", false}, {"auto_pixels", 1024*1024},
            {"max_new_tokens", 1024}, {"temperature", 0.0}, {"top_p", 0.95},
            {"max_images", 3}, {"interleave_max_new_tokens", 4096},
        };
        res.set_content(j.dump(), "application/json");
    });

    svr.Get(R"(/images/([^/]+))", [&](const httplib::Request & req, httplib::Response & res) {
        const std::string name = req.matches[1];
        if (name.find("..") != std::string::npos) {
            res.status = 404;
            return;
        }
        const fs::path path = out_root / name;
        std::ifstream f(path, std::ios::binary);
        if (!f) {
            res.status = 404;
            res.set_content("image not found", "text/plain");
            return;
        }
        std::string data((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        res.set_content(data, "image/png");
    });

    // native protocol, kept for existing clients
    svr.Post("/generate", [&](const httplib::Request & req, httplib::Response & res) {
        sensenova_gen_params p;
        std::string response_format = "png";
        std::string err;
        try {
            const nlohmann::json j = nlohmann::json::parse(req.body);
            if (!j.is_object()) {
                json_error_legacy(res, 400, "request body must be a JSON object");
                return;
            }
            p.prompt = j.value("prompt", std::string());
            p.width  = j.value("width", p.width);
            p.height = j.value("height", p.height);
            p.steps  = j.value("steps", p.steps);
            p.seed   = j.value("seed", p.seed);
            p.cfg    = j.value("cfg", p.cfg);
            p.img_cfg = j.value("img_cfg", p.img_cfg);
            p.cfg_norm = j.value("cfg_norm", p.cfg_norm);
            p.shift  = j.value("shift", p.shift);
            p.think  = j.value("think", p.think);
            p.max_think_tokens = j.value("max_think_tokens", p.max_think_tokens);
            p.auto_size_pixels = j.value("auto_pixels", p.auto_size_pixels);
            if (!set_json_images(j, p.images, err)) {
                json_error_legacy(res, 400, err);
                return;
            }
            response_format = j.value("response_format", response_format);
        } catch (const std::exception & e) {
            json_error_legacy(res, 400, std::string("invalid request: ") + e.what());
            return;
        }

        if (p.prompt.empty()) {
            json_error_legacy(res, 400, "prompt is required");
            return;
        }
        if ((p.width == 0) != (p.height == 0)) {
            json_error_legacy(res, 400, "width and height must be set together");
            return;
        }
        if (p.width != 0 && p.width % 32 != 0) {
            json_error_legacy(res, 400, "width and height must be multiples of 32");
            return;
        }
        if (p.width < 0 || p.height < 0) {
            json_error_legacy(res, 400, "width and height must be positive");
            return;
        }
        if (p.steps <= 0) {
            json_error_legacy(res, 400, "steps must be positive");
            return;
        }
        if (response_format != "png" && response_format != "b64_json") {
            json_error_legacy(res, 400, "response_format must be png or b64_json");
            return;
        }

        gen_output out;
        if (!generate_png(*model, mu, p, out, err)) {
            json_error_legacy(res, 500, err);
            return;
        }

        if (response_format == "b64_json") {
            nlohmann::json j;
            j["image"]      = base64_encode(out.png.data(), out.png.size());
            j["width"]      = out.width;
            j["height"]     = out.height;
            j["steps"]      = p.steps;
            j["seed"]       = p.seed;
            j["elapsed_ms"] = out.elapsed_ms;
            if (p.think) {
                j["think"] = out.think_text;
            }
            res.set_content(j.dump(), "application/json");
        } else {
            res.set_header("X-Elapsed-Ms", std::to_string(out.elapsed_ms));
            if (p.think) {
                res.set_header("X-Think-Text", base64_encode(
                        reinterpret_cast<const uint8_t *>(out.think_text.data()), out.think_text.size()));
            }
            res.set_content(reinterpret_cast<const char *>(out.png.data()), out.png.size(), "image/png");
        }
    });

    // OpenAI Chat API, visual understanding
    svr.Post("/v1/chat/completions", [&](const httplib::Request & req, httplib::Response & res) {
        sensenova_qa_params p;
        std::string model_name = a.model_name;
        std::string err;
        try {
            const nlohmann::json j = nlohmann::json::parse(req.body);
            if (!j.is_object()) {
                json_error_openai(res, 400, "request body must be a JSON object");
                return;
            }
            if (j.value("stream", false)) {
                json_error_openai(res, 400, "stream is not supported");
                return;
            }
            if (j.contains("model") && j["model"].is_string()) {
                model_name = j["model"].get<std::string>();
            }
            if (!parse_chat_request(j, p, err)) {
                json_error_openai(res, 400, err);
                return;
            }
            for (const char * key : {"max_completion_tokens", "max_tokens"}) {
                if (j.contains(key) && j[key].is_number_integer()) {
                    p.max_new_tokens = j[key].get<int>();
                }
            }
            p.temperature      = j.value("temperature", p.temperature);
            p.top_p            = j.value("top_p", p.top_p);
            p.top_k            = j.value("top_k", p.top_k);
            p.repeat_penalty   = j.value("repetition_penalty", p.repeat_penalty);
            p.seed             = j.value("seed", p.seed);
            p.input_max_pixels = j.value("input_max_pixels", p.input_max_pixels);
        } catch (const std::exception & e) {
            json_error_openai(res, 400, std::string("invalid request: ") + e.what());
            return;
        }

        if (p.max_new_tokens < 1 || p.max_new_tokens > 16384) {
            json_error_openai(res, 400, "max_tokens must be between 1 and 16384");
            return;
        }
        if (p.temperature < 0.0f || p.top_k < 0 || p.top_p <= 0.0f || p.top_p > 1.0f || p.repeat_penalty <= 0.0f) {
            json_error_openai(res, 400, "invalid sampling parameters");
            return;
        }

        sensenova_qa_result out;
        const auto t0 = std::chrono::steady_clock::now();
        {
            std::lock_guard<std::mutex> lock(mu);
            if (!sensenova_understand(*model, p, out, err)) {
                json_error_openai(res, 500, err, "server_error");
                return;
            }
        }
        const int64_t elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - t0).count();

        nlohmann::json message = {
            {"role", "assistant"},
            {"content", out.answer},
        };
        if (!out.think_text.empty()) {
            message["reasoning_content"] = out.think_text;
        }
        nlohmann::json choice = {
            {"index", 0},
            {"message", message},
            {"finish_reason", out.truncated ? "length" : "stop"},
        };
        nlohmann::json j = {
            {"id", "chatcmpl-" + random_hex(16)},
            {"object", "chat.completion"},
            {"created", (int64_t) std::time(nullptr)},
            {"model", model_name},
            {"choices", nlohmann::json::array({choice})},
            {"usage", {
                {"prompt_tokens", out.n_prompt_tokens},
                {"completion_tokens", out.n_generated},
                {"total_tokens", out.n_prompt_tokens + out.n_generated},
            }},
        };
        res.set_header("X-Elapsed-Ms", std::to_string(elapsed_ms));
        res.set_content(j.dump(), "application/json");
    });

    // OpenAI Images API
    svr.Post("/v1/images/generations", [&](const httplib::Request & req, httplib::Response & res) {
        sensenova_gen_params p;
        int n = 1;
        std::string model_name = a.model_name;
        std::string response_format = "b64_json";
        std::string err;
        try {
            const nlohmann::json j = nlohmann::json::parse(req.body);
            if (!j.is_object()) {
                json_error_openai(res, 400, "request body must be a JSON object");
                return;
            }
            p.prompt = j.value("prompt", std::string());
            n = j.value("n", 1);
            if (j.contains("model") && !j["model"].is_null()) {
                model_name = j["model"].get<std::string>();
            }
            response_format = j.value("response_format", response_format);
            const std::string output_format = j.value("output_format", std::string("png"));
            if (output_format != "png") {
                json_error_openai(res, 400, "output_format: only png is supported");
                return;
            }
            if (!parse_size(j.value("size", std::string("1024x1024")), p.width, p.height, err)) {
                json_error_openai(res, 400, err);
                return;
            }
            p.seed  = j.value("seed", p.seed);
            p.steps = j.value("num_steps", p.steps);
            p.cfg   = j.value("cfg_scale", p.cfg);
            p.shift = j.value("timestep_shift", p.shift);
            p.think = j.value("think", p.think);
            p.max_think_tokens = j.value("max_think_tokens", p.max_think_tokens);
            p.cfg_norm = j.value("cfg_norm", p.cfg_norm);
        } catch (const std::exception & e) {
            json_error_openai(res, 400, std::string("invalid request: ") + e.what());
            return;
        }

        if (p.prompt.empty()) {
            json_error_openai(res, 400, "prompt is required");
            return;
        }
        if (n < 1 || n > 4) {
            json_error_openai(res, 400, "n must be between 1 and 4");
            return;
        }
        if (p.steps < 1 || p.steps > 100) {
            json_error_openai(res, 400, "num_steps must be between 1 and 100");
            return;
        }
        if (response_format != "b64_json" && response_format != "url") {
            json_error_openai(res, 400, "response_format must be b64_json or url");
            return;
        }
        if (p.cfg_norm != "none" && p.cfg_norm != "global" && p.cfg_norm != "channel") {
            json_error_openai(res, 400, "cfg_norm must be none, global or channel");
            return;
        }

        nlohmann::json data = nlohmann::json::array();
        int64_t elapsed_ms = 0;
        for (int i = 0; i < n; ++i) {
            sensenova_gen_params q = p;
            if (p.seed < 0) {
                q.seed = random_seed();
            } else {
                q.seed = (int) (((int64_t) p.seed + i) & 0x7fffffff);
            }

            gen_output out;
            if (!generate_png(*model, mu, q, out, err)) {
                json_error_openai(res, 500, err, "server_error");
                return;
            }
            elapsed_ms += out.elapsed_ms;

            nlohmann::json item;
            if (response_format == "b64_json") {
                item["b64_json"] = base64_encode(out.png.data(), out.png.size());
            } else {
                const std::string name = "openai_" + random_hex(12) + ".png";
                if (!write_file(out_root / name, out.png, err)) {
                    json_error_openai(res, 500, err, "server_error");
                    return;
                }
                item["url"] = request_base_url(req, a) + "/images/" + name;
            }
            if (!out.think_text.empty()) {
                item["revised_prompt"] = out.think_text;
            }
            data.push_back(std::move(item));
        }

        nlohmann::json out = {
            {"created", (int64_t) std::time(nullptr)},
            {"data", data},
            {"model", model_name},
        };
        res.set_header("X-Elapsed-Ms", std::to_string(elapsed_ms));
        res.set_content(out.dump(), "application/json");
    });

    // OpenAI Images API, image editing
    svr.Post("/v1/images/edits", [&](const httplib::Request & req, httplib::Response & res) {
        sensenova_gen_params p;
        int n = 1;
        std::string model_name = a.model_name;
        std::string response_format = "b64_json";
        std::string size = "auto";
        std::string err;

        if (req.is_multipart_form_data()) {
            auto field = [&](const char * key, const std::string & def) {
                return req.form.has_field(key) ? req.form.get_field(key) : def;
            };
            auto field_int = [&](const char * key, int def) {
                return req.form.has_field(key) ? atoi(req.form.get_field(key).c_str()) : def;
            };
            auto field_float = [&](const char * key, float def) {
                return req.form.has_field(key) ? (float) atof(req.form.get_field(key).c_str()) : def;
            };

            p.prompt    = field("prompt", std::string());
            n           = field_int("n", 1);
            model_name  = field("model", a.model_name);
            response_format = field("response_format", response_format);
            size        = field("size", size);
            p.seed      = field_int("seed", p.seed);
            p.steps     = field_int("num_steps", p.steps);
            p.cfg       = field_float("cfg_scale", p.cfg);
            p.img_cfg   = field_float("img_cfg_scale", p.img_cfg);
            p.cfg_norm  = field("cfg_norm", p.cfg_norm);
            p.shift     = field_float("timestep_shift", p.shift);
            p.input_max_pixels = field_int("input_max_pixels", p.input_max_pixels);
            p.auto_size_pixels = field_int("auto_pixels", p.auto_size_pixels);
            const std::string think = field("think", "false");
            p.think = think == "true" || think == "1";
            p.max_think_tokens = field_int("max_think_tokens", p.max_think_tokens);
            const std::string output_format = field("output_format", "png");
            if (output_format != "png") {
                json_error_openai(res, 400, "output_format: only png is supported");
                return;
            }

            for (const char * key : {"image", "image[]"}) {
                for (const auto & f : req.form.get_files(key)) {
                    sensenova_image_input img;
                    img.data.assign(f.content.begin(), f.content.end());
                    p.images.push_back(std::move(img));
                }
            }
        } else {
            try {
                const nlohmann::json j = nlohmann::json::parse(req.body);
                if (!j.is_object()) {
                    json_error_openai(res, 400, "request body must be a JSON object");
                    return;
                }
                p.prompt = j.value("prompt", std::string());
                n = j.value("n", 1);
                if (j.contains("model") && !j["model"].is_null()) {
                    model_name = j["model"].get<std::string>();
                }
                response_format = j.value("response_format", response_format);
                size = j.value("size", size);
                p.seed     = j.value("seed", p.seed);
                p.steps    = j.value("num_steps", p.steps);
                p.cfg      = j.value("cfg_scale", p.cfg);
                p.img_cfg  = j.value("img_cfg_scale", p.img_cfg);
                p.cfg_norm = j.value("cfg_norm", p.cfg_norm);
                p.shift    = j.value("timestep_shift", p.shift);
                p.input_max_pixels = j.value("input_max_pixels", p.input_max_pixels);
                p.auto_size_pixels = j.value("auto_pixels", p.auto_size_pixels);
                p.think    = j.value("think", p.think);
                p.max_think_tokens = j.value("max_think_tokens", p.max_think_tokens);
                const std::string output_format = j.value("output_format", std::string("png"));
                if (output_format != "png") {
                    json_error_openai(res, 400, "output_format: only png is supported");
                    return;
                }
                if (!set_json_images(j, p.images, err)) {
                    json_error_openai(res, 400, err);
                    return;
                }
            } catch (const std::exception & e) {
                json_error_openai(res, 400, std::string("invalid request: ") + e.what());
                return;
            }
        }

        if (p.prompt.empty()) {
            json_error_openai(res, 400, "prompt is required");
            return;
        }
        if (p.images.empty()) {
            json_error_openai(res, 400, "at least one input image is required");
            return;
        }
        if (n < 1 || n > 4) {
            json_error_openai(res, 400, "n must be between 1 and 4");
            return;
        }
        if (p.steps < 1 || p.steps > 100) {
            json_error_openai(res, 400, "num_steps must be between 1 and 100");
            return;
        }
        if (response_format != "b64_json" && response_format != "url") {
            json_error_openai(res, 400, "response_format must be b64_json or url");
            return;
        }
        if (p.cfg_norm != "none" && p.cfg_norm != "global" && p.cfg_norm != "channel") {
            json_error_openai(res, 400, "cfg_norm must be none, global or channel");
            return;
        }

        std::string s = size;
        for (char & c : s) {
            c = (char) tolower((unsigned char) c);
        }
        if (s.empty() || s == "auto") {
            p.width  = 0;
            p.height = 0;
        } else if (!parse_size(s, p.width, p.height, err)) {
            json_error_openai(res, 400, err);
            return;
        }

        nlohmann::json data = nlohmann::json::array();
        int64_t elapsed_ms = 0;
        for (int i = 0; i < n; ++i) {
            sensenova_gen_params q = p;
            if (p.seed < 0) {
                q.seed = random_seed();
            } else {
                q.seed = (int) (((int64_t) p.seed + i) & 0x7fffffff);
            }

            gen_output out;
            if (!generate_png(*model, mu, q, out, err)) {
                json_error_openai(res, 500, err, "server_error");
                return;
            }
            elapsed_ms += out.elapsed_ms;

            nlohmann::json item;
            if (response_format == "b64_json") {
                item["b64_json"] = base64_encode(out.png.data(), out.png.size());
            } else {
                const std::string name = "edit_" + random_hex(12) + ".png";
                if (!write_file(out_root / name, out.png, err)) {
                    json_error_openai(res, 500, err, "server_error");
                    return;
                }
                item["url"] = request_base_url(req, a) + "/images/" + name;
            }
            if (!out.think_text.empty()) {
                item["revised_prompt"] = out.think_text;
            }
            data.push_back(std::move(item));
        }

        openai_images_response(res, data, elapsed_ms, model_name);
    });

    // private protocol used by omni-note
    svr.Post("/v1/infographic/generate", [&](const httplib::Request & req, httplib::Response & res) {
        sensenova_gen_params p;
        std::string output_dir;
        std::string err;
        try {
            const nlohmann::json j = nlohmann::json::parse(req.body);
            if (!j.is_object()) {
                private_error(res, "request body must be a JSON object");
                return;
            }
            p.prompt = j.value("prompt", std::string());
            p.width  = j.value("width", p.width);
            p.height = j.value("height", p.height);
            p.seed   = j.value("seed", p.seed);
            p.steps  = j.value("num_steps", p.steps);
            p.cfg    = j.value("cfg_scale", p.cfg);
            p.img_cfg = j.value("img_cfg_scale", p.img_cfg);
            p.cfg_norm = j.value("cfg_norm", p.cfg_norm);
            p.shift  = j.value("timestep_shift", p.shift);
            p.think  = j.value("think", p.think);
            p.max_think_tokens = j.value("max_think_tokens", p.max_think_tokens);
            p.auto_size_pixels = j.value("auto_pixels", p.auto_size_pixels);
            output_dir = j.value("output_dir", std::string());
            if (!set_json_images(j, p.images, err)) {
                private_error(res, err);
                return;
            }
        } catch (const std::exception & e) {
            private_error(res, std::string("invalid request: ") + e.what());
            return;
        }

        if (p.prompt.empty()) {
            private_error(res, "prompt is required");
            return;
        }
        if ((p.width == 0) != (p.height == 0)) {
            private_error(res, "width and height must be set together");
            return;
        }
        if (p.width != 0 && p.width % 32 != 0) {
            private_error(res, "width and height must be multiples of 32");
            return;
        }
        if (p.width < 0 || p.height < 0) {
            private_error(res, "width and height must be positive");
            return;
        }
        if (p.steps < 1 || p.steps > 100) {
            private_error(res, "num_steps must be between 1 and 100");
            return;
        }
        if (!output_dir.empty() && !a.allow_output_dir) {
            private_error(res, "output_dir is disabled, start server with --allow-output-dir");
            return;
        }

        fs::path dir = out_root;
        if (!output_dir.empty()) {
            std::error_code dir_ec;
            dir = fs::absolute(output_dir, dir_ec).lexically_normal();
            if (dir_ec) {
                private_error(res, "invalid output_dir: " + dir_ec.message());
                return;
            }
            fs::create_directories(dir, dir_ec);
            if (dir_ec) {
                private_error(res, "cannot create output_dir: " + dir_ec.message());
                return;
            }
        }

        sensenova_gen_params q = p;
        if (q.seed < 0) {
            q.seed = random_seed();
        }

        gen_output out;
        if (!generate_png(*model, mu, q, out, err)) {
            private_error(res, err);
            return;
        }

        const std::string name = "infographic_" + random_hex(12) + ".png";
        if (!write_file(dir / name, out.png, err)) {
            private_error(res, err);
            return;
        }

        nlohmann::json j = {
            {"status", "success"},
            {"image_url", output_dir.empty() ? "/images/" + name : ""},
            {"image_path", (dir / name).string()},
            {"generation_time", out.elapsed_ms / 1000.0},
            {"width", out.width},
            {"height", out.height},
            {"message", ""},
        };
        if (p.think) {
            j["think"] = out.think_text;
        }
        res.set_header("X-Elapsed-Ms", std::to_string(out.elapsed_ms));
        res.set_content(j.dump(), "application/json");
    });

    // interleaved text and image generation
    svr.Post("/v1/interleave/generate", [&](const httplib::Request & req, httplib::Response & res) {
        sensenova_interleave_params p;
        std::string response_format = "b64_json";
        std::string output_dir;
        std::string model_name = a.model_name;
        std::string err;
        try {
            const nlohmann::json j = nlohmann::json::parse(req.body);
            if (!j.is_object()) {
                json_error_legacy(res, 400, "request body must be a JSON object");
                return;
            }
            p.prompt = j.value("prompt", std::string());
            if (j.contains("model") && j["model"].is_string()) {
                model_name = j["model"].get<std::string>();
            }
            p.width  = j.value("width", p.width);
            p.height = j.value("height", p.height);
            p.auto_pixels = j.value("auto_pixels", p.auto_pixels);
            p.steps  = j.value("num_steps", p.steps);
            p.seed   = j.value("seed", p.seed);
            p.shift  = j.value("timestep_shift", p.shift);
            p.max_images = j.value("max_images", p.max_images);
            p.max_new_tokens = j.value("max_new_tokens", p.max_new_tokens);
            p.think  = j.value("think", p.think);
            p.temperature = j.value("temperature", p.temperature);
            p.top_k = j.value("top_k", p.top_k);
            p.top_p = j.value("top_p", p.top_p);
            p.repeat_penalty = j.value("repetition_penalty", p.repeat_penalty);
            p.input_max_pixels = j.value("input_max_pixels", p.input_max_pixels);
            response_format = j.value("response_format", response_format);
            output_dir = j.value("output_dir", std::string());
            if (!set_json_images(j, p.images, err)) {
                json_error_legacy(res, 400, err);
                return;
            }
        } catch (const std::exception & e) {
            json_error_legacy(res, 400, std::string("invalid request: ") + e.what());
            return;
        }

        if (p.prompt.empty()) {
            json_error_legacy(res, 400, "prompt is required");
            return;
        }
        if ((p.width == 0) != (p.height == 0)) {
            json_error_legacy(res, 400, "width and height must be set together");
            return;
        }
        if (p.width != 0 && p.width % 32 != 0) {
            json_error_legacy(res, 400, "width and height must be multiples of 32");
            return;
        }
        if (p.width < 0 || p.height < 0) {
            json_error_legacy(res, 400, "width and height must be positive");
            return;
        }
        if (p.steps < 1 || p.steps > 100) {
            json_error_legacy(res, 400, "num_steps must be between 1 and 100");
            return;
        }
        if (p.max_images < 1 || p.max_images > 8) {
            json_error_legacy(res, 400, "max_images must be between 1 and 8");
            return;
        }
        if (p.max_new_tokens < 1 || p.max_new_tokens > 16384) {
            json_error_legacy(res, 400, "max_new_tokens must be between 1 and 16384");
            return;
        }
        if (p.temperature < 0.0f || p.top_k < 0 || p.top_p <= 0.0f || p.top_p > 1.0f ||
                p.repeat_penalty <= 0.0f) {
            json_error_legacy(res, 400, "invalid sampling parameters");
            return;
        }
        if (response_format != "b64_json" && response_format != "url") {
            json_error_legacy(res, 400, "response_format must be b64_json or url");
            return;
        }
        if (!output_dir.empty() && !a.allow_output_dir) {
            json_error_legacy(res, 400, "output_dir is disabled, start server with --allow-output-dir");
            return;
        }
        if (p.width == 0 && !p.images.empty() && p.auto_pixels < 512*512) {
            json_error_legacy(res, 400, "auto_pixels must be at least 262144");
            return;
        }

        fs::path dir = out_root;
        if (!output_dir.empty()) {
            std::error_code dir_ec;
            dir = fs::absolute(output_dir, dir_ec).lexically_normal();
            if (dir_ec) {
                json_error_legacy(res, 400, "invalid output_dir: " + dir_ec.message());
                return;
            }
            fs::create_directories(dir, dir_ec);
            if (dir_ec) {
                json_error_legacy(res, 400, "cannot create output_dir: " + dir_ec.message());
                return;
            }
        }

        sensenova_interleave_params q = p;
        if (q.seed < 0) {
            q.seed = random_seed();
        }

        sensenova_interleave_result out;
        const auto t0 = std::chrono::steady_clock::now();
        {
            std::lock_guard<std::mutex> lock(mu);
            if (!sensenova_interleave(*model, q, out, err)) {
                json_error_legacy(res, 500, err);
                return;
            }
        }
        const int64_t elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - t0).count();

        std::vector<gen_output> images(out.images.size());
        for (size_t i = 0; i < out.images.size(); ++i) {
            images[i].width  = out.images[i].width;
            images[i].height = out.images[i].height;
            if (!png_encode_rgb(out.images[i].rgb.data(), out.images[i].width, out.images[i].height,
                                images[i].png, err)) {
                json_error_legacy(res, 500, err);
                return;
            }
        }

        std::vector<std::string> locs;
        bool use_paths = false;
        if (response_format == "url") {
            use_paths = !output_dir.empty();
            const std::string stem = "interleave_" + random_hex(12);
            for (size_t i = 0; i < images.size(); ++i) {
                const std::string name = stem + "_" + std::to_string(i) + ".png";
                if (!write_file(dir / name, images[i].png, err)) {
                    json_error_legacy(res, 500, err);
                    return;
                }
                locs.push_back(use_paths ? (dir / name).string()
                                         : request_base_url(req, a) + "/images/" + name);
            }
        }

        nlohmann::json j = {
            {"id", "interleave-" + random_hex(16)},
            {"created", (int64_t) std::time(nullptr)},
            {"model", model_name},
            {"text", out.text},
            {"answer", out.answer},
            {"segments", interleave_segments(out.text, images, locs, use_paths)},
            {"finish_reason", out.truncated ? "length" : "stop"},
            {"usage", {
                {"prompt_tokens", out.n_prompt_tokens},
                {"text_tokens", out.n_generated},
                {"images", (int) out.images.size()},
                {"total_tokens", out.n_prompt_tokens + out.n_generated},
            }},
        };
        if (p.think) {
            j["think"] = out.think_text;
        }
        res.set_header("X-Elapsed-Ms", std::to_string(elapsed_ms));
        res.set_content(j.dump(), "application/json");
    });

    // model lifecycle hooks used by the omni-note manager
    svr.Post("/v1/model/load", [](const httplib::Request &, httplib::Response & res) {
        res.set_content(nlohmann::json({{"status", "success"}, {"message", "model already loaded"}}).dump(),
                        "application/json");
    });

    svr.Post("/v1/model/unload", [](const httplib::Request &, httplib::Response & res) {
        res.set_content(nlohmann::json({{"status", "error"}, {"message", "unload is not supported, stop the server to free VRAM"}}).dump(),
                        "application/json");
    });

    svr.set_logger([](const httplib::Request & req, const httplib::Response & res) {
        fprintf(stderr, "%s %s -> %d\n", req.method.c_str(), req.path.c_str(), res.status);
    });

    g_server = &svr;
    std::signal(SIGINT, on_signal);
    std::signal(SIGTERM, on_signal);

    fprintf(stderr, "listening on http://%s:%d (model: %s, output: %s)\n",
            a.host.c_str(), a.port, a.model_name.c_str(), out_root.string().c_str());
    if (!svr.listen(a.host, a.port)) {
        fprintf(stderr, "error: cannot listen on %s:%d\n", a.host.c_str(), a.port);
        return 1;
    }
    fprintf(stderr, "server stopped\n");
    return 0;
}
