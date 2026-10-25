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
            "  POST /v1/images/generations   OpenAI Images API\n"
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

static bool generate_png(
        sensenova_model & model,
        std::mutex & mu,
        const sensenova_gen_params & p,
        std::vector<uint8_t> & png,
        int64_t & elapsed_ms,
        std::string & err) {
    std::vector<uint8_t> rgb;
    const auto t0 = std::chrono::steady_clock::now();
    {
        std::lock_guard<std::mutex> lock(mu);
        if (!sensenova_generate(model, p, rgb, err)) {
            return false;
        }
    }
    if (!png_encode_rgb(rgb.data(), p.width, p.height, png, err)) {
        return false;
    }
    elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
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

    svr.set_payload_max_length(1 << 20);
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
        j["defaults"] = {
            {"width", 1024}, {"height", 1024}, {"steps", 8},
            {"seed", 42}, {"cfg", 1.0}, {"shift", 3.0},
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
            p.shift  = j.value("shift", p.shift);
            response_format = j.value("response_format", response_format);
        } catch (const std::exception & e) {
            json_error_legacy(res, 400, std::string("invalid request: ") + e.what());
            return;
        }

        if (p.prompt.empty()) {
            json_error_legacy(res, 400, "prompt is required");
            return;
        }
        if (p.width <= 0 || p.height <= 0 || p.width % 32 != 0 || p.height % 32 != 0) {
            json_error_legacy(res, 400, "width and height must be positive multiples of 32");
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

        std::vector<uint8_t> png;
        int64_t elapsed_ms = 0;
        if (!generate_png(*model, mu, p, png, elapsed_ms, err)) {
            json_error_legacy(res, 500, err);
            return;
        }

        if (response_format == "b64_json") {
            nlohmann::json j;
            j["image"]      = base64_encode(png.data(), png.size());
            j["width"]      = p.width;
            j["height"]     = p.height;
            j["steps"]      = p.steps;
            j["seed"]       = p.seed;
            j["elapsed_ms"] = elapsed_ms;
            res.set_content(j.dump(), "application/json");
        } else {
            res.set_header("X-Elapsed-Ms", std::to_string(elapsed_ms));
            res.set_content(reinterpret_cast<const char *>(png.data()), png.size(), "image/png");
        }
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
            if (j.contains("cfg_norm") && !j["cfg_norm"].is_null()) {
                if (j["cfg_norm"].get<std::string>() != "none") {
                    json_error_openai(res, 400, "cfg_norm: only none is supported");
                    return;
                }
            }
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

        nlohmann::json data = nlohmann::json::array();
        int64_t elapsed_ms = 0;
        for (int i = 0; i < n; ++i) {
            sensenova_gen_params q = p;
            if (p.seed < 0) {
                q.seed = random_seed();
            } else {
                q.seed = (int) (((int64_t) p.seed + i) & 0x7fffffff);
            }

            std::vector<uint8_t> png;
            int64_t item_ms = 0;
            if (!generate_png(*model, mu, q, png, item_ms, err)) {
                json_error_openai(res, 500, err, "server_error");
                return;
            }
            elapsed_ms += item_ms;

            if (response_format == "b64_json") {
                data.push_back({{"b64_json", base64_encode(png.data(), png.size())}});
            } else {
                const std::string name = "openai_" + random_hex(12) + ".png";
                if (!write_file(out_root / name, png, err)) {
                    json_error_openai(res, 500, err, "server_error");
                    return;
                }
                data.push_back({{"url", request_base_url(req, a) + "/images/" + name}});
            }
        }

        nlohmann::json out = {
            {"created", (int64_t) std::time(nullptr)},
            {"data", data},
            {"model", model_name},
        };
        res.set_header("X-Elapsed-Ms", std::to_string(elapsed_ms));
        res.set_content(out.dump(), "application/json");
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
            p.shift  = j.value("timestep_shift", p.shift);
            output_dir = j.value("output_dir", std::string());
            if (j.contains("cfg_norm") && !j["cfg_norm"].is_null()) {
                if (j["cfg_norm"].get<std::string>() != "none") {
                    private_error(res, "cfg_norm: only none is supported");
                    return;
                }
            }
        } catch (const std::exception & e) {
            private_error(res, std::string("invalid request: ") + e.what());
            return;
        }

        if (p.prompt.empty()) {
            private_error(res, "prompt is required");
            return;
        }
        if (p.width <= 0 || p.height <= 0 || p.width % 32 != 0 || p.height % 32 != 0) {
            private_error(res, "width and height must be positive multiples of 32");
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

        std::vector<uint8_t> png;
        int64_t elapsed_ms = 0;
        if (!generate_png(*model, mu, q, png, elapsed_ms, err)) {
            private_error(res, err);
            return;
        }

        const std::string name = "infographic_" + random_hex(12) + ".png";
        if (!write_file(dir / name, png, err)) {
            private_error(res, err);
            return;
        }

        nlohmann::json out = {
            {"status", "success"},
            {"image_url", output_dir.empty() ? "/images/" + name : ""},
            {"image_path", (dir / name).string()},
            {"generation_time", elapsed_ms / 1000.0},
            {"width", p.width},
            {"height", p.height},
            {"message", ""},
        };
        res.set_header("X-Elapsed-Ms", std::to_string(elapsed_ms));
        res.set_content(out.dump(), "application/json");
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
