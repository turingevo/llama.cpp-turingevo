#include "pipeline.h"
#include "png.h"

#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

struct gen_args {
    std::string model;
    std::string fm;
    std::string prompt;
    std::string output = "sensenova-u1.png";
    int width  = 1024;
    int height = 1024;
    int steps  = 8;
    int seed   = 42;
    int ngl    = 999;
    int threads = 0;
    float cfg   = 1.0f;
    float shift = 3.0f;
};

static void print_usage(const char * prog) {
    fprintf(stderr,
            "usage: %s -m MODEL.gguf --fm FM.gguf -p PROMPT [options]\n"
            "\n"
            "options:\n"
            "  -m, --model PATH     language model GGUF (required)\n"
            "      --fm PATH        FM sidecar GGUF (required)\n"
            "  -p, --prompt TEXT    prompt (required)\n"
            "  -o, --output PATH    output PNG (default: sensenova-u1.png)\n"
            "      --width N        image width, multiple of 32 (default: 1024)\n"
            "      --height N       image height, multiple of 32 (default: 1024)\n"
            "      --steps N        denoising steps (default: 8)\n"
            "      --cfg F          classifier-free guidance scale (default: 1.0)\n"
            "      --shift F        timestep shift (default: 3.0)\n"
            "      --seed N         random seed (default: 42)\n"
            "  -ngl N               layers to offload to GPU (default: 999)\n"
            "  -t, --threads N      CPU threads (default: number of cores)\n",
            prog);
}

static bool parse_args(int argc, char ** argv, gen_args & a) {
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
        } else if (arg == "-p" || arg == "--prompt") {
            const char * v = next(arg.c_str()); if (!v) return false; a.prompt = v;
        } else if (arg == "-o" || arg == "--output") {
            const char * v = next(arg.c_str()); if (!v) return false; a.output = v;
        } else if (arg == "--width") {
            const char * v = next(arg.c_str()); if (!v) return false; a.width = atoi(v);
        } else if (arg == "--height") {
            const char * v = next(arg.c_str()); if (!v) return false; a.height = atoi(v);
        } else if (arg == "--steps") {
            const char * v = next(arg.c_str()); if (!v) return false; a.steps = atoi(v);
        } else if (arg == "--cfg") {
            const char * v = next(arg.c_str()); if (!v) return false; a.cfg = (float) atof(v);
        } else if (arg == "--shift") {
            const char * v = next(arg.c_str()); if (!v) return false; a.shift = (float) atof(v);
        } else if (arg == "--seed") {
            const char * v = next(arg.c_str()); if (!v) return false; a.seed = atoi(v);
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

    if (a.model.empty() || a.fm.empty() || a.prompt.empty()) {
        print_usage(argv[0]);
        return false;
    }
    return true;
}

int main(int argc, char ** argv) {
    gen_args a;
    if (!parse_args(argc, argv, a)) {
        return 1;
    }
    if (a.width % 32 != 0 || a.height % 32 != 0 || a.width <= 0 || a.height <= 0) {
        fprintf(stderr, "error: width and height must be positive multiples of 32\n");
        return 1;
    }
    if (a.steps <= 0) {
        fprintf(stderr, "error: steps must be positive\n");
        return 1;
    }

    std::string err;
    std::unique_ptr<sensenova_model> model = sensenova_model_load(a.model, a.fm, a.ngl, a.threads, err);
    if (!model) {
        fprintf(stderr, "error: %s\n", err.c_str());
        return 1;
    }

    sensenova_gen_params p;
    p.prompt = a.prompt;
    p.width  = a.width;
    p.height = a.height;
    p.steps  = a.steps;
    p.seed   = a.seed;
    p.cfg    = a.cfg;
    p.shift  = a.shift;

    std::vector<uint8_t> rgb;
    if (!sensenova_generate(*model, p, rgb, err)) {
        fprintf(stderr, "error: %s\n", err.c_str());
        return 1;
    }

    if (!png_write_rgb(a.output, rgb.data(), a.width, a.height, err)) {
        fprintf(stderr, "error: cannot write %s: %s\n", a.output.c_str(), err.c_str());
        return 1;
    }
    fprintf(stderr, "wrote %s\n", a.output.c_str());
    return 0;
}
