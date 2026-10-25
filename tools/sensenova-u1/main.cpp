#include "pipeline.h"
#include "png.h"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

struct gen_args {
    std::string model;
    std::string fm;
    std::string prompt;
    std::string output = "sensenova-u1.png";
    std::vector<std::string> images;
    int width  = 0;
    int height = 0;
    int steps  = 8;
    int seed   = 42;
    int ngl    = 999;
    int threads = 0;
    float cfg   = 1.0f;
    float img_cfg = 1.0f;
    float shift = 3.0f;
    std::string cfg_norm = "none";
    bool think = false;
    bool print_think = false;
    std::string think_output;
    int max_think_tokens = 1024;
    int input_max_pixels = 0;
    int auto_size_pixels = 1024*1024;
};

static void print_usage(const char * prog) {
    fprintf(stderr,
            "usage: %s -m MODEL.gguf --fm FM.gguf -p PROMPT [options]\n"
            "\n"
            "options:\n"
            "  -m, --model PATH     language model GGUF (required)\n"
            "      --fm PATH        FM sidecar GGUF (required)\n"
            "  -p, --prompt TEXT    prompt (required)\n"
            "  -i, --image PATH     input image for editing, repeatable\n"
            "  -o, --output PATH    output PNG (default: sensenova-u1.png)\n"
            "      --width N        image width, multiple of 32 (default: auto)\n"
            "      --height N       image height, multiple of 32 (default: auto)\n"
            "      --steps N        denoising steps (default: 8)\n"
            "      --cfg F          classifier-free guidance scale (default: 1.0)\n"
            "      --img-cfg F      image guidance scale for editing (default: 1.0)\n"
            "      --cfg-norm MODE  clamp the CFG norm: none, global, channel (default: none)\n"
            "      --shift F        timestep shift (default: 3.0)\n"
            "      --seed N         random seed (default: 42)\n"
            "      --think          generate a think block before denoising\n"
            "      --max-think-tokens N  think budget (default: 1024)\n"
            "      --print-think    print the think text to stdout\n"
            "      --think-output PATH   save the think text to a file\n"
            "      --input-max-pixels N  per input image budget (default: auto)\n"
            "      --auto-pixels N  output pixels when width and height are auto (default: 1048576)\n"
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
        } else if (arg == "-i" || arg == "--image") {
            const char * v = next(arg.c_str()); if (!v) return false; a.images.push_back(v);
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
        } else if (arg == "--img-cfg") {
            const char * v = next(arg.c_str()); if (!v) return false; a.img_cfg = (float) atof(v);
        } else if (arg == "--cfg-norm") {
            const char * v = next(arg.c_str()); if (!v) return false; a.cfg_norm = v;
        } else if (arg == "--shift") {
            const char * v = next(arg.c_str()); if (!v) return false; a.shift = (float) atof(v);
        } else if (arg == "--seed") {
            const char * v = next(arg.c_str()); if (!v) return false; a.seed = atoi(v);
        } else if (arg == "--think") {
            a.think = true;
        } else if (arg == "--max-think-tokens") {
            const char * v = next(arg.c_str()); if (!v) return false; a.max_think_tokens = atoi(v);
        } else if (arg == "--print-think") {
            a.print_think = true;
        } else if (arg == "--think-output") {
            const char * v = next(arg.c_str()); if (!v) return false; a.think_output = v;
        } else if (arg == "--input-max-pixels") {
            const char * v = next(arg.c_str()); if (!v) return false; a.input_max_pixels = atoi(v);
        } else if (arg == "--auto-pixels") {
            const char * v = next(arg.c_str()); if (!v) return false; a.auto_size_pixels = atoi(v);
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
    if (a.max_think_tokens <= 0) {
        fprintf(stderr, "error: max think tokens must be positive\n");
        return false;
    }
    return true;
}

int main(int argc, char ** argv) {
    gen_args a;
    if (!parse_args(argc, argv, a)) {
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
    p.img_cfg = a.img_cfg;
    p.shift  = a.shift;
    p.cfg_norm = a.cfg_norm;
    p.think  = a.think;
    p.max_think_tokens = a.max_think_tokens;
    p.input_max_pixels = a.input_max_pixels;
    p.auto_size_pixels = a.auto_size_pixels;
    for (const std::string & path : a.images) {
        sensenova_image_input in;
        in.path = path;
        p.images.push_back(std::move(in));
    }

    sensenova_result res;
    if (!sensenova_generate(*model, p, res, err)) {
        fprintf(stderr, "error: %s\n", err.c_str());
        return 1;
    }

    if (a.print_think) {
        printf("%s\n", res.think_text.c_str());
    }
    if (!a.think_output.empty()) {
        std::ofstream f(a.think_output);
        if (!f) {
            fprintf(stderr, "error: cannot write %s\n", a.think_output.c_str());
            return 1;
        }
        f << res.think_text;
    }

    if (!png_write_rgb(a.output, res.rgb.data(), res.width, res.height, err)) {
        fprintf(stderr, "error: cannot write %s: %s\n", a.output.c_str(), err.c_str());
        return 1;
    }
    fprintf(stderr, "wrote %s\n", a.output.c_str());
    return 0;
}
