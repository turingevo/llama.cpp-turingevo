#pragma once

#include "fm.h"
#include "llama.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

// model and FM sidecar, loaded once and shared by all generations
struct sensenova_model {
    llama_model * model = nullptr;
    ggml_backend_t fm_backend = nullptr;
    std::string fm_path;
    int n_embd = 0;
    int ngl = 999;
    int n_threads = 0;

    // the FM graphs are built for one image size, so the sidecar is cached per size
    std::unique_ptr<fm_context> fm;
    int fm_w = 0;
    int fm_h = 0;

    ~sensenova_model();

    bool load_fm(int width, int height, std::string & err);
};

struct sensenova_image_input {
    std::string path;            // file path, or empty when data is set
    std::vector<uint8_t> data;   // encoded image bytes
};

struct sensenova_gen_params {
    std::string prompt;
    int width  = 0; // 0 = auto
    int height = 0;
    int steps  = 8;
    int seed   = 42;
    float cfg     = 1.0f;
    float img_cfg = 1.0f;
    float shift   = 3.0f;
    std::string cfg_norm = "none"; // none, global, channel

    bool think = false;
    int max_think_tokens = 1024;

    // per input image, 0 = auto: min(2048^2, 4096^2/n)
    int input_max_pixels = 0;

    // output pixels for auto size, which follows the first input aspect.
    // the reference uses 2048^2, which needs more VRAM than a 16 GB card
    // has with a Q5 model
    int auto_size_pixels = 1024*1024;

    std::vector<sensenova_image_input> images;
};

struct sensenova_result {
    std::vector<uint8_t> rgb; // [height][width][3]
    int width  = 0;
    int height = 0;
    std::string think_text;
};

std::unique_ptr<sensenova_model> sensenova_model_load(
        const std::string & model_path,
        const std::string & fm_path,
        int ngl,
        int n_threads,
        std::string & err);

bool sensenova_generate(
        sensenova_model & model,
        const sensenova_gen_params & params,
        sensenova_result & out,
        std::string & err);
