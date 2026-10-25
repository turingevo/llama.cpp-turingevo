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

struct sensenova_gen_params {
    std::string prompt;
    int width  = 1024;
    int height = 1024;
    int steps  = 8;
    int seed   = 42;
    float cfg   = 1.0f;
    float shift = 3.0f;
};

std::unique_ptr<sensenova_model> sensenova_model_load(
        const std::string & model_path,
        const std::string & fm_path,
        int ngl,
        int n_threads,
        std::string & err);

// rgb is [height][width][3]
bool sensenova_generate(
        sensenova_model & model,
        const sensenova_gen_params & params,
        std::vector<uint8_t> & rgb,
        std::string & err);
