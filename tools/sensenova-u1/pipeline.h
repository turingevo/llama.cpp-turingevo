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
    bool fm_cpu = false;

    // the FM graphs are built for one image size, so the sidecar is cached per size
    std::unique_ptr<fm_context> fm;
    int fm_w = 0;
    int fm_h = 0;

    ~sensenova_model();

    bool load_fm(int width, int height, std::string & err);

    // weights only, for input image encoding without generation graphs
    bool load_fm_und(std::string & err);
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

struct sensenova_qa_params {
    std::string question;
    int max_new_tokens = 1024;

    // 0 keeps greedy decoding, the other fields apply only when temperature > 0
    float temperature      = 0.0f;
    int   top_k            = 0;
    float top_p            = 0.95f;
    float repeat_penalty   = 1.0f;
    int   seed             = 42;

    // per input image, 0 = auto: min(2048^2, 4096^2/n)
    int input_max_pixels = 0;

    std::vector<sensenova_image_input> images;
};

struct sensenova_qa_result {
    std::string text;       // full reply, think block included
    std::string think_text; // reasoning text, empty when the reply has no think block
    std::string answer;     // reply with the think block removed
    int n_prompt_tokens = 0;
    int n_generated = 0;
    bool truncated = false;
};

struct sensenova_interleave_params {
    std::string prompt;
    std::string system_prompt; // empty keeps the built-in interleave system message

    int width  = 0; // 0 = auto: follow the first input image, else auto_pixels at 1:1
    int height = 0;
    int auto_pixels = 1024*1024;
    int steps  = 8;
    int seed   = 42;
    float shift = 3.0f;

    int max_images     = 3;
    int max_new_tokens = 4096;
    bool think = true; // let the model reason, images can then appear inside the think block

    // 0 keeps greedy decoding, the other fields apply only when temperature > 0
    float temperature    = 0.0f;
    int   top_k          = 0;
    float top_p          = 0.95f;
    float repeat_penalty = 1.0f;

    // per input image, 0 = auto: min(2048^2, 4096^2/n)
    int input_max_pixels = 0;

    std::vector<sensenova_image_input> images;
};

struct sensenova_interleave_result {
    std::string text;       // generated text, "<image>" marks each generated image
    std::string think_text; // reasoning text, empty when the reply has no think block
    std::string answer;     // text with the think block removed
    std::vector<sensenova_result> images;
    int n_prompt_tokens = 0;
    int n_generated = 0;
    bool truncated = false;
};

std::unique_ptr<sensenova_model> sensenova_model_load(
        const std::string & model_path,
        const std::string & fm_path,
        int ngl,
        int n_threads,
        std::string & err,
        const std::vector<std::string> & cpu_tensors = {});

bool sensenova_generate(
        sensenova_model & model,
        const sensenova_gen_params & params,
        sensenova_result & out,
        std::string & err);

// visual understanding: answer a question about the input images
bool sensenova_understand(
        sensenova_model & model,
        const sensenova_qa_params & params,
        sensenova_qa_result & out,
        std::string & err);

// interleaved text and image generation: the model emits text and <img> tags,
// each tag is replaced by a generated image that the model then reads back
bool sensenova_interleave(
        sensenova_model & model,
        const sensenova_interleave_params & params,
        sensenova_interleave_result & out,
        std::string & err);
