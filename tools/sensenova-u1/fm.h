#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#include <cstdint>
#include <string>
#include <vector>

// parameters read from the FM sidecar GGUF
struct fm_params {
    bool head_conv = true;
    int patch_size = 16;
    int merge_size = 2;
    int n_embd = 4096;
    int vision_n_embd = 1024;
    float rope_freq_base = 10000.0f;

    float noise_scale = 1.0f;
    int noise_scale_base_seq_len = 64;
    float noise_scale_max = 16.0f;
    bool add_noise_scale_emb = true;
    std::string noise_scale_mode = "resolution";
};

// vision patch embedder graph for one image size
struct fm_vision_encoder {
    int img_w = 0;
    int img_h = 0;

    ggml_context * ctx = nullptr;
    ggml_cgraph  * gf  = nullptr;
    ggml_gallocr_t ga  = nullptr;
    ggml_tensor  * in  = nullptr;
    ggml_tensor  * out = nullptr;

    ggml_context * ctx_pos = nullptr;
    ggml_backend_buffer_t buf_pos = nullptr;
    ggml_tensor * pos_x = nullptr;
    ggml_tensor * pos_y = nullptr;

    ~fm_vision_encoder();

    bool build(ggml_backend_t backend, const fm_params & p, int img_w, int img_h,
            ggml_tensor * patch_w, ggml_tensor * patch_b,
            ggml_tensor * dense_w, ggml_tensor * dense_b,
            std::string & err);

    // encode img [img_w, img_h, 3] to out [n_embd, img_w/2, img_h/2]
    bool encode(ggml_backend_t backend, const float * img, float * out_t, std::string & err);
};

// vision patch embedder, timestep embedders and pixel head of SenseNova-U1
struct fm_context {
    fm_params p;

    ggml_backend_t backend = nullptr;
    ggml_context * ctx_w   = nullptr;
    ggml_backend_buffer_t buf_w = nullptr;

    // generation branch patch embedder
    ggml_tensor * patch_w = nullptr;
    ggml_tensor * patch_b = nullptr;
    ggml_tensor * dense_w = nullptr;
    ggml_tensor * dense_b = nullptr;

    // understanding branch patch embedder, used for input images
    ggml_tensor * patch_w_und = nullptr;
    ggml_tensor * patch_b_und = nullptr;
    ggml_tensor * dense_w_und = nullptr;
    ggml_tensor * dense_b_und = nullptr;

    ggml_tensor * ts0_w = nullptr;
    ggml_tensor * ts0_b = nullptr;
    ggml_tensor * ts2_w = nullptr;
    ggml_tensor * ts2_b = nullptr;
    ggml_tensor * ns0_w = nullptr;
    ggml_tensor * ns0_b = nullptr;
    ggml_tensor * ns2_w = nullptr;
    ggml_tensor * ns2_b = nullptr;

    ggml_tensor * conv1_w = nullptr;
    ggml_tensor * conv1_b = nullptr;
    ggml_tensor * conv2_w = nullptr;
    ggml_tensor * conv2_b = nullptr;

    ggml_tensor * fc0_w = nullptr;
    ggml_tensor * fc0_b = nullptr;
    ggml_tensor * fc2_w = nullptr;
    ggml_tensor * fc2_b = nullptr;

    // encode the generated image at the output size
    fm_vision_encoder enc_gen;

    // timestep / noise scale embedders: 2 x [256] -> [n_embd]
    ggml_context * ctx_ts   = nullptr;
    ggml_cgraph  * gf_ts    = nullptr;
    ggml_gallocr_t ga_ts    = nullptr;
    ggml_tensor  * in_ts    = nullptr;
    ggml_tensor  * in_ns    = nullptr;
    ggml_tensor  * out_ts   = nullptr;

    // conv pixel head, layer 1 (ps1 is done on the host)
    ggml_context * ctx_h1   = nullptr;
    ggml_cgraph  * gf_h1    = nullptr;
    ggml_gallocr_t ga_h1    = nullptr;
    ggml_tensor  * in_h1    = nullptr;
    ggml_tensor  * out_h1   = nullptr;

    // conv pixel head, layer 2 (ps2 is done on the host)
    ggml_context * ctx_h2   = nullptr;
    ggml_cgraph  * gf_h2    = nullptr;
    ggml_gallocr_t ga_h2    = nullptr;
    ggml_tensor  * in_h2    = nullptr;
    ggml_tensor  * out_h2   = nullptr;

    // linear pixel head
    ggml_context * ctx_hl   = nullptr;
    ggml_cgraph  * gf_hl    = nullptr;
    ggml_gallocr_t ga_hl    = nullptr;
    ggml_tensor  * in_hl    = nullptr;
    ggml_tensor  * out_hl   = nullptr;

    bool load(const std::string & path, ggml_backend_t backend, int img_w, int img_h, std::string & err);
    ~fm_context();

    bool has_und_vision() const { return patch_w_und != nullptr; }

    bool encode_image(const float * img, float * out, std::string & err);

    // encode an input image with the understanding branch embedder,
    // the encoder graph is built and freed per call
    bool encode_ref_image(const float * img, int img_w, int img_h, float * out, std::string & err);

    bool embed_cond(float t, float noise_ratio, float * out, std::string & err);
    bool head_conv1(const float * in, float * out, std::string & err);
    bool head_conv2(const float * in, float * out, std::string & err);
    bool head_linear(const float * in, int n_tokens, float * out, std::string & err);
};
