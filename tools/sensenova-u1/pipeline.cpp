#include "pipeline.h"

#include "ggml-backend.h"
#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

// system prompt used by the text-to-image branch of the reference implementation
static const char * SYSTEM_MESSAGE_FOR_GEN =
    "You are an image generation and editing assistant that accurately understands and executes "
    "user intent.\n\nYou support two modes:\n\n1. Think Mode:\nIf the task requires reasoning, you "
    "MUST start with a  thinking</think> block. Put all reasoning inside the block using plain text. "
    "DO NOT include any image tags. Keep it reasonable and directly useful for producing the final "
    "image.\n\n2. Non-Think Mode:\nIf no reasoning is needed, directly produce the final image.\n\n"
    "Task Types:\n\nA. Text-to-Image Generation:\n"
    "- Generate a high-quality image based on the user's description.\n"
    "- Ensure visual clarity, semantic consistency, and completeness.\n"
    "- DO NOT introduce elements that contradict or override the user's intent.\n\n"
    "B. Image Editing:\n"
    "- Use the provided image(s) as input or reference for modification or transformation.\n"
    "- The result can be an edited image or a new image based on the reference(s).\n"
    "- Preserve all unspecified attributes unless explicitly changed.\n\n"
    "General Rules:\n"
    "- For any visible text in the image, follow the language specified for the rendered text in the "
    "user's description, not the language of the prompt. If no language is specified, use the user's "
    "input language.";

// deterministic normal distribution, independent of the standard library
struct rng_normal {
    uint64_t state;

    explicit rng_normal(uint64_t seed) : state(seed ? seed : 0x9e3779b97f4a7c15ull) {}

    uint64_t next_u64() {
        uint64_t z = (state += 0x9e3779b97f4a7c15ull);
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
        return z ^ (z >> 31);
    }

    float next() {
        // Box-Muller, using the top 53 bits for the uniform samples
        const double u1 = ((next_u64() >> 11) + 1.0) / 9007199254740994.0;
        const double u2 = ((next_u64() >> 11) + 0.5) / 9007199254740992.0;
        return (float) (sqrt(-2.0*log(u1))*cos(2.0*M_PI*u2));
    }
};

static std::string build_cond_prompt(const std::string & prompt) {
    std::string q;
    q += "<|im_start|>system\n";
    q += SYSTEM_MESSAGE_FOR_GEN;
    q += "<|im_end|>\n";
    q += "<|im_start|>user\n";
    q += prompt;
    q += "<|im_end|>\n";
    q += "<|im_start|>assistant\n";
    q += " thinking\n\n</think>\n\n<img>";
    return q;
}

static std::string build_uncond_prompt() {
    return "<|im_start|>user\n<|im_end|>\n<|im_start|>assistant\n<img>";
}

static bool tokenize_prompt(
        const llama_vocab * vocab,
        const std::string & text,
        std::vector<llama_token> & out) {
    const int n = -llama_tokenize(vocab, text.c_str(), (int32_t) text.size(), nullptr, 0, false, true);
    if (n <= 0) {
        fprintf(stderr, "error: cannot tokenize prompt\n");
        return false;
    }
    out.resize(n);
    if (llama_tokenize(vocab, text.c_str(), (int32_t) text.size(), out.data(), n, false, true) < 0) {
        fprintf(stderr, "error: cannot tokenize prompt\n");
        return false;
    }
    return true;
}

// decode a prefix, leaving its KV in the cache of the given sequence
static bool decode_prefix(llama_context * ctx, llama_seq_id seq, const std::vector<llama_token> & tokens) {
    std::vector<llama_pos>      pos(tokens.size());
    std::vector<int32_t>        n_seq_id(tokens.size(), 1);
    std::vector<llama_seq_id *> seq_id(tokens.size());
    std::vector<int8_t>         logits(tokens.size(), 0);

    for (size_t i = 0; i < tokens.size(); ++i) {
        pos[i]    = (llama_pos) i;
        seq_id[i] = &seq;
    }

    llama_batch batch = {};
    batch.n_tokens = (int32_t) tokens.size();
    batch.token    = const_cast<llama_token *>(tokens.data());
    batch.pos      = pos.data();
    batch.n_seq_id = n_seq_id.data();
    batch.seq_id   = seq_id.data();
    batch.logits   = logits.data();

    return llama_decode(ctx, batch) == 0;
}

// run one generation pass: image embeddings -> last hidden state, in place
static bool decode_image(
        llama_context * ctx,
        llama_seq_id seq,
        int prefix_len,
        int token_h,
        int token_w,
        const float * embeds,
        float * out,
        int n_embd) {
    const int n_tokens = token_h*token_w;

    std::vector<llama_pos>      pos((size_t) n_tokens*4);
    std::vector<int32_t>        n_seq_id(n_tokens, 1);
    std::vector<llama_seq_id *> seq_id(n_tokens);
    std::vector<int8_t>         logits(n_tokens, 1);

    for (int i = 0; i < n_tokens; ++i) {
        pos[(size_t) 0*n_tokens + i] = prefix_len;
        pos[(size_t) 1*n_tokens + i] = i / token_w;
        pos[(size_t) 2*n_tokens + i] = i % token_w;
        pos[(size_t) 3*n_tokens + i] = 0;
        seq_id[i] = &seq;
    }

    llama_batch batch = {};
    batch.n_tokens = n_tokens;
    batch.embd     = const_cast<float *>(embeds);
    batch.pos      = pos.data();
    batch.n_seq_id = n_seq_id.data();
    batch.seq_id   = seq_id.data();
    batch.logits   = logits.data();

    if (llama_decode(ctx, batch) != 0) {
        return false;
    }

    const float * e = llama_get_embeddings(ctx);
    if (e == nullptr) {
        fprintf(stderr, "error: no embeddings returned by the generation pass\n");
        return false;
    }
    memcpy(out, e, (size_t) n_tokens*n_embd*sizeof(float));
    return true;
}

// channel-major [c][h][w] -> interleaved [w][h][c], as expected by the vision patch embedder
static void to_whc(const std::vector<float> & img, std::vector<float> & out, int w, int h) {
    for (int c = 0; c < 3; ++c) {
        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                out[(size_t) x + (size_t) w*(y + (size_t) h*c)] = img[(size_t) c*w*h + (size_t) y*w + x];
            }
        }
    }
}

// pixel shuffle by r on a channel-major [c][h][w] buffer
static void pixel_shuffle(const float * in, std::vector<float> & out, int c, int h, int w, int r) {
    const int oc = c/(r*r);
    const int oh = h*r;
    const int ow = w*r;
    out.resize((size_t) oc*oh*ow);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            for (int dy = 0; dy < r; ++dy) {
                for (int dx = 0; dx < r; ++dx) {
                    const int oy = y*r + dy;
                    const int ox = x*r + dx;
                    for (int oc_i = 0; oc_i < oc; ++oc_i) {
                        const int ic = oc_i*r*r + dy*r + dx;
                        out[(size_t) oc_i*oh*ow + (size_t) oy*ow + ox] = in[(size_t) ic*h*w + (size_t) y*w + x];
                    }
                }
            }
        }
    }
}

// gather the last hidden state into the pixel head input layout: [c][h][w]
static void embeds_to_head(const float * embeds, std::vector<float> & out, int n_embd, int token_h, int token_w) {
    out.resize((size_t) n_embd*token_h*token_w);
    for (int j = 0; j < token_h*token_w; ++j) {
        const int y = j / token_w;
        const int x = j % token_w;
        for (int c = 0; c < n_embd; ++c) {
            out[(size_t) c*token_h*token_w + (size_t) y*token_w + x] = embeds[(size_t) j*n_embd + c];
        }
    }
}

static bool run_head(
        fm_context & fm,
        const float * embeds,
        std::vector<float> & pixels,
        int H,
        int W,
        std::string & err) {
    const int token_h = H/32;
    const int token_w = W/32;
    const int n_embd  = fm.p.n_embd;

    if (!fm.p.head_conv) {
        std::vector<float> x_pred((size_t) 3072*token_h*token_w);
        if (!fm.head_linear(embeds, token_h*token_w, x_pred.data(), err)) {
            return false;
        }
        pixels.resize((size_t) 3*H*W);
        for (int j = 0; j < token_h*token_w; ++j) {
            const int th = j / token_w;
            const int tw = j % token_w;
            for (int p = 0; p < 32; ++p) {
                for (int q = 0; q < 32; ++q) {
                    for (int c = 0; c < 3; ++c) {
                        pixels[(size_t) c*H*W + (size_t) (th*32 + p)*W + (tw*32 + q)] =
                            x_pred[(size_t) j*3072 + (size_t) (p*32 + q)*3 + c];
                    }
                }
            }
        }
        return true;
    }

    std::vector<float> cur;
    embeds_to_head(embeds, cur, n_embd, token_h, token_w);

    std::vector<float> shuffled;
    pixel_shuffle(cur.data(), shuffled, n_embd, token_h, token_w, 2);

    std::vector<float> h1((size_t) (n_embd/4)*(2*token_h)*(2*token_w));
    if (!fm.head_conv1(shuffled.data(), h1.data(), err)) {
        return false;
    }

    pixel_shuffle(h1.data(), shuffled, n_embd/4, 2*token_h, 2*token_w, 2);

    std::vector<float> h2((size_t) 192*(4*token_h)*(4*token_w));
    if (!fm.head_conv2(shuffled.data(), h2.data(), err)) {
        return false;
    }

    // final pixel shuffle by 8: [192][4h][4w] -> [3][32h][32w]
    const int sh = 4*token_h;
    const int sw = 4*token_w;
    pixels.resize((size_t) 3*H*W);
    for (int y = 0; y < sh; ++y) {
        for (int x = 0; x < sw; ++x) {
            for (int dy = 0; dy < 8; ++dy) {
                for (int dx = 0; dx < 8; ++dx) {
                    const int oy = y*8 + dy;
                    const int ox = x*8 + dx;
                    for (int c = 0; c < 3; ++c) {
                        const int ic = c*64 + dy*8 + dx;
                        pixels[(size_t) c*H*W + (size_t) oy*W + ox] = h2[(size_t) ic*sh*sw + (size_t) y*sw + x];
                    }
                }
            }
        }
    }
    return true;
}

sensenova_model::~sensenova_model() {
    // released before the backend, as the FM buffers live on it
    fm.reset();
    if (fm_backend != nullptr) {
        ggml_backend_free(fm_backend);
    }
    if (model != nullptr) {
        llama_model_free(model);
    }
}

bool sensenova_model::load_fm(int width, int height, std::string & err) {
    if (fm && fm_w == width && fm_h == height) {
        return true;
    }

    if (fm_backend == nullptr) {
        ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
        if (dev == nullptr) {
            dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        }
        if (dev == nullptr) {
            err = "no backend device available";
            return false;
        }
        fm_backend = ggml_backend_dev_init(dev, nullptr);
        if (fm_backend == nullptr) {
            err = "cannot create FM backend";
            return false;
        }
    }

    fm.reset();
    fm_w = 0;
    fm_h = 0;

    auto next = std::make_unique<fm_context>();
    if (!next->load(fm_path, fm_backend, width, height, err)) {
        return false;
    }
    if (next->p.n_embd != n_embd) {
        err = "embedding size mismatch: model " + std::to_string(n_embd) +
              ", FM " + std::to_string(next->p.n_embd);
        return false;
    }

    fm = std::move(next);
    fm_w = width;
    fm_h = height;
    return true;
}

std::unique_ptr<sensenova_model> sensenova_model_load(
        const std::string & model_path,
        const std::string & fm_path,
        int ngl,
        int n_threads,
        std::string & err) {
    llama_backend_init();
    ggml_backend_load_all();

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = ngl;

    llama_model * model = llama_model_load_from_file(model_path.c_str(), mparams);
    if (model == nullptr) {
        err = "cannot load model " + model_path;
        return nullptr;
    }

    auto out = std::make_unique<sensenova_model>();
    out->model = model;
    out->fm_path = fm_path;
    out->n_embd = llama_model_n_embd(model);
    out->ngl = ngl;
    out->n_threads = n_threads;
    return out;
}

bool sensenova_generate(
        sensenova_model & model,
        const sensenova_gen_params & p,
        std::vector<uint8_t> & rgb,
        std::string & err) {
    if (p.prompt.empty()) {
        err = "prompt is empty";
        return false;
    }
    if (p.width <= 0 || p.height <= 0 || p.width % 32 != 0 || p.height % 32 != 0) {
        err = "width and height must be positive multiples of 32";
        return false;
    }
    if (p.steps <= 0) {
        err = "steps must be positive";
        return false;
    }

    const int token_h = p.height/32;
    const int token_w = p.width/32;
    const int n_tokens = token_h*token_w;
    const int n_patch  = (p.height/16)*(p.width/16);

    const llama_vocab * vocab = llama_model_get_vocab(model.model);
    const int n_embd = model.n_embd;

    std::vector<llama_token> cond;
    std::vector<llama_token> uncond;
    if (!tokenize_prompt(vocab, build_cond_prompt(p.prompt), cond)) {
        err = "cannot tokenize prompt";
        return false;
    }
    const bool needs_cfg = p.cfg > 1.0f;
    if (needs_cfg && !tokenize_prompt(vocab, build_uncond_prompt(), uncond)) {
        err = "cannot tokenize prompt";
        return false;
    }

    fprintf(stderr, "prompt tokens: %zu, image tokens: %d (%dx%d), steps: %d, cfg: %.2f\n",
            cond.size(), n_tokens, p.width, p.height, p.steps, p.cfg);

    llama_context_params cparams = llama_context_default_params();
    const uint32_t n_batch = (uint32_t) std::max(n_tokens, (int) std::max(cond.size(), uncond.size()));

    cparams.n_ctx      = (uint32_t) (cond.size() + uncond.size() + n_tokens + 16);
    cparams.n_batch    = n_batch;
    cparams.n_ubatch   = n_batch;
    cparams.n_seq_max  = 2;
    cparams.kv_unified = true;
    cparams.embeddings = true;
    cparams.no_perf    = false;
    if (model.n_threads > 0) {
        cparams.n_threads       = model.n_threads;
        cparams.n_threads_batch = model.n_threads;
    }

    llama_context * ctx_raw = llama_init_from_model(model.model, cparams);
    if (ctx_raw == nullptr) {
        err = "cannot create context";
        return false;
    }
    struct ctx_deleter {
        void operator()(llama_context * c) const { llama_free(c); }
    };
    std::unique_ptr<llama_context, ctx_deleter> ctx_holder(ctx_raw);
    llama_context * ctx = ctx_holder.get();

    if (!decode_prefix(ctx, 0, cond)) {
        err = "condition prefix decode failed";
        return false;
    }
    if (needs_cfg && !decode_prefix(ctx, 1, uncond)) {
        err = "unconditional prefix decode failed";
        return false;
    }

    llama_memory_t mem = llama_get_memory(ctx);

    // the generation pass is fully bidirectional inside the image block
    llama_set_causal_attn(ctx, false);

    if (!model.load_fm(p.width, p.height, err)) {
        err = "cannot load FM sidecar: " + err;
        return false;
    }
    fm_context & fm = *model.fm;

    float noise_scale = fm.p.noise_scale;
    if (fm.p.noise_scale_mode == "resolution" || fm.p.noise_scale_mode == "dynamic" ||
        fm.p.noise_scale_mode == "dynamic_sqrt") {
        const float scale = sqrtf((float) n_patch/(float) (fm.p.merge_size*fm.p.merge_size)/
                                  (float) fm.p.noise_scale_base_seq_len);
        noise_scale = scale*fm.p.noise_scale;
        if (fm.p.noise_scale_mode == "dynamic_sqrt") {
            noise_scale = sqrtf(noise_scale);
        }
    }
    if (noise_scale > fm.p.noise_scale_max) {
        noise_scale = fm.p.noise_scale_max;
    }

    // timestep schedule
    std::vector<float> timesteps(p.steps + 1);
    for (int i = 0; i <= p.steps; ++i) {
        const float t = (float) i/p.steps;
        float sigma = 1.0f - t;
        sigma = p.shift*sigma/(1.0f + (p.shift - 1.0f)*sigma);
        timesteps[i] = 1.0f - sigma;
    }

    rng_normal rng((uint64_t) p.seed);
    std::vector<float> img((size_t) 3*p.width*p.height);
    for (size_t i = 0; i < img.size(); ++i) {
        img[i] = noise_scale*rng.next();
    }

    std::vector<float> img_whc((size_t) 3*p.width*p.height);
    std::vector<float> image_embeds((size_t) n_tokens*n_embd);
    std::vector<float> ts_emb(n_embd);
    std::vector<float> hidden((size_t) n_tokens*n_embd);
    std::vector<float> hidden_uncond;
    if (needs_cfg) {
        hidden_uncond.resize((size_t) n_tokens*n_embd);
    }
    std::vector<float> x_pred;
    std::vector<float> v_cond((size_t) 3*p.width*p.height);
    std::vector<float> v_uncond;
    if (needs_cfg) {
        v_uncond.resize((size_t) 3*p.width*p.height);
    }

    const float noise_ratio = noise_scale/fm.p.noise_scale_max;

    for (int step = 0; step < p.steps; ++step) {
        const float t      = timesteps[step];
        const float t_next = timesteps[step + 1];

        to_whc(img, img_whc, p.width, p.height);
        if (!fm.encode_image(img_whc.data(), image_embeds.data(), err)) {
            err = "vision encode failed: " + err;
            return false;
        }
        if (!fm.embed_cond(t, noise_ratio, ts_emb.data(), err)) {
            err = "timestep embedding failed: " + err;
            return false;
        }
        for (int j = 0; j < n_tokens; ++j) {
            float * e = image_embeds.data() + (size_t) j*n_embd;
            for (int c = 0; c < n_embd; ++c) {
                e[c] += ts_emb[c];
            }
        }

        if (!decode_image(ctx, 0, (int) cond.size(), token_h, token_w, image_embeds.data(), hidden.data(), n_embd)) {
            err = "generation pass failed";
            return false;
        }
        llama_memory_seq_rm(mem, 0, (llama_pos) cond.size(), -1);

        if (!run_head(fm, hidden.data(), x_pred, p.height, p.width, err)) {
            err = "pixel head failed: " + err;
            return false;
        }

        const float denom = std::max(1.0f - t, 0.05f);
        for (size_t i = 0; i < v_cond.size(); ++i) {
            v_cond[i] = (x_pred[i] - img[i])/denom;
        }

        if (needs_cfg) {
            if (!decode_image(ctx, 1, (int) uncond.size(), token_h, token_w, image_embeds.data(), hidden_uncond.data(), n_embd)) {
                err = "unconditional generation pass failed";
                return false;
            }
            llama_memory_seq_rm(mem, 1, (llama_pos) uncond.size(), -1);

            if (!run_head(fm, hidden_uncond.data(), x_pred, p.height, p.width, err)) {
                err = "pixel head failed: " + err;
                return false;
            }
            for (size_t i = 0; i < v_uncond.size(); ++i) {
                v_uncond[i] = (x_pred[i] - img[i])/denom;
            }
            for (size_t i = 0; i < v_cond.size(); ++i) {
                v_cond[i] = v_uncond[i] + p.cfg*(v_cond[i] - v_uncond[i]);
            }
        }

        const float dt = t_next - t;
        for (size_t i = 0; i < img.size(); ++i) {
            img[i] += dt*v_cond[i];
        }

        fprintf(stderr, "step %d/%d: t=%.4f -> %.4f\n", step + 1, p.steps, t, t_next);
    }

    // normalized [-1, 1] space -> 8-bit RGB
    rgb.resize((size_t) 3*p.width*p.height);
    for (int c = 0; c < 3; ++c) {
        for (int y = 0; y < p.height; ++y) {
            for (int x = 0; x < p.width; ++x) {
                float v = img[(size_t) c*p.width*p.height + (size_t) y*p.width + x]*0.5f + 0.5f;
                v = std::min(1.0f, std::max(0.0f, v));
                rgb[(size_t) (y*p.width + x)*3 + c] = (uint8_t) lroundf(v*255.0f);
            }
        }
    }
    return true;
}
