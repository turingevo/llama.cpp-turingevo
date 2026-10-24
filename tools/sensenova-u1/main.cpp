#include "fm.h"
#include "png.h"

#include "ggml-backend.h"
#include "ggml.h"
#include "llama.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
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

    const int token_h = a.height/32;
    const int token_w = a.width/32;
    const int n_tokens = token_h*token_w;
    const int n_patch  = (a.height/16)*(a.width/16);

    llama_backend_init();
    ggml_backend_load_all();

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = a.ngl;

    llama_model * model = llama_model_load_from_file(a.model.c_str(), mparams);
    if (model == nullptr) {
        fprintf(stderr, "error: cannot load model %s\n", a.model.c_str());
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_embd = llama_model_n_embd(model);

    std::vector<llama_token> cond;
    std::vector<llama_token> uncond;
    if (!tokenize_prompt(vocab, build_cond_prompt(a.prompt), cond)) {
        return 1;
    }
    const bool needs_cfg = a.cfg > 1.0f;
    if (needs_cfg && !tokenize_prompt(vocab, build_uncond_prompt(), uncond)) {
        return 1;
    }

    fprintf(stderr, "prompt tokens: %zu, image tokens: %d (%dx%d), steps: %d, cfg: %.2f\n",
            cond.size(), n_tokens, a.width, a.height, a.steps, a.cfg);

    llama_context_params cparams = llama_context_default_params();
    const uint32_t n_batch = (uint32_t) std::max(n_tokens, (int) std::max(cond.size(), uncond.size()));

    cparams.n_ctx     = (uint32_t) (cond.size() + uncond.size() + n_tokens + 16);
    cparams.n_batch   = n_batch;
    cparams.n_ubatch  = n_batch;
    cparams.n_seq_max = 2;
    cparams.kv_unified = true;
    cparams.embeddings = true;
    cparams.no_perf    = false;
    if (a.threads > 0) {
        cparams.n_threads       = a.threads;
        cparams.n_threads_batch = a.threads;
    }

    llama_context * ctx = llama_init_from_model(model, cparams);
    if (ctx == nullptr) {
        fprintf(stderr, "error: cannot create context\n");
        return 1;
    }

    if (!decode_prefix(ctx, 0, cond)) {
        fprintf(stderr, "error: condition prefix decode failed\n");
        return 1;
    }
    if (needs_cfg && !decode_prefix(ctx, 1, uncond)) {
        fprintf(stderr, "error: unconditional prefix decode failed\n");
        return 1;
    }

    llama_memory_t mem = llama_get_memory(ctx);

    // the generation pass is fully bidirectional inside the image block
    llama_set_causal_attn(ctx, false);

    // FM sidecar on the same device class as the model, when possible
    ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (dev == nullptr) {
        dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    }
    if (dev == nullptr) {
        fprintf(stderr, "error: no backend device available\n");
        return 1;
    }
    ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
    if (backend == nullptr) {
        fprintf(stderr, "error: cannot create FM backend\n");
        return 1;
    }

    // released before the backend, as the FM buffers live on it
    std::unique_ptr<fm_context> fm_holder = std::make_unique<fm_context>();
    fm_context & fm = *fm_holder;

    std::string err;
    if (!fm.load(a.fm, backend, a.width, a.height, err)) {
        fprintf(stderr, "error: cannot load FM sidecar: %s\n", err.c_str());
        return 1;
    }

    if (fm.p.n_embd != n_embd) {
        fprintf(stderr, "error: embedding size mismatch: model %d, FM %d\n", n_embd, fm.p.n_embd);
        return 1;
    }

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
    std::vector<float> timesteps(a.steps + 1);
    for (int i = 0; i <= a.steps; ++i) {
        const float t = (float) i/a.steps;
        float sigma = 1.0f - t;
        sigma = a.shift*sigma/(1.0f + (a.shift - 1.0f)*sigma);
        timesteps[i] = 1.0f - sigma;
    }

    rng_normal rng((uint64_t) a.seed);
    std::vector<float> img((size_t) 3*a.width*a.height);
    for (size_t i = 0; i < img.size(); ++i) {
        img[i] = noise_scale*rng.next();
    }

    std::vector<float> img_whc((size_t) 3*a.width*a.height);
    std::vector<float> image_embeds((size_t) n_tokens*n_embd);
    std::vector<float> ts_emb(n_embd);
    std::vector<float> hidden((size_t) n_tokens*n_embd);
    std::vector<float> hidden_uncond;
    if (needs_cfg) {
        hidden_uncond.resize((size_t) n_tokens*n_embd);
    }
    std::vector<float> x_pred;
    std::vector<float> v_cond((size_t) 3*a.width*a.height);
    std::vector<float> v_uncond;
    if (needs_cfg) {
        v_uncond.resize((size_t) 3*a.width*a.height);
    }

    const float noise_ratio = noise_scale/fm.p.noise_scale_max;

    for (int step = 0; step < a.steps; ++step) {
        const float t      = timesteps[step];
        const float t_next = timesteps[step + 1];

        to_whc(img, img_whc, a.width, a.height);
        if (!fm.encode_image(img_whc.data(), image_embeds.data(), err)) {
            fprintf(stderr, "error: vision encode failed: %s\n", err.c_str());
            return 1;
        }
        if (!fm.embed_cond(t, noise_ratio, ts_emb.data(), err)) {
            fprintf(stderr, "error: timestep embedding failed: %s\n", err.c_str());
            return 1;
        }
        for (int j = 0; j < n_tokens; ++j) {
            float * e = image_embeds.data() + (size_t) j*n_embd;
            for (int c = 0; c < n_embd; ++c) {
                e[c] += ts_emb[c];
            }
        }

        if (!decode_image(ctx, 0, (int) cond.size(), token_h, token_w, image_embeds.data(), hidden.data(), n_embd)) {
            fprintf(stderr, "error: generation pass failed\n");
            return 1;
        }
        llama_memory_seq_rm(mem, 0, (llama_pos) cond.size(), -1);

        if (!run_head(fm, hidden.data(), x_pred, a.height, a.width, err)) {
            fprintf(stderr, "error: pixel head failed: %s\n", err.c_str());
            return 1;
        }

        const float denom = std::max(1.0f - t, 0.05f);
        for (size_t i = 0; i < v_cond.size(); ++i) {
            v_cond[i] = (x_pred[i] - img[i])/denom;
        }

        if (needs_cfg) {
            if (!decode_image(ctx, 1, (int) uncond.size(), token_h, token_w, image_embeds.data(), hidden_uncond.data(), n_embd)) {
                fprintf(stderr, "error: unconditional generation pass failed\n");
                return 1;
            }
            llama_memory_seq_rm(mem, 1, (llama_pos) uncond.size(), -1);

            if (!run_head(fm, hidden_uncond.data(), x_pred, a.height, a.width, err)) {
                fprintf(stderr, "error: pixel head failed: %s\n", err.c_str());
                return 1;
            }
            for (size_t i = 0; i < v_uncond.size(); ++i) {
                v_uncond[i] = (x_pred[i] - img[i])/denom;
            }
            for (size_t i = 0; i < v_cond.size(); ++i) {
                v_cond[i] = v_uncond[i] + a.cfg*(v_cond[i] - v_uncond[i]);
            }
        }

        const float dt = t_next - t;
        for (size_t i = 0; i < img.size(); ++i) {
            img[i] += dt*v_cond[i];
        }

        fprintf(stderr, "step %d/%d: t=%.4f -> %.4f\n", step + 1, a.steps, t, t_next);
    }

    // normalized [-1, 1] space -> 8-bit RGB
    std::vector<uint8_t> rgb((size_t) 3*a.width*a.height);
    for (int c = 0; c < 3; ++c) {
        for (int y = 0; y < a.height; ++y) {
            for (int x = 0; x < a.width; ++x) {
                float v = img[(size_t) c*a.width*a.height + (size_t) y*a.width + x]*0.5f + 0.5f;
                v = std::min(1.0f, std::max(0.0f, v));
                rgb[(size_t) (y*a.width + x)*3 + c] = (uint8_t) lroundf(v*255.0f);
            }
        }
    }

    if (!png_write_rgb(a.output, rgb.data(), a.width, a.height, err)) {
        fprintf(stderr, "error: cannot write %s: %s\n", a.output.c_str(), err.c_str());
        return 1;
    }
    fprintf(stderr, "wrote %s\n", a.output.c_str());

    fm_holder.reset();
    ggml_backend_free(backend);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();

    return 0;
}
