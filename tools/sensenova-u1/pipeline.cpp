#include "pipeline.h"

#include "ggml-backend.h"
#include "ggml.h"
#include "image.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

// system prompt used by the generation branch of the reference implementation
static const char * SYSTEM_MESSAGE_FOR_GEN =
    "You are an image generation and editing assistant that accurately understands and executes "
    "user intent.\n\nYou support two modes:\n\n1. Think Mode:\nIf the task requires reasoning, you "
    "MUST start with a <think></think> block. Put all reasoning inside the block using plain text. "
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
    "- For any visible text in the image, follow the language specified for the rendered text in "
    "the user's description, not the language of the prompt. If no language is specified, use the "
    "user's input language.";

// system prompt used by the interleave branch of the reference implementation
static const char * SYSTEM_MESSAGE_FOR_INTERLEAVE =
    "You are a multimodal assistant capable of reasoning with both text and images. You support "
    "two modes:\n\nThink Mode: When reasoning is needed, you MUST start with a <think></think> "
    "block and place all reasoning inside it. You MUST interleave text with generated images using "
    "tags like <image1>, <image2>. Images can ONLY be generated between <think> and </think>, and "
    "may be referenced in the final answer.\n\nNon-Think Mode: When no reasoning is needed, "
    "directly provide the answer without reasoning. Do not use tags like <image1>, <image2>; "
    "present any images naturally alongside the text.\n\nAfter the think block, always provide a "
    "concise, user-facing final answer. The answer may include text, images, or both. Match the "
    "user's language in both reasoning and the final answer.";

static const int GRID = 32;

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

static bool token_id(const llama_vocab * vocab, const char * text, llama_token & id) {
    std::vector<llama_token> ids;
    if (!tokenize_prompt(vocab, text, ids) || ids.size() != 1) {
        return false;
    }
    id = ids[0];
    return true;
}

static int count_placeholders(const std::string & text) {
    int n = 0;
    for (size_t pos = text.find("<image>"); pos != std::string::npos; pos = text.find("<image>", pos + 7)) {
        ++n;
    }
    return n;
}

static std::string image_block(int n_tok) {
    std::string block = "<img>";
    block.reserve(5 + (size_t) n_tok*13 + 6);
    for (int i = 0; i < n_tok; ++i) {
        block += "<IMG_CONTEXT>";
    }
    block += "</img>";
    return block;
}

static std::string expand_images(std::string text, const std::vector<int> & n_tok) {
    size_t pos = 0;
    for (int n : n_tok) {
        pos = text.find("<image>", pos);
        if (pos == std::string::npos) {
            break;
        }
        const std::string block = image_block(n);
        text.replace(pos, 7, block);
        pos += block.size();
    }
    return text;
}

// add placeholders for images the prompt does not reference
static std::string add_missing_placeholders(const std::string & prompt, int n_images) {
    const int have = count_placeholders(prompt);
    if (n_images <= have) {
        return prompt;
    }

    std::string pre;
    if (have == 0 && n_images > 1) {
        for (int i = 0; i < n_images; ++i) {
            pre += "Image-" + std::to_string(i + 1) + ":<image>\n";
        }
    } else {
        for (int i = have; i < n_images; ++i) {
            pre += "<image>\n";
        }
    }
    return pre + prompt;
}

static std::string build_cond_prompt(const std::string & prompt, bool think) {
    std::string q;
    q += "<|im_start|>system\n";
    q += SYSTEM_MESSAGE_FOR_GEN;
    q += "<|im_end|>\n";
    q += "<|im_start|>user\n";
    q += prompt;
    q += "<|im_end|>\n";
    q += "<|im_start|>assistant\n";
    q += think ? "<think>\n" : "<think>\n\n</think>\n\n<img>";
    return q;
}

// neo1_0 template without a system message
static std::string build_plain_prompt(const std::string & user) {
    return "<|im_start|>user\n" + user + "<|im_end|>\n<|im_start|>assistant\n<img>";
}

// single turn chat prompt for visual understanding
static std::string build_chat_prompt(const std::string & question) {
    return "<|im_start|>user\n" + question + "<|im_end|>\n<|im_start|>assistant\n";
}

// chat prompt for interleave, the model decides when to emit <img>
static std::string build_interleave_prompt(const std::string & system, const std::string & user, bool think) {
    std::string q;
    q += "<|im_start|>system\n";
    q += system;
    q += "<|im_end|>\n";
    q += "<|im_start|>user\n";
    q += user;
    q += "<|im_end|>\n";
    q += "<|im_start|>assistant\n";
    if (!think) {
        q += "<think>\n\n</think>\n\n";
    }
    return q;
}

// the interleave prompt prepends missing placeholders without labels
static std::string add_missing_placeholders_plain(const std::string & prompt, int n_images) {
    std::string pre;
    for (int i = count_placeholders(prompt); i < n_images; ++i) {
        pre += "<image>\n";
    }
    return pre + prompt;
}

// one token of a prefix, with its M-RoPE position
struct prefix_item {
    llama_token tok = 0;
    int img = -1;     // input image index, -1 for text
    int img_pos = 0;  // token offset inside the image block
    llama_pos t = 0;
    llama_pos h = 0;
    llama_pos w = 0;
};

struct prefix_spec {
    std::vector<llama_token> tokens;
    std::vector<prefix_item> items;
    llama_pos max_t = -1;
};

// assign M-RoPE positions the same way as the reference get_thw_indexes
static bool build_prefix(
        const llama_vocab * vocab,
        const std::string & text,
        const std::vector<std::pair<int, int>> & grid, // token_h, token_w per image
        prefix_spec & spec,
        std::string & err) {
    if (!tokenize_prompt(vocab, text, spec.tokens)) {
        err = "cannot tokenize prompt";
        return false;
    }

    llama_token img_start = 0;
    llama_token img_context = 0;
    if (!token_id(vocab, "<img>", img_start) || !token_id(vocab, "<IMG_CONTEXT>", img_context)) {
        err = "cannot find the image tokens in the vocabulary";
        return false;
    }

    spec.items.resize(spec.tokens.size());

    llama_pos t = -1;
    int next_img = 0;
    int inner = 0;

    for (size_t i = 0; i < spec.tokens.size(); ++i) {
        const llama_token tok = spec.tokens[i];
        const bool is_context = tok == img_context;
        const bool prev_start = i > 0 && spec.tokens[i - 1] == img_start;

        t += (prev_start ? 1 : 0) + (is_context ? 0 : 1);

        prefix_item & it = spec.items[i];
        it.tok = tok;
        it.t = t;

        if (is_context) {
            if (prev_start) {
                it.img = next_img++;
                inner = 0;
            } else {
                it.img = next_img - 1;
            }
            if (it.img < 0 || it.img >= (int) grid.size()) {
                err = "image tokens without a matching image";
                return false;
            }
            const int token_w = grid[it.img].second;
            it.img_pos = inner++;
            it.h = it.img_pos/token_w;
            it.w = it.img_pos%token_w;
        }
    }

    if (next_img != (int) grid.size()) {
        err = "image count does not match the prompt placeholders";
        return false;
    }

    std::vector<int> seen(grid.size(), 0);
    for (const prefix_item & it : spec.items) {
        if (it.img >= 0) {
            ++seen[it.img];
        }
    }
    for (size_t i = 0; i < grid.size(); ++i) {
        if (seen[i] != grid[i].first*grid[i].second) {
            err = "image token count is inconsistent";
            return false;
        }
    }

    spec.max_t = spec.items.empty() ? -1 : spec.items.back().t;
    return true;
}

static bool decode_text_run(
        llama_context * ctx,
        llama_seq_id seq,
        const prefix_item * items,
        int n,
        bool logits_last) {
    llama_set_causal_attn(ctx, true);
    llama_set_gen_pass(ctx, false);
    llama_set_embeddings(ctx, false);

    std::vector<llama_token>   tokens((size_t) n);
    std::vector<llama_pos>     pos((size_t) n*4);
    std::vector<int32_t>       n_seq_id((size_t) n, 1);
    std::vector<llama_seq_id *> seq_id((size_t) n);
    std::vector<int8_t>        logits((size_t) n, 0);

    for (int i = 0; i < n; ++i) {
        tokens[(size_t) i] = items[i].tok;
        pos[(size_t) 0*n + i] = items[i].t;
        pos[(size_t) 1*n + i] = items[i].h;
        pos[(size_t) 2*n + i] = items[i].w;
        pos[(size_t) 3*n + i] = 0;
        seq_id[(size_t) i] = &seq;
    }
    if (logits_last) {
        logits[(size_t) n - 1] = 1;
    }

    llama_batch batch = {};
    batch.n_tokens = n;
    batch.token    = tokens.data();
    batch.pos      = pos.data();
    batch.n_seq_id = n_seq_id.data();
    batch.seq_id   = seq_id.data();
    batch.logits   = logits.data();

    return llama_decode(ctx, batch) == 0;
}

static bool decode_embd_run(
        llama_context * ctx,
        llama_seq_id seq,
        const float * embeds,
        const prefix_item * items,
        int n) {
    llama_set_causal_attn(ctx, false);
    llama_set_gen_pass(ctx, false);
    llama_set_embeddings(ctx, false);

    std::vector<llama_pos>     pos((size_t) n*4);
    std::vector<int32_t>       n_seq_id((size_t) n, 1);
    std::vector<llama_seq_id *> seq_id((size_t) n);
    std::vector<int8_t>        logits((size_t) n, 0);

    for (int i = 0; i < n; ++i) {
        pos[(size_t) 0*n + i] = items[i].t;
        pos[(size_t) 1*n + i] = items[i].h;
        pos[(size_t) 2*n + i] = items[i].w;
        pos[(size_t) 3*n + i] = 0;
        seq_id[(size_t) i] = &seq;
    }

    llama_batch batch = {};
    batch.n_tokens = n;
    batch.embd     = const_cast<float *>(embeds);
    batch.pos      = pos.data();
    batch.n_seq_id = n_seq_id.data();
    batch.seq_id   = seq_id.data();
    batch.logits   = logits.data();

    return llama_decode(ctx, batch) == 0;
}

static bool decode_prefix(
        llama_context * ctx,
        llama_seq_id seq,
        const prefix_spec & spec,
        const std::vector<std::vector<float>> & img_embeds,
        int n_embd,
        bool logits_last) {
    size_t i = 0;
    while (i < spec.items.size()) {
        const int img = spec.items[i].img;
        size_t j = i + 1;
        while (j < spec.items.size() && spec.items[j].img == img) {
            ++j;
        }

        const int n = (int) (j - i);
        if (img < 0) {
            if (!decode_text_run(ctx, seq, &spec.items[i], n, logits_last && j == spec.items.size())) {
                return false;
            }
        } else {
            const float * e = img_embeds[img].data() + (size_t) spec.items[i].img_pos*n_embd;
            if (!decode_embd_run(ctx, seq, e, &spec.items[i], n)) {
                return false;
            }
        }
        i = j;
    }
    return true;
}

static bool decode_one_token(
        llama_context * ctx,
        llama_seq_id seq,
        llama_token tok,
        llama_pos t) {
    llama_set_causal_attn(ctx, true);
    llama_set_gen_pass(ctx, false);
    llama_set_embeddings(ctx, false);

    llama_pos pos[4] = { t, 0, 0, 0 };
    int32_t n_seq_id = 1;
    llama_seq_id * seq_id = &seq;
    int8_t logits = 1;

    llama_batch batch = {};
    batch.n_tokens = 1;
    batch.token    = &tok;
    batch.pos      = pos;
    batch.n_seq_id = &n_seq_id;
    batch.seq_id   = &seq_id;
    batch.logits   = &logits;

    return llama_decode(ctx, batch) == 0;
}

static llama_token argmax_token(const float * logits, int n_vocab) {
    int best = 0;
    float best_v = logits[0];
    for (int i = 1; i < n_vocab; ++i) {
        if (logits[i] > best_v) {
            best_v = logits[i];
            best = i;
        }
    }
    return best;
}

static std::string detokenize(const llama_vocab * vocab, const std::vector<llama_token> & ids) {
    if (ids.empty()) {
        return "";
    }
    std::string text(std::max<size_t>(ids.size(), 64), '\0');
    int32_t n = llama_detokenize(vocab, ids.data(), (int32_t) ids.size(), text.data(), (int32_t) text.size(), false, true);
    if (n < 0) {
        text.resize((size_t) -n);
        n = llama_detokenize(vocab, ids.data(), (int32_t) ids.size(), text.data(), (int32_t) text.size(), false, true);
    }
    if (n < 0) {
        return "";
    }
    text.resize((size_t) n);
    return text;
}

struct sampler_deleter {
    void operator()(llama_sampler * s) const { llama_sampler_free(s); }
};

// null keeps greedy decoding, which is what the vendor defaults to
static std::unique_ptr<llama_sampler, sampler_deleter> make_sampler(
        float temperature,
        int top_k,
        float top_p,
        float repeat_penalty,
        int seed,
        int n_vocab) {
    if (temperature <= 0.0f) {
        return nullptr;
    }
    llama_sampler * smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
    if (repeat_penalty != 1.0f) {
        llama_sampler_chain_add(smpl, llama_sampler_init_penalties(
                n_vocab, 64, repeat_penalty, 0.0f, 0.0f));
    }
    if (top_k > 0) {
        llama_sampler_chain_add(smpl, llama_sampler_init_top_k(top_k));
    }
    if (top_p < 1.0f) {
        llama_sampler_chain_add(smpl, llama_sampler_init_top_p(top_p, 1));
    }
    llama_sampler_chain_add(smpl, llama_sampler_init_temp(temperature));
    llama_sampler_chain_add(smpl, llama_sampler_init_dist((uint32_t) seed));
    return std::unique_ptr<llama_sampler, sampler_deleter>(smpl);
}

// AR loop, stops at <|im_end|>, max_tokens, or stop_tok when >= 0
static bool generate_tokens(
        llama_context * ctx,
        const llama_vocab * vocab,
        llama_seq_id seq,
        llama_pos & t,
        int max_tokens,
        llama_token stop_tok,
        llama_sampler * smpl,
        std::vector<llama_token> & ids,
        std::string & err) {
    llama_token eos = 0;
    if (!token_id(vocab, "<|im_end|>", eos)) {
        err = "cannot find the end token in the vocabulary";
        return false;
    }

    const int n_vocab = llama_vocab_n_tokens(vocab);

    auto sample = [&]() -> llama_token {
        if (smpl != nullptr) {
            return llama_sampler_sample(smpl, ctx, -1);
        }
        const float * logits = llama_get_logits_ith(ctx, -1);
        if (logits == nullptr) {
            return LLAMA_TOKEN_NULL;
        }
        return argmax_token(logits, n_vocab);
    };

    llama_token next = sample();
    for (int i = 0; i < max_tokens; ++i) {
        if (next == LLAMA_TOKEN_NULL) {
            err = ids.empty() ? "no logits after the prefix pass" : "no logits during generation";
            return false;
        }
        if (next == eos) {
            break;
        }
        ids.push_back(next);
        ++t;
        if (!decode_one_token(ctx, seq, next, t)) {
            err = "token decode failed";
            return false;
        }
        if (stop_tok >= 0 && next == stop_tok) {
            break;
        }
        next = sample();
    }
    return true;
}

static bool generate_think(
        llama_context * ctx,
        const llama_vocab * vocab,
        llama_seq_id seq,
        llama_pos & t,
        int max_tokens,
        std::string & text,
        std::string & err) {
    llama_token think_end = 0;
    if (!token_id(vocab, "</think>", think_end)) {
        err = "cannot find the think tokens in the vocabulary";
        return false;
    }

    std::vector<llama_token> ids;
    if (!generate_tokens(ctx, vocab, seq, t, max_tokens, think_end, nullptr, ids, err)) {
        return false;
    }
    text = detokenize(vocab, ids);
    return true;
}

// run one generation pass: image embeddings -> last hidden state, in place
static bool decode_image(
        llama_context * ctx,
        llama_seq_id seq,
        llama_pos t_img,
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
        pos[(size_t) 0*n_tokens + i] = t_img;
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

// one denoising step for one sequence: generation pass, then drop the KV rows
static bool predict_velocity(
        fm_context & fm,
        llama_context * ctx,
        llama_seq_id seq,
        llama_pos t_img,
        int token_h,
        int token_w,
        int width,
        int height,
        const float * image_embeds,
        const float * img,
        float denom,
        std::vector<float> & hidden,
        std::vector<float> & x_pred,
        std::vector<float> & v,
        std::string & err) {
    if (!decode_image(ctx, seq, t_img, token_h, token_w, image_embeds, hidden.data(), fm.p.n_embd)) {
        err = "generation pass decode failed";
        return false;
    }
    llama_memory_seq_rm(llama_get_memory(ctx), seq, t_img, -1);
    if (!run_head(fm, hidden.data(), x_pred, height, width, err)) {
        return false;
    }
    for (size_t i = 0; i < v.size(); ++i) {
        v[i] = (x_pred[i] - img[i])/denom;
    }
    return true;
}

// [-1, 1] channel-major image -> patch embedder layout, ImageNet normalized
static void gen_to_fm_input(const std::vector<float> & img, std::vector<float> & out, int w, int h) {
    static const float mean[3] = { 0.485f, 0.456f, 0.406f };
    static const float std_[3] = { 0.229f, 0.224f, 0.225f };
    out.resize((size_t) 3*w*h);
    for (int c = 0; c < 3; ++c) {
        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                const float v = img[(size_t) c*w*h + (size_t) y*w + x]*0.5f + 0.5f;
                out[(size_t) x + (size_t) w*y + (size_t) w*h*c] = (v - mean[c])/std_[c];
            }
        }
    }
}

static void to_rgb(const std::vector<float> & img, int width, int height, std::vector<uint8_t> & rgb) {
    rgb.resize((size_t) 3*width*height);
    for (int c = 0; c < 3; ++c) {
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                float v = img[(size_t) c*width*height + (size_t) y*width + x]*0.5f + 0.5f;
                v = std::min(1.0f, std::max(0.0f, v));
                rgb[(size_t) (y*width + x)*3 + c] = (uint8_t) lroundf(v*255.0f);
            }
        }
    }
}

static void build_timesteps(int steps, float shift, std::vector<float> & timesteps) {
    timesteps.resize((size_t) steps + 1);
    for (int i = 0; i <= steps; ++i) {
        const float t = (float) i/steps;
        float sigma = 1.0f - t;
        sigma = shift*sigma/(1.0f + (shift - 1.0f)*sigma);
        timesteps[(size_t) i] = 1.0f - sigma;
    }
}

static float noise_scale_for(const fm_params & p, int width, int height) {
    float noise_scale = p.noise_scale;
    if (p.noise_scale_mode == "resolution" || p.noise_scale_mode == "dynamic" ||
        p.noise_scale_mode == "dynamic_sqrt") {
        const int n_patch = (height/16)*(width/16);
        const float scale = sqrtf((float) n_patch/(float) (p.merge_size*p.merge_size)/
                                  (float) p.noise_scale_base_seq_len);
        noise_scale = scale*p.noise_scale;
        if (p.noise_scale_mode == "dynamic_sqrt") {
            noise_scale = sqrtf(noise_scale);
        }
    }
    return std::min(noise_scale, p.noise_scale_max);
}

// clamp the CFG result to the conditional norm, matching the reference
static void apply_cfg_norm(
        std::vector<float> & v,
        const std::vector<float> & v_cond,
        int width,
        int height,
        const std::string & mode) {
    if (mode == "global") {
        double a = 0.0;
        double b = 0.0;
        for (size_t i = 0; i < v.size(); ++i) {
            a += (double) v_cond[i]*v_cond[i];
            b += (double) v[i]*v[i];
        }
        const float scale = std::min(1.0f, (float) (sqrt(a)/(sqrt(b) + 1e-8)));
        for (float & x : v) {
            x *= scale;
        }
    } else if (mode == "channel") {
        for (int c = 0; c < 3; ++c) {
            for (int y = 0; y < height; ++y) {
                const size_t row = (size_t) c*width*height + (size_t) y*width;
                double a = 0.0;
                double b = 0.0;
                for (int x = 0; x < width; ++x) {
                    a += (double) v_cond[row + x]*v_cond[row + x];
                    b += (double) v[row + x]*v[row + x];
                }
                const float scale = std::min(1.0f, (float) (sqrt(a)/(sqrt(b) + 1e-8)));
                for (int x = 0; x < width; ++x) {
                    v[row + x] *= scale;
                }
            }
        }
    }
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
        ggml_backend_dev_t dev = nullptr;
        if (!fm_cpu) {
            dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
        }
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

bool sensenova_model::load_fm_und(std::string & err) {
    return load_fm(0, 0, err);
}

std::unique_ptr<sensenova_model> sensenova_model_load(
        const std::string & model_path,
        const std::string & fm_path,
        int ngl,
        int n_threads,
        std::string & err,
        const std::vector<std::string> & cpu_tensors) {
    llama_backend_init();
    ggml_backend_load_all();

    std::vector<llama_model_tensor_buft_override> overrides;
    for (const auto & pattern : cpu_tensors) {
        overrides.push_back({ pattern.c_str(), ggml_backend_cpu_buffer_type() });
    }
    overrides.push_back({ nullptr, nullptr });

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = ngl;
    if (!cpu_tensors.empty()) {
        mparams.tensor_buft_overrides = overrides.data();
    }

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

struct ctx_deleter {
    void operator()(llama_context * c) const { llama_free(c); }
};

static std::unique_ptr<llama_context, ctx_deleter> create_context(
        sensenova_model & model,
        uint32_t n_ctx,
        uint32_t n_batch,
        uint32_t n_seq_max,
        bool embeddings,
        std::string & err) {
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx      = n_ctx;
    cparams.n_batch    = n_batch;
    cparams.n_ubatch   = n_batch;
    // a text context only needs the logits of the last token, otherwise the pp reserve
    // graph is sized for n_batch logits; embedding contexts output every token
    cparams.n_outputs_max = embeddings ? n_batch : 1;
    cparams.n_seq_max  = n_seq_max;
    cparams.kv_unified = true;
    cparams.embeddings = embeddings;
    cparams.no_perf    = false;
    if (model.n_threads > 0) {
        cparams.n_threads       = model.n_threads;
        cparams.n_threads_batch = model.n_threads;
    }

    std::unique_ptr<llama_context, ctx_deleter> ctx(llama_init_from_model(model.model, cparams));
    if (!ctx) {
        err = "cannot create context";
    }
    return ctx;
}

// decode and resize the input images to the pixel budget
static bool prepare_input_images(
        const std::vector<sensenova_image_input> & src,
        int input_max_pixels,
        std::vector<sensenova_image> & out,
        std::string & err) {
    const int n_images = (int) src.size();
    const int min_pixels = 512*512;

    out.resize((size_t) n_images);
    for (int i = 0; i < n_images; ++i) {
        sensenova_image raw;
        bool ok = false;
        if (!src[i].path.empty()) {
            ok = image_load_file(src[i].path, raw, err);
        } else {
            ok = image_load_memory(src[i].data.data(), src[i].data.size(), raw, err);
        }
        if (!ok) {
            err = "cannot load input image: " + err;
            return false;
        }

        const int max_pixels = input_max_pixels > 0
            ? input_max_pixels
            : std::min(2048*2048, (4096*4096)/std::max(1, n_images));
        if (max_pixels < min_pixels) {
            err = "input_max_pixels must be at least 512*512";
            return false;
        }
        if (!image_prepare(raw, min_pixels, max_pixels, out[(size_t) i], err)) {
            err = "cannot resize input image: " + err;
            return false;
        }
    }
    return true;
}

// encode the input images with the understanding branch vision embedder
static bool encode_input_images(
        sensenova_model & model,
        const std::vector<sensenova_image> & inputs,
        std::vector<std::vector<float>> & embeds,
        std::vector<std::pair<int, int>> & grid, // token_h, token_w
        std::vector<int> & n_tok,
        std::string & err) {
    const int n_images = (int) inputs.size();
    embeds.resize((size_t) n_images);
    grid.resize((size_t) n_images);
    n_tok.resize((size_t) n_images);

    std::vector<float> img_whc;
    for (int i = 0; i < n_images; ++i) {
        const sensenova_image & img = inputs[(size_t) i];
        const int th = img.h/GRID;
        const int tw = img.w/GRID;
        grid[(size_t) i] = { th, tw };
        n_tok[(size_t) i] = th*tw;

        image_to_fm_input(img, img_whc);
        embeds[(size_t) i].resize((size_t) th*tw*model.n_embd);
        if (!model.fm->encode_ref_image(img_whc.data(), img.w, img.h, embeds[(size_t) i].data(), err)) {
            err = "input image encode failed: " + err;
            return false;
        }
    }
    return true;
}

bool sensenova_generate(
        sensenova_model & model,
        const sensenova_gen_params & p,
        sensenova_result & out,
        std::string & err) {
    if (p.prompt.empty()) {
        err = "prompt is empty";
        return false;
    }
    if (p.steps <= 0) {
        err = "steps must be positive";
        return false;
    }
    if (p.cfg_norm != "none" && p.cfg_norm != "global" && p.cfg_norm != "channel") {
        err = "cfg_norm must be none, global or channel";
        return false;
    }
    if (p.images.empty() && p.img_cfg != 1.0f) {
        err = "img_cfg requires input images";
        return false;
    }

    const int n_images = (int) p.images.size();
    const int min_pixels = 512*512;

    std::vector<sensenova_image> inputs;
    if (!prepare_input_images(p.images, p.input_max_pixels, inputs, err)) {
        return false;
    }

    // resolve the output size: explicit, first input aspect, or 1024x1024
    int width  = p.width;
    int height = p.height;
    if (width == 0 && height == 0) {
        if (n_images > 0) {
            const int target = p.auto_size_pixels;
            if (target < min_pixels) {
                err = "auto_size_pixels must be at least 512*512";
                return false;
            }
            if (!image_plan_size(inputs[0].w, inputs[0].h, target, target, width, height, err)) {
                err = "cannot derive the output size: " + err;
                return false;
            }
        } else {
            width  = 1024;
            height = 1024;
        }
    } else if (width <= 0 || height <= 0) {
        err = "width and height must be set together";
        return false;
    }
    if (width % GRID != 0 || height % GRID != 0) {
        err = "width and height must be positive multiples of 32";
        return false;
    }

    const int token_h = height/GRID;
    const int token_w = width/GRID;
    const int n_tokens = token_h*token_w;

    const llama_vocab * vocab = llama_model_get_vocab(model.model);
    const int n_embd = model.n_embd;

    const bool editing = n_images > 0;
    const bool needs_cfg = editing ? !(p.cfg == 1.0f && p.img_cfg == 1.0f) : p.cfg > 1.0f;
    const bool needs_img_condition = editing && needs_cfg && (p.img_cfg == 1.0f || p.cfg != p.img_cfg);
    const bool needs_uncond = editing ? (needs_cfg && p.img_cfg != 1.0f) : needs_cfg;

    if (!model.load_fm(width, height, err)) {
        err = "cannot load FM sidecar: " + err;
        return false;
    }
    fm_context & fm = *model.fm;

    if (editing && !fm.has_und_vision()) {
        err = "the FM sidecar has no vision weights for input images, regenerate it with retag.py";
        return false;
    }

    // encode the input images once, with the understanding branch embedder
    std::vector<std::vector<float>> img_embeds;
    std::vector<std::pair<int, int>> img_grid; // token_h, token_w
    std::vector<int> img_tok;
    if (!encode_input_images(model, inputs, img_embeds, img_grid, img_tok, err)) {
        return false;
    }

    // build the three prefixes
    const std::string question = add_missing_placeholders(p.prompt, n_images);
    const std::string question_full = expand_images(question, img_tok);

    prefix_spec cond_spec;
    if (!build_prefix(vocab, build_cond_prompt(question_full, p.think), img_grid, cond_spec, err)) {
        err = "condition prompt: " + err;
        return false;
    }

    prefix_spec img_cond_spec;
    if (needs_img_condition) {
        std::string placeholders;
        for (int i = 0; i < n_images; ++i) {
            placeholders += "<image>";
        }
        if (!build_prefix(vocab, build_plain_prompt(expand_images(placeholders, img_tok)), img_grid, img_cond_spec, err)) {
            err = "image condition prompt: " + err;
            return false;
        }
    }

    prefix_spec uncond_spec;
    if (needs_uncond) {
        if (!build_prefix(vocab, build_plain_prompt(""), {}, uncond_spec, err)) {
            err = "unconditional prompt: " + err;
            return false;
        }
    }

    // largest batch: the generation block, a prefix, or one input image block
    uint32_t n_batch = (uint32_t) std::max<int>(n_tokens, (int) cond_spec.items.size());
    n_batch = std::max<uint32_t>(n_batch, (uint32_t) uncond_spec.items.size());
    n_batch = std::max<uint32_t>(n_batch, (uint32_t) img_cond_spec.items.size());
    for (int n : img_tok) {
        n_batch = std::max<uint32_t>(n_batch, (uint32_t) n);
    }

    const bool think_slack = p.think;
    uint32_t n_ctx = (uint32_t) (cond_spec.items.size() + uncond_spec.items.size() +
            img_cond_spec.items.size() + n_tokens + 16);
    if (think_slack) {
        n_ctx += (uint32_t) p.max_think_tokens + 8;
    }
    n_ctx = std::max<uint32_t>(n_ctx, n_batch + n_tokens + 16);

    fprintf(stderr, "prompt tokens: cond %zu, img_cond %zu, uncond %zu, image tokens: %d (%dx%d), steps: %d, cfg: %.2f, img_cfg: %.2f%s\n",
            cond_spec.items.size(), img_cond_spec.items.size(), uncond_spec.items.size(),
            n_tokens, width, height, p.steps, p.cfg, p.img_cfg, p.think ? ", think" : "");

    auto ctx_holder = create_context(model, n_ctx, n_batch, 3, true, err);
    if (!ctx_holder) {
        return false;
    }
    llama_context * ctx = ctx_holder.get();

    const llama_seq_id seq_cond     = 0;
    const llama_seq_id seq_img_cond = 1;
    const llama_seq_id seq_uncond   = 2;

    if (!decode_prefix(ctx, seq_cond, cond_spec, img_embeds, n_embd, p.think)) {
        err = "condition prefix decode failed";
        return false;
    }
    if (needs_img_condition && !decode_prefix(ctx, seq_img_cond, img_cond_spec, img_embeds, n_embd, false)) {
        err = "image condition prefix decode failed";
        return false;
    }
    if (needs_uncond && !decode_prefix(ctx, seq_uncond, uncond_spec, img_embeds, n_embd, false)) {
        err = "unconditional prefix decode failed";
        return false;
    }

    llama_pos cond_t = cond_spec.max_t;
    if (p.think) {
        if (!generate_think(ctx, vocab, seq_cond, cond_t, p.max_think_tokens, out.think_text, err)) {
            err = "think generation failed: " + err;
            return false;
        }

        std::vector<llama_token> tail;
        if (!tokenize_prompt(vocab, "\n\n<img>", tail)) {
            err = "cannot tokenize the image start";
            return false;
        }
        for (llama_token tok : tail) {
            ++cond_t;
            if (!decode_one_token(ctx, seq_cond, tok, cond_t)) {
                err = "cannot append the image start";
                return false;
            }
        }
    }

    const llama_pos cond_img_t     = cond_t + 1;
    const llama_pos img_cond_img_t = img_cond_spec.max_t + 1;
    const llama_pos uncond_img_t   = uncond_spec.max_t + 1;

    // the generation pass is fully bidirectional inside the image block
    llama_set_causal_attn(ctx, false);
    llama_set_gen_pass(ctx, true);
    llama_set_embeddings(ctx, true);

    const float noise_scale = noise_scale_for(fm.p, width, height);

    // timestep schedule
    std::vector<float> timesteps;
    build_timesteps(p.steps, p.shift, timesteps);

    rng_normal rng((uint64_t) p.seed);
    std::vector<float> img((size_t) 3*width*height);
    for (size_t i = 0; i < img.size(); ++i) {
        img[i] = noise_scale*rng.next();
    }

    std::vector<float> img_whc_gen((size_t) 3*width*height);
    std::vector<float> image_embeds((size_t) n_tokens*n_embd);
    std::vector<float> ts_emb(n_embd);
    std::vector<float> hidden((size_t) n_tokens*n_embd);
    std::vector<float> hidden_img_cond;
    std::vector<float> hidden_uncond;
    if (needs_img_condition) {
        hidden_img_cond.resize((size_t) n_tokens*n_embd);
    }
    if (needs_uncond) {
        hidden_uncond.resize((size_t) n_tokens*n_embd);
    }
    std::vector<float> x_pred;
    std::vector<float> v_cond((size_t) 3*width*height);
    std::vector<float> v_ref;
    std::vector<float> v_img_cond;
    std::vector<float> v_uncond;
    if (needs_cfg) {
        v_ref.resize((size_t) 3*width*height);
    }
    if (needs_img_condition) {
        v_img_cond.resize((size_t) 3*width*height);
    }
    if (needs_uncond) {
        v_uncond.resize((size_t) 3*width*height);
    }

    const float noise_ratio = noise_scale/fm.p.noise_scale_max;

    for (int step = 0; step < p.steps; ++step) {
        const float t      = timesteps[step];
        const float t_next = timesteps[step + 1];

        to_whc(img, img_whc_gen, width, height);
        if (!fm.encode_image(img_whc_gen.data(), image_embeds.data(), err)) {
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

        const float denom = std::max(1.0f - t, 0.05f);

        auto predict = [&](llama_seq_id seq, llama_pos t_img, std::vector<float> & hid, std::vector<float> & v) -> bool {
            return predict_velocity(fm, ctx, seq, t_img, token_h, token_w, width, height,
                    image_embeds.data(), img.data(), denom, hid, x_pred, v, err);
        };

        if (!predict(seq_cond, cond_img_t, hidden, v_cond)) {
            err = "condition generation pass failed: " + err;
            return false;
        }

        if (needs_cfg) {
            if (needs_img_condition) {
                if (!predict(seq_img_cond, img_cond_img_t, hidden_img_cond, v_img_cond)) {
                    err = "image condition generation pass failed: " + err;
                    return false;
                }
            }
            if (needs_uncond) {
                if (!predict(seq_uncond, uncond_img_t, hidden_uncond, v_uncond)) {
                    err = "unconditional generation pass failed: " + err;
                    return false;
                }
            }

            std::copy(v_cond.begin(), v_cond.end(), v_ref.begin());

            if (!editing) {
                for (size_t i = 0; i < v_cond.size(); ++i) {
                    v_cond[i] = v_uncond[i] + p.cfg*(v_cond[i] - v_uncond[i]);
                }
            } else if (p.img_cfg == 1.0f) {
                for (size_t i = 0; i < v_cond.size(); ++i) {
                    v_cond[i] = v_img_cond[i] + p.cfg*(v_cond[i] - v_img_cond[i]);
                }
            } else if (p.cfg == p.img_cfg) {
                for (size_t i = 0; i < v_cond.size(); ++i) {
                    v_cond[i] = v_uncond[i] + p.cfg*(v_cond[i] - v_uncond[i]);
                }
            } else {
                for (size_t i = 0; i < v_cond.size(); ++i) {
                    v_cond[i] = v_uncond[i] + p.cfg*(v_cond[i] - v_img_cond[i]) +
                        p.img_cfg*(v_img_cond[i] - v_uncond[i]);
                }
            }

            if (p.cfg > 1.0f || p.img_cfg > 1.0f) {
                apply_cfg_norm(v_cond, v_ref, width, height, p.cfg_norm);
            }
        }

        const float dt = t_next - t;
        for (size_t i = 0; i < img.size(); ++i) {
            img[i] += dt*v_cond[i];
        }

        fprintf(stderr, "step %d/%d: t=%.4f -> %.4f\n", step + 1, p.steps, t, t_next);
    }

    // normalized [-1, 1] space -> 8-bit RGB
    out.width  = width;
    out.height = height;
    to_rgb(img, width, height, out.rgb);
    return true;
}

static void trim_text(std::string & s) {
    const char * ws = " \t\r\n";
    const size_t b = s.find_first_not_of(ws);
    if (b == std::string::npos) {
        s.clear();
        return;
    }
    const size_t e = s.find_last_not_of(ws);
    s = s.substr(b, e - b + 1);
}

static void split_think(const std::string & text, std::string & think, std::string & answer) {
    const std::string open  = "<think>";
    const std::string close = "</think>";
    const size_t b = text.find(open);
    if (b == std::string::npos) {
        think.clear();
        answer = text;
        trim_text(answer);
        return;
    }
    const size_t e = text.find(close, b + open.size());
    if (e == std::string::npos) {
        // reply ran out of budget inside the think block
        think = text.substr(b + open.size());
        answer.clear();
        trim_text(think);
        return;
    }
    think  = text.substr(b + open.size(), e - b - open.size());
    answer = text.substr(e + close.size());
    trim_text(think);
    trim_text(answer);
}

bool sensenova_understand(
        sensenova_model & model,
        const sensenova_qa_params & p,
        sensenova_qa_result & out,
        std::string & err) {
    if (p.question.empty()) {
        err = "question is empty";
        return false;
    }
    if (p.max_new_tokens <= 0) {
        err = "max_new_tokens must be positive";
        return false;
    }

    const int n_images = (int) p.images.size();

    std::vector<sensenova_image> inputs;
    if (!prepare_input_images(p.images, p.input_max_pixels, inputs, err)) {
        return false;
    }

    std::vector<std::vector<float>> img_embeds;
    std::vector<std::pair<int, int>> img_grid;
    std::vector<int> img_tok;
    if (n_images > 0) {
        if (!model.load_fm_und(err)) {
            err = "cannot load FM sidecar: " + err;
            return false;
        }
        if (!model.fm->has_und_vision()) {
            err = "the FM sidecar has no vision weights for input images, regenerate it with retag.py";
            return false;
        }
        if (!encode_input_images(model, inputs, img_embeds, img_grid, img_tok, err)) {
            return false;
        }
    }

    const std::string question = add_missing_placeholders(p.question, n_images);
    const std::string question_full = expand_images(question, img_tok);

    const llama_vocab * vocab = llama_model_get_vocab(model.model);
    prefix_spec spec;
    if (!build_prefix(vocab, build_chat_prompt(question_full), img_grid, spec, err)) {
        err = "question prompt: " + err;
        return false;
    }

    uint32_t n_batch = (uint32_t) spec.items.size();
    for (int n : img_tok) {
        n_batch = std::max<uint32_t>(n_batch, (uint32_t) n);
    }
    const uint32_t n_ctx = (uint32_t) spec.items.size() + (uint32_t) p.max_new_tokens + 8;

    auto ctx_holder = create_context(model, n_ctx, std::max<uint32_t>(n_batch, 1), 1, false, err);
    if (!ctx_holder) {
        return false;
    }
    llama_context * ctx = ctx_holder.get();

    const llama_seq_id seq = 0;
    if (!decode_prefix(ctx, seq, spec, img_embeds, model.n_embd, true)) {
        err = "question prefix decode failed";
        return false;
    }

    auto smpl = make_sampler(p.temperature, p.top_k, p.top_p, p.repeat_penalty, p.seed,
            llama_vocab_n_tokens(vocab));

    std::vector<llama_token> ids;
    llama_pos t = spec.max_t;
    if (!generate_tokens(ctx, vocab, seq, t, p.max_new_tokens, -1, smpl.get(), ids, err)) {
        err = "answer generation failed: " + err;
        return false;
    }

    out.text = detokenize(vocab, ids);
    trim_text(out.text);
    split_think(out.text, out.think_text, out.answer);
    out.n_prompt_tokens = (int) spec.items.size();
    out.n_generated = (int) ids.size();
    out.truncated = out.n_generated >= p.max_new_tokens;
    return true;
}

bool sensenova_interleave(
        sensenova_model & model,
        const sensenova_interleave_params & p,
        sensenova_interleave_result & out,
        std::string & err) {
    if (p.prompt.empty()) {
        err = "prompt is empty";
        return false;
    }
    if (p.steps <= 0) {
        err = "steps must be positive";
        return false;
    }
    if (p.max_images <= 0) {
        err = "max_images must be positive";
        return false;
    }
    if (p.max_new_tokens <= 0) {
        err = "max_new_tokens must be positive";
        return false;
    }

    const int n_images = (int) p.images.size();
    const int min_pixels = 512*512;

    std::vector<sensenova_image> inputs;
    if (!prepare_input_images(p.images, p.input_max_pixels, inputs, err)) {
        return false;
    }

    // resolve the output size: explicit, first input aspect, or 1024x1024
    int width  = p.width;
    int height = p.height;
    if (width == 0 && height == 0) {
        if (n_images > 0) {
            const int target = p.auto_pixels;
            if (target < min_pixels) {
                err = "auto_pixels must be at least 512*512";
                return false;
            }
            if (!image_plan_size(inputs[0].w, inputs[0].h, target, target, width, height, err)) {
                err = "cannot derive the output size: " + err;
                return false;
            }
        } else {
            width  = 1024;
            height = 1024;
        }
    } else if (width <= 0 || height <= 0) {
        err = "width and height must be set together";
        return false;
    }
    if (width % GRID != 0 || height % GRID != 0) {
        err = "width and height must be positive multiples of 32";
        return false;
    }

    const int token_h = height/GRID;
    const int token_w = width/GRID;
    const int n_tokens = token_h*token_w;

    const llama_vocab * vocab = llama_model_get_vocab(model.model);
    const int n_embd = model.n_embd;

    if (!model.load_fm(width, height, err)) {
        err = "cannot load FM sidecar: " + err;
        return false;
    }
    fm_context & fm = *model.fm;
    if (!fm.has_und_vision()) {
        err = "the FM sidecar has no vision weights, regenerate it with retag.py";
        return false;
    }

    std::vector<std::vector<float>> img_embeds;
    std::vector<std::pair<int, int>> img_grid; // token_h, token_w
    std::vector<int> img_tok;
    if (!encode_input_images(model, inputs, img_embeds, img_grid, img_tok, err)) {
        return false;
    }

    if (count_placeholders(p.prompt) > n_images) {
        err = "the prompt has more <image> placeholders than input images";
        return false;
    }

    const std::string question = expand_images(add_missing_placeholders_plain(p.prompt, n_images), img_tok);
    const std::string system = p.system_prompt.empty() ? SYSTEM_MESSAGE_FOR_INTERLEAVE : p.system_prompt;

    prefix_spec spec;
    if (!build_prefix(vocab, build_interleave_prompt(system, question, p.think), img_grid, spec, err)) {
        err = "prompt: " + err;
        return false;
    }

    llama_token img_start = 0;
    llama_token img_end   = 0;
    llama_token eos       = 0;
    if (!token_id(vocab, "<img>", img_start) || !token_id(vocab, "</img>", img_end)) {
        err = "cannot find the image tokens in the vocabulary";
        return false;
    }
    if (!token_id(vocab, "<|im_end|>", eos)) {
        err = "cannot find the end token in the vocabulary";
        return false;
    }

    // largest batch: one prefix run, one input image block, or one generated image block
    uint32_t n_batch = (uint32_t) spec.items.size();
    n_batch = std::max<uint32_t>(n_batch, (uint32_t) n_tokens);
    for (int n : img_tok) {
        n_batch = std::max<uint32_t>(n_batch, (uint32_t) n);
    }

    const uint32_t n_ctx = (uint32_t) spec.items.size() + (uint32_t) p.max_new_tokens +
            (uint32_t) p.max_images * (uint32_t) (n_tokens + 2) + 8;

    fprintf(stderr, "interleave: prompt %zu tokens, image %dx%d (%d tokens), steps %d, max images %d, text budget %d%s\n",
            spec.items.size(), width, height, n_tokens, p.steps, p.max_images, p.max_new_tokens,
            p.think ? ", think" : "");

    auto ctx_holder = create_context(model, n_ctx, std::max<uint32_t>(n_batch, 1), 1, true, err);
    if (!ctx_holder) {
        return false;
    }
    llama_context * ctx = ctx_holder.get();

    const llama_seq_id seq = 0;
    if (!decode_prefix(ctx, seq, spec, img_embeds, n_embd, true)) {
        err = "prompt decode failed";
        return false;
    }

    auto smpl = make_sampler(p.temperature, p.top_k, p.top_p, p.repeat_penalty, p.seed,
            llama_vocab_n_tokens(vocab));

    const int n_vocab = llama_vocab_n_tokens(vocab);
    auto sample = [&]() -> llama_token {
        if (smpl != nullptr) {
            return llama_sampler_sample(smpl.get(), ctx, -1);
        }
        const float * logits = llama_get_logits_ith(ctx, -1);
        if (logits == nullptr) {
            return LLAMA_TOKEN_NULL;
        }
        return argmax_token(logits, n_vocab);
    };

    const float noise_scale = noise_scale_for(fm.p, width, height);
    const float noise_ratio = noise_scale/fm.p.noise_scale_max;

    std::vector<float> timesteps;
    build_timesteps(p.steps, p.shift, timesteps);

    rng_normal rng((uint64_t) p.seed);

    std::vector<float> image_embeds((size_t) n_tokens*n_embd);
    std::vector<float> ts_emb(n_embd);
    std::vector<float> hidden((size_t) n_tokens*n_embd);
    std::vector<float> x_pred;
    std::vector<float> v((size_t) 3*width*height);
    std::vector<float> img((size_t) 3*width*height);
    std::vector<float> img_whc((size_t) 3*width*height);
    std::vector<float> und_whc((size_t) 3*width*height);
    std::vector<float> und_embeds((size_t) n_tokens*n_embd);
    std::vector<prefix_item> append_items((size_t) n_tokens);
    std::vector<llama_token> text_ids;

    llama_pos t = spec.max_t;
    int n_generated = 0;
    bool truncated = false;

    llama_token next = sample();
    while (true) {
        if (next == LLAMA_TOKEN_NULL) {
            err = "no logits after the prompt pass";
            return false;
        }
        if (next == eos) {
            break;
        }

        if (next == img_start) {
            if ((int) out.images.size() >= p.max_images) {
                break;
            }

            const int img_idx = (int) out.images.size();
            fprintf(stderr, "image %d/%d: %dx%d, %d steps\n", img_idx + 1, p.max_images, width, height, p.steps);

            ++t;
            if (!decode_one_token(ctx, seq, img_start, t)) {
                err = "cannot append the image start";
                return false;
            }

            for (size_t i = 0; i < img.size(); ++i) {
                img[i] = noise_scale*rng.next();
            }

            llama_set_causal_attn(ctx, false);
            llama_set_gen_pass(ctx, true);
            llama_set_embeddings(ctx, true);

            for (int step = 0; step < p.steps; ++step) {
                const float t_cur  = timesteps[(size_t) step];
                const float t_next = timesteps[(size_t) step + 1];

                to_whc(img, img_whc, width, height);
                if (!fm.encode_image(img_whc.data(), image_embeds.data(), err)) {
                    err = "vision encode failed: " + err;
                    return false;
                }
                if (!fm.embed_cond(t_cur, noise_ratio, ts_emb.data(), err)) {
                    err = "timestep embedding failed: " + err;
                    return false;
                }
                for (int j = 0; j < n_tokens; ++j) {
                    float * e = image_embeds.data() + (size_t) j*n_embd;
                    for (int c = 0; c < n_embd; ++c) {
                        e[c] += ts_emb[c];
                    }
                }

                const float denom = std::max(1.0f - t_cur, 0.05f);
                if (!predict_velocity(fm, ctx, seq, t + 1, token_h, token_w, width, height,
                        image_embeds.data(), img.data(), denom, hidden, x_pred, v, err)) {
                    err = "image generation failed: " + err;
                    return false;
                }

                const float dt = t_next - t_cur;
                for (size_t i = 0; i < img.size(); ++i) {
                    img[i] += dt*v[i];
                }
                fprintf(stderr, "  step %d/%d: t=%.4f -> %.4f\n", step + 1, p.steps, t_cur, t_next);
            }

            sensenova_result res;
            res.width  = width;
            res.height = height;
            to_rgb(img, width, height, res.rgb);
            out.images.push_back(std::move(res));

            // the model reads the generated image back as an input image
            gen_to_fm_input(img, und_whc, width, height);
            if (!fm.encode_ref_image(und_whc.data(), width, height, und_embeds.data(), err)) {
                err = "cannot re-encode the generated image: " + err;
                return false;
            }
            for (int i = 0; i < n_tokens; ++i) {
                prefix_item & it = append_items[(size_t) i];
                it.t = t + 1;
                it.h = i/token_w;
                it.w = i%token_w;
            }
            if (!decode_embd_run(ctx, seq, und_embeds.data(), append_items.data(), n_tokens)) {
                err = "cannot append the generated image";
                return false;
            }
            if (!decode_one_token(ctx, seq, img_end, t + 2)) {
                err = "cannot append the image end";
                return false;
            }
            t += 2;

            out.text += detokenize(vocab, text_ids);
            text_ids.clear();
            out.text += "<image>";

            next = sample();
            continue;
        }

        if (n_generated >= p.max_new_tokens) {
            truncated = true;
            break;
        }
        text_ids.push_back(next);
        ++n_generated;
        ++t;
        if (!decode_one_token(ctx, seq, next, t)) {
            err = "token decode failed";
            return false;
        }
        next = sample();
    }

    out.text += detokenize(vocab, text_ids);
    split_think(out.text, out.think_text, out.answer);
    out.n_prompt_tokens = (int) spec.items.size();
    out.n_generated = n_generated;
    out.truncated = truncated;
    return true;
}
