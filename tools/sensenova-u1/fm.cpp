#include "fm.h"

#include "gguf.h"

#include <cmath>
#include <cstring>
#include <cstdio>

namespace {

ggml_context * make_ctx(size_t nbytes) {
    ggml_init_params ip = {
        /*.mem_size   =*/ nbytes,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    return ggml_init(ip);
}

bool run_graph(ggml_backend_t backend, ggml_cgraph * gf, std::string & err) {
    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        err = "graph compute failed";
        return false;
    }
    return true;
}

bool get_kv_u32(gguf_context * gf, const char * key, uint32_t & out, std::string & err) {
    const int64_t id = gguf_find_key(gf, key);
    if (id < 0) {
        err = std::string("missing key: ") + key;
        return false;
    }
    out = gguf_get_val_u32(gf, id);
    return true;
}

bool get_kv_f32(gguf_context * gf, const char * key, float & out, std::string & err) {
    const int64_t id = gguf_find_key(gf, key);
    if (id < 0) {
        err = std::string("missing key: ") + key;
        return false;
    }
    out = gguf_get_val_f32(gf, id);
    return true;
}

bool get_kv_bool(gguf_context * gf, const char * key, bool & out, std::string & err) {
    const int64_t id = gguf_find_key(gf, key);
    if (id < 0) {
        err = std::string("missing key: ") + key;
        return false;
    }
    out = gguf_get_val_bool(gf, id);
    return true;
}

bool get_kv_str(gguf_context * gf, const char * key, std::string & out, std::string & err) {
    const int64_t id = gguf_find_key(gf, key);
    if (id < 0) {
        err = std::string("missing key: ") + key;
        return false;
    }
    out = gguf_get_val_str(gf, id);
    return true;
}

} // namespace

fm_vision_encoder::~fm_vision_encoder() {
    if (ga)      ggml_gallocr_free(ga);
    if (ctx)     ggml_free(ctx);
    if (buf_pos) ggml_backend_buffer_free(buf_pos);
    if (ctx_pos) ggml_free(ctx_pos);
}

bool fm_vision_encoder::build(
        ggml_backend_t backend,
        const fm_params & p,
        int img_w_,
        int img_h_,
        ggml_tensor * patch_w,
        ggml_tensor * patch_b,
        ggml_tensor * dense_w,
        ggml_tensor * dense_b,
        std::string & err) {
    img_w = img_w_;
    img_h = img_h_;

    if (img_w <= 0 || img_h <= 0 || img_w % 32 != 0 || img_h % 32 != 0) {
        err = "image size must be a positive multiple of 32";
        return false;
    }

    const int64_t pxx = img_w/p.patch_size;
    const int64_t pxy = img_h/p.patch_size;
    const int64_t n_patch = pxx*pxy;

    ctx_pos = make_ctx(1u << 20);
    if (ctx_pos == nullptr) {
        err = "cannot create position context";
        return false;
    }
    pos_x = ggml_new_tensor_1d(ctx_pos, GGML_TYPE_I32, n_patch);
    pos_y = ggml_new_tensor_1d(ctx_pos, GGML_TYPE_I32, n_patch);

    buf_pos = ggml_backend_alloc_ctx_tensors(ctx_pos, backend);
    if (buf_pos == nullptr) {
        err = "cannot allocate position buffer";
        return false;
    }
    {
        std::vector<int32_t> x(n_patch), y(n_patch);
        for (int64_t j = 0; j < n_patch; ++j) {
            x[j] = (int32_t) (j % pxx);
            y[j] = (int32_t) (j / pxx);
        }
        ggml_backend_tensor_set(pos_x, x.data(), 0, n_patch*sizeof(int32_t));
        ggml_backend_tensor_set(pos_y, y.data(), 0, n_patch*sizeof(int32_t));
    }

    // vision patch embedder: conv 16/16, rope, conv 2/2
    ctx = make_ctx(128u << 20);
    if (ctx == nullptr) {
        err = "cannot create vision embedder context";
        return false;
    }
    {
        ggml_context * c = ctx;
        in = ggml_new_tensor_3d(c, GGML_TYPE_F32, img_w, img_h, 3);
        ggml_set_input(in);

        ggml_tensor * x = ggml_conv_2d_direct(c, patch_w, in, p.patch_size, p.patch_size, 0, 0, 1, 1);
        x = ggml_add(c, x, ggml_reshape_3d(c, patch_b, 1, 1, p.vision_n_embd));
        x = ggml_gelu_erf(c, x);

        x = ggml_cont(c, ggml_permute(c, x, 1, 2, 0, 3));
        x = ggml_reshape_3d(c, x, p.vision_n_embd, 1, n_patch);

        const int64_t n_half = p.vision_n_embd/2;
        ggml_tensor * xa = ggml_view_3d(c, x, n_half, 1, n_patch, x->nb[1], x->nb[2], 0);
        ggml_tensor * xb = ggml_view_3d(c, x, n_half, 1, n_patch, x->nb[1], x->nb[2], n_half*x->nb[0]);

        xa = ggml_rope_ext(c, xa, pos_x, nullptr, n_half, GGML_ROPE_TYPE_NORMAL, 0,
                p.rope_freq_base, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
        xb = ggml_rope_ext(c, xb, pos_y, nullptr, n_half, GGML_ROPE_TYPE_NORMAL, 0,
                p.rope_freq_base, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);

        ggml_tensor * r = ggml_concat(c, xa, xb, 0);
        r = ggml_reshape_3d(c, r, p.vision_n_embd, pxx, pxy);
        r = ggml_cont(c, ggml_permute(c, r, 2, 0, 1, 3));

        ggml_tensor * y = ggml_conv_2d_direct(c, dense_w, r, p.merge_size, p.merge_size, 0, 0, 1, 1);
        y = ggml_add(c, y, ggml_reshape_3d(c, dense_b, 1, 1, p.n_embd));

        out = ggml_cont(c, ggml_permute(c, y, 1, 2, 0, 3));
        ggml_set_output(out);
    }
    if (out->ne[0] != p.n_embd || out->ne[1] != pxx/2 || out->ne[2] != pxy/2) {
        err = "unexpected vision embedder output shape";
        return false;
    }

    gf = ggml_new_graph_custom(ctx, 4096, false);
    ggml_build_forward_expand(gf, out);
    ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_alloc_graph(ga, gf)) {
        err = "cannot allocate vision embedder graph";
        return false;
    }

    return true;
}

bool fm_vision_encoder::encode(ggml_backend_t backend, const float * img, float * out_t, std::string & err) {
    ggml_backend_tensor_set(in, img, 0, ggml_nbytes(in));
    if (!run_graph(backend, gf, err)) {
        return false;
    }
    ggml_backend_tensor_get(out, out_t, 0, ggml_nbytes(out));
    return true;
}

fm_context::~fm_context() {
    if (ga_ts)  ggml_gallocr_free(ga_ts);
    if (ga_h1)  ggml_gallocr_free(ga_h1);
    if (ga_h2)  ggml_gallocr_free(ga_h2);
    if (ga_hl)  ggml_gallocr_free(ga_hl);

    if (ctx_ts)  ggml_free(ctx_ts);
    if (ctx_h1)  ggml_free(ctx_h1);
    if (ctx_h2)  ggml_free(ctx_h2);
    if (ctx_hl)  ggml_free(ctx_hl);

    if (buf_w)  ggml_backend_buffer_free(buf_w);
    if (ctx_w)  ggml_free(ctx_w);
}

bool fm_context::load(
        const std::string & path,
        ggml_backend_t backend_,
        int img_w,
        int img_h,
        std::string & err) {
    backend = backend_;

    // 0 x 0 loads the weights only, for callers that do not generate images
    const bool with_image = img_w != 0 || img_h != 0;
    if (with_image && (img_w <= 0 || img_h <= 0 || img_w % 32 != 0 || img_h % 32 != 0)) {
        err = "image size must be a positive multiple of 32";
        return false;
    }

    ggml_context * ctx_load = nullptr;
    gguf_init_params gp = {
        /*.no_alloc =*/ false,
        /*.ctx      =*/ &ctx_load,
    };
    gguf_context * gf = gguf_init_from_file(path.c_str(), gp);
    if (gf == nullptr || ctx_load == nullptr) {
        err = "cannot load FM GGUF: " + path;
        if (gf) gguf_free(gf);
        return false;
    }

    std::string head_type;
    uint32_t u32 = 0;

    bool ok = true;
    ok = ok && get_kv_str (gf, "sensenova_u1_fm.head_type", head_type, err);
    ok = ok && get_kv_u32 (gf, "sensenova_u1_fm.patch_size", u32, err);
    if (ok) p.patch_size = (int) u32;
    ok = ok && get_kv_u32 (gf, "sensenova_u1_fm.merge_size", u32, err);
    if (ok) p.merge_size = (int) u32;
    ok = ok && get_kv_u32 (gf, "sensenova_u1_fm.n_embd", u32, err);
    if (ok) p.n_embd = (int) u32;
    ok = ok && get_kv_u32 (gf, "sensenova_u1_fm.vision_n_embd", u32, err);
    if (ok) p.vision_n_embd = (int) u32;
    ok = ok && get_kv_f32 (gf, "sensenova_u1_fm.rope_freq_base", p.rope_freq_base, err);
    ok = ok && get_kv_f32 (gf, "sensenova_u1_fm.noise_scale", p.noise_scale, err);
    ok = ok && get_kv_u32 (gf, "sensenova_u1_fm.noise_scale_base_seq_len", u32, err);
    if (ok) p.noise_scale_base_seq_len = (int) u32;
    ok = ok && get_kv_f32 (gf, "sensenova_u1_fm.noise_scale_max_value", p.noise_scale_max, err);
    ok = ok && get_kv_bool(gf, "sensenova_u1_fm.add_noise_scale_embedding", p.add_noise_scale_emb, err);
    ok = ok && get_kv_str (gf, "sensenova_u1_fm.noise_scale_mode", p.noise_scale_mode, err);
    if (!ok) {
        gguf_free(gf);
        return false;
    }

    p.head_conv = head_type == "conv";
    if (!p.head_conv && head_type != "linear") {
        err = "unknown head type: " + head_type;
        gguf_free(gf);
        return false;
    }
    if (p.patch_size != 16 || p.merge_size != 2) {
        err = "unsupported patch / merge size";
        gguf_free(gf);
        return false;
    }

    const int64_t pxx = with_image ? img_w/p.patch_size : 0;
    const int64_t pxy = with_image ? img_h/p.patch_size : 0;
    const int64_t n_tok = pxx*pxy/(p.merge_size*p.merge_size);

    ctx_w = make_ctx(4u << 20);
    if (ctx_w == nullptr) {
        err = "cannot create FM weight context";
        gguf_free(gf);
        return false;
    }

    // create weight tensors with the shapes stored in the sidecar, then copy
    std::vector<ggml_tensor *> dst_list;
    std::vector<ggml_tensor *> src_list;

    auto take = [&](const char * name, int n_dims, const int64_t * exp_ne) -> ggml_tensor * {
        const int64_t id = gguf_find_tensor(gf, name);
        if (id < 0) {
            err = std::string("missing tensor: ") + name;
            return nullptr;
        }
        ggml_tensor * src = ggml_get_tensor(ctx_load, name);
        if (src->type != GGML_TYPE_F16 && src->type != GGML_TYPE_F32) {
            err = std::string("unsupported tensor type: ") + name;
            return nullptr;
        }
        for (int i = 0; i < n_dims; ++i) {
            if (src->ne[i] != exp_ne[i]) {
                err = std::string("unexpected shape: ") + name;
                return nullptr;
            }
        }
        ggml_tensor * dst = ggml_new_tensor(ctx_w, src->type, GGML_MAX_DIMS, src->ne);
        dst_list.push_back(dst);
        src_list.push_back(src);
        return dst;
    };

    const int64_t ne_patch_w[4] = { p.patch_size, p.patch_size, 3, p.vision_n_embd };
    const int64_t ne_dense_w[4] = { p.merge_size, p.merge_size, p.vision_n_embd, p.n_embd };
    patch_w = take("vision.patch_embd.weight", 4, ne_patch_w);
    ok = patch_w != nullptr;
    {
        const int64_t ne[1] = { p.vision_n_embd };
        ok = ok && (patch_b = take("vision.patch_embd.bias", 1, ne)) != nullptr;
    }
    ok = ok && (dense_w = take("vision.dense_embd.weight", 4, ne_dense_w)) != nullptr;
    {
        const int64_t ne[1] = { p.n_embd };
        ok = ok && (dense_b = take("vision.dense_embd.bias", 1, ne)) != nullptr;
    }

    // input images use a separate vision embedder, not all sidecars have it
    const bool has_und =
        gguf_find_tensor(gf, "vision_und.patch_embd.weight") >= 0 ||
        gguf_find_tensor(gf, "vision_und.patch_embd.bias")   >= 0 ||
        gguf_find_tensor(gf, "vision_und.dense_embd.weight") >= 0 ||
        gguf_find_tensor(gf, "vision_und.dense_embd.bias")   >= 0;
    if (has_und) {
        patch_w_und = take("vision_und.patch_embd.weight", 4, ne_patch_w);
        ok = patch_w_und != nullptr;
        {
            const int64_t ne[1] = { p.vision_n_embd };
            ok = ok && (patch_b_und = take("vision_und.patch_embd.bias", 1, ne)) != nullptr;
        }
        ok = ok && (dense_w_und = take("vision_und.dense_embd.weight", 4, ne_dense_w)) != nullptr;
        {
            const int64_t ne[1] = { p.n_embd };
            ok = ok && (dense_b_und = take("vision_und.dense_embd.bias", 1, ne)) != nullptr;
        }
    }

    {
        const int64_t ne0[2] = { 256, p.n_embd };
        const int64_t ne1[2] = { p.n_embd, p.n_embd };
        const int64_t neb[1] = { p.n_embd };
        ok = ok && (ts0_w = take("timestep.mlp.0.weight", 2, ne0)) != nullptr;
        ok = ok && (ts0_b = take("timestep.mlp.0.bias", 1, neb)) != nullptr;
        ok = ok && (ts2_w = take("timestep.mlp.2.weight", 2, ne1)) != nullptr;
        ok = ok && (ts2_b = take("timestep.mlp.2.bias", 1, neb)) != nullptr;
        if (p.add_noise_scale_emb) {
            ok = ok && (ns0_w = take("noise_scale.mlp.0.weight", 2, ne0)) != nullptr;
            ok = ok && (ns0_b = take("noise_scale.mlp.0.bias", 1, neb)) != nullptr;
            ok = ok && (ns2_w = take("noise_scale.mlp.2.weight", 2, ne1)) != nullptr;
            ok = ok && (ns2_b = take("noise_scale.mlp.2.bias", 1, neb)) != nullptr;
        }
    }

    if (p.head_conv) {
        // conv layers run after a host-side pixel shuffle, so their input channels are n_embd/4 and n_embd/16
        const int64_t ne1[4] = { 3, 3, p.n_embd/4,  p.n_embd/4 };
        const int64_t ne2[4] = { 3, 3, p.n_embd/16, 192 };
        const int64_t b1[1]   = { p.n_embd/4 };
        const int64_t b2[1]   = { 192 };
        ok = ok && (conv1_w = take("head.conv1.weight", 4, ne1)) != nullptr;
        ok = ok && (conv1_b = take("head.conv1.bias", 1, b1)) != nullptr;
        ok = ok && (conv2_w = take("head.conv2.weight", 4, ne2)) != nullptr;
        ok = ok && (conv2_b = take("head.conv2.bias", 1, b2)) != nullptr;
    } else {
        const int64_t ne0[2] = { p.n_embd, 4096 };
        const int64_t ne2[2] = { 4096, 3*(2*p.patch_size)*(2*p.patch_size) };
        const int64_t b0[1]  = { 4096 };
        const int64_t b2[1]  = { 3*(2*p.patch_size)*(2*p.patch_size) };
        ok = ok && (fc0_w = take("head.fc0.weight", 2, ne0)) != nullptr;
        ok = ok && (fc0_b = take("head.fc0.bias", 1, b0)) != nullptr;
        ok = ok && (fc2_w = take("head.fc2.weight", 2, ne2)) != nullptr;
        ok = ok && (fc2_b = take("head.fc2.bias", 1, b2)) != nullptr;
    }

    if (!ok) {
        gguf_free(gf);
        return false;
    }

    buf_w = ggml_backend_alloc_ctx_tensors(ctx_w, backend);
    if (buf_w == nullptr) {
        err = "cannot allocate FM weight buffer";
        gguf_free(gf);
        return false;
    }
    for (size_t i = 0; i < dst_list.size(); ++i) {
        ggml_backend_tensor_set(dst_list[i], src_list[i]->data, 0, ggml_nbytes(src_list[i]));
    }

    if (with_image && !enc_gen.build(backend, p, img_w, img_h, patch_w, patch_b, dense_w, dense_b, err)) {
        gguf_free(gf);
        return false;
    }

    // timestep and noise scale embedders
    ctx_ts = make_ctx(8u << 20);
    {
        ggml_context * c = ctx_ts;
        in_ts = ggml_new_tensor_1d(c, GGML_TYPE_F32, 256);
        in_ns = ggml_new_tensor_1d(c, GGML_TYPE_F32, 256);
        ggml_set_input(in_ts);
        ggml_set_input(in_ns);

        ggml_tensor * h = ggml_add(c, ggml_mul_mat(c, ts0_w, in_ts), ts0_b);
        h = ggml_silu(c, h);
        h = ggml_add(c, ggml_mul_mat(c, ts2_w, h), ts2_b);

        if (p.add_noise_scale_emb) {
            ggml_tensor * g = ggml_add(c, ggml_mul_mat(c, ns0_w, in_ns), ns0_b);
            g = ggml_silu(c, g);
            g = ggml_add(c, ggml_mul_mat(c, ns2_w, g), ns2_b);
            h = ggml_add(c, h, g);
        }

        out_ts = h;
        ggml_set_output(out_ts);
    }
    gf_ts = ggml_new_graph_custom(ctx_ts, 4096, false);
    ggml_build_forward_expand(gf_ts, out_ts);
    ga_ts = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_alloc_graph(ga_ts, gf_ts)) {
        err = "cannot allocate timestep graph";
        gguf_free(gf);
        return false;
    }

    if (with_image && p.head_conv) {
        const int64_t hw = pxx/2;
        const int64_t hh = pxy/2;

        ctx_h1 = make_ctx(16u << 20);
        {
            ggml_context * c = ctx_h1;
            in_h1 = ggml_new_tensor_3d(c, GGML_TYPE_F32, 2*hw, 2*hh, p.n_embd/4);
            ggml_set_input(in_h1);

            ggml_tensor * y = ggml_conv_2d_direct(c, conv1_w, in_h1, 1, 1, 1, 1, 1, 1);
            y = ggml_add(c, y, ggml_reshape_3d(c, conv1_b, 1, 1, p.n_embd/4));
            out_h1 = ggml_gelu_erf(c, y);
            ggml_set_output(out_h1);
        }
        gf_h1 = ggml_new_graph_custom(ctx_h1, 4096, false);
        ggml_build_forward_expand(gf_h1, out_h1);
        ga_h1 = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        if (!ggml_gallocr_alloc_graph(ga_h1, gf_h1)) {
            err = "cannot allocate head conv1 graph";
            gguf_free(gf);
            return false;
        }

        ctx_h2 = make_ctx(16u << 20);
        {
            ggml_context * c = ctx_h2;
            in_h2 = ggml_new_tensor_3d(c, GGML_TYPE_F32, 4*hw, 4*hh, p.n_embd/16);
            ggml_set_input(in_h2);

            ggml_tensor * y = ggml_conv_2d_direct(c, conv2_w, in_h2, 1, 1, 1, 1, 1, 1);
            out_h2 = ggml_add(c, y, ggml_reshape_3d(c, conv2_b, 1, 1, 192));
            ggml_set_output(out_h2);
        }
        gf_h2 = ggml_new_graph_custom(ctx_h2, 4096, false);
        ggml_build_forward_expand(gf_h2, out_h2);
        ga_h2 = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        if (!ggml_gallocr_alloc_graph(ga_h2, gf_h2)) {
            err = "cannot allocate head conv2 graph";
            gguf_free(gf);
            return false;
        }
    } else if (with_image) {
        ctx_hl = make_ctx(16u << 20);
        {
            ggml_context * c = ctx_hl;
            in_hl = ggml_new_tensor_2d(c, GGML_TYPE_F32, p.n_embd, n_tok);
            ggml_set_input(in_hl);

            ggml_tensor * h = ggml_add(c, ggml_mul_mat(c, fc0_w, in_hl), fc0_b);
            h = ggml_gelu_erf(c, h);
            out_hl = ggml_add(c, ggml_mul_mat(c, fc2_w, h), fc2_b);
            ggml_set_output(out_hl);
        }
        gf_hl = ggml_new_graph_custom(ctx_hl, 4096, false);
        ggml_build_forward_expand(gf_hl, out_hl);
        ga_hl = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        if (!ggml_gallocr_alloc_graph(ga_hl, gf_hl)) {
            err = "cannot allocate head linear graph";
            gguf_free(gf);
            return false;
        }
    }

    gguf_free(gf);
    return true;
}

bool fm_context::encode_image(const float * img, float * out, std::string & err) {
    return enc_gen.encode(backend, img, out, err);
}

bool fm_context::encode_ref_image(const float * img, int img_w, int img_h, float * out, std::string & err) {
    if (!has_und_vision()) {
        err = "FM sidecar has no understanding vision weights, regenerate it with retag.py";
        return false;
    }

    fm_vision_encoder enc;
    if (!enc.build(backend, p, img_w, img_h, patch_w_und, patch_b_und, dense_w_und, dense_b_und, err)) {
        return false;
    }
    return enc.encode(backend, img, out, err);
}

bool fm_context::embed_cond(float t, float noise_ratio, float * out, std::string & err) {
    float ts[256];
    float ns[256];
    const int n_half = 128;
    for (int j = 0; j < n_half; ++j) {
        const float freq = expf(-logf(10000.0f)*(float) j/n_half);
        ts[j]          = cosf(t*freq);
        ts[j + n_half] = sinf(t*freq);
        ns[j]          = cosf(noise_ratio*freq);
        ns[j + n_half] = sinf(noise_ratio*freq);
    }
    ggml_backend_tensor_set(in_ts, ts, 0, sizeof(ts));
    ggml_backend_tensor_set(in_ns, ns, 0, sizeof(ns));
    if (!run_graph(backend, gf_ts, err)) {
        return false;
    }
    ggml_backend_tensor_get(out_ts, out, 0, p.n_embd*sizeof(float));
    return true;
}

bool fm_context::head_conv1(const float * in, float * out, std::string & err) {
    ggml_backend_tensor_set(in_h1, in, 0, ggml_nbytes(in_h1));
    if (!run_graph(backend, gf_h1, err)) {
        return false;
    }
    ggml_backend_tensor_get(out_h1, out, 0, ggml_nbytes(out_h1));
    return true;
}

bool fm_context::head_conv2(const float * in, float * out, std::string & err) {
    ggml_backend_tensor_set(in_h2, in, 0, ggml_nbytes(in_h2));
    if (!run_graph(backend, gf_h2, err)) {
        return false;
    }
    ggml_backend_tensor_get(out_h2, out, 0, ggml_nbytes(out_h2));
    return true;
}

bool fm_context::head_linear(const float * in, int n_tokens, float * out, std::string & err) {
    if ((int) in_hl->ne[1] != n_tokens) {
        err = "unexpected token count for the linear head";
        return false;
    }
    ggml_backend_tensor_set(in_hl, in, 0, ggml_nbytes(in_hl));
    if (!run_graph(backend, gf_hl, err)) {
        return false;
    }
    ggml_backend_tensor_get(out_hl, out, 0, ggml_nbytes(out_hl));
    return true;
}
