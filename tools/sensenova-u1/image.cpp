#include "image.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wcast-qual"
#endif
#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

namespace {

constexpr int k_grid = 32;
constexpr int k_max_image_pixels = 100*1024*1024;

const float k_mean[3] = { 0.485f, 0.456f, 0.406f };
const float k_std [3] = { 0.229f, 0.224f, 0.225f };

// Python round() rounds half to even
int round_factor(float v) {
    return (int) nearbyintf(v/k_grid)*k_grid;
}

int floor_factor(float v) {
    return (int) floorf(v/k_grid)*k_grid;
}

int ceil_factor(float v) {
    return (int) ceilf(v/k_grid)*k_grid;
}

// stb decodes RGBA; composite over white like PIL paste with the alpha mask
bool decode(const unsigned char * data, int len, sensenova_image & img, std::string & err) {
    int w = 0;
    int h = 0;
    int n = 0;
    if (stbi_info_from_memory(data, len, &w, &h, &n) == 0) {
        err = "unsupported image format";
        return false;
    }
    if (w <= 0 || h <= 0) {
        err = "invalid image size";
        return false;
    }
    if ((int64_t) w*h > k_max_image_pixels) {
        err = "image is too large";
        return false;
    }

    unsigned char * px = stbi_load_from_memory(data, len, &w, &h, &n, 4);
    if (px == nullptr) {
        err = "cannot decode image";
        return false;
    }

    img.w = w;
    img.h = h;
    img.rgb.resize((size_t) w*h*3);
    for (size_t i = 0; i < (size_t) w*h; ++i) {
        const unsigned a = px[i*4 + 3];
        for (int c = 0; c < 3; ++c) {
            const unsigned v = px[i*4 + c];
            img.rgb[i*3 + c] = (uint8_t) ((v*a + 255u*(255u - a) + 127u)/255u);
        }
    }

    stbi_image_free(px);
    return true;
}

// triangle filter, widened when downscaling to avoid aliasing
struct taps {
    int i0 = 0;
    std::vector<float> w;
};

std::vector<taps> make_taps(int src_n, int dst_n) {
    std::vector<taps> out((size_t) dst_n);

    const float scale = (float) dst_n/src_n;
    const float filter_scale = std::max(1.0f, 1.0f/scale);
    const float support = filter_scale;

    for (int i = 0; i < dst_n; ++i) {
        const float center = ((float) i + 0.5f)/scale - 0.5f;

        int i0 = (int) floorf(center - support + 0.5f);
        int i1 = (int) ceilf (center + support - 0.5f);
        i0 = std::max(i0, 0);
        i1 = std::min(i1, src_n - 1);

        taps & t = out[(size_t) i];
        t.i0 = i0;
        t.w.resize((size_t) (i1 - i0 + 1));

        float sum = 0.0f;
        for (int j = i0; j <= i1; ++j) {
            const float v = std::max(0.0f, 1.0f - fabsf((float) j - center)/filter_scale);
            t.w[(size_t) (j - i0)] = v;
            sum += v;
        }
        if (sum > 0.0f) {
            for (float & v : t.w) {
                v /= sum;
            }
        }
    }
    return out;
}

} // namespace

bool image_load_file(const std::string & path, sensenova_image & img, std::string & err) {
    FILE * f = fopen(path.c_str(), "rb");
    if (f == nullptr) {
        err = "cannot open image: " + path;
        return false;
    }
    std::vector<uint8_t> data;
    uint8_t buf[64*1024];
    size_t n = 0;
    while ((n = fread(buf, 1, sizeof(buf), f)) > 0) {
        data.insert(data.end(), buf, buf + n);
        if (data.size() > (size_t) std::numeric_limits<int>::max()) {
            fclose(f);
            err = "image file is too large";
            return false;
        }
    }
    fclose(f);

    if (data.empty()) {
        err = "image file is empty: " + path;
        return false;
    }
    return decode(data.data(), (int) data.size(), img, err);
}

bool image_load_memory(const void * data, size_t size, sensenova_image & img, std::string & err) {
    if (data == nullptr || size == 0) {
        err = "empty image data";
        return false;
    }
    if (size > (size_t) std::numeric_limits<int>::max()) {
        err = "image data is too large";
        return false;
    }
    return decode((const unsigned char *) data, (int) size, img, err);
}

bool image_plan_size(int w, int h, int min_pixels, int max_pixels, int & out_w, int & out_h, std::string & err) {
    if (w <= 0 || h <= 0) {
        err = "invalid image size";
        return false;
    }
    if (min_pixels <= 0 || max_pixels < min_pixels) {
        err = "invalid pixel bounds";
        return false;
    }
    if ((float) std::max(w, h)/std::min(w, h) > 200.0f) {
        err = "image aspect ratio is too extreme";
        return false;
    }

    int h_bar = std::max(k_grid, round_factor((float) h));
    int w_bar = std::max(k_grid, round_factor((float) w));

    if ((int64_t) h_bar*w_bar > max_pixels) {
        const float beta = sqrtf((float) w*h/max_pixels);
        h_bar = std::max(k_grid, floor_factor((float) h/beta));
        w_bar = std::max(k_grid, floor_factor((float) w/beta));
    } else if ((int64_t) h_bar*w_bar < min_pixels) {
        const float beta = sqrtf((float) min_pixels/((float) w*h));
        h_bar = ceil_factor((float) h*beta);
        w_bar = ceil_factor((float) w*beta);
    }

    out_w = w_bar;
    out_h = h_bar;
    return true;
}

bool image_resize(const sensenova_image & src, int w, int h, sensenova_image & out, std::string & err) {
    if (w <= 0 || h <= 0 || src.w <= 0 || src.h <= 0) {
        err = "invalid resize size";
        return false;
    }
    if (src.w == w && src.h == h) {
        out = src;
        return true;
    }

    const std::vector<taps> tx = make_taps(src.w, w);
    const std::vector<taps> ty = make_taps(src.h, h);

    // horizontal pass: [src.h][w][3]
    std::vector<float> tmp((size_t) src.h*w*3);
    for (int y = 0; y < src.h; ++y) {
        const uint8_t * row = src.rgb.data() + (size_t) y*src.w*3;
        for (int x = 0; x < w; ++x) {
            const taps & t = tx[(size_t) x];
            float r = 0.0f;
            float g = 0.0f;
            float b = 0.0f;
            for (size_t k = 0; k < t.w.size(); ++k) {
                const uint8_t * p = row + (size_t) (t.i0 + k)*3;
                r += t.w[k]*p[0];
                g += t.w[k]*p[1];
                b += t.w[k]*p[2];
            }
            float * dst = tmp.data() + ((size_t) y*w + x)*3;
            dst[0] = r;
            dst[1] = g;
            dst[2] = b;
        }
    }

    // vertical pass
    out.w = w;
    out.h = h;
    out.rgb.resize((size_t) w*h*3);
    for (int y = 0; y < h; ++y) {
        const taps & t = ty[(size_t) y];
        for (int x = 0; x < w; ++x) {
            float r = 0.0f;
            float g = 0.0f;
            float b = 0.0f;
            for (size_t k = 0; k < t.w.size(); ++k) {
                const float * p = tmp.data() + ((size_t) (t.i0 + k)*w + x)*3;
                r += t.w[k]*p[0];
                g += t.w[k]*p[1];
                b += t.w[k]*p[2];
            }
            uint8_t * dst = out.rgb.data() + ((size_t) y*w + x)*3;
            dst[0] = (uint8_t) std::min(255.0f, std::max(0.0f, r + 0.5f));
            dst[1] = (uint8_t) std::min(255.0f, std::max(0.0f, g + 0.5f));
            dst[2] = (uint8_t) std::min(255.0f, std::max(0.0f, b + 0.5f));
        }
    }
    return true;
}

bool image_prepare(const sensenova_image & src, int min_pixels, int max_pixels, sensenova_image & out, std::string & err) {
    int w = 0;
    int h = 0;
    if (!image_plan_size(src.w, src.h, min_pixels, max_pixels, w, h, err)) {
        return false;
    }
    return image_resize(src, w, h, out, err);
}

void image_to_fm_input(const sensenova_image & img, std::vector<float> & out) {
    const size_t n = (size_t) img.w*img.h;
    out.resize(n*3);
    for (int y = 0; y < img.h; ++y) {
        for (int x = 0; x < img.w; ++x) {
            const uint8_t * p = img.rgb.data() + ((size_t) y*img.w + x)*3;
            const size_t base = (size_t) x + (size_t) img.w*y;
            for (int c = 0; c < 3; ++c) {
                out[base + (size_t) img.w*img.h*c] = ((float) p[c]/255.0f - k_mean[c])/k_std[c];
            }
        }
    }
}
