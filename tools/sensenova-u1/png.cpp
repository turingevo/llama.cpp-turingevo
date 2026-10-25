#include "png.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

uint32_t crc32_update(uint32_t crc, const uint8_t * data, size_t len) {
    static uint32_t table[256];
    static bool init = false;
    if (!init) {
        for (uint32_t i = 0; i < 256; ++i) {
            uint32_t c = i;
            for (int k = 0; k < 8; ++k) {
                c = (c & 1) ? 0xedb88320u ^ (c >> 1) : c >> 1;
            }
            table[i] = c;
        }
        init = true;
    }
    crc ^= 0xffffffffu;
    for (size_t i = 0; i < len; ++i) {
        crc = table[(crc ^ data[i]) & 0xff] ^ (crc >> 8);
    }
    return crc ^ 0xffffffffu;
}

uint32_t adler32_update(uint32_t adler, const uint8_t * data, size_t len) {
    uint32_t a = adler & 0xffff;
    uint32_t b = adler >> 16;
    for (size_t i = 0; i < len; ++i) {
        a = (a + data[i]) % 65521;
        b = (b + a) % 65521;
    }
    return (b << 16) | a;
}

void put_u32(std::vector<uint8_t> & out, uint32_t v) {
    out.push_back((uint8_t) (v >> 24));
    out.push_back((uint8_t) (v >> 16));
    out.push_back((uint8_t) (v >> 8));
    out.push_back((uint8_t) v);
}

void put_chunk(std::vector<uint8_t> & out, const char * type, const std::vector<uint8_t> & data) {
    put_u32(out, (uint32_t) data.size());
    const size_t start = out.size();
    out.insert(out.end(), type, type + 4);
    out.insert(out.end(), data.begin(), data.end());
    put_u32(out, crc32_update(0, out.data() + start, out.size() - start));
}

} // namespace

bool png_encode_rgb(
        const uint8_t * rgb,
        int w,
        int h,
        std::vector<uint8_t> & out,
        std::string & err) {
    if (w <= 0 || h <= 0) {
        err = "invalid image size";
        return false;
    }

    const size_t stride = (size_t) w*3;
    const size_t raw_size = (size_t) h*(stride + 1);

    std::vector<uint8_t> raw(raw_size);
    for (int y = 0; y < h; ++y) {
        raw[(size_t) y*(stride + 1)] = 0; // filter type none
        memcpy(raw.data() + (size_t) y*(stride + 1) + 1, rgb + (size_t) y*stride, stride);
    }

    // zlib stream with stored deflate blocks
    std::vector<uint8_t> z;
    z.push_back(0x78);
    z.push_back(0x01);
    for (size_t off = 0; off < raw_size;) {
        const size_t n = std::min<size_t>(65535, raw_size - off);
        const bool last = off + n == raw_size;
        z.push_back(last ? 1 : 0);
        z.push_back((uint8_t) (n & 0xff));
        z.push_back((uint8_t) (n >> 8));
        z.push_back((uint8_t) (~n & 0xff));
        z.push_back((uint8_t) ((~n >> 8) & 0xff));
        z.insert(z.end(), raw.begin() + off, raw.begin() + off + n);
        off += n;
    }
    put_u32(z, adler32_update(1, raw.data(), raw_size));

    std::vector<uint8_t> ihdr;
    put_u32(ihdr, (uint32_t) w);
    put_u32(ihdr, (uint32_t) h);
    ihdr.push_back(8); // bit depth
    ihdr.push_back(2); // color type RGB
    ihdr.push_back(0); // compression
    ihdr.push_back(0); // filter
    ihdr.push_back(0); // interlace

    static const uint8_t sig[8] = { 0x89, 'P', 'N', 'G', '\r', '\n', 0x1a, '\n' };
    out.assign(sig, sig + 8);
    put_chunk(out, "IHDR", ihdr);
    put_chunk(out, "IDAT", z);
    put_chunk(out, "IEND", {});
    return true;
}

bool png_write_rgb(
        const std::string & path,
        const uint8_t * rgb,
        int w,
        int h,
        std::string & err) {
    std::vector<uint8_t> png;
    if (!png_encode_rgb(rgb, w, h, png, err)) {
        return false;
    }

    FILE * f = fopen(path.c_str(), "wb");
    if (f == nullptr) {
        err = "cannot open " + path;
        return false;
    }
    const bool ok = fwrite(png.data(), 1, png.size(), f) == png.size();
    fclose(f);
    if (!ok) {
        err = "failed to write " + path;
    }
    return ok;
}
