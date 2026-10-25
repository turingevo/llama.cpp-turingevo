#pragma once

#include <cstdint>
#include <string>
#include <vector>

// encode an 8-bit RGB image, rows are top-down
bool png_encode_rgb(
        const uint8_t * rgb,
        int w,
        int h,
        std::vector<uint8_t> & out,
        std::string & err);

// write an 8-bit RGB image, rows are top-down
bool png_write_rgb(
        const std::string & path,
        const uint8_t * rgb,
        int w,
        int h,
        std::string & err);
