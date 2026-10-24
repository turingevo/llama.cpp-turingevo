#pragma once

#include <cstdint>
#include <string>

// write an 8-bit RGB image, rows are top-down
bool png_write_rgb(
        const std::string & path,
        const uint8_t * rgb,
        int w,
        int h,
        std::string & err);
