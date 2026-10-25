#pragma once

#include <cstdint>
#include <string>
#include <vector>

// 8-bit RGB image
struct sensenova_image {
    int w = 0;
    int h = 0;
    std::vector<uint8_t> rgb; // [h][w][3]
};

bool image_load_file(const std::string & path, sensenova_image & img, std::string & err);
bool image_load_memory(const void * data, size_t size, sensenova_image & img, std::string & err);

// closest size divisible by 32 inside the pixel bounds, aspect ratio preserved
bool image_plan_size(int w, int h, int min_pixels, int max_pixels, int & out_w, int & out_h, std::string & err);

// antialiased bilinear resize to an exact size
bool image_resize(const sensenova_image & src, int w, int h, sensenova_image & out, std::string & err);

// plan a size inside the pixel bounds and resize to it
bool image_prepare(const sensenova_image & src, int min_pixels, int max_pixels, sensenova_image & out, std::string & err);

// ImageNet normalization, written in the patch embedder layout [w][h][3]
void image_to_fm_input(const sensenova_image & img, std::vector<float> & out);
