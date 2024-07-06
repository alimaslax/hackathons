#pragma once

#include <vector>
#include <cstdint>

struct sam_point
{
    float x = 0;
    float y = 0;
};

struct sam_image_u8
{
    int nx = 0;
    int ny = 0;
    std::vector<uint8_t> data;
};

struct sam_image_f32
{
    int nx = 0;
    int ny = 0;
    std::vector<float> data;
};