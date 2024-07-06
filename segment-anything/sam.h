//"sam.h"
#pragma once

#include <vector>
#include <cstdint>
#include <string>
#include <thread>
#include <memory>

#define _USE_MATH_DEFINES

#include "sam_image.h"
#include "sam_params.h"
#include "sam_encoder.h"
#include "sam_ggml_state.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>

#if defined(_MSC_VER)
#pragma warning(disable : 4244 4267) // possible loss of data
#endif

struct sam_ggml_state;

struct sam_ggml_model;

struct sam_state
{
    std::unique_ptr<sam_ggml_state> state;
    std::unique_ptr<sam_ggml_model> model;
    int t_load_ms = 0;
    int t_compute_img_ms = 0;
    int t_compute_masks_ms = 0;
};

// Main function declarations
std::shared_ptr<sam_state> sam_load_model(
    const sam_params &params);

std::shared_ptr<sam_state> sam_load_model(const sam_params &params);
std::vector<sam_image_u8> sam_compute_masks(const sam_image_u8 &img, int n_threads, sam_point pt, sam_state &state, int mask_on_val = 255, int mask_off_val = 0);
void sam_deinit(sam_state &state);