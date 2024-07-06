//"sam.h"
#pragma once

#define _USE_MATH_DEFINES

#include "sam_encoder.h"
#include "sam_image.h"
#include "sam_params.h"
#include "sam_utils.h"
#include "sam_graph.h"
#include "sam_ggml_state.h"
#include "sam_postprocess.h"

#include <vector>
#include <cstdint>
#include <string>
#include <thread>
#include <memory>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>

#if defined(_MSC_VER)
#pragma warning(disable : 4244 4267) // possible loss of data
#endif

struct sam_state
{
    std::unique_ptr<sam_ggml_state> state;
    std::unique_ptr<sam_ggml_model> model;
    int t_load_ms = 0;
    int t_compute_img_ms = 0;
    int t_compute_masks_ms = 0;
};

bool sam_ggml_model_load(const std::string &fname, sam_ggml_model &model);

bool sam_image_preprocess(const sam_image_u8 &img, sam_image_f32 &res);

bool sam_compute_embd_img(
    const sam_image_u8 &img,
    int n_threads,
    sam_state &state);

std::vector<sam_image_u8> sam_compute_masks(
    const sam_image_u8 &img,
    int n_threads,
    sam_point pt,
    sam_state &state,
    int mask_on_val,
    int mask_off_val);

void sam_deinit(sam_state &state);

// Main function declarations
std::shared_ptr<sam_state> sam_load_model(
    const sam_params &params);

std::shared_ptr<sam_state> sam_load_model(const sam_params &params);
std::vector<sam_image_u8> sam_compute_masks(const sam_image_u8 &img, int n_threads, sam_point pt, sam_state &state, int mask_on_val = 255, int mask_off_val = 0);