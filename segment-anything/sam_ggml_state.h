#pragma once

#include "ggml.h"

struct sam_ggml_state
{
    struct ggml_tensor *embd_img = {};
    struct ggml_context *ctx_img = {};

    struct ggml_tensor *low_res_masks = {};
    struct ggml_tensor *iou_predictions = {};
    struct ggml_context *ctx_masks = {};

    // struct ggml_tensor * tmp_save = {};

    struct ggml_allocr *allocr = {};
};