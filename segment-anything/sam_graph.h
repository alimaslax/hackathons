#pragma once

#include "sam_ggml_state.h"
#include "sam_encoder.h"
#include "sam_image.h"

struct ggml_cgraph *sam_build_fast_graph(
    const sam_ggml_model &model,
    sam_ggml_state &state,
    int nx,
    int ny,
    sam_point point);