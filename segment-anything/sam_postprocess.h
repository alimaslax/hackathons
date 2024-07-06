#pragma once

#include "sam_params.h"
#include "sam_ggml_state.h"
#include "sam_image.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>

std::vector<sam_image_u8> sam_postprocess_masks(
    const sam_hparams &hparams,
    int nx,
    int ny,
    const sam_ggml_state &state,
    int mask_on_val,
    int mask_off_val);