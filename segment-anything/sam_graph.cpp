// sam_graph.h
#pragma once

#include "sam_graph.h"
#include "sam_encoder.h"

struct ggml_cgraph *sam_build_fast_graph(
    const sam_ggml_model &model,
    sam_ggml_state &state,
    int nx,
    int ny,
    sam_point point)
{

    // since we are using ggml-alloc, this buffer only needs enough space to hold the ggml_tensor and ggml_cgraph structs, but not the tensor data
    static size_t buf_size = ggml_tensor_overhead() * GGML_MAX_NODES + ggml_graph_overhead();
    static std::vector<uint8_t> buf(buf_size);

    struct ggml_init_params ggml_params = {
        /*.mem_size   =*/buf.size(),
        /*.mem_buffer =*/buf.data(),
        /*.no_alloc   =*/true, // skip allocating as we use ggml_alloc to allocate exact memory requirements
    };

    struct ggml_context *ctx0 = ggml_init(ggml_params);
    struct ggml_cgraph *gf = ggml_new_graph(ctx0);

    prompt_encoder_result enc_res = sam_encode_prompt(model, ctx0, gf, state, nx, ny, point);
    if (!enc_res.embd_prompt_sparse || !enc_res.embd_prompt_dense)
    {
        fprintf(stderr, "%s: failed to encode prompt\n", __func__);
        return {};
    }

    struct ggml_tensor *pe_img_dense = sam_fill_dense_pe(model, ctx0, gf, state);
    if (!pe_img_dense)
    {
        fprintf(stderr, "%s: failed to get dense positional encoding\n", __func__);
        return {};
    }

    if (!sam_decode_mask(model, enc_res, pe_img_dense, ctx0, gf, state))
    {
        fprintf(stderr, "%s: failed to decode mask\n", __func__);
        return {};
    }

    ggml_free(ctx0);

    return gf;
}
