#pragma once

#include "sam_utils.h"
#include "sam_ggml_state.h"
#include "sam_image.h"
#include "sam_params.h"
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>

struct sam_layer_enc
{
    struct ggml_tensor *norm1_w;
    struct ggml_tensor *norm1_b;

    struct ggml_tensor *rel_pos_w;
    struct ggml_tensor *rel_pos_h;

    struct ggml_tensor *qkv_w;
    struct ggml_tensor *qkv_b;

    struct ggml_tensor *proj_w;
    struct ggml_tensor *proj_b;

    struct ggml_tensor *norm2_w;
    struct ggml_tensor *norm2_b;

    struct ggml_tensor *mlp_lin1_w;
    struct ggml_tensor *mlp_lin1_b;

    struct ggml_tensor *mlp_lin2_w;
    struct ggml_tensor *mlp_lin2_b;
};

struct sam_encoder_image
{
    struct ggml_tensor *pe;

    struct ggml_tensor *proj_w;
    struct ggml_tensor *proj_b;

    struct ggml_tensor *neck_conv_0;
    struct ggml_tensor *neck_norm_0_w;
    struct ggml_tensor *neck_norm_0_b;
    struct ggml_tensor *neck_conv_1;
    struct ggml_tensor *neck_norm_1_w;
    struct ggml_tensor *neck_norm_1_b;

    std::vector<sam_layer_enc> layers;
};

struct sam_encoder_prompt
{
    struct ggml_tensor *pe;

    struct ggml_tensor *not_a_pt_embd_w;
    std::vector<struct ggml_tensor *> pt_embd;

    struct ggml_tensor *no_mask_embd_w;
    // std::vector<struct ggml_tensor *> mask_down_w;
    // std::vector<struct ggml_tensor *> mask_down_b;
};

struct prompt_encoder_result
{
    struct ggml_tensor *embd_prompt_sparse = {};
    struct ggml_tensor *embd_prompt_dense = {};
};

struct sam_layer_dec_transformer_attn
{
    // q_proj
    struct ggml_tensor *q_w;
    struct ggml_tensor *q_b;

    // k_proj
    struct ggml_tensor *k_w;
    struct ggml_tensor *k_b;

    // v_proj
    struct ggml_tensor *v_w;
    struct ggml_tensor *v_b;

    // out_proj
    struct ggml_tensor *out_w;
    struct ggml_tensor *out_b;
};

struct sam_layer_dec_transformer
{
    sam_layer_dec_transformer_attn self_attn;

    // norm1
    struct ggml_tensor *norm1_w;
    struct ggml_tensor *norm1_b;

    sam_layer_dec_transformer_attn cross_attn_token_to_img;

    // norm2
    struct ggml_tensor *norm2_w;
    struct ggml_tensor *norm2_b;

    // mlp.lin1
    struct ggml_tensor *mlp_lin1_w;
    struct ggml_tensor *mlp_lin1_b;

    // mlp.lin2
    struct ggml_tensor *mlp_lin2_w;
    struct ggml_tensor *mlp_lin2_b;

    // norm3
    struct ggml_tensor *norm3_w;
    struct ggml_tensor *norm3_b;

    // norm4
    struct ggml_tensor *norm4_w;
    struct ggml_tensor *norm4_b;

    sam_layer_dec_transformer_attn cross_attn_img_to_token;
};

struct sam_layer_dec_output_hypernet_mlps
{
    // mlps_*.layers.0
    struct ggml_tensor *w_0;
    struct ggml_tensor *b_0;

    // mlps_*.layers.1
    struct ggml_tensor *w_1;
    struct ggml_tensor *b_1;

    // mlps_*.layers.2
    struct ggml_tensor *w_2;
    struct ggml_tensor *b_2;
};

struct sam_decoder_mask
{
    std::vector<sam_layer_dec_transformer> transformer_layers;

    // trasnformer.final_attn_token_to_image
    sam_layer_dec_transformer_attn transformer_final_attn_token_to_img;

    // transformer.norm_final
    struct ggml_tensor *transformer_norm_final_w;
    struct ggml_tensor *transformer_norm_final_b;

    // output_upscaling.0
    struct ggml_tensor *output_upscaling_0_w;
    struct ggml_tensor *output_upscaling_0_b;

    // output_upscaling.1
    struct ggml_tensor *output_upscaling_1_w;
    struct ggml_tensor *output_upscaling_1_b;

    // output_upscaling.3
    struct ggml_tensor *output_upscaling_3_w;
    struct ggml_tensor *output_upscaling_3_b;

    // output_hypernetworks_mlps
    std::vector<sam_layer_dec_output_hypernet_mlps> output_hypernet_mlps;

    // iou_prediction_head.0
    struct ggml_tensor *iou_prediction_head_0_w;
    struct ggml_tensor *iou_prediction_head_0_b;

    // iou_prediction_head.1
    struct ggml_tensor *iou_prediction_head_1_w;
    struct ggml_tensor *iou_prediction_head_1_b;

    // iou_prediction_head.2
    struct ggml_tensor *iou_prediction_head_2_w;
    struct ggml_tensor *iou_prediction_head_2_b;

    // iou_token.weight
    struct ggml_tensor *iou_token_w;

    // mask_tokens.weight
    struct ggml_tensor *mask_tokens_w;
};

struct sam_ggml_model
{
    sam_hparams hparams;

    sam_encoder_image enc_img;
    sam_encoder_prompt enc_prompt;
    sam_decoder_mask dec;

    ggml_backend_t backend = {};
    ggml_backend_buffer_t buffer = {};

    //
    struct ggml_context *ctx;
    std::map<std::string, struct ggml_tensor *> tensors;
};

// encoding functions
struct ggml_tensor *sam_fill_dense_pe(const sam_ggml_model &model, struct ggml_context *ctx0, struct ggml_cgraph *gf, sam_ggml_state &state);
struct ggml_tensor *sam_layer_norm_2d(struct ggml_context *ctx0, struct ggml_tensor *layer, int n_channels, struct ggml_tensor *w, struct ggml_tensor *b, float eps);
struct ggml_cgraph *sam_encode_image(const sam_ggml_model &model, sam_ggml_state &state, const sam_image_f32 &img);
prompt_encoder_result sam_encode_prompt(const sam_ggml_model &model, struct ggml_context *ctx0, struct ggml_cgraph *gf, sam_ggml_state &state, int nx, int ny, sam_point point);

// decoding functions
bool sam_decode_mask(
    const sam_ggml_model &model,
    const prompt_encoder_result &prompt,
    struct ggml_tensor *pe_img,
    struct ggml_context *ctx0,
    struct ggml_cgraph *gf,
    sam_ggml_state &state);

struct ggml_tensor *sam_decode_mask_mlp_relu_3(
    struct ggml_tensor *in,
    struct ggml_tensor *w_0,
    struct ggml_tensor *b_0,
    struct ggml_tensor *w_1,
    struct ggml_tensor *b_1,
    struct ggml_tensor *w_2,
    struct ggml_tensor *b_2,
    struct ggml_context *ctx0);