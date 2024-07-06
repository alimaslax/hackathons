// sam_encoder.h
#pragma once

#include "sam_utils.h"
#include "sam_ggml_state.h"
#include "sam_image.h"
#include "sam_encoder.h"
#include "sam_params.h"
#include "ggml.h"

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

struct ggml_tensor *sam_decode_mask_transformer_attn(
    const sam_layer_dec_transformer_attn &attn,
    struct ggml_tensor *queries,
    struct ggml_tensor *keys,
    struct ggml_tensor *values,
    struct ggml_context *ctx0,
    const sam_ggml_model &model)
{
    const auto &hparams = model.hparams;
    const int n_head = hparams.n_dec_heads;

    struct ggml_tensor *Qcur = {};
    struct ggml_tensor *Kcur = {};
    struct ggml_tensor *Vcur = {};

    Qcur = ggml_mul_mat(ctx0, attn.q_w, queries);
    Qcur = ggml_add_inplace(ctx0, Qcur, attn.q_b);

    Kcur = ggml_mul_mat(ctx0, attn.k_w, keys);
    Kcur = ggml_add_inplace(ctx0, Kcur, attn.k_b);

    Vcur = ggml_mul_mat(ctx0, attn.v_w, values);
    Vcur = ggml_add_inplace(ctx0, Vcur, attn.v_b);

    struct ggml_tensor *Q = {};
    struct ggml_tensor *K = {};
    struct ggml_tensor *V = {};

    Q = ggml_reshape_4d(ctx0, Qcur, Qcur->ne[0] / n_head, n_head, Qcur->ne[1], Qcur->ne[2]);
    Q = ggml_cont(ctx0, ggml_permute(ctx0, Q, 0, 2, 1, 3));

    K = ggml_reshape_4d(ctx0, Kcur, Kcur->ne[0] / n_head, n_head, Kcur->ne[1], Kcur->ne[2]);
    K = ggml_cont(ctx0, ggml_permute(ctx0, K, 0, 2, 1, 3));

    V = ggml_reshape_4d(ctx0, Vcur, Vcur->ne[0] / n_head, n_head, Vcur->ne[1], Vcur->ne[2]);
    V = ggml_cont(ctx0, ggml_permute(ctx0, V, 0, 2, 1, 3));

    // Q * K
    struct ggml_tensor *KQ = ggml_mul_mat(ctx0, K, Q);

    struct ggml_tensor *KQ_scaled =
        ggml_scale_inplace(ctx0,
                           KQ,
                           ggml_new_f32(ctx0, 1.0f / sqrt(float(Q->ne[0]))));

    struct ggml_tensor *KQ_soft_max = ggml_soft_max_inplace(ctx0, KQ_scaled);

    struct ggml_tensor *KQV = ggml_mul_mat(ctx0, KQ_soft_max, ggml_cont(ctx0, ggml_transpose(ctx0, V)));

    struct ggml_tensor *KQV_merged = ggml_cont(ctx0, ggml_transpose(ctx0, KQV));
    KQV_merged = ggml_cont(ctx0, ggml_permute(ctx0, KQV_merged, 0, 2, 1, 3));
    KQV_merged = ggml_reshape_3d(ctx0, KQV_merged, KQV_merged->ne[0] * KQV_merged->ne[1], KQV_merged->ne[2], KQV_merged->ne[3]);
    KQV_merged = ggml_mul_mat(ctx0, attn.out_w, KQV_merged);
    KQV_merged = ggml_add_inplace(ctx0, KQV_merged, attn.out_b);

    return KQV_merged;
}

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