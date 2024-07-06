#include "sam_encoder.h"
#include "sam_utils.h"
#include "sam.h"

struct ggml_tensor *sam_fill_dense_pe(
    const sam_ggml_model &model,
    struct ggml_context *ctx0,
    struct ggml_cgraph *gf,
    sam_ggml_state &state)
{
    const auto &hparams = model.hparams;
    const auto &enc = model.enc_prompt;

    const int32_t n_img_embd = hparams.n_img_embd();
    const float n_img_embd_inv = 1.0f / n_img_embd;

    struct ggml_tensor *xy_embed_stacked = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, 2, n_img_embd, n_img_embd);
    ggml_allocr_alloc(state.allocr, xy_embed_stacked);

    if (!ggml_allocr_is_measure(state.allocr))
    {
        float *data = (float *)ggml_get_data(xy_embed_stacked);
        for (int i = 0; i < n_img_embd; ++i)
        {
            const int row = 2 * i * n_img_embd;
            const float y_val = 2 * (i + 0.5f) * n_img_embd_inv - 1;
            for (int j = 0; j < n_img_embd; ++j)
            {
                const float x_val = 2 * (j + 0.5f) * n_img_embd_inv - 1;
                data[row + 2 * j + 0] = x_val;
                data[row + 2 * j + 1] = y_val;
            }
        }
    }

    struct ggml_tensor *cur = ggml_mul_mat(ctx0, ggml_cont(ctx0, ggml_transpose(ctx0, enc.pe)), xy_embed_stacked);

    cur = ggml_scale(ctx0, cur, ggml_new_f32(ctx0, float(2.0f * M_PI)));

    // concat
    // ref: https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/prompt_encoder.py#L192
    {
        struct ggml_tensor *t_sin = ggml_map_custom1(ctx0, cur, ggml_sam_sin, GGML_N_TASKS_MAX, NULL);
        struct ggml_tensor *t_cos = ggml_map_custom1(ctx0, cur, ggml_sam_cos, GGML_N_TASKS_MAX, NULL);

        cur = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, t_sin->ne[0] + t_cos->ne[0], cur->ne[1], cur->ne[2]);

        ggml_build_forward_expand(gf, ggml_cpy(ctx0, t_sin, ggml_view_3d(ctx0, cur, t_sin->ne[0], t_sin->ne[1], t_sin->ne[2], cur->nb[1], cur->nb[2], 0)));
        ggml_build_forward_expand(gf, ggml_cpy(ctx0, t_cos, ggml_view_3d(ctx0, cur, t_sin->ne[0], t_sin->ne[1], t_sin->ne[2], cur->nb[1], cur->nb[2], t_sin->nb[1])));
    }

    struct ggml_tensor *pe_img_dense = ggml_cont(ctx0, ggml_permute(ctx0, cur, 2, 0, 1, 3));
    ggml_build_forward_expand(gf, pe_img_dense);

    return pe_img_dense;
}

struct ggml_tensor *sam_layer_norm_2d(
    struct ggml_context *ctx0,
    struct ggml_tensor *layer,
    int n_channels,
    struct ggml_tensor *w,
    struct ggml_tensor *b,
    float eps)
{
    // LayerNorm2d
    // normalize along channel dimmension
    // TODO: better implementation
    layer = ggml_permute(ctx0,
                         ggml_norm(ctx0, ggml_cont(ctx0, ggml_permute(ctx0, layer, 1, 2, 0, 3)), eps),
                         2, 0, 1, 3);

    layer = ggml_add(ctx0,
                     ggml_mul(ctx0,
                              ggml_repeat(ctx0, ggml_reshape_3d(ctx0, w, 1, 1, n_channels), layer),
                              layer),
                     ggml_repeat(ctx0, ggml_reshape_3d(ctx0, b, 1, 1, n_channels), layer));

    return layer;
}

struct ggml_cgraph *sam_encode_image(
    const sam_ggml_model &model,
    sam_ggml_state &state,
    const sam_image_f32 &img)
{

    const auto &hparams = model.hparams;
    const auto &enc = model.enc_img;

    const int32_t n_enc_state = hparams.n_enc_state;
    const int32_t n_enc_layer = hparams.n_enc_layer;
    const int32_t n_enc_head = hparams.n_enc_head;
    const int32_t n_enc_head_dim = hparams.n_enc_head_dim();
    const int32_t n_enc_out_chans = hparams.n_enc_out_chans;
    const int32_t n_img_size = hparams.n_img_size();
    const int32_t n_window_size = hparams.n_window_size();

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

    struct ggml_tensor *inp = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, n_img_size, n_img_size, 3, 1);
    ggml_allocr_alloc(state.allocr, inp);
    if (!ggml_allocr_is_measure(state.allocr))
    {
        float *data = (float *)ggml_get_data(inp);

        const int nx = img.nx;
        const int ny = img.ny;
        const int n = nx * ny;

        GGML_ASSERT(nx == n_img_size && ny == n_img_size);

        for (int k = 0; k < 3; k++)
        {
            for (int y = 0; y < ny; y++)
            {
                for (int x = 0; x < nx; x++)
                {
                    data[k * n + y * nx + x] = img.data[3 * (y * nx + x) + k];
                }
            }
        }
    }

    // ref: https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/image_encoder.py#L392
    struct ggml_tensor *cur = ggml_conv_2d_sk_p0(ctx0, enc.proj_w, inp);
    cur = ggml_add_inplace(ctx0,
                           cur,
                           ggml_repeat(ctx0, enc.proj_b, cur));

    // ref: https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/image_encoder.py#L394
    // keep in F32
    cur = ggml_cont(ctx0,
                    ggml_permute(ctx0, cur, 1, 2, 0, 3));

    // convert to F16
    // cur = ggml_cpy(ctx0,
    //        ggml_permute(ctx0, cur, 1, 2, 0, 3),
    //        ggml_new_tensor_3d(ctx0, GGML_TYPE_F16, n_enc_state, n_img_embd, n_img_embd));

    // ref: https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/image_encoder.py#L108-L109
    cur = ggml_add_inplace(ctx0, cur, enc.pe);

    struct ggml_tensor *inpL = cur;

    for (int il = 0; il < n_enc_layer; ++il)
    {
        const auto &layer = enc.layers[il];

        // norm
        // ref: https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/image_encoder.py#L168
        {
            cur = ggml_norm(ctx0, inpL, hparams.eps);

            // cur = ln_0_w*cur + ln_0_b
            cur = ggml_mul(ctx0, cur, layer.norm1_w);
            cur = ggml_add_inplace(ctx0, cur, layer.norm1_b);
        }

        const int64_t w0 = cur->ne[1];
        const int64_t h0 = cur->ne[2];

        if (hparams.is_global_attn(il) == false)
        {
            // local attention layer - apply window partition
            // ref: https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/image_encoder.py#L169-L172
            cur = ggml_win_part(ctx0, cur, n_window_size);
        }

        const int64_t W = cur->ne[1];
        const int64_t H = cur->ne[2];

        // self-attention
        {
            cur = ggml_mul_mat(ctx0, layer.qkv_w, cur);
            cur = ggml_add_inplace(ctx0, cur, layer.qkv_b);

            // split qkv into separate tensors
            // ref: https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/image_encoder.py#L225-L229
            const int B = cur->ne[3];

            cur = ggml_reshape_4d(ctx0, cur, n_enc_state, 3, W * H, B);
            cur = ggml_cont(ctx0, ggml_permute(ctx0, cur, 0, 3, 1, 2));

            struct ggml_tensor *Q;
            struct ggml_tensor *K;
            struct ggml_tensor *V;

            Q = ggml_view_3d(ctx0, cur, n_enc_state, W * H, B, cur->nb[1], cur->nb[2], 0 * cur->nb[3]);
            Q = ggml_reshape_4d(ctx0, Q, n_enc_head_dim, n_enc_head, W * H, B);
            Q = ggml_cont(ctx0, ggml_permute(ctx0, Q, 0, 2, 1, 3));
            Q = ggml_reshape_3d(ctx0, Q, n_enc_head_dim, W * H, B * n_enc_head);

            K = ggml_view_3d(ctx0, cur, n_enc_state, W * H, B, cur->nb[1], cur->nb[2], 1 * cur->nb[3]);
            K = ggml_reshape_4d(ctx0, K, n_enc_head_dim, n_enc_head, W * H, B);
            K = ggml_cont(ctx0, ggml_permute(ctx0, K, 0, 2, 1, 3));
            K = ggml_reshape_3d(ctx0, K, n_enc_head_dim, W * H, B * n_enc_head);

            V = ggml_view_3d(ctx0, cur, n_enc_state, W * H, B, cur->nb[1], cur->nb[2], 2 * cur->nb[3]);
            V = ggml_reshape_4d(ctx0, V, n_enc_head_dim, n_enc_head, W * H, B);
            V = ggml_cont(ctx0, ggml_permute(ctx0, V, 1, 2, 0, 3)); // transposed
            V = ggml_reshape_3d(ctx0, V, W * H, n_enc_head_dim, B * n_enc_head);

            struct ggml_tensor *KQ = ggml_mul_mat(ctx0, K, Q);

            struct ggml_tensor *KQ_scaled =
                ggml_scale_inplace(ctx0,
                                   KQ,
                                   ggml_new_f32(ctx0, 1.0f / sqrtf(n_enc_head_dim)));

            struct ggml_tensor *rw = ggml_get_rel_pos(ctx0, layer.rel_pos_w, W, W);
            struct ggml_tensor *rh = ggml_get_rel_pos(ctx0, layer.rel_pos_h, H, H);

            struct ggml_tensor *q_r = ggml_reshape_4d(ctx0, Q, n_enc_head_dim, W, H, B * n_enc_head);

            struct ggml_tensor *rel_w = ggml_cont(ctx0, ggml_permute(ctx0,
                                                                     ggml_mul_mat(ctx0,
                                                                                  rw,
                                                                                  ggml_cont(ctx0, ggml_permute(ctx0, q_r, 0, 2, 1, 3))),
                                                                     0, 2, 1, 3));
            struct ggml_tensor *rel_h = ggml_mul_mat(ctx0, rh, q_r);

            struct ggml_tensor *attn = ggml_add_rel_pos_inplace(ctx0, KQ_scaled, rel_w, rel_h);

            struct ggml_tensor *KQ_soft_max = ggml_soft_max_inplace(ctx0, attn);

            struct ggml_tensor *KQV = ggml_mul_mat(ctx0, V, KQ_soft_max);

            cur =
                ggml_reshape_4d(ctx0,
                                ggml_cont(ctx0,
                                          ggml_permute(ctx0,
                                                       ggml_reshape_4d(ctx0, KQV, n_enc_head_dim, W * H, n_enc_head, B),
                                                       0, 2, 1, 3)),
                                n_enc_state, W, H, B);

            cur = ggml_mul_mat(ctx0, layer.proj_w, cur);
            cur = ggml_add_inplace(ctx0, cur, layer.proj_b);
        }

        if (hparams.is_global_attn(il) == false)
        {
            // local attention layer - reverse window partition
            cur = ggml_win_unpart(ctx0, cur, w0, h0, n_window_size);
        }

        cur = ggml_add_inplace(ctx0, cur, inpL);

        struct ggml_tensor *inpFF = cur;

        // feed-forward network
        {
            // norm
            {
                cur = ggml_norm(ctx0, inpFF, hparams.eps);

                // cur = mlp_ln_w*cur + mlp_ln_b
                cur = ggml_mul(ctx0, cur, layer.norm2_w);
                cur = ggml_add_inplace(ctx0, cur, layer.norm2_b);
            }

            // fully connected
            cur = ggml_mul_mat(ctx0, layer.mlp_lin1_w, cur);
            cur = ggml_add_inplace(ctx0, cur, layer.mlp_lin1_b);

            // GELU activation
            cur = ggml_gelu(ctx0, cur);

            // projection
            cur = ggml_mul_mat(ctx0, layer.mlp_lin2_w, cur);
            cur = ggml_add_inplace(ctx0, cur, layer.mlp_lin2_b);
        }

        inpL = ggml_add(ctx0, cur, inpFF);
    }

    cur = ggml_cont(ctx0, ggml_permute(ctx0, inpL, 2, 0, 1, 3));

    cur = ggml_conv_2d_sk_p0(ctx0, enc.neck_conv_0, cur);

    cur = sam_layer_norm_2d(ctx0, cur, n_enc_out_chans, enc.neck_norm_0_w, enc.neck_norm_0_b, hparams.eps);

    cur = ggml_conv_2d_s1_ph(ctx0, enc.neck_conv_1, cur);

    cur = sam_layer_norm_2d(ctx0, cur, n_enc_out_chans, enc.neck_norm_1_w, enc.neck_norm_1_b, hparams.eps);

    cur = ggml_cpy(state.ctx_img, cur, state.embd_img);

    ggml_build_forward_expand(gf, cur);
    ggml_disconnect_node_from_graph(state.embd_img);

    // ggml_graph_print(&gf);

    ggml_free(ctx0);

    return gf;
}

// encode a prompt
//
// - points
// - boxes
// - masks
//
// TODO: currently just encode a single point for simplicity
//
prompt_encoder_result sam_encode_prompt(
    const sam_ggml_model &model,
    struct ggml_context *ctx0,
    struct ggml_cgraph *gf,
    sam_ggml_state &state,
    int nx,
    int ny,
    sam_point point)
{

    const auto &hparams = model.hparams;
    const auto &enc = model.enc_prompt;

    // transform points
    // ref: https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/automatic_mask_generator.py#L276
    {
        const int nmax = std::max(nx, ny);

        const float scale = hparams.n_img_size() / (float)nmax;

        const int nx_new = int(nx * scale + 0.5f);
        const int ny_new = int(ny * scale + 0.5f);

        point.x = point.x * (float(nx_new) / nx) + 0.5f;
        point.y = point.y * (float(ny_new) / ny) + 0.5f;
    }

    struct ggml_tensor *inp = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, 2, 2);

    ggml_allocr_alloc(state.allocr, inp);
    if (!ggml_allocr_is_measure(state.allocr))
    {
        // set the input by converting the [0, 1] coordinates to [-1, 1]
        float *data = (float *)inp->data;

        data[0] = 2.0f * (point.x / hparams.n_img_size()) - 1.0f;
        data[1] = 2.0f * (point.y / hparams.n_img_size()) - 1.0f;

        // padding
        // ref: https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/prompt_encoder.py#L81-L85
        data[2] = 2.0f * (0.0f) - 1.0f;
        data[3] = 2.0f * (0.0f) - 1.0f;
    }

    struct ggml_tensor *cur = ggml_mul_mat(ctx0, ggml_cont(ctx0, ggml_transpose(ctx0, enc.pe)), inp);

    cur = ggml_scale(ctx0, cur, ggml_new_f32(ctx0, float(2.0f * M_PI)));

    // concat
    // ref: https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/prompt_encoder.py#L192
    {
        struct ggml_tensor *t_sin = ggml_map_custom1(ctx0, cur, ggml_sam_sin, GGML_N_TASKS_MAX, NULL);
        struct ggml_tensor *t_cos = ggml_map_custom1(ctx0, cur, ggml_sam_cos, GGML_N_TASKS_MAX, NULL);

        cur = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, t_sin->ne[0] + t_cos->ne[0], cur->ne[1]);

        ggml_build_forward_expand(gf, ggml_cpy(ctx0, t_sin, ggml_view_2d(ctx0, cur, t_sin->ne[0], t_sin->ne[1], cur->nb[1], 0)));
        ggml_build_forward_expand(gf, ggml_cpy(ctx0, t_cos, ggml_view_2d(ctx0, cur, t_sin->ne[0], t_sin->ne[1], cur->nb[1], t_sin->nb[1])));

        // overwrite label == -1 with not_a_point_embed.weight
        // ref: https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/prompt_encoder.py#L86
        // TODO: extend for multiple points
        ggml_build_forward_expand(gf, ggml_cpy(ctx0, enc.not_a_pt_embd_w, ggml_view_2d(ctx0, cur, cur->ne[0], 1, cur->nb[1], cur->nb[1])));
    }

    // add point_embeddings[1] to label == 1
    // ref: https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/prompt_encoder.py#L90
    struct ggml_tensor *v = ggml_view_2d(ctx0, cur, cur->ne[0], 1, cur->nb[1], 0);
    ggml_build_forward_expand(gf, ggml_cpy(ctx0, ggml_add_inplace(ctx0, v, enc.pt_embd[1]), v));

    struct ggml_tensor *embd_prompt_sparse = cur;
    ggml_build_forward_expand(gf, embd_prompt_sparse);

    struct ggml_tensor *embd_prompt_dense = ggml_repeat(ctx0,
                                                        ggml_cont(ctx0,
                                                                  ggml_view_3d(ctx0, enc.no_mask_embd_w,
                                                                               1, 1, enc.no_mask_embd_w->ne[0], enc.no_mask_embd_w->nb[0], enc.no_mask_embd_w->nb[0], 0)),
                                                        ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, hparams.n_img_embd(), hparams.n_img_embd(), hparams.n_enc_out_chans));

    ggml_build_forward_expand(gf, embd_prompt_dense);

    // printf("used_mem = %zu\n", ggml_used_mem(ctx0));

    prompt_encoder_result res;
    res.embd_prompt_sparse = embd_prompt_sparse;
    res.embd_prompt_dense = embd_prompt_dense;
    return res;
}

// decoding functions

bool sam_decode_mask(
    const sam_ggml_model &model,
    const prompt_encoder_result &prompt,
    struct ggml_tensor *pe_img,
    struct ggml_context *ctx0,
    struct ggml_cgraph *gf,
    sam_ggml_state &state)
{

    const auto &hparams = model.hparams;
    const auto &dec = model.dec;
    const int n_img_embd = hparams.n_img_embd();

    struct ggml_tensor *tokens = {};
    {
        // Concatenate output tokens
        // ref: https://github.com/facebookresearch/segment-anything/blob/6fdee8f2727f4506cfbbe553e23b895e27956588/segment_anything/modeling/mask_decoder.py#L120
        const auto &sparse = prompt.embd_prompt_sparse;

        tokens = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, dec.iou_token_w->ne[0], dec.iou_token_w->ne[1] + dec.mask_tokens_w->ne[1] + sparse->ne[1], sparse->ne[2]);

        const size_t offsets[3] = {0, dec.iou_token_w->ne[1] * tokens->nb[1], dec.iou_token_w->ne[1] * tokens->nb[1] + dec.mask_tokens_w->ne[1] * tokens->nb[1]};
        ggml_build_forward_expand(gf, ggml_cpy(ctx0, dec.iou_token_w, ggml_view_2d(ctx0, tokens, tokens->ne[0], dec.iou_token_w->ne[1], tokens->nb[1], offsets[0])));
        ggml_build_forward_expand(gf, ggml_cpy(ctx0, dec.mask_tokens_w, ggml_view_2d(ctx0, tokens, tokens->ne[0], dec.mask_tokens_w->ne[1], tokens->nb[1], offsets[1])));
        ggml_build_forward_expand(gf, ggml_cpy(ctx0, sparse, ggml_view_2d(ctx0, tokens, tokens->ne[0], sparse->ne[1], tokens->nb[1], offsets[2])));
        // TODO: Sparse prompt embeddings can have more than one point
    }

    struct ggml_tensor *src = {};
    struct ggml_tensor *pos_src = {};
    int srcNE[4] = {0, 0, 0, 0};
    {
        // Expand per-image data in the batch direction to be per-mask
        // ref: https://github.com/facebookresearch/segment-anything/blob/6fdee8f2727f4506cfbbe553e23b895e27956588/segment_anything/modeling/mask_decoder.py#L125
        src = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, state.embd_img->ne[0], state.embd_img->ne[1], state.embd_img->ne[2], tokens->ne[2]);

        src = ggml_add(ctx0,
                       ggml_repeat(ctx0,
                                   state.embd_img,
                                   src),
                       prompt.embd_prompt_dense);

        srcNE[0] = src->ne[0];
        srcNE[1] = src->ne[1];
        srcNE[2] = src->ne[2];
        srcNE[3] = src->ne[3];

        // flatten & permute
        // ref: https://github.com/facebookresearch/segment-anything/blob/6fdee8f2727f4506cfbbe553e23b895e27956588/segment_anything/modeling/transformer.py#L83
        src = ggml_cont(ctx0, ggml_permute(ctx0,
                                           ggml_view_3d(ctx0,
                                                        src,
                                                        src->ne[0] * src->ne[1],
                                                        src->ne[2],
                                                        src->ne[3],
                                                        src->nb[2],
                                                        src->nb[3],
                                                        0),
                                           1, 0, 2, 3));

        pos_src = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, pe_img->ne[0], pe_img->ne[1], pe_img->ne[2], tokens->ne[2]);
        pos_src = ggml_repeat(ctx0,
                              pe_img,
                              pos_src);

        // flatten & permute
        // ref: https://github.com/facebookresearch/segment-anything/blob/6fdee8f2727f4506cfbbe553e23b895e27956588/segment_anything/modeling/transformer.py#L83
        pos_src = ggml_cont(ctx0, ggml_permute(ctx0,
                                               ggml_view_3d(ctx0,
                                                            pos_src,
                                                            pos_src->ne[0] * pos_src->ne[1],
                                                            pos_src->ne[2],
                                                            pos_src->ne[3],
                                                            pos_src->nb[2],
                                                            pos_src->nb[3],
                                                            0),
                                               1, 0, 2, 3));
    }

    struct ggml_tensor *queries = tokens;
    struct ggml_tensor *keys = src;
    {
        // Run the transformer
        // ref: https://github.com/facebookresearch/segment-anything/blob/6fdee8f2727f4506cfbbe553e23b895e27956588/segment_anything/modeling/transformer.py#L62
        for (int i = 0; i < int(model.dec.transformer_layers.size()); ++i)
        {
            const auto &tfm_layer = model.dec.transformer_layers[i];

            // Self attention block
            // ref: https://github.com/facebookresearch/segment-anything/blob/6fdee8f2727f4506cfbbe553e23b895e27956588/segment_anything/modeling/transformer.py#L154
            const bool skip_first_layer_pe = i == 0;
            if (skip_first_layer_pe)
            {
                queries = sam_decode_mask_transformer_attn(tfm_layer.self_attn, queries, queries, queries, ctx0, model);
            }
            else
            {
                struct ggml_tensor *q_0 = ggml_add(ctx0, queries, tokens);

                struct ggml_tensor *self_attn = sam_decode_mask_transformer_attn(tfm_layer.self_attn, q_0, q_0, queries, ctx0, model);
                queries = ggml_add(ctx0, queries, self_attn);
            }

            queries = ggml_norm(ctx0, queries, hparams.eps_decoder_transformer);
            queries = ggml_add_inplace(ctx0,
                                       ggml_mul(ctx0, queries, tfm_layer.norm1_w),
                                       tfm_layer.norm1_b);

            // Cross attention block, tokens attending to image embedding
            // ref: https://github.com/facebookresearch/segment-anything/blob/6fdee8f2727f4506cfbbe553e23b895e27956588/segment_anything/modeling/transformer.py#L163
            struct ggml_tensor *q_1 = ggml_add(ctx0, queries, tokens);
            struct ggml_tensor *k_1 = ggml_add(ctx0, keys, pos_src);

            struct ggml_tensor *cross_attn_token_to_img = sam_decode_mask_transformer_attn(tfm_layer.cross_attn_token_to_img, q_1, k_1, keys, ctx0, model);

            queries = ggml_add_inplace(ctx0, queries, cross_attn_token_to_img);
            queries = ggml_norm_inplace(ctx0, queries, hparams.eps_decoder_transformer);
            queries = ggml_add_inplace(ctx0,
                                       ggml_mul(ctx0, queries, tfm_layer.norm2_w),
                                       tfm_layer.norm2_b);

            // MLP block
            // ref: https://github.com/facebookresearch/segment-anything/blob/6fdee8f2727f4506cfbbe553e23b895e27956588/segment_anything/modeling/transformer.py#L170
            struct ggml_tensor *mlp_out = ggml_mul_mat(ctx0,
                                                       tfm_layer.mlp_lin1_w,
                                                       queries);

            mlp_out = ggml_add_inplace(ctx0, mlp_out, tfm_layer.mlp_lin1_b);

            // RELU activation
            mlp_out = ggml_relu_inplace(ctx0, mlp_out);
            mlp_out = ggml_mul_mat(ctx0, tfm_layer.mlp_lin2_w, mlp_out);

            mlp_out = ggml_add_inplace(ctx0, mlp_out, tfm_layer.mlp_lin2_b);

            queries = ggml_add_inplace(ctx0, queries, mlp_out);
            queries = ggml_norm_inplace(ctx0, queries, hparams.eps_decoder_transformer);
            queries = ggml_add_inplace(ctx0,
                                       ggml_mul(ctx0, queries, tfm_layer.norm3_w),
                                       tfm_layer.norm3_b);

            // Cross attention block, image embedding attending to tokens
            // ref: https://github.com/facebookresearch/segment-anything/blob/6fdee8f2727f4506cfbbe553e23b895e27956588/segment_anything/modeling/transformer.py#L175
            struct ggml_tensor *q_2 = ggml_add(ctx0, queries, tokens);
            struct ggml_tensor *k_2 = ggml_add(ctx0, keys, pos_src);

            struct ggml_tensor *cross_attn_img_to_token = sam_decode_mask_transformer_attn(tfm_layer.cross_attn_img_to_token, k_2, q_2, queries, ctx0, model);
            keys = ggml_add_inplace(ctx0, keys, cross_attn_img_to_token);
            keys = ggml_norm_inplace(ctx0, keys, hparams.eps_decoder_transformer);
            keys = ggml_add_inplace(ctx0,
                                    ggml_mul(ctx0, keys, tfm_layer.norm4_w),
                                    tfm_layer.norm4_b);
        }

        // Apply the final attention layer from the points to the image
        // ref: https://github.com/facebookresearch/segment-anything/blob/6fdee8f2727f4506cfbbe553e23b895e27956588/segment_anything/modeling/transformer.py#L99
        struct ggml_tensor *q = ggml_add(ctx0, queries, tokens);
        struct ggml_tensor *k = ggml_add(ctx0, keys, pos_src);

        struct ggml_tensor *final_attn_token_to_img = sam_decode_mask_transformer_attn(dec.transformer_final_attn_token_to_img, q, k, keys, ctx0, model);

        queries = ggml_add_inplace(ctx0, queries, final_attn_token_to_img);
        queries = ggml_norm_inplace(ctx0, queries, hparams.eps_decoder_transformer);
        queries = ggml_add_inplace(ctx0,
                                   ggml_mul(ctx0, queries, dec.transformer_norm_final_w),
                                   dec.transformer_norm_final_b);
    }

    struct ggml_tensor *iou_pred = ggml_view_2d(ctx0, queries, queries->ne[0], queries->ne[2], queries->nb[2], 0);
    const int num_mask_tokens = 4; // num_multimask_outputs + 1
    struct ggml_tensor *mask_tokens_out = ggml_view_3d(ctx0, queries, queries->ne[0], num_mask_tokens, queries->ne[2], queries->nb[1], num_mask_tokens * queries->nb[1], queries->nb[1]);

    // Upscale mask embeddings and predict masks using the mask tokens
    // ref: https://github.com/facebookresearch/segment-anything/blob/6fdee8f2727f4506cfbbe553e23b895e27956588/segment_anything/modeling/mask_decoder.py#L136
    keys = ggml_cont(ctx0, ggml_transpose(ctx0, keys));
    keys = ggml_view_4d(ctx0, keys, srcNE[0], srcNE[1], srcNE[2], srcNE[3], srcNE[0] * keys->nb[0], keys->nb[1], keys->nb[2], 0);
    // ggml_build_forward_expand(gf, keys);
    struct ggml_tensor *upscaled_embedding = {};
    {
        // ConvTranspose2d
        keys = ggml_conv_transpose_2d_p0(ctx0, dec.output_upscaling_0_w, keys, 2);
        keys = ggml_add_inplace(ctx0, keys, ggml_repeat(ctx0, ggml_reshape_3d(ctx0, dec.output_upscaling_0_b, 1, 1, dec.output_upscaling_0_b->ne[0]), keys));

        keys = sam_layer_norm_2d(ctx0, keys, n_img_embd, dec.output_upscaling_1_w, dec.output_upscaling_1_b, hparams.eps);

        // GELU activation
        keys = ggml_gelu_inplace(ctx0, keys);

        // ConvTranspose2d
        keys = ggml_conv_transpose_2d_p0(ctx0, dec.output_upscaling_3_w, keys, 2);
        keys = ggml_add_inplace(ctx0, ggml_repeat(ctx0, ggml_reshape_3d(ctx0, dec.output_upscaling_3_b, 1, 1, dec.output_upscaling_3_b->ne[0]), keys), keys);
        // GELU activation
        keys = ggml_gelu_inplace(ctx0, keys);
        upscaled_embedding = ggml_reshape_3d(ctx0, keys, keys->ne[0] * keys->ne[1], keys->ne[2], keys->ne[3]);
        upscaled_embedding = ggml_cont(ctx0, ggml_transpose(ctx0, upscaled_embedding)); // TODO: Shouldn't be needed
    }

    struct ggml_tensor *hyper_in = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, n_img_embd / 2, num_mask_tokens, mask_tokens_out->ne[2]);

    for (int i = 0; i < num_mask_tokens; ++i)
    {
        const auto &mlp = dec.output_hypernet_mlps[i];
        struct ggml_tensor *in = ggml_view_2d(ctx0, mask_tokens_out, mask_tokens_out->ne[0], mask_tokens_out->ne[2], mask_tokens_out->nb[1], i * mask_tokens_out->nb[1]);
        struct ggml_tensor *out = sam_decode_mask_mlp_relu_3(in, mlp.w_0, mlp.b_0, mlp.w_1, mlp.b_1, mlp.w_2, mlp.b_2, ctx0);
        ggml_build_forward_expand(gf, ggml_cpy(ctx0, out, ggml_view_2d(ctx0, hyper_in, hyper_in->ne[0], hyper_in->ne[2], hyper_in->nb[1], i * hyper_in->nb[1])));
    }

    struct ggml_tensor *masks = ggml_mul_mat(ctx0, hyper_in, upscaled_embedding);
    masks = ggml_cont(ctx0, ggml_transpose(ctx0, masks)); // TODO: Shouldn't be needed
    masks = ggml_reshape_4d(ctx0, masks, keys->ne[0], keys->ne[1], masks->ne[1], keys->ne[3]);

    // Generate mask quality predictions
    // ref: https://github.com/facebookresearch/segment-anything/blob/6fdee8f2727f4506cfbbe553e23b895e27956588/segment_anything/modeling/mask_decoder.py#L146
    iou_pred = sam_decode_mask_mlp_relu_3(iou_pred, dec.iou_prediction_head_0_w, dec.iou_prediction_head_0_b, dec.iou_prediction_head_1_w, dec.iou_prediction_head_1_b, dec.iou_prediction_head_2_w, dec.iou_prediction_head_2_b, ctx0);

    // Select the correct mask or masks for output
    // ref: https://github.com/facebookresearch/segment-anything/blob/6fdee8f2727f4506cfbbe553e23b895e27956588/segment_anything/modeling/mask_decoder.py#L101
    iou_pred = ggml_cpy(state.ctx_masks, ggml_view_1d(ctx0, iou_pred, iou_pred->ne[0] - 1, iou_pred->nb[0]), state.iou_predictions);
    masks = ggml_view_4d(ctx0, masks, masks->ne[0], masks->ne[1], masks->ne[2] - 1, masks->ne[3],
                         masks->nb[1], masks->nb[2], masks->nb[3], masks->nb[2] /* offset*/);
    masks = ggml_cpy(state.ctx_masks, masks, state.low_res_masks);

    ggml_build_forward_expand(gf, masks);
    ggml_build_forward_expand(gf, iou_pred);

    ggml_disconnect_node_from_graph(state.low_res_masks);
    ggml_disconnect_node_from_graph(state.iou_predictions);

    return true;
}

struct ggml_tensor *sam_decode_mask_mlp_relu_3(
    struct ggml_tensor *in,
    struct ggml_tensor *w_0,
    struct ggml_tensor *b_0,
    struct ggml_tensor *w_1,
    struct ggml_tensor *b_1,
    struct ggml_tensor *w_2,
    struct ggml_tensor *b_2,
    struct ggml_context *ctx0)
{

    struct ggml_tensor *cur = {};
    cur = ggml_mul_mat(ctx0, w_0, in);
    cur = ggml_add_inplace(ctx0, cur, b_0);

    cur = ggml_relu_inplace(ctx0, cur);

    cur = ggml_mul_mat(ctx0, w_1, cur);
    cur = ggml_add_inplace(ctx0, cur, b_1);

    cur = ggml_relu_inplace(ctx0, cur);

    cur = ggml_mul_mat(ctx0, w_2, cur);
    cur = ggml_add_inplace(ctx0, cur, b_2);

    return cur;
}