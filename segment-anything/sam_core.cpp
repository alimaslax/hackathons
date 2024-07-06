#include "sam.h"
#include "sam_utils.h"
#include "sam_graph.h"
#include "sam_encoder.h"
#include "sam_ggml_state.h"
#include "sam_postprocess.h"

// Implementations of sam_load_model, sam_compute_masks, sam_deinit

bool sam_ggml_model_load(const std::string &fname, sam_ggml_model &model)
{
    fprintf(stderr, "%s: loading model from '%s' - please wait ...\n", __func__, fname.c_str());

    auto fin = std::ifstream(fname, std::ios::binary);
    if (!fin)
    {
        fprintf(stderr, "%s: failed to open '%s'\n", __func__, fname.c_str());
        return false;
    }

    // verify magic
    {
        uint32_t magic;
        fin.read((char *)&magic, sizeof(magic));
        if (magic != 0x67676d6c)
        {
            fprintf(stderr, "%s: invalid model file '%s' (bad magic)\n", __func__, fname.c_str());
            return false;
        }
    }

    // load hparams
    {
        auto &hparams = model.hparams;

        fin.read((char *)&hparams.n_enc_state, sizeof(hparams.n_enc_state));
        fin.read((char *)&hparams.n_enc_layer, sizeof(hparams.n_enc_layer));
        fin.read((char *)&hparams.n_enc_head, sizeof(hparams.n_enc_head));
        fin.read((char *)&hparams.n_enc_out_chans, sizeof(hparams.n_enc_out_chans));
        fin.read((char *)&hparams.n_pt_embd, sizeof(hparams.n_pt_embd));
        fin.read((char *)&hparams.ftype, sizeof(hparams.ftype));

        const int32_t qntvr = hparams.ftype / GGML_QNT_VERSION_FACTOR;

        printf("%s: n_enc_state      = %d\n", __func__, hparams.n_enc_state);
        printf("%s: n_enc_layer      = %d\n", __func__, hparams.n_enc_layer);
        printf("%s: n_enc_head       = %d\n", __func__, hparams.n_enc_head);
        printf("%s: n_enc_out_chans  = %d\n", __func__, hparams.n_enc_out_chans);
        printf("%s: n_pt_embd        = %d\n", __func__, hparams.n_pt_embd);
        printf("%s: ftype            = %d\n", __func__, hparams.ftype);
        printf("%s: qntvr            = %d\n", __func__, qntvr);

        hparams.ftype %= GGML_QNT_VERSION_FACTOR;
    }

    // for the big tensors, we have the option to store the data in 16-bit floats or quantized
    // in order to save memory and also to speed up the computation
    ggml_type wtype = ggml_ftype_to_ggml_type((ggml_ftype)(model.hparams.ftype));
    if (wtype == GGML_TYPE_COUNT)
    {
        fprintf(stderr, "%s: invalid model file '%s' (bad ftype value %d)\n",
                __func__, fname.c_str(), model.hparams.ftype);
        return false;
    }

    auto &ctx = model.ctx;

    const size_t buf_size = [&]()
    {
        size_t buf_size = 0;

        const auto &hparams = model.hparams;

        const int32_t n_enc_state = hparams.n_enc_state;
        const int32_t n_enc_layer = hparams.n_enc_layer;
        const int32_t n_enc_head_dim = hparams.n_enc_head_dim();
        const int32_t n_enc_out_chans = hparams.n_enc_out_chans;
        const int32_t n_pt_embd = hparams.n_pt_embd;

        const int32_t n_enc_layer_local = hparams.global_attn_indices().size();
        const int32_t n_enc_layer_global = n_enc_layer - n_enc_layer_local;

        const int32_t n_img_embd = hparams.n_img_embd();
        const int32_t n_window_size = hparams.n_window_size();
        const int32_t n_patch_size = hparams.n_patch_size();

        // image encoder
        {
            buf_size += n_enc_state * n_img_embd * n_img_embd * ggml_type_sizef(GGML_TYPE_F32);

            buf_size += n_enc_state * 3 * n_patch_size * n_patch_size * ggml_type_sizef(GGML_TYPE_F16);
            buf_size += n_enc_state * ggml_type_sizef(GGML_TYPE_F32);

            buf_size += n_enc_state * n_enc_out_chans * 1 * 1 * ggml_type_sizef(GGML_TYPE_F16);
            buf_size += n_enc_out_chans * n_enc_out_chans * 3 * 3 * ggml_type_sizef(GGML_TYPE_F16);

            buf_size += n_enc_out_chans * ggml_type_sizef(GGML_TYPE_F32);
            buf_size += n_enc_out_chans * ggml_type_sizef(GGML_TYPE_F32);

            buf_size += n_enc_out_chans * ggml_type_sizef(GGML_TYPE_F32);
            buf_size += n_enc_out_chans * ggml_type_sizef(GGML_TYPE_F32);
        }

        // image encoder layers
        {
            buf_size += n_enc_layer * n_enc_state * ggml_type_sizef(GGML_TYPE_F32);
            buf_size += n_enc_layer * n_enc_state * ggml_type_sizef(GGML_TYPE_F32);

            buf_size += n_enc_layer_global * n_enc_head_dim * (2 * n_img_embd - 1) * ggml_type_sizef(GGML_TYPE_F16);
            buf_size += n_enc_layer_global * n_enc_head_dim * (2 * n_img_embd - 1) * ggml_type_sizef(GGML_TYPE_F16);

            buf_size += n_enc_layer_local * n_enc_head_dim * (2 * n_window_size - 1) * ggml_type_sizef(GGML_TYPE_F16);
            buf_size += n_enc_layer_local * n_enc_head_dim * (2 * n_window_size - 1) * ggml_type_sizef(GGML_TYPE_F16);

            buf_size += n_enc_layer * 3 * n_enc_state * n_enc_state * ggml_type_sizef(GGML_TYPE_F16);
            buf_size += n_enc_layer * 3 * n_enc_state * ggml_type_sizef(GGML_TYPE_F32);

            buf_size += n_enc_layer * n_enc_state * n_enc_state * ggml_type_sizef(GGML_TYPE_F16);
            buf_size += n_enc_layer * n_enc_state * ggml_type_sizef(GGML_TYPE_F32);

            buf_size += n_enc_layer * n_enc_state * ggml_type_sizef(GGML_TYPE_F32);
            buf_size += n_enc_layer * n_enc_state * ggml_type_sizef(GGML_TYPE_F32);

            buf_size += n_enc_layer * 4 * n_enc_state * n_enc_state * ggml_type_sizef(GGML_TYPE_F16);
            buf_size += n_enc_layer * 4 * n_enc_state * ggml_type_sizef(GGML_TYPE_F32);

            buf_size += n_enc_layer * 4 * n_enc_state * n_enc_state * ggml_type_sizef(GGML_TYPE_F16);
            buf_size += n_enc_layer * 4 * n_enc_state * ggml_type_sizef(GGML_TYPE_F32);
        }

        buf_size += (8 + 14 * n_enc_layer) * ggml_tensor_overhead();

        // prompt encoder
        {
            buf_size += n_enc_out_chans * ggml_type_sizef(GGML_TYPE_F16); // 2*(n_enc_out_chans/2)

            buf_size += n_enc_out_chans * ggml_type_sizef(GGML_TYPE_F32);
            buf_size += n_pt_embd * n_enc_out_chans * ggml_type_sizef(GGML_TYPE_F32);
        }

        buf_size += (2 + n_pt_embd) * ggml_tensor_overhead();

        // mask decoder
        {
            // transformer
            {
                const int tfm_layers_count = 2;
                const int qkv_count = 3;
                const int norm_count = 4;
                const int n_hypernet_mpls_count = 4;

                // self_attn
                buf_size += tfm_layers_count * qkv_count * n_enc_state * n_enc_state * ggml_type_sizef(GGML_TYPE_F16);
                buf_size += tfm_layers_count * qkv_count * n_enc_state * ggml_type_sizef(GGML_TYPE_F32);
                buf_size += tfm_layers_count * n_enc_state * ggml_type_sizef(GGML_TYPE_F32);

                // all norms
                buf_size += tfm_layers_count * norm_count * n_enc_state * ggml_type_sizef(GGML_TYPE_F32);
                buf_size += tfm_layers_count * norm_count * n_enc_state * ggml_type_sizef(GGML_TYPE_F32);

                // cross_attn_token_to_img
                buf_size += tfm_layers_count * qkv_count * n_enc_state * (n_enc_state / 2) * ggml_type_sizef(GGML_TYPE_F16);
                buf_size += tfm_layers_count * qkv_count * (n_enc_state / 2) * ggml_type_sizef(GGML_TYPE_F32);
                buf_size += tfm_layers_count * n_enc_state * ggml_type_sizef(GGML_TYPE_F32);

                // mlp
                buf_size += tfm_layers_count * 8 * n_enc_out_chans * n_enc_out_chans * ggml_type_sizef(GGML_TYPE_F16);
                buf_size += tfm_layers_count * 8 * n_enc_out_chans * ggml_type_sizef(GGML_TYPE_F32);
                buf_size += tfm_layers_count * n_enc_out_chans * 8 * n_enc_out_chans * ggml_type_sizef(GGML_TYPE_F16);
                buf_size += tfm_layers_count * n_enc_out_chans * ggml_type_sizef(GGML_TYPE_F32);

                // cross_attn_img_to_token
                buf_size += tfm_layers_count * qkv_count * n_enc_state * (n_enc_state / 2) * ggml_type_sizef(GGML_TYPE_F16);
                buf_size += tfm_layers_count * qkv_count * (n_enc_state / 2) * ggml_type_sizef(GGML_TYPE_F32);
                buf_size += tfm_layers_count * n_enc_state * ggml_type_sizef(GGML_TYPE_F32);

                // transformer_final_attn_token_to_img
                buf_size += qkv_count * n_enc_state * (n_enc_state / 2) * ggml_type_sizef(GGML_TYPE_F16);
                buf_size += qkv_count * (n_enc_state / 2) * ggml_type_sizef(GGML_TYPE_F32);
                buf_size += n_enc_state * ggml_type_sizef(GGML_TYPE_F32);

                // transformer_norm_final
                buf_size += norm_count * n_enc_state * ggml_type_sizef(GGML_TYPE_F32);
                buf_size += norm_count * n_enc_state * ggml_type_sizef(GGML_TYPE_F32);

                // output_upscaling
                buf_size += n_enc_out_chans * n_img_embd * 2 * 2 * ggml_type_sizef(GGML_TYPE_F16);
                buf_size += 3 * n_img_embd * ggml_type_sizef(GGML_TYPE_F32);
                buf_size += n_enc_out_chans * n_img_embd * (n_img_embd / 2) * 2 * 2 * ggml_type_sizef(GGML_TYPE_F16);
                buf_size += (n_img_embd / 2) * ggml_type_sizef(GGML_TYPE_F32);

                // output_hypernetworks_mlps
                buf_size += n_hypernet_mpls_count * 2 * n_enc_out_chans * n_enc_out_chans * ggml_type_sizef(GGML_TYPE_F16);
                buf_size += n_hypernet_mpls_count * 2 * n_enc_out_chans * ggml_type_sizef(GGML_TYPE_F32);
                buf_size += n_hypernet_mpls_count * n_enc_out_chans * (n_img_embd / 2) * ggml_type_sizef(GGML_TYPE_F16);
                buf_size += n_hypernet_mpls_count * (n_img_embd / 2) * ggml_type_sizef(GGML_TYPE_F32);

                // iou_prediction_head
                buf_size += 2 * n_enc_out_chans * n_enc_out_chans * ggml_type_sizef(GGML_TYPE_F16);
                buf_size += 2 * n_enc_out_chans * ggml_type_sizef(GGML_TYPE_F32);
                buf_size += n_pt_embd * n_enc_out_chans * ggml_type_sizef(GGML_TYPE_F16);
                buf_size += n_pt_embd * ggml_type_sizef(GGML_TYPE_F32);

                // iou_token_w
                buf_size += n_enc_out_chans * ggml_type_sizef(GGML_TYPE_F32);

                // mask_tokens_w
                buf_size += n_pt_embd * n_enc_out_chans * ggml_type_sizef(GGML_TYPE_F32);
            }
        }
        fprintf(stderr, "ggml buffer size = %6.2f MB\n", buf_size / (1024.0 * 1024.0));

        return buf_size;
    }();

    // create the ggml context
    {
        struct ggml_init_params params = {
            /*.mem_size   =*/ggml_tensor_overhead() * GGML_MAX_NODES,
            /*.mem_buffer =*/NULL,
            /*.no_alloc   =*/true,
        };

        ctx = ggml_init(params);
        if (!ctx)
        {
            fprintf(stderr, "%s: ggml_init() failed\n", __func__);
            return false;
        }
    }

    // initialize backend & allocate buffers
    {
        if (!model.backend)
        {
            printf("Using CPU backend\n");
            model.backend = ggml_backend_cpu_init();
            if (!model.backend)
            {
                fprintf(stderr, "%s: ggml_backend_cpu_init() failed\n", __func__);
                return false;
            }
        }

        model.buffer = ggml_backend_alloc_buffer(model.backend, buf_size);
    }

    // prepare memory for the weights
    {
        const auto &hparams = model.hparams;

        const int32_t n_enc_state = hparams.n_enc_state;
        const int32_t n_enc_layer = hparams.n_enc_layer;
        const int32_t n_enc_head_dim = hparams.n_enc_head_dim();
        const int32_t n_enc_out_chans = hparams.n_enc_out_chans;
        const int32_t n_pt_embd = hparams.n_pt_embd;

        const int32_t n_img_embd = hparams.n_img_embd();
        const int32_t n_window_size = hparams.n_window_size();
        const int32_t n_patch_size = hparams.n_patch_size();

        model.enc_img.layers.resize(n_enc_layer);

        // image encoder
        {
            auto &enc = model.enc_img;

            enc.pe = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, n_enc_state, n_img_embd, n_img_embd, 1);

            enc.proj_w = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, n_patch_size, n_patch_size, 3, n_enc_state);
            enc.proj_b = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, 1, n_enc_state);

            enc.neck_conv_0 = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 1, 1, n_enc_state, n_enc_out_chans);
            enc.neck_conv_1 = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 3, 3, n_enc_out_chans, n_enc_out_chans);

            enc.neck_norm_0_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_enc_out_chans);
            enc.neck_norm_0_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_enc_out_chans);

            enc.neck_norm_1_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_enc_out_chans);
            enc.neck_norm_1_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_enc_out_chans);

            model.tensors["image_encoder.pos_embed"] = enc.pe;

            model.tensors["image_encoder.patch_embed.proj.weight"] = enc.proj_w;
            model.tensors["image_encoder.patch_embed.proj.bias"] = enc.proj_b;

            model.tensors["image_encoder.neck.0.weight"] = enc.neck_conv_0;
            model.tensors["image_encoder.neck.2.weight"] = enc.neck_conv_1;

            model.tensors["image_encoder.neck.1.weight"] = enc.neck_norm_0_w;
            model.tensors["image_encoder.neck.1.bias"] = enc.neck_norm_0_b;

            model.tensors["image_encoder.neck.3.weight"] = enc.neck_norm_1_w;
            model.tensors["image_encoder.neck.3.bias"] = enc.neck_norm_1_b;

            for (int i = 0; i < n_enc_layer; ++i)
            {
                auto &layer = enc.layers[i];

                layer.norm1_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_enc_state);
                layer.norm1_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_enc_state);

                if (hparams.is_global_attn(i))
                {
                    layer.rel_pos_w = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, n_enc_head_dim, 2 * n_img_embd - 1);
                    layer.rel_pos_h = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, n_enc_head_dim, 2 * n_img_embd - 1);
                }
                else
                {
                    layer.rel_pos_w = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, n_enc_head_dim, 2 * n_window_size - 1);
                    layer.rel_pos_h = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, n_enc_head_dim, 2 * n_window_size - 1);
                }

                layer.qkv_w = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, n_enc_state, 3 * n_enc_state);
                layer.qkv_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 3 * n_enc_state);

                layer.proj_w = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, n_enc_state, n_enc_state);
                layer.proj_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_enc_state);

                layer.norm2_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_enc_state);
                layer.norm2_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_enc_state);

                layer.mlp_lin1_w = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, n_enc_state, 4 * n_enc_state);
                layer.mlp_lin1_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4 * n_enc_state);

                layer.mlp_lin2_w = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 4 * n_enc_state, n_enc_state);
                layer.mlp_lin2_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_enc_state);

                model.tensors["image_encoder.blocks." + std::to_string(i) + ".norm1.weight"] = layer.norm1_w;
                model.tensors["image_encoder.blocks." + std::to_string(i) + ".norm1.bias"] = layer.norm1_b;

                model.tensors["image_encoder.blocks." + std::to_string(i) + ".attn.rel_pos_w"] = layer.rel_pos_w;
                model.tensors["image_encoder.blocks." + std::to_string(i) + ".attn.rel_pos_h"] = layer.rel_pos_h;

                model.tensors["image_encoder.blocks." + std::to_string(i) + ".attn.qkv.weight"] = layer.qkv_w;
                model.tensors["image_encoder.blocks." + std::to_string(i) + ".attn.qkv.bias"] = layer.qkv_b;

                model.tensors["image_encoder.blocks." + std::to_string(i) + ".attn.proj.weight"] = layer.proj_w;
                model.tensors["image_encoder.blocks." + std::to_string(i) + ".attn.proj.bias"] = layer.proj_b;

                model.tensors["image_encoder.blocks." + std::to_string(i) + ".norm2.weight"] = layer.norm2_w;
                model.tensors["image_encoder.blocks." + std::to_string(i) + ".norm2.bias"] = layer.norm2_b;

                model.tensors["image_encoder.blocks." + std::to_string(i) + ".mlp.lin1.weight"] = layer.mlp_lin1_w;
                model.tensors["image_encoder.blocks." + std::to_string(i) + ".mlp.lin1.bias"] = layer.mlp_lin1_b;

                model.tensors["image_encoder.blocks." + std::to_string(i) + ".mlp.lin2.weight"] = layer.mlp_lin2_w;
                model.tensors["image_encoder.blocks." + std::to_string(i) + ".mlp.lin2.bias"] = layer.mlp_lin2_b;
            }
        }

        // prompt encoder
        {
            auto &enc = model.enc_prompt;

            enc.pe = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_enc_out_chans / 2, 2);

            enc.not_a_pt_embd_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_enc_out_chans);
            enc.no_mask_embd_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_enc_out_chans);

            model.tensors["prompt_encoder.pe_layer.positional_encoding_gaussian_matrix"] = enc.pe;
            model.tensors["prompt_encoder.not_a_point_embed.weight"] = enc.not_a_pt_embd_w;
            model.tensors["prompt_encoder.no_mask_embed.weight"] = enc.no_mask_embd_w;

            enc.pt_embd.resize(n_pt_embd);
            for (int i = 0; i < n_pt_embd; i++)
            {
                enc.pt_embd[i] = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_enc_out_chans);

                model.tensors["prompt_encoder.point_embeddings." + std::to_string(i) + ".weight"] = enc.pt_embd[i];
            }
        }

        // mask decoder
        {
            auto &dec = model.dec;
            auto &tfm_layers = dec.transformer_layers;

            const int tfm_layers_count = 2;
            tfm_layers.resize(tfm_layers_count);
            for (int i = 0; i < tfm_layers_count; ++i)
            {
                auto &l = tfm_layers[i];
                l.self_attn.q_w = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, n_enc_out_chans, n_enc_out_chans);
                l.self_attn.q_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_enc_out_chans);
                l.self_attn.k_w = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, n_enc_out_chans, n_enc_out_chans);
                l.self_attn.k_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_enc_out_chans);
                l.self_attn.v_w = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, n_enc_out_chans, n_enc_out_chans);
                l.self_attn.v_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_enc_out_chans);
                l.self_attn.out_w = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, n_enc_out_chans, n_enc_out_chans);
                l.self_attn.out_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_enc_out_chans);

                l.norm1_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_enc_out_chans);
                l.norm1_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_enc_out_chans);

                l.cross_attn_token_to_img.q_w = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, n_enc_out_chans, n_enc_out_chans / 2);
                l.cross_attn_token_to_img.q_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_enc_out_chans / 2);
                l.cross_attn_token_to_img.k_w = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, n_enc_out_chans, n_enc_out_chans / 2);
                l.cross_attn_token_to_img.k_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_enc_out_chans / 2);
                l.cross_attn_token_to_img.v_w = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, n_enc_out_chans, n_enc_out_chans / 2);
                l.cross_attn_token_to_img.v_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_enc_out_chans / 2);
                l.cross_attn_token_to_img.out_w = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, n_enc_out_chans / 2, n_enc_out_chans);
                l.cross_attn_token_to_img.out_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_enc_out_chans);

                l.norm2_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_enc_out_chans);
                l.norm2_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_enc_out_chans);

                l.mlp_lin1_w = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, n_enc_out_chans, 8 * n_enc_out_chans);
                l.mlp_lin1_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8 * n_enc_out_chans);
                l.mlp_lin2_w = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 8 * n_enc_out_chans, n_enc_out_chans);
                l.mlp_lin2_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_enc_out_chans);

                l.norm3_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_enc_out_chans);
                l.norm3_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_enc_out_chans);

                l.norm4_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_enc_out_chans);
                l.norm4_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_enc_out_chans);

                l.cross_attn_img_to_token.q_w = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, n_enc_out_chans, n_enc_out_chans / 2);
                l.cross_attn_img_to_token.q_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_enc_out_chans / 2);
                l.cross_attn_img_to_token.k_w = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, n_enc_out_chans, n_enc_out_chans / 2);
                l.cross_attn_img_to_token.k_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_enc_out_chans / 2);
                l.cross_attn_img_to_token.v_w = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, n_enc_out_chans, n_enc_out_chans / 2);
                l.cross_attn_img_to_token.v_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_enc_out_chans / 2);
                l.cross_attn_img_to_token.out_w = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, n_enc_out_chans / 2, n_enc_out_chans);
                l.cross_attn_img_to_token.out_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_enc_out_chans);

                const auto prefix = "mask_decoder.transformer.layers." + std::to_string(i) + ".";
                model.tensors[prefix + "self_attn.q_proj.weight"] = l.self_attn.q_w;
                model.tensors[prefix + "self_attn.q_proj.bias"] = l.self_attn.q_b;
                model.tensors[prefix + "self_attn.k_proj.weight"] = l.self_attn.k_w;
                model.tensors[prefix + "self_attn.k_proj.bias"] = l.self_attn.k_b;
                model.tensors[prefix + "self_attn.v_proj.weight"] = l.self_attn.v_w;
                model.tensors[prefix + "self_attn.v_proj.bias"] = l.self_attn.v_b;
                model.tensors[prefix + "self_attn.out_proj.weight"] = l.self_attn.out_w;
                model.tensors[prefix + "self_attn.out_proj.bias"] = l.self_attn.out_b;

                model.tensors[prefix + "norm1.weight"] = l.norm1_w;
                model.tensors[prefix + "norm1.bias"] = l.norm1_b;

                model.tensors[prefix + "cross_attn_token_to_image.q_proj.weight"] = l.cross_attn_token_to_img.q_w;
                model.tensors[prefix + "cross_attn_token_to_image.q_proj.bias"] = l.cross_attn_token_to_img.q_b;
                model.tensors[prefix + "cross_attn_token_to_image.k_proj.weight"] = l.cross_attn_token_to_img.k_w;
                model.tensors[prefix + "cross_attn_token_to_image.k_proj.bias"] = l.cross_attn_token_to_img.k_b;
                model.tensors[prefix + "cross_attn_token_to_image.v_proj.weight"] = l.cross_attn_token_to_img.v_w;
                model.tensors[prefix + "cross_attn_token_to_image.v_proj.bias"] = l.cross_attn_token_to_img.v_b;
                model.tensors[prefix + "cross_attn_token_to_image.out_proj.weight"] = l.cross_attn_token_to_img.out_w;
                model.tensors[prefix + "cross_attn_token_to_image.out_proj.bias"] = l.cross_attn_token_to_img.out_b;

                model.tensors[prefix + "norm2.weight"] = l.norm2_w;
                model.tensors[prefix + "norm2.bias"] = l.norm2_b;

                model.tensors[prefix + "mlp.lin1.weight"] = l.mlp_lin1_w;
                model.tensors[prefix + "mlp.lin1.bias"] = l.mlp_lin1_b;
                model.tensors[prefix + "mlp.lin2.weight"] = l.mlp_lin2_w;
                model.tensors[prefix + "mlp.lin2.bias"] = l.mlp_lin2_b;

                model.tensors[prefix + "norm3.weight"] = l.norm3_w;
                model.tensors[prefix + "norm3.bias"] = l.norm3_b;
                model.tensors[prefix + "norm4.weight"] = l.norm4_w;
                model.tensors[prefix + "norm4.bias"] = l.norm4_b;

                model.tensors[prefix + "cross_attn_image_to_token.q_proj.weight"] = l.cross_attn_img_to_token.q_w;
                model.tensors[prefix + "cross_attn_image_to_token.q_proj.bias"] = l.cross_attn_img_to_token.q_b;
                model.tensors[prefix + "cross_attn_image_to_token.k_proj.weight"] = l.cross_attn_img_to_token.k_w;
                model.tensors[prefix + "cross_attn_image_to_token.k_proj.bias"] = l.cross_attn_img_to_token.k_b;
                model.tensors[prefix + "cross_attn_image_to_token.v_proj.weight"] = l.cross_attn_img_to_token.v_w;
                model.tensors[prefix + "cross_attn_image_to_token.v_proj.bias"] = l.cross_attn_img_to_token.v_b;
                model.tensors[prefix + "cross_attn_image_to_token.out_proj.weight"] = l.cross_attn_img_to_token.out_w;
                model.tensors[prefix + "cross_attn_image_to_token.out_proj.bias"] = l.cross_attn_img_to_token.out_b;
            }

            dec.transformer_final_attn_token_to_img.q_w = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, n_enc_out_chans, n_enc_out_chans / 2);
            dec.transformer_final_attn_token_to_img.q_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_enc_out_chans / 2);
            dec.transformer_final_attn_token_to_img.k_w = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, n_enc_out_chans, n_enc_out_chans / 2);
            dec.transformer_final_attn_token_to_img.k_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_enc_out_chans / 2);
            dec.transformer_final_attn_token_to_img.v_w = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, n_enc_out_chans, n_enc_out_chans / 2);
            dec.transformer_final_attn_token_to_img.v_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_enc_out_chans / 2);
            dec.transformer_final_attn_token_to_img.out_w = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, n_enc_out_chans / 2, n_enc_out_chans);
            dec.transformer_final_attn_token_to_img.out_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_enc_out_chans);

            model.tensors["mask_decoder.transformer.final_attn_token_to_image.q_proj.weight"] = dec.transformer_final_attn_token_to_img.q_w;
            model.tensors["mask_decoder.transformer.final_attn_token_to_image.q_proj.bias"] = dec.transformer_final_attn_token_to_img.q_b;
            model.tensors["mask_decoder.transformer.final_attn_token_to_image.k_proj.weight"] = dec.transformer_final_attn_token_to_img.k_w;
            model.tensors["mask_decoder.transformer.final_attn_token_to_image.k_proj.bias"] = dec.transformer_final_attn_token_to_img.k_b;
            model.tensors["mask_decoder.transformer.final_attn_token_to_image.v_proj.weight"] = dec.transformer_final_attn_token_to_img.v_w;
            model.tensors["mask_decoder.transformer.final_attn_token_to_image.v_proj.bias"] = dec.transformer_final_attn_token_to_img.v_b;
            model.tensors["mask_decoder.transformer.final_attn_token_to_image.out_proj.weight"] = dec.transformer_final_attn_token_to_img.out_w;
            model.tensors["mask_decoder.transformer.final_attn_token_to_image.out_proj.bias"] = dec.transformer_final_attn_token_to_img.out_b;

            dec.transformer_norm_final_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_enc_out_chans);
            dec.transformer_norm_final_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_enc_out_chans);

            model.tensors["mask_decoder.transformer.norm_final_attn.weight"] = dec.transformer_norm_final_w;
            model.tensors["mask_decoder.transformer.norm_final_attn.bias"] = dec.transformer_norm_final_b;

            dec.output_upscaling_0_w = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 2, 2, n_img_embd, n_enc_out_chans);
            dec.output_upscaling_0_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_img_embd);
            dec.output_upscaling_1_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_img_embd);
            dec.output_upscaling_1_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_img_embd);
            dec.output_upscaling_3_w = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 2, 2, n_img_embd / 2, n_img_embd);
            dec.output_upscaling_3_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_img_embd / 2);

            model.tensors["mask_decoder.output_upscaling.0.weight"] = dec.output_upscaling_0_w;
            model.tensors["mask_decoder.output_upscaling.0.bias"] = dec.output_upscaling_0_b;
            model.tensors["mask_decoder.output_upscaling.1.weight"] = dec.output_upscaling_1_w;
            model.tensors["mask_decoder.output_upscaling.1.bias"] = dec.output_upscaling_1_b;
            model.tensors["mask_decoder.output_upscaling.3.weight"] = dec.output_upscaling_3_w;
            model.tensors["mask_decoder.output_upscaling.3.bias"] = dec.output_upscaling_3_b;

            const int n_hypernet_mpls_count = 4;
            dec.output_hypernet_mlps.resize(n_hypernet_mpls_count);
            for (int i = 0; i < n_hypernet_mpls_count; ++i)
            {
                auto &mlp = dec.output_hypernet_mlps[i];

                mlp.w_0 = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, n_enc_out_chans, n_enc_out_chans);
                mlp.b_0 = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_enc_out_chans);
                mlp.w_1 = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, n_enc_out_chans, n_enc_out_chans);
                mlp.b_1 = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_enc_out_chans);
                mlp.w_2 = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, n_enc_out_chans, n_img_embd / 2);
                mlp.b_2 = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_img_embd / 2);

                const auto prefix = "mask_decoder.output_hypernetworks_mlps." + std::to_string(i) + ".";
                model.tensors[prefix + "layers.0.weight"] = mlp.w_0;
                model.tensors[prefix + "layers.0.bias"] = mlp.b_0;
                model.tensors[prefix + "layers.1.weight"] = mlp.w_1;
                model.tensors[prefix + "layers.1.bias"] = mlp.b_1;
                model.tensors[prefix + "layers.2.weight"] = mlp.w_2;
                model.tensors[prefix + "layers.2.bias"] = mlp.b_2;
            }

            dec.iou_prediction_head_0_w = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, n_enc_out_chans, n_enc_out_chans);
            dec.iou_prediction_head_0_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_enc_out_chans);
            dec.iou_prediction_head_1_w = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, n_enc_out_chans, n_enc_out_chans);
            dec.iou_prediction_head_1_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_enc_out_chans);
            dec.iou_prediction_head_2_w = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, n_enc_out_chans, n_pt_embd);
            dec.iou_prediction_head_2_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_pt_embd);

            dec.iou_token_w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_enc_out_chans, 1);
            dec.mask_tokens_w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_enc_out_chans, n_pt_embd);

            model.tensors["mask_decoder.iou_prediction_head.layers.0.weight"] = dec.iou_prediction_head_0_w;
            model.tensors["mask_decoder.iou_prediction_head.layers.0.bias"] = dec.iou_prediction_head_0_b;
            model.tensors["mask_decoder.iou_prediction_head.layers.1.weight"] = dec.iou_prediction_head_1_w;
            model.tensors["mask_decoder.iou_prediction_head.layers.1.bias"] = dec.iou_prediction_head_1_b;
            model.tensors["mask_decoder.iou_prediction_head.layers.2.weight"] = dec.iou_prediction_head_2_w;
            model.tensors["mask_decoder.iou_prediction_head.layers.2.bias"] = dec.iou_prediction_head_2_b;

            model.tensors["mask_decoder.iou_token.weight"] = dec.iou_token_w;
            model.tensors["mask_decoder.mask_tokens.weight"] = dec.mask_tokens_w;
        }
    }

    // load weights
    {
        ggml_allocr *alloc = ggml_allocr_new_from_buffer(model.buffer);

        int n_tensors = 0;
        size_t total_size = 0;

        fprintf(stderr, "%s: ", __func__);

        while (true)
        {
            int32_t n_dims;
            int32_t length;
            int32_t ftype;

            fin.read(reinterpret_cast<char *>(&n_dims), sizeof(n_dims));
            fin.read(reinterpret_cast<char *>(&length), sizeof(length));
            fin.read(reinterpret_cast<char *>(&ftype), sizeof(ftype));

            if (fin.eof())
            {
                break;
            }

            int64_t nelements = 1;
            int64_t ne[4] = {1, 1, 1, 1};
            for (int i = 0; i < n_dims; ++i)
            {
                int32_t ne_cur;
                fin.read(reinterpret_cast<char *>(&ne_cur), sizeof(ne_cur));
                ne[i] = ne_cur;
                nelements *= ne[i];
            }

            std::string name(length, 0);
            fin.read(&name[0], length);

            if (model.tensors.find(name) == model.tensors.end())
            {
                fprintf(stderr, "%s: unknown tensor '%s' in model file\n", __func__, name.c_str());
                return false;
            }

            auto tensor = model.tensors[name];
            ggml_set_name(tensor, name.c_str());

            if (ggml_nelements(tensor) != nelements)
            {
                fprintf(stderr, "%s: tensor '%s' has wrong size in model file: got %d, expected %d\n",
                        __func__, name.c_str(), (int)nelements, (int)ggml_nelements(tensor));
                return false;
            }

            if (tensor->ne[0] != ne[0] || tensor->ne[1] != ne[1] || tensor->ne[2] != ne[2] || tensor->ne[3] != ne[3])
            {
                fprintf(stderr, "%s: tensor '%s' has wrong shape in model file: got [%d, %d, %d, %d], expected [%d, %d, %d, %d]\n",
                        __func__, name.c_str(),
                        (int)ne[0], (int)ne[1], (int)ne[2], (int)ne[3],
                        (int)tensor->ne[0], (int)tensor->ne[1], (int)tensor->ne[2], (int)tensor->ne[3]);
                return false;
            }

            size_t bpe = 0;

            switch (ftype)
            {
            case 0:
                bpe = ggml_type_size(GGML_TYPE_F32);
                break;
            case 1:
                bpe = ggml_type_size(GGML_TYPE_F16);
                break;
            case 2:
                bpe = ggml_type_size(GGML_TYPE_Q4_0);
                assert(ne[0] % 64 == 0);
                break;
            case 3:
                bpe = ggml_type_size(GGML_TYPE_Q4_1);
                assert(ne[0] % 64 == 0);
                break;
            default:
            {
                fprintf(stderr, "%s: unknown ftype %d in model file\n", __func__, ftype);
                return false;
            }
            };

            if ((nelements * bpe) / ggml_blck_size(tensor->type) != ggml_nbytes(tensor))
            {
                fprintf(stderr, "%s: tensor '%s' has wrong size in model file: got %zu, expected %zu\n",
                        __func__, name.data(), ggml_nbytes(tensor), (size_t)nelements * bpe);
                return false;
            }

            ggml_allocr_alloc(alloc, tensor);
            fin.read(reinterpret_cast<char *>(tensor->data), ggml_nbytes(tensor));

            total_size += ggml_nbytes(tensor);
            if (++n_tensors % 8 == 0)
            {
                fprintf(stderr, ".");
                fflush(stdout);
            }
        }

        if (n_tensors != model.tensors.size())
        {
            fprintf(stderr, "%s: model file has %d tensors, but %d tensors were expected\n", __func__, n_tensors, (int)model.tensors.size());
            return false;
        }

        fprintf(stderr, " done\n");

        fprintf(stderr, "%s: model size = %8.2f MB / num tensors = %d\n", __func__, total_size / 1024.0 / 1024.0, n_tensors);

        ggml_allocr_free(alloc);
    }

    fin.close();

    return true;
}

std::shared_ptr<sam_state> sam_load_model(const sam_params &params)
{
    ggml_time_init();
    const int64_t t_start_ms = ggml_time_ms();

    sam_state state;
    state.model = std::make_unique<sam_ggml_model>();
    state.state = std::make_unique<sam_ggml_state>();
    if (!sam_ggml_model_load(params.model, *state.model))
    {
        fprintf(stderr, "%s: failed to load model from '%s'\n", __func__, params.model.c_str());
        return {};
    }

    state.t_load_ms = ggml_time_ms() - t_start_ms;

    return std::make_unique<sam_state>(std::move(state));
}

std::vector<sam_image_u8> sam_compute_masks(
    const sam_image_u8 &img,
    int n_threads,
    sam_point pt,
    sam_state &state,
    int mask_on_val,
    int mask_off_val)
{
    if (!state.model || !state.state)
    {
        return {};
    }

    const int64_t t_start_ms = ggml_time_ms();

    static size_t buf_size = 256u * 1024 * 1024;

    struct ggml_init_params ggml_params = {
        /*.mem_size   =*/buf_size,
        /*.mem_buffer =*/NULL,
        /*.no_alloc   =*/false,
    };

    auto &st = *state.state;
    auto &model = *state.model;

    st.ctx_masks = ggml_init(ggml_params);

    st.low_res_masks = ggml_new_tensor_3d(st.ctx_masks, GGML_TYPE_F32,
                                          model.hparams.n_enc_out_chans, model.hparams.n_enc_out_chans, 3);

    st.iou_predictions = ggml_new_tensor_1d(st.ctx_masks, GGML_TYPE_F32, 3);

    const size_t alignment = ggml_backend_get_alignment(model.backend);
    st.allocr = ggml_allocr_new_measure(alignment);

    // measure memory requirements for the graph
    struct ggml_cgraph *gf_measure = sam_build_fast_graph(model, st, img.nx, img.ny, pt);
    if (!gf_measure)
    {
        fprintf(stderr, "%s: failed to build fast graph to measure\n", __func__);
        return {};
    }

    size_t alloc_size = ggml_allocr_alloc_graph(st.allocr, gf_measure);
    ggml_allocr_free(st.allocr);

    // recreate allocator with exact memory requirements
    ggml_backend_buffer_t buf_compute = ggml_backend_alloc_buffer(model.backend, alloc_size);
    st.allocr = ggml_allocr_new_from_buffer(buf_compute);

    // compute the graph with the measured exact memory requirements from above
    ggml_allocr_reset(st.allocr);

    struct ggml_cgraph *gf = sam_build_fast_graph(model, st, img.nx, img.ny, pt);
    if (!gf)
    {
        fprintf(stderr, "%s: failed to build fast graph\n", __func__);
        return {};
    }

    ggml_allocr_alloc_graph(st.allocr, gf);

    ggml_graph_compute_helper(model.backend, gf, n_threads);

    // print_t_f32("iou_predictions", st.iou_predictions);
    // print_t_f32("low_res_masks", st.low_res_masks);

    std::vector<sam_image_u8> masks = sam_postprocess_masks(model.hparams, img.nx, img.ny, st, mask_on_val, mask_off_val);

    ggml_allocr_free(st.allocr);
    ggml_free(st.ctx_masks);
    ggml_backend_buffer_free(buf_compute);

    st.allocr = {};
    st.ctx_masks = {};
    st.low_res_masks = {};
    st.iou_predictions = {};

    state.t_compute_masks_ms = ggml_time_ms() - t_start_ms;

    return masks;
}

void sam_deinit(sam_state &state)
{
    if (state.state)
    {
        if (state.state->ctx_img)
        {
            ggml_free(state.state->ctx_img);
        }
        state.model.reset();
        state.state.reset();
    }

    if (state.model)
    {
        if (state.model->backend)
        {
            ggml_backend_free(state.model->backend);
            ggml_backend_buffer_free(state.model->buffer);
        }
    }
}