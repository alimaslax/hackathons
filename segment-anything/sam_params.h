#pragma once

#include <string>
#include <thread>
#include <vector>

struct sam_params
{
    int32_t seed = -1; // RNG seed
    int32_t n_threads = std::min(4, (int32_t)std::thread::hardware_concurrency());

    std::string model = "../checkpoints/ggml-model-f16-b.bin"; // model path
    std::string fname_inp = "../img.jpg";
    std::string fname_out = "img.out";
};

struct sam_hparams
{
    int32_t n_enc_state = 768;
    int32_t n_enc_layer = 12;
    int32_t n_enc_head = 12;
    int32_t n_enc_out_chans = 256;
    int32_t n_pt_embd = 4;
    int32_t n_dec_heads = 8;
    int32_t ftype = 1;
    float mask_threshold = 0.f;
    float iou_threshold = 0.85f;             // Default in PyTorch is 0.88f
    float stability_score_threshold = 0.90f; // Default in PyTorch is 0.95f
    float stability_score_offset = 1.0f;
    float eps = 1e-6f;
    float eps_decoder_transformer = 1e-5f;

    int32_t n_enc_head_dim() const { return n_enc_state / n_enc_head; }
    int32_t n_img_size() const { return 1024; }
    int32_t n_window_size() const { return 14; }
    int32_t n_patch_size() const { return 16; }
    int32_t n_img_embd() const { return n_img_size() / n_patch_size(); }

    std::vector<int32_t> global_attn_indices() const
    {
        switch (n_enc_state)
        {
        case 768:
            return {2, 5, 8, 11};
        case 1024:
            return {5, 11, 17, 23};
        case 1280:
            return {7, 15, 23, 31};
        default:
        {
            printf("%s: unsupported n_enc_state = %d\n", __func__, n_enc_state);
        }
        break;
        };

        return {};
    }

    bool is_global_attn(int32_t layer) const
    {
        const auto indices = global_attn_indices();

        for (const auto &idx : indices)
        {
            if (layer == idx)
            {
                return true;
            }
        }

        return false;
    }
};