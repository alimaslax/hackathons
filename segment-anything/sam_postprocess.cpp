#include "sam_postprocess.h"

// Implement sam_postprocess_masks function
std::vector<sam_image_u8> sam_postprocess_masks(
    const sam_hparams &hparams,
    int nx,
    int ny,
    const sam_ggml_state &state,
    int mask_on_val,
    int mask_off_val)
{
    if (state.low_res_masks->ne[2] == 0)
        return {};
    if (state.low_res_masks->ne[2] != state.iou_predictions->ne[0])
    {
        printf("Error: number of masks (%d) does not match number of iou predictions (%d)\n", (int)state.low_res_masks->ne[2], (int)state.iou_predictions->ne[0]);
        return {};
    }

    const int n_img_size = hparams.n_img_size();
    const float mask_threshold = hparams.mask_threshold;
    const float iou_threshold = hparams.iou_threshold;
    const float stability_score_threshold = hparams.stability_score_threshold;
    const float intersection_threshold = mask_threshold + hparams.stability_score_offset;
    const float union_threshold = mask_threshold - hparams.stability_score_offset;

    const int ne0 = state.low_res_masks->ne[0];
    const int ne1 = state.low_res_masks->ne[1];
    const int ne2 = state.low_res_masks->ne[2];

    // Remove padding and upscale masks to the original image size.
    // ref: https://github.com/facebookresearch/segment-anything/blob/efeab7296ab579d4a261e554eca80faf6b33924a/segment_anything/modeling/sam.py#L140

    const float preprocess_scale = std::max(nx, ny) / float(n_img_size);
    const int cropped_nx = int(nx / preprocess_scale + 0.5f);
    const int cropped_ny = int(ny / preprocess_scale + 0.5f);

    const float scale_x_1 = (float)ne0 / (float)n_img_size;
    const float scale_y_1 = (float)ne1 / (float)n_img_size;

    const float scale_x_2 = float(cropped_nx) / float(nx);
    const float scale_y_2 = float(cropped_ny) / float(ny);

    const auto iou_data = (float *)state.iou_predictions->data;

    std::map<float, sam_image_u8, std::greater<float>> res_map;
    for (int i = 0; i < ne2; ++i)
    {
        if (iou_threshold > 0.f && iou_data[i] < iou_threshold)
        {
            printf("Skipping mask %d with iou %f below threshold %f\n", i, iou_data[i], iou_threshold);
            continue; // Filtering masks with iou below the threshold
        }

        std::vector<float> mask_data(n_img_size * n_img_size);
        {
            const float *data = (float *)state.low_res_masks->data + i * ne0 * ne1;

            for (int iy = 0; iy < n_img_size; ++iy)
            {
                for (int ix = 0; ix < n_img_size; ++ix)
                {
                    const float sx = std::max(scale_x_1 * (ix + 0.5f) - 0.5f, 0.0f);
                    const float sy = std::max(scale_y_1 * (iy + 0.5f) - 0.5f, 0.0f);

                    const int x0 = std::max(0, (int)sx);
                    const int y0 = std::max(0, (int)sy);

                    const int x1 = std::min(x0 + 1, ne0 - 1);
                    const int y1 = std::min(y0 + 1, ne1 - 1);

                    const float dx = sx - x0;
                    const float dy = sy - y0;

                    const int j00 = y0 * ne0 + x0;
                    const int j01 = y0 * ne0 + x1;
                    const int j10 = y1 * ne0 + x0;
                    const int j11 = y1 * ne0 + x1;

                    const float v00 = data[j00];
                    const float v01 = data[j01];
                    const float v10 = data[j10];
                    const float v11 = data[j11];

                    const float v0 = (1 - dx) * v00 + dx * v01;
                    const float v1 = (1 - dx) * v10 + dx * v11;

                    const float v = (1 - dy) * v0 + dy * v1;

                    mask_data[iy * n_img_size + ix] = v;
                }
            }
        }

        int intersections = 0;
        int unions = 0;
        sam_image_u8 res;
        int min_iy = ny;
        int max_iy = 0;
        int min_ix = nx;
        int max_ix = 0;
        {
            const float *data = mask_data.data();

            res.nx = nx;
            res.ny = ny;
            res.data.resize(nx * ny, mask_off_val);

            for (int iy = 0; iy < ny; ++iy)
            {
                for (int ix = 0; ix < nx; ++ix)
                {
                    const float sx = std::max(scale_x_2 * (ix + 0.5f) - 0.5f, 0.0f);
                    const float sy = std::max(scale_y_2 * (iy + 0.5f) - 0.5f, 0.0f);

                    const int x0 = std::max(0, (int)sx);
                    const int y0 = std::max(0, (int)sy);

                    const int x1 = std::min(x0 + 1, cropped_nx - 1);
                    const int y1 = std::min(y0 + 1, cropped_ny - 1);

                    const float dx = sx - x0;
                    const float dy = sy - y0;

                    const int j00 = y0 * n_img_size + x0;
                    const int j01 = y0 * n_img_size + x1;
                    const int j10 = y1 * n_img_size + x0;
                    const int j11 = y1 * n_img_size + x1;

                    const float v00 = data[j00];
                    const float v01 = data[j01];
                    const float v10 = data[j10];
                    const float v11 = data[j11];

                    const float v0 = (1 - dx) * v00 + dx * v01;
                    const float v1 = (1 - dx) * v10 + dx * v11;

                    const float v = (1 - dy) * v0 + dy * v1;

                    if (v > intersection_threshold)
                    {
                        intersections++;
                    }
                    if (v > union_threshold)
                    {
                        unions++;
                    }
                    if (v > mask_threshold)
                    {
                        min_iy = std::min(min_iy, iy);
                        max_iy = std::max(max_iy, iy);
                        min_ix = std::min(min_ix, ix);
                        max_ix = std::max(max_ix, ix);

                        res.data[iy * nx + ix] = mask_on_val;
                    }
                }
            }
        }

        const float stability_score = float(intersections) / float(unions);
        if (stability_score_threshold > 0.f && stability_score < stability_score_threshold)
        {
            printf("Skipping mask %d with stability score %f below threshold %f\n", i, stability_score, stability_score_threshold);
            continue; // Filtering masks with stability score below the threshold
        }

        printf("Mask %d: iou = %f, stability_score = %f, bbox (%d, %d), (%d, %d)\n",
               i, iou_data[i], stability_score, min_ix, max_ix, min_iy, max_iy);

        res_map[iou_data[i] + stability_score] = std::move(res);
    }

    std::vector<sam_image_u8> res;
    for (auto &mask : res_map)
    {
        res.push_back(std::move(mask.second));
    }

    return res;
}