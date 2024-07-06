#include "sam_utils.h"
#include "sam.h"

static void ggml_graph_compute_helper(ggml_backend_t backend, ggml_cgraph *graph, int n_threads)
{
    ggml_backend_cpu_set_n_threads(backend, n_threads);
    ggml_backend_graph_compute(backend, graph);
}

static void ggml_disconnect_node_from_graph(ggml_tensor *t)
{
    t->op = GGML_OP_NONE;
    for (int i = 0; i < GGML_MAX_SRC; i++)
    {
        t->src[i] = NULL;
    }
}

static void ggml_sam_sin(struct ggml_tensor *dst, const struct ggml_tensor *src, int ith, int nth, void *userdata)
{
    GGML_ASSERT(userdata == NULL);
    GGML_ASSERT(ggml_are_same_shape(dst, src));
    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_is_contiguous(src));

    const float *src_data = ggml_get_data_f32(src);
    float *dst_data = ggml_get_data_f32(dst);

    const int ne = (int)ggml_nelements(dst);
    const int dr = (ne + nth - 1) / nth;
    const int ie0 = dr * ith;
    const int ie1 = std::min(ie0 + dr, ne);

    for (int i = ie0; i < ie1; ++i)
    {
        dst_data[i] = sinf(src_data[i]);
    }
}

static void ggml_sam_cos(struct ggml_tensor *dst, const struct ggml_tensor *src, int ith, int nth, void *userdata)
{
    GGML_ASSERT(userdata == NULL);
    GGML_ASSERT(ggml_are_same_shape(dst, src));
    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_is_contiguous(src));

    const float *src_data = ggml_get_data_f32(src);
    float *dst_data = ggml_get_data_f32(dst);

    const int ne = (int)ggml_nelements(dst);
    const int dr = (ne + nth - 1) / nth;
    const int ie0 = dr * ith;
    const int ie1 = std::min(ie0 + dr, ne);

    for (int i = ie0; i < ie1; ++i)
    {
        dst_data[i] = cosf(src_data[i]);
    }
}