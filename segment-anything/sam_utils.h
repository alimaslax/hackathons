#pragma once

#include "ggml.h"
#include "ggml-backend.h"

static void ggml_graph_compute_helper(ggml_backend_t backend, ggml_cgraph *graph, int n_threads);
static void ggml_disconnect_node_from_graph(ggml_tensor *t);

// Add declarations for the new functions
static void ggml_sam_sin(struct ggml_tensor *dst, const struct ggml_tensor *src, int ith, int nth, void *userdata);
static void ggml_sam_cos(struct ggml_tensor *dst, const struct ggml_tensor *src, int ith, int nth, void *userdata);

// Add declarations for using these operations in the GGML graph
struct ggml_tensor *ggml_sam_sin_op(struct ggml_context *ctx, struct ggml_tensor *src);
struct ggml_tensor *ggml_sam_cos_op(struct ggml_context *ctx, struct ggml_tensor *src);