#pragma once

#include "ggml-backend.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>

void ggml_graph_compute_helper(ggml_backend_t backend, ggml_cgraph *graph, int n_threads);
void ggml_disconnect_node_from_graph(ggml_tensor *t);

// Add declarations for the new functions
void ggml_sam_sin(struct ggml_tensor *dst, const struct ggml_tensor *src, int ith, int nth, void *userdata);
void ggml_sam_cos(struct ggml_tensor *dst, const struct ggml_tensor *src, int ith, int nth, void *userdata);

// Add declarations for using these operations in the GGML graph
struct ggml_tensor *ggml_sam_sin_op(struct ggml_context *ctx, struct ggml_tensor *src);
struct ggml_tensor *ggml_sam_cos_op(struct ggml_context *ctx, struct ggml_tensor *src);