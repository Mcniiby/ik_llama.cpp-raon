//
// Grouped small-batch MoE matvec (2..8 tokens): each expert's weights are decoded once per batch and applied to
// every token routed to that expert. See moe-grouped.cu.
// MIT license
// SPDX-License-Identifier: MIT
//
#pragma once

#include "common.cuh"

struct moeg_params {
    const char * vx_u;      // expert weights (up, or the only matrix when !fused)
    const char * vx_g;      // gate weights (fused only)
    const char * vy;        // q8_1 activations
    float      * dst;
    const char * ids;       // int32 expert ids, [n_ids, n_tok]
    const float * bias;     // optional per-expert bias (non-fused only), row e at bias + e*bias_nb1 bytes
    int64_t ncols_x;        // row length of the weights (= activation length)
    int64_t nrows_x;        // output rows
    uint64_t nb02;          // bytes between experts
    uint64_t ids_nb0, ids_nb1;
    uint64_t y_nb_tok, y_nb_k;      // q8_1 row of slot (t, k) at vy + t*y_nb_tok + k*y_nb_k
    uint64_t dst_nb_tok, dst_nb_k;  // float row of slot (t, k) at dst + t*dst_nb_tok + k*dst_nb_k (bytes)
    uint64_t bias_nb1;
    int n_tok;
    int n_ids;
    ggml_unary_op unary_op; // fused only
    float limit, limit2;    // raw op params (mapped like the mmvq path inside)
};

// GGML_CUDA_MOE_GROUPED=0 turns the grouped path off (read once)
bool ggml_cuda_moe_grouped_enabled();

// whether the grouped kernel takes this case
bool ggml_cuda_moe_grouped_supported(ggml_type type, bool fused, int64_t ncols_x, int64_t nrows_x, int n_tok, int n_ids);

// one launch, no host sync (CUDA-graph safe)
void ggml_cuda_moe_grouped_mmvq(ggml_type type, bool fused, const moeg_params & p, cudaStream_t stream);
