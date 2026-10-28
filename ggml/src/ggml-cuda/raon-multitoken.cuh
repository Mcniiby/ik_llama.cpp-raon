#pragma once

#include "common.cuh"

// Multi-token (2-8) fusions. ADD whose result feeds a fusable mHC boundary: one launch for all five nodes.
// Returns the last consumed node index or -1. RAON_HC_BOUNDARY_ADD=0: off.
int ggml_cuda_raon_hc_boundary_add(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int ia);
