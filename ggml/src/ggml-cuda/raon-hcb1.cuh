#pragma once

#include "common.cuh"

// mHC sublayer boundary, single-token kernel; tried before ggml_cuda_raon_hc_boundary. Returns the last fused node or -1.
//   RAON_HCB1=0            off
//   RAON_HCB1_VARIANT=n    thread layout, see raon_hcb1_launch (default 14)
//   RAON_HCB1_PREFETCH=1   prefetch the next boundary's hc_fn into L2
// RAON_HC_BOUNDARY=0 disables both.
int ggml_cuda_raon_hcb1(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int i);
