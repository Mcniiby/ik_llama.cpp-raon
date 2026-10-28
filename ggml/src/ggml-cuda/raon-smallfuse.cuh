#pragma once

#include "common.cuh"

// Decode graph-pattern fusions. Each returns the index of the last node it consumed, or -1.
// Env switches (default on):
//   RAON_HC_BOUNDARY=0   mHC boundary (HC_POST -> MUL_MAT -> RAON_HC_PRE -> FUSED_RMS_NORM)
//   RAON_HC_COLLAPSE=0   final HC_POST -> stream sum -> SCALE -> FUSED_RMS_NORM
//   RAON_F16_MV=0        small F16 matvecs
//   RAON_MLA_FUSE=0      MLA small ops (also read by build_raon1.cpp)

int ggml_cuda_raon_hc_boundary(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int i);
int ggml_cuda_raon_hc_collapse(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int i);
int ggml_cuda_raon_f16_mv(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int i);
int ggml_cuda_raon_rms_norm_mul(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int i);
int ggml_cuda_raon_sigmoid_mul(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int i);

// fusion hit counters (GGML_CUDA_GRAPH_STATS)
extern int64_t g_raon_fused_hc_boundary;
extern int64_t g_raon_fused_hc_collapse;
extern int64_t g_raon_fused_f16_mv;
extern int64_t g_raon_fused_mla;
