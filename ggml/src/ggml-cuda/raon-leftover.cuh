#pragma once

#include "common.cuh"

// RAON_FB_FOLD (default on): the one-token RDA ssm_f_b matvec is computed per head by RDA_DECODE's prep kernel.
struct raon_fb_pending {
    const half  * w   = nullptr;   // f_b weight, row stride s1 halves
    const float * x   = nullptr;   // f_a output, 128 floats
    float       * dst = nullptr;   // the MUL_MAT output RDA reads as f
    int64_t       s1  = 0;
    int           nrows = 0;
};
extern raon_fb_pending g_raon_fb_pending;
bool ggml_cuda_raon_fb_fold_enabled();
// launches the deferred f_b matvec (used when RDA_DECODE cannot consume it)
void ggml_cuda_raon_fb_launch(const raon_fb_pending & fb, cudaStream_t stream);
