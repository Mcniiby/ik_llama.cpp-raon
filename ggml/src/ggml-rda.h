#pragma once

// RDA decode step (GGML_OP_RDA_DECODE): scalar math and op params shared by the CPU op and the CUDA kernel,
// matching the unfused graph:
//   conv : y = sum_j w[c*4 + j] * win[j], win = [state(3), x]  (ssm_conv), then SiLU
//   q, k : l2 normed per head, x / max(|x|, eps_l2); q also scaled by 1/sqrt(S)
//   g    : lower_bound * sigmoid(-((f + dt_b) * a))     (log decay per channel)
//   beta : sigmoid(b)                                    (erase gate)
//   vs   : exp(softplus(-b) - softplus(-(b + rda)))      (write/erase ratio, scales V)
//   S    = S * diag(exp(min(g, 50)));  v_new = beta * (v * vs - S k);  S += v_new k^T (clamped +-1e6)
//   o    = S q;  out = rms_norm(o, eps_rms) * norm * sigmoid(z)

#include <math.h>
#include <stdint.h>

#ifdef __CUDACC__
#define GGML_RDA_FN static __device__ __forceinline__
#else
#define GGML_RDA_FN static inline
#endif

// op_params (int32 slots)
#define GGML_RDA_P_NSEQS   0   // number of token groups
#define GGML_RDA_P_RESET   1   // bit g: zero the state of group g before its first token
#define GGML_RDA_P_LB      2   // f32 decay lower bound
#define GGML_RDA_P_EPS_L2  3   // f32
#define GGML_RDA_P_EPS_RMS 4   // f32
#define GGML_RDA_P_SLOTS   5   // int32[n_seqs] state row per group
#define GGML_RDA_MAX_SEQS  8
#define GGML_RDA_MAX_TOKENS 8  // tokens per group the CUDA kernel stages in shared memory
#define GGML_RDA_D_CONV    4
#define GGML_RDA_P_FLAGS  13   // int32 flags (after the GGML_RDA_MAX_SEQS slots)
#define GGML_RDA_F_NO_STATE_WRITE 1 // keep the state slot untouched (spec verify)

GGML_RDA_FN float ggml_rda_sigmoid(float x) {
    return 1.0f/(1.0f + expf(-x));
}

// same threshold as ggml_compute_softplus_f32
GGML_RDA_FN float ggml_rda_softplus(float x) {
    return x > 20.0f ? x : logf(1.0f + expf(x));
}

GGML_RDA_FN float ggml_rda_silu(float x) {
    return x/(1.0f + expf(-x));
}

GGML_RDA_FN float ggml_rda_v_scale(float b, float d) {
    return expf(ggml_rda_softplus(-b) - ggml_rda_softplus(-(b + d)));
}

// decay multiplier exp(min(g, 50)) for one channel
GGML_RDA_FN float ggml_rda_decay(float f, float dt, float a, float lower_bound) {
    const float g = lower_bound*ggml_rda_sigmoid(-((f + dt)*a));
    return expf(fminf(g, 50.0f));
}

GGML_RDA_FN float ggml_rda_clamp_state(float s) {
    return fminf(fmaxf(s, -1e6f), 1e6f);
}
