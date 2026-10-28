#pragma once

// mHC pre-mix coefficients for one token, shared by the CPU op and the CUDA kernel (same arithmetic order).
// Mirrors raon1_build_hc_pre in src/graphs/build_raon1.cpp:
//   m'   = m * r                       (r = 1/rms(x) when the rms is folded in, else 1)
//   pre  = sigmoid(m'[0:S]*a0 + b[0:S])
//   post = mag * sigmoid(m'[S:2S]*a1 + b[S:2S])
//   z    = m'[2S:]*a2 + b[2S:]          ((S-1)^2 entries)
//   e    = H z H^T
//   rho  = 1 + max(0, max_j(-(e_j / C_j)))   C = 0.5*I + 0.5/S
//   phi  = e/rho + C
//   comb = (phi - I)*sigmoid(a3) + I    comb[dst + src*S]
// S <= 8.

#include <math.h>

#ifdef __CUDACC__
#define RAON_HC_FN static __device__ __forceinline__
#define RAON_HC_UNROLL _Pragma("unroll")
#else
#define RAON_HC_FN static inline
#define RAON_HC_UNROLL
#endif

RAON_HC_FN float raon_hc_sigmoid(float v) {
    return 1.0f/(1.0f + expf(-v));
}

// Helmert basis H [S, S-1]: orthonormal columns, each orthogonal to the all-ones vector
RAON_HC_FN float raon_hc_helmert(int i, int j) {
    const float v = i <= j ? 1.0f : i == j + 1 ? -(float) (j + 1) : 0.0f;
    return v/sqrtf((float) ((j + 1)*(j + 2)));
}

RAON_HC_FN void raon_hc_coeffs(const int S, const float * m, const float r,
        const float * scale, const float * base, const float mag,
        float * pre, float * post, float * comb) {
    const int nm = S - 1;

    RAON_HC_UNROLL
    for (int i = 0; i < S; ++i) {
        pre[i]  = raon_hc_sigmoid((m[i]*r)*scale[0] + base[i]);
        post[i] = mag*raon_hc_sigmoid((m[S + i]*r)*scale[1] + base[S + i]);
    }

    float z[49];
    RAON_HC_UNROLL
    for (int k = 0; k < nm*nm; ++k) {
        z[k] = (m[2*S + k]*r)*scale[2] + base[2*S + k];
    }

    float e1[56];
    RAON_HC_UNROLL
    for (int n = 0; n < nm; ++n) {
        RAON_HC_UNROLL
        for (int i = 0; i < S; ++i) {
            float sum = 0.0f;
            RAON_HC_UNROLL
            for (int k = 0; k < nm; ++k) {
                sum += raon_hc_helmert(i, k)*z[k + n*nm];
            }
            e1[i + n*S] = sum;
        }
    }

    float e[64];
    float rho = -INFINITY;
    RAON_HC_UNROLL
    for (int i = 0; i < S; ++i) {
        RAON_HC_UNROLL
        for (int p = 0; p < S; ++p) {
            float sum = 0.0f;
            RAON_HC_UNROLL
            for (int k = 0; k < nm; ++k) {
                sum += raon_hc_helmert(p, k)*e1[i + k*S];
            }
            e[p + i*S] = sum;
            const float c = (p == i ? 0.5f : 0.0f) + 0.5f/S;
            const float ratio = -(sum/c);
            rho = ratio > rho ? ratio : rho;
        }
    }
    rho = (rho > 0.0f ? rho : 0.0f) + 1.0f;

    const float gamma = raon_hc_sigmoid(scale[3]);
    RAON_HC_UNROLL
    for (int i = 0; i < S; ++i) {
        RAON_HC_UNROLL
        for (int p = 0; p < S; ++p) {
            const float id  = p == i ? 1.0f : 0.0f;
            const float c   = 0.5f*id + 0.5f/S;
            const float phi = e[p + i*S]/rho + c;
            comb[p + i*S] = (phi - id)*gamma + id;
        }
    }
}
