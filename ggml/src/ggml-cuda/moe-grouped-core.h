//
// Grouped small-batch MoE matvec: shared host/device core.
//
// Copyright (C) 2024 Iwan Kawrakow (the IQ3_KS / IQ4_KS_R16 dot-product math is a decode/apply split of
//   vec_dot_iq3_ks_q8_1 and vec_dot_iq4_ks_r16_q8_1 in template-instances/mmvq-instance-*.cu)
// Grouping after Strata's native_expert_grouped / EXPDEC kernels (MIT).
// MIT license
// SPDX-License-Identifier: MIT
//
// Plain C++ that also compiles for the device, so a CPU check runs exactly the code the kernels run.
//
#pragma once

#include <stdint.h>
#include <math.h>

#if defined(__CUDACC__)
#define MOEG_HD __host__ __device__ __forceinline__
#else
#define MOEG_HD inline
#endif

// at most this many (token, expert) slots per launch: 8 tokens x 16 experts
#define MOEG_MAX_SLOTS 128

// byte offsets inside the quant blocks (checked with static_asserts in moe-grouped.cu)
#define MOEG_Q8_1_BYTES     36   // half d, half s, int8 qs[32]
#define MOEG_Q8_1_WORDS      9
#define MOEG_IQ3KS_BYTES   102   // uint16 extra, uint8 scales[4], uint8 qs[64], uint8 qh[32]
#define MOEG_IQ3KS_SCALES    2
#define MOEG_IQ3KS_QS        6
#define MOEG_IQ3KS_QH       70
#define MOEG_IQ4KSR16_BYTES 272  // uint8 scales[16], uint8 qs[256]

MOEG_HD float moeg_h2f(uint16_t h) {
#if defined(__CUDA_ARCH__)
    return __half2float(__ushort_as_half(h));
#else
    const uint32_t sign = (uint32_t)(h & 0x8000) << 16;
    uint32_t exp  = (h >> 10) & 0x1f;
    uint32_t mant = h & 0x3ff;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            exp = 127 - 15 + 1;
            while (!(mant & 0x400)) { mant <<= 1; --exp; }
            mant &= 0x3ff;
            bits = sign | (exp << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7f800000 | (mant << 13);
    } else {
        bits = sign | ((exp + 127 - 15) << 23) | (mant << 13);
    }
    union { uint32_t u; float f; } c; c.u = bits;
    return c.f;
#endif
}

MOEG_HD int moeg_dp4a(int a, int b, int c) {
#if defined(__CUDA_ARCH__)
    return __dp4a(a, b, c);
#else
    const int8_t * pa = (const int8_t *)&a;
    const int8_t * pb = (const int8_t *)&b;
    return c + pa[0]*pb[0] + pa[1]*pb[1] + pa[2]*pb[2] + pa[3]*pb[3];
#endif
}

// 16-entry int8 codebook lookup of the 8 nibbles of q4: low nibbles -> x, high nibbles -> y.
// values must be 4-byte aligned (the kernels stage the codebook in shared memory).
MOEG_HD void moeg_table16(uint32_t q4, const int8_t * values, int & x, int & y) {
#if defined(__CUDA_ARCH__)
    // same __byte_perm lookup as get_int_from_table_16 (iqk_mmvq_templates.cuh)
    const uint32_t * values32 = (const uint32_t *)values;
    uint32_t mask = (0x32103210 | ((q4 & 0x88888888) >> 1));
    uint32_t v1 = __byte_perm(values32[0], values32[1], q4);
    uint32_t v2 = __byte_perm(values32[2], values32[3], q4);
    uint32_t v3 = __byte_perm(v1, v2, mask);
    v1 = __byte_perm(values32[0], values32[1], q4 >> 16);
    v2 = __byte_perm(values32[2], values32[3], q4 >> 16);
    uint32_t v4 = __byte_perm(v1, v2, mask >> 16);
    x = (int)__byte_perm(v3, v4, 0x6420);
    y = (int)__byte_perm(v3, v4, 0x7531);
#else
    uint32_t a = 0, b = 0;
    for (int i = 0; i < 4; ++i) {
        a |= (uint32_t)(uint8_t)values[(q4 >> (8*i + 0)) & 0xf] << 8*i;
        b |= (uint32_t)(uint8_t)values[(q4 >> (8*i + 4)) & 0xf] << 8*i;
    }
    x = (int)a; y = (int)b;
#endif
}

// Slot grouping. Slot s = t*n_ids + k is token t's k-th expert. The block of slot s is the group's leader if no
// earlier slot picked the same expert; it then lists every slot (in increasing order, itself first) that did.
// Returns the number of listed slots, 0 if this block has nothing to do.
MOEG_HD int moeg_group(const int * ids, int n_slots, int slot, int * list) {
    const int e = ids[slot];
    if (e < 0) return 0;
    for (int s = 0; s < slot; ++s) if (ids[s] == e) return 0;
    int n = 0;
    for (int s = slot; s < n_slots; ++s) if (ids[s] == e) list[n++] = s;
    return n;
}

// ---------------------------------------------------------------------------------------------------------------
// IQ3_KS: one lane does block kbx, sub-index kqs = 4*(lane%8), as vec_dot_iq3_ks_q8_1 with vdr 4.
struct moeg_iq3ks_slice {
    int   v[8];   // v[4*i + k]: the 4 int8 weights the lane dots with q8 block (4*ib128 + k), word i
    int   sc[4];  // block scales (signed, incl. the extra bit)
    float d;      // row scale
};

MOEG_HD void moeg_iq3ks_decode(const uint8_t * row, int kbx, int kqs, const int8_t * tab, moeg_iq3ks_slice & w) {
    w.d = moeg_h2f(*(const uint16_t *)row);
    const uint8_t * b = row + 2 + kbx*MOEG_IQ3KS_BYTES;

    const int iqs   = kqs/4;
    const int ib128 = iqs/4;
    const int il8   = iqs%4;

    const uint16_t * ql = (const uint16_t *)(b + MOEG_IQ3KS_QS) + 16*ib128 + 4*il8;
    const uint16_t * qh = (const uint16_t *)(b + MOEG_IQ3KS_QH) + 4*il8;

    const uint16_t extra = *(const uint16_t *)b >> 4*ib128;
    const uint32_t extra_v = uint32_t(extra >> 8) * 0x01010101;

    const uint32_t extra32_1 = ((extra_v << 3) & 0x08080808) | ((extra_v << 5) & 0x80808080);
    const uint32_t extra32_2 = ((extra_v << 2) & 0x08080808) | ((extra_v << 4) & 0x80808080);

    for (int i = 0; i < 2; ++i) {
        const uint32_t vl = ql[2*i+0] | ((uint32_t)ql[2*i+1] << 16);
        const uint32_t vh = ((qh[2*i+0] | ((uint32_t)qh[2*i+1] << 16)) >> 4*ib128);
        const uint32_t val1 = ((vl >> 0) & 0x33333333) | extra32_1 | ((vh << 2) & 0x04040404) | ((vh << 4) & 0x40404040);
        const uint32_t val2 = ((vl >> 2) & 0x33333333) | extra32_2 | ((vh << 1) & 0x04040404) | ((vh << 3) & 0x40404040);
        moeg_table16(val1, tab, w.v[4*i+0], w.v[4*i+2]);
        moeg_table16(val2, tab, w.v[4*i+1], w.v[4*i+3]);
    }
    const uint16_t * sl16 = (const uint16_t *)(b + MOEG_IQ3KS_SCALES);
    const uint32_t s32 = ((sl16[0] | ((uint32_t)sl16[1] << 16)) >> 4*ib128) & 0x0f0f0f0f;
    for (int k = 0; k < 4; ++k) {
        w.sc[k] = (int)((s32 >> 8*k) & 0xff) - 16 + ((extra << (4 - k)) & 0x10);
    }
}

// yb: the q8_1 block aligned with kbx (i.e. block 8*kbx of the activation row), 4-byte aligned
MOEG_HD float moeg_iq3ks_apply(const moeg_iq3ks_slice & w, const uint8_t * yb, int kqs) {
    const int iqs   = kqs/4;
    const int ib128 = iqs/4;
    const int il8   = iqs%4;
    int sumi[4] = {0, 0, 0, 0};
    float ds[4];
    for (int k = 0; k < 4; ++k) {
        const uint8_t * q8b = yb + (4*ib128 + k)*MOEG_Q8_1_BYTES;
        const int * q8 = (const int *)(q8b + 4) + 2*il8;
        sumi[k] = moeg_dp4a(w.v[k],   q8[0], sumi[k]);
        sumi[k] = moeg_dp4a(w.v[4+k], q8[1], sumi[k]);
        ds[k] = moeg_h2f(*(const uint16_t *)q8b);
    }
    return w.d * (ds[0] * w.sc[0] * sumi[0] + ds[1] * w.sc[1] * sumi[1] +
                  ds[2] * w.sc[2] * sumi[2] + ds[3] * w.sc[3] * sumi[3]);
}

// IQ4_KS_R16: one lane does column block kbx, half il (0/1), rows ir..ir+3 of the 16-row group,
// as vec_dot_iq4_ks_r16_q8_1.
struct moeg_iq4ksr16_slice {
    int   v[4][4];  // per row: lookup(q4[0]).x/.y, lookup(q4[16]).x/.y
    int   sc[4];
    int   sh[4];
    float d[4];
};

MOEG_HD void moeg_iq4ksr16_decode(const uint8_t * grp, int kbx, int il, int ir, const int8_t * tab, moeg_iq4ksr16_slice & w) {
    const float * dptr = (const float *)grp;
    const uint8_t * b = grp + 16*sizeof(float) + kbx*MOEG_IQ4KSR16_BYTES;
    const uint32_t sas = *(const uint32_t *)(b + ir);
    const int * qs = (const int *)(b + 16) + 32*il + ir;
    for (int j = 0; j < 4; ++j) {
        const uint32_t sb = (sas >> 8*j) & 0xff;
        w.sh[j] = (int)((sb & 1) << 1);
        w.sc[j] = (int)(sb & 0xfe) - 127;
        w.d[j]  = dptr[ir + j];
        moeg_table16((uint32_t)qs[j],      tab, w.v[j][0], w.v[j][1]);
        moeg_table16((uint32_t)qs[j + 16], tab, w.v[j][2], w.v[j][3]);
    }
}

// yb: q8_1 block kbx of the activation row; adds the lane's partial sums of 4 rows into res
MOEG_HD void moeg_iq4ksr16_apply(const moeg_iq4ksr16_slice & w, const uint8_t * yb, int il, float * res) {
    const float d8x = moeg_h2f(((const uint16_t *)yb)[0]);
    const float d8y = moeg_h2f(((const uint16_t *)yb)[1]);
    const int * q8 = (const int *)(yb + 4) + 2*il;
    for (int j = 0; j < 4; ++j) {
        int sumi = moeg_dp4a(w.v[j][0], q8[0], moeg_dp4a(w.v[j][1], q8[4], 0));
        sumi = moeg_dp4a(w.v[j][2], q8[1], moeg_dp4a(w.v[j][3], q8[5], sumi));
        const int isum = sumi * w.sc[j];
        res[j] += w.d[j] * (d8x * isum + d8y * w.sh[j] * w.sc[j]);
    }
}

// Fused up/gate epilogue, same formulas as iqk_fused_mul_mat_vec_q_kernel (the grouped path takes no biases).
// op: the ggml_unary_op enum as int, so this header needs no ggml.h. limit is already mapped (> 1e-6 or INFINITY).
struct moeg_unary { int silu, relu, gelu, situ, swiglu_oai; };

MOEG_HD float moeg_epilogue(int op, const moeg_unary & ops, float u, float g, float limit, float limit2) {
    if (op == ops.swiglu_oai) {
        const float alpha = 1.702f;
        const float lim = 7.0f;
        g = fminf(g, lim);
        u = fmaxf(fminf(u, lim), -lim);
        return g / (1.0f + expf(-g * alpha)) * (1.0f + u);
    }
    if (op == ops.silu) {
        g = g/(1 + expf(-g));
        g = fminf(g, limit);
        return fmaxf(-limit, fminf(limit, u))*g;
    }
    if (op == ops.relu) return fmaxf(g, 0.0f) * u;
    if (op == ops.situ) {
        // ggml_cuda_situ (common.cuh)
        const float a = limit*tanhf(g*(1.0f/limit)) * (1.0f/(1.0f + expf(-g)));
        return a * (limit2 > 0.0f ? limit2*tanhf(u*(1.0f/limit2)) : u);
    }
    // GELU
    const float GELU_COEF_A    = 0.044715f;
    const float SQRT_2_OVER_PI = 0.79788456080286535587989211986876f;
    return 0.5f*g*u*(1.0f + tanhf(SQRT_2_OVER_PI*g*(1.0f + GELU_COEF_A*g*g)));
}
