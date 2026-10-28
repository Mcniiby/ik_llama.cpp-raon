// Raon-1 MoE block: dimensions, block formats, and host/device-portable helpers.
// Builds with nvcc and plain g++, so the CPU check runs the same dot-product code as the GPU kernel.
#pragma once
#include <stdint.h>
#include <string.h>
#include <math.h>

#if defined(__CUDACC__)
#include <cuda_fp16.h>
#define MK_HD __host__ __device__ __forceinline__
#else
#define MK_HD inline
#endif

constexpr int MK_H   = 2560;   // hidden
constexpr int MK_F   = 1088;   // expert inner width
constexpr int MK_E   = 128;    // routed experts
constexpr int MK_K   = 6;      // routed experts per token
constexpr int MK_NX  = MK_K + 1; // routed + shared expert slots

constexpr int QK_K_  = 256;
constexpr int QK8    = 32;

// q8_1: half d, half s (= sum of the float inputs), 32 x int8. Layout-compatible with ik block_q8_1.
struct mk_q8_1 { uint16_t d; uint16_t s; int8_t qs[32]; };
static_assert(sizeof(mk_q8_1) == 36, "q8_1");

// IQ3_KS: per-row half d (row meta, 2 bytes), then QK_K blocks
struct mk_iq3_ks { uint16_t extra; uint8_t scales[4]; uint8_t qs[64]; uint8_t qh[32]; };
static_assert(sizeof(mk_iq3_ks) == 102, "iq3_ks");
constexpr int IQ3KS_ROW   = 2 + (MK_H / QK_K_) * 102;          // 1022 bytes (2560 cols)
constexpr size_t IQ3KS_EXP = (size_t)MK_F * IQ3KS_ROW;          // 1111936 bytes / expert matrix

// IQ4_KS_R16: per 16-row group: 16 x float d, then (ncols/32) blocks of {scales[16], qs[256]}
struct mk_iq4_ks_r16 { uint8_t scales[16]; uint8_t qs[256]; };
static_assert(sizeof(mk_iq4_ks_r16) == 272, "iq4_ks_r16");
constexpr int IQ4R16_GROUP = 16 * 4 + (MK_F / 32) * 272;        // 9312 bytes (1088 cols, 16 rows)
constexpr size_t IQ4R16_EXP = (size_t)(MK_H / 16) * IQ4R16_GROUP; // 1489920 bytes / expert

// Q6_K
struct mk_q6_K { uint8_t ql[128]; uint8_t qh[64]; int8_t scales[16]; uint16_t d; };
static_assert(sizeof(mk_q6_K) == 210, "q6_K");
constexpr int Q6K_ROW = (MK_H / QK_K_) * 210;                    // 2100 bytes
constexpr size_t Q6K_MAT = (size_t)MK_F * Q6K_ROW;

// Q8_0
struct mk_q8_0 { uint16_t d; int8_t qs[32]; };
static_assert(sizeof(mk_q8_0) == 34, "q8_0");
constexpr int Q80_ROW = (MK_F / 32) * 34;                        // 1156 bytes
constexpr size_t Q80_MAT = (size_t)MK_H * Q80_ROW;

constexpr size_t ROUTER_BYTES = (size_t)MK_E * MK_H * 4;

// bytes one decode token touches in one MoE layer (weights only)
constexpr size_t MK_LAYER_BYTES = ROUTER_BYTES + 2 * Q6K_MAT + Q80_MAT + (size_t)MK_K * (2 * IQ3KS_EXP + IQ4R16_EXP);

// tables (from ik ggml-common.h)
#define MK_IQ3NL_INIT { -63, -40, -23, -10, 1, 13, 28,  47, -59, -36, -19,  -6, 5, 17, 32,  51 }
#define MK_IQ4K_INIT  { -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113, \
                        -123, -100, -79, -61, -45, -31, -18,  -6, 5, 17, 29, 42, 57, 73, 93, 117 }
alignas(4) static const int8_t mk_iq3nl_values_h[16] = MK_IQ3NL_INIT;
alignas(4) static const int8_t mk_iq4k_values_h[32]  = MK_IQ4K_INIT;
#if defined(__CUDACC__)
// global memory like ik's GGML_TABLE (constant-bank pointers are not generic)
static __device__ const int8_t mk_iq3nl_values_d[16] __attribute__((aligned(4))) = MK_IQ3NL_INIT;
static __device__ const int8_t mk_iq4k_values_d[32]  __attribute__((aligned(4))) = MK_IQ4K_INIT;
#endif
#if defined(__CUDA_ARCH__)
#define mk_iq3nl_values mk_iq3nl_values_d
#define mk_iq4k_values  mk_iq4k_values_d
#else
#define mk_iq3nl_values mk_iq3nl_values_h
#define mk_iq4k_values  mk_iq4k_values_h
#endif

static inline float mk_h2f_host(uint16_t h) {
    const uint32_t sign = (uint32_t)(h & 0x8000) << 16;
    uint32_t exp = (h >> 10) & 0x1f, man = h & 0x3ff, bits;
    if (exp == 0) {
        const float v = ldexpf((float)man, -24); // zero or subnormal
        return sign ? -v : v;
    } else if (exp == 31) bits = sign | 0x7f800000u | (man << 13);
    else bits = sign | ((exp + 112) << 23) | (man << 13);
    float f; memcpy(&f, &bits, 4); return f;
}
static inline uint16_t mk_f2h_host(float f) { // round to nearest even
    uint32_t x; memcpy(&x, &f, 4);
    const uint32_t sign = (x >> 16) & 0x8000;
    int32_t exp = (int32_t)((x >> 23) & 0xff) - 127 + 15;
    uint32_t man = x & 0x7fffff;
    if (((x >> 23) & 0xff) == 0xff) return (uint16_t)(sign | 0x7c00 | (man ? 0x200 : 0));
    if (exp >= 31) return (uint16_t)(sign | 0x7c00);
    if (exp <= 0) {
        if (exp < -10) return (uint16_t)sign;
        man |= 0x800000;
        const int shift = 14 - exp;
        uint32_t h = man >> shift; const uint32_t rem = man & ((1u << shift) - 1), half = 1u << (shift - 1);
        if (rem > half || (rem == half && (h & 1))) ++h;
        return (uint16_t)(sign | h);
    }
    uint32_t h = ((uint32_t)exp << 10) | (man >> 13);
    const uint32_t rem = man & 0x1fff;
    if (rem > 0x1000 || (rem == 0x1000 && (h & 1))) ++h;
    return (uint16_t)(sign | h);
}

MK_HD float mk_h2f(uint16_t h) {
#if defined(__CUDA_ARCH__)
    return __half2float(__ushort_as_half(h));
#else
    return mk_h2f_host(h);
#endif
}
MK_HD uint16_t mk_f2h(float f) {
#if defined(__CUDA_ARCH__)
    return __half_as_ushort(__float2half(f));
#else
    return mk_f2h_host(f);
#endif
}

MK_HD int mk_dp4a(int a, int b, int c) {
#if defined(__CUDA_ARCH__)
    return __dp4a(a, b, c);
#else
    const int8_t * a8 = (const int8_t *)&a; const int8_t * b8 = (const int8_t *)&b;
    return c + a8[0]*b8[0] + a8[1]*b8[1] + a8[2]*b8[2] + a8[3]*b8[3];
#endif
}
MK_HD uint32_t mk_byte_perm(uint32_t x, uint32_t y, uint32_t s) {
#if defined(__CUDA_ARCH__)
    return __byte_perm(x, y, s);
#else
    uint8_t b[8]; memcpy(b, &x, 4); memcpy(b + 4, &y, 4);
    uint32_t r = 0;
    for (int n = 0; n < 4; ++n) r |= (uint32_t)b[(s >> (4*n)) & 7] << (8*n);
    return r;
#endif
}
MK_HD uint32_t mk_vsub4(uint32_t a, uint32_t b) { // per-byte wrap-around subtract
#if defined(__CUDA_ARCH__)
    return __vsub4(a, b);
#else
    uint32_t r = 0;
    for (int n = 0; n < 4; ++n) r |= (uint32_t)(uint8_t)(((a >> 8*n) & 0xff) - ((b >> 8*n) & 0xff)) << 8*n;
    return r;
#endif
}
MK_HD int mk_vsubss4(int a, int b) { // per-byte signed saturating subtract
#if defined(__CUDA_ARCH__)
    return __vsubss4(a, b);
#else
    uint32_t r = 0;
    for (int n = 0; n < 4; ++n) {
        int v = (int)(int8_t)((a >> 8*n) & 0xff) - (int)(int8_t)((b >> 8*n) & 0xff);
        v = v < -128 ? -128 : v > 127 ? 127 : v;
        r |= (uint32_t)(uint8_t)v << 8*n;
    }
    return (int)r;
#endif
}
MK_HD int mk_get_int_b2(const void * x, int i32) { // 2-byte aligned int load
    const uint16_t * x16 = (const uint16_t *)x;
    return (int)(x16[2*i32 + 0] | ((uint32_t)x16[2*i32 + 1] << 16));
}
MK_HD int mk_get_int_b4(const void * x, int i32) {
    return ((const int *)x)[i32];
}

// IEEE float division regardless of --prec-div / -use_fast_math (ik builds CUDA with -use_fast_math)
#if defined(__CUDA_ARCH__)
#define MK_DIV(a, b) __fdiv_rn((a), (b))
#else
#define MK_DIV(a, b) ((a) / (b))
#endif

// SiTU (ik common.cuh ggml_cuda_situ)
MK_HD float mk_situ(float g, float u, float beta, float lb) {
    const float a = beta*tanhf(g*MK_DIV(1.0f, beta)) * MK_DIV(1.0f, 1.0f + expf(-g));
    return a * (lb > 0.0f ? lb*tanhf(u*MK_DIV(1.0f, lb)) : u);
}

