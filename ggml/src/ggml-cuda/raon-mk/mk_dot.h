// Dot-product device functions copied from ik_llama.cpp ggml-cuda (iqk_mmvq, mmvq, vecdotq, quantize).
// Intrinsics go through mk_* shims so the code also builds on the host; half types are uint16_t;
// vec_dot_iq4_ks_r16 takes the 4-row sub-group `ir` explicitly instead of deriving it from blockIdx.x.
// Copyright (C) 2023-2024 The ggml authors, (C) 2024 Iwan Kawrakow. MIT license.
#pragma once
#include "mk_common.h"

struct mk_int2 { int x, y; };

MK_HD mk_int2 mk_get_int_from_table_16(const int & q4, const int8_t * values) {
    uint32_t v1, v2, v3, v4, mask;
    const uint32_t * values32 = (const uint32_t *)values;
    mask = (0x32103210 | (((uint32_t)q4 & 0x88888888) >> 1));
    v1 = mk_byte_perm(values32[0], values32[1], q4);
    v2 = mk_byte_perm(values32[2], values32[3], q4);
    v3 = mk_byte_perm(v1, v2, mask);
    v1 = mk_byte_perm(values32[0], values32[1], (uint32_t)q4 >> 16);
    v2 = mk_byte_perm(values32[2], values32[3], (uint32_t)q4 >> 16);
    v4 = mk_byte_perm(v1, v2, mask >> 16);
    mk_int2 r; r.x = (int)mk_byte_perm(v3, v4, 0x6420); r.y = (int)mk_byte_perm(v3, v4, 0x7531);
    return r;
}

// vbq: start of the IQ3_KS row (row-meta half d first). bq8_1: q8_1 block aligned with kbx (= &y[8*kbx]).
// iiqs = 4*(lane%8) as in ik's mmvq (VDR 4, qi 32): one call covers 32 of the 256 quants.
MK_HD void mk_vec_dot_iq3_ks_q8_1(const void * __restrict__ vbq, const mk_q8_1 * __restrict__ bq8_1,
        const int kbx, const int iiqs, float * result) {
    float d = mk_h2f(*(const uint16_t *)vbq);
    const mk_iq3_ks * bq3 = (const mk_iq3_ks *)((const char *)vbq + 2) + kbx;

    int iqs = iiqs/4;
    const int ib128 = iqs/4;
    const int il8   = iqs%4;

    const uint16_t * ql = (const uint16_t *)bq3->qs + 16*ib128 + 4*il8;
    const uint16_t * qh = (const uint16_t *)bq3->qh + 4*il8;

    uint16_t extra = bq3->extra >> 4*ib128;
    uint32_t extra_v = uint32_t(extra >> 8) * 0x01010101;

    uint32_t extra32_1 = ((extra_v << 3) & 0x08080808) | ((extra_v << 5) & 0x80808080);
    uint32_t extra32_2 = ((extra_v << 2) & 0x08080808) | ((extra_v << 4) & 0x80808080);

    const int * q8;
    int sumi[4] = {0, 0, 0, 0};
    for (int i = 0; i < 2; ++i) {
        uint32_t vl = ql[2*i+0] | ((uint32_t)ql[2*i+1] << 16);
        uint32_t vh = ((qh[2*i+0] | ((uint32_t)qh[2*i+1] << 16)) >> 4*ib128);

        uint32_t val1 = ((vl >> 0) & 0x33333333) | extra32_1 | ((vh << 2) & 0x04040404) | ((vh << 4) & 0x40404040);
        uint32_t val2 = ((vl >> 2) & 0x33333333) | extra32_2 | ((vh << 1) & 0x04040404) | ((vh << 3) & 0x40404040);
        mk_int2 v1 = mk_get_int_from_table_16((int)val1, mk_iq3nl_values);
        mk_int2 v2 = mk_get_int_from_table_16((int)val2, mk_iq3nl_values);

        q8 = (const int *)bq8_1[4*ib128+0].qs + 2*il8;
        sumi[0] = mk_dp4a(v1.x, q8[i], sumi[0]);
        q8 += sizeof(mk_q8_1)/4;
        sumi[1] = mk_dp4a(v2.x, q8[i], sumi[1]);
        q8 += sizeof(mk_q8_1)/4;
        sumi[2] = mk_dp4a(v1.y, q8[i], sumi[2]);
        q8 += sizeof(mk_q8_1)/4;
        sumi[3] = mk_dp4a(v2.y, q8[i], sumi[3]);
    }
    const uint16_t * sl16 = (const uint16_t *)bq3->scales;
    int32_t aux32 = (int32_t)mk_vsub4(((sl16[0] | ((uint32_t)sl16[1] << 16)) >> 4*ib128) & 0x0f0f0f0f, 0x10101010);
    const int8_t * a8 = (const int8_t *)&aux32;
    *result += d * (mk_h2f(bq8_1[4*ib128+0].d) * (a8[0] + ((extra << 4) & 0x10)) * sumi[0] +
                    mk_h2f(bq8_1[4*ib128+1].d) * (a8[1] + ((extra << 3) & 0x10)) * sumi[1] +
                    mk_h2f(bq8_1[4*ib128+2].d) * (a8[2] + ((extra << 2) & 0x10)) * sumi[2] +
                    mk_h2f(bq8_1[4*ib128+3].d) * (a8[3] + ((extra << 1) & 0x10)) * sumi[3]);
}

// vbq: start of a 16-row IQ4_KS_R16 group (16 float d first). ir in {0,4,8,12}: which 4 rows.
// iqs = 8*il, il in {0,1}: which half of the 32 columns. Adds into result[0..3] (rows ir..ir+3).
MK_HD void mk_vec_dot_iq4_ks_r16_q8_1(const void * __restrict__ vbq, const mk_q8_1 * __restrict__ bq8_1,
        const int kbx, const int iqs, const int ir, float * result) {
    const float * dptr = (const float *)vbq;
    const mk_iq4_ks_r16 * bq4 = (const mk_iq4_ks_r16 *)(dptr + 16) + kbx;

    const int il = iqs / 8;

    const float d8x = mk_h2f(bq8_1->d), d8y = mk_h2f(bq8_1->s);
    const int32_t * q8 = (const int *)bq8_1->qs + 2*il;

    const uint32_t * sas = (const uint32_t *)(bq4->scales + ir);
    uint32_t shifts[1];
    int32_t  scales[1];
    shifts[0] = (sas[0] & 0x01010101) << 1;
    scales[0] = (int32_t)mk_vsub4(sas[0] & 0xfefefefe, 0x7f7f7f7f);
    const uint8_t * shifts8 = (const uint8_t *)shifts;
    const int8_t  * scales8 = (const  int8_t *)scales;
    int32_t isum[4];
#if defined(__CUDA_ARCH__)
    #pragma unroll
#endif
    for (int j = 0; j < 4; ++j) {
        const int * q4 = (const int *)bq4->qs + 32*il + ir + j;
        int sumi = 0;
        mk_int2 val = mk_get_int_from_table_16(q4[0], mk_iq4k_values);
        sumi = mk_dp4a(val.x, q8[0], mk_dp4a(val.y, q8[4], sumi));
        val = mk_get_int_from_table_16(q4[16], mk_iq4k_values);
        sumi = mk_dp4a(val.x, q8[1], mk_dp4a(val.y, q8[5], sumi));
        isum[j] = sumi * scales8[j];
    }
#if defined(__CUDA_ARCH__)
    #pragma unroll
#endif
    for (int j = 0; j < 4; ++j) result[j] += dptr[ir+j] * (d8x * isum[j] + d8y * shifts8[j] * scales8[j]);
}

// Q6_K: vbq = row start, kbx = super-block within the row, iqs = lane (qi 32, vdr 1).
MK_HD float mk_vec_dot_q6_K_q8_1(const void * __restrict__ vbq, const mk_q8_1 * __restrict__ bq8_1,
        const int kbx, const int iqs) {
    constexpr int MK_QR6_K = 2, MK_QI6_K = 32, MK_QI8_1 = 8;
    const mk_q6_K * bq6_K = (const mk_q6_K *)vbq + kbx;

    const int bq8_offset = 2 * MK_QR6_K * (iqs / (MK_QI6_K/2)) + (iqs % (MK_QI6_K/2)) / (MK_QI6_K/4);
    const int scale_offset = (MK_QI6_K/4) * (iqs / (MK_QI6_K/2)) + (iqs % (MK_QI6_K/2)) / (MK_QI6_K/8);
    const int vh_shift = 2 * ((iqs % (MK_QI6_K/2)) / (MK_QI6_K/4));

    const int vl = mk_get_int_b2(bq6_K->ql, iqs);
    const int vh = mk_get_int_b2(bq6_K->qh, (MK_QI6_K/4) * (iqs / (MK_QI6_K/2)) + iqs % (MK_QI6_K/4)) >> vh_shift;

    const int8_t * scales = bq6_K->scales + scale_offset;

    float sumf = 0.0f;
    for (int i = 0; i < MK_QR6_K; ++i) {
        const int u   = mk_get_int_b4(bq8_1[bq8_offset + 2*i].qs, iqs % MK_QI8_1);
        const float d8 = mk_h2f(bq8_1[bq8_offset + 2*i].d);
        const int sc = scales[4*i];
        const int vil = (vl >> (4*i)) & 0x0F0F0F0F;
        const int vih = ((vh >> (4*i)) << 4) & 0x30303030;
        const int vi = mk_vsubss4((vil | vih), 0x20202020);
        sumf += d8 * (mk_dp4a(vi, u, 0) * sc);
    }
    return mk_h2f(bq6_K->d) * sumf;
}

// Q8_0: vbq = row start, kbx = 32-block within the row, iqs = 2*(lane%4) (qi 8, vdr 2).
MK_HD float mk_vec_dot_q8_0_q8_1(const void * __restrict__ vbq, const mk_q8_1 * __restrict__ bq8_1,
        const int kbx, const int iqs) {
    const mk_q8_0 * bq8_0 = (const mk_q8_0 *)vbq + kbx;
    int sumi = 0;
    for (int i = 0; i < 2; ++i) {
        const int v = mk_get_int_b2(bq8_0->qs, iqs + i);
        const int u = mk_get_int_b4(bq8_1->qs, iqs + i);
        sumi = mk_dp4a(v, u, sumi);
    }
    return mk_h2f(bq8_0->d) * mk_h2f(bq8_1->d) * (float)sumi;
}

// Host q8_1 quantization of n values (n % 32 == 0), same math as ik quantize_q8_1.
static inline void mk_quantize_q8_1_host(const float * x, mk_q8_1 * y, int n) {
    for (int ib = 0; ib < n/32; ++ib) {
        float amax = 0.0f, sum = 0.0f;
        for (int j = 0; j < 32; ++j) { amax = fmaxf(amax, fabsf(x[32*ib+j])); sum += x[32*ib+j]; }
        const float d = amax / 127;
        for (int j = 0; j < 32; ++j) y[ib].qs[j] = amax == 0.0f ? 0 : (int8_t)roundf(x[32*ib+j] / d);
        y[ib].d = mk_f2h_host(d);
        y[ib].s = mk_f2h_host(sum);
    }
}
