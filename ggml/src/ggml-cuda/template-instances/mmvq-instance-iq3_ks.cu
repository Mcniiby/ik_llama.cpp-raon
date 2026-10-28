#include "../iqk_mmvq_templates.cuh"

__device__ __forceinline__ void vec_dot_iq3_ks_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iiqs, float * result) {

    float d = __half2float(*(const half *)vbq);
    const block_iq3_ks * bq3 = (const block_iq3_ks *)((const char *)vbq + sizeof(half)) + kbx;

    int iqs = iiqs/4;
    const int ib128 = iqs/4;  // 0 or 1. 0 works on quants 0...127, 1 on quants 128...255
                              // Each thread processes 8 quants in each of the 4 32-blocks
    const int il8   = iqs%4;  // 0...3. 0 works on quants 0...7, 1 on quants 8...15, 2 on 16...23, 3 on 24...31

    const uint16_t * ql = (const uint16_t *)bq3->qs + 16*ib128 + 4*il8;
    const uint16_t * qh = (const uint16_t *)bq3->qh + 4*il8;

    uint16_t extra = bq3->extra >> 4*ib128;
    uint32_t extra_v = uint32_t(extra >> 8) * 0x01010101;

    uint32_t extra32_1 = ((extra_v << 3) & 0x08080808) | ((extra_v << 5) & 0x80808080);
    uint32_t extra32_2 = ((extra_v << 2) & 0x08080808) | ((extra_v << 4) & 0x80808080);

    const int * q8;
    int sumi[4] = {0, 0, 0, 0};
    for (int i = 0; i < 2; ++i) {
        uint32_t vl = ql[2*i+0] | (ql[2*i+1] << 16);
        uint32_t vh = ((qh[2*i+0] | (qh[2*i+1] << 16)) >> 4*ib128);

        uint32_t val1 = ((vl >> 0) & 0x33333333) | extra32_1 | ((vh << 2) & 0x04040404) | ((vh << 4) & 0x40404040);
        uint32_t val2 = ((vl >> 2) & 0x33333333) | extra32_2 | ((vh << 1) & 0x04040404) | ((vh << 3) & 0x40404040);
        int2 v1 = get_int_from_table_16(val1, iq3nl_values);
        int2 v2 = get_int_from_table_16(val2, iq3nl_values);

        q8 = (const int *)bq8_1[4*ib128+0].qs + 2*il8;
        sumi[0] = ggml_cuda_dp4a(v1.x, q8[i], sumi[0]);

        q8 += sizeof(block_q8_1)/4;
        sumi[1] = ggml_cuda_dp4a(v2.x, q8[i], sumi[1]);

        q8 += sizeof(block_q8_1)/4;
        sumi[2] = ggml_cuda_dp4a(v1.y, q8[i], sumi[2]);

        q8 += sizeof(block_q8_1)/4;
        sumi[3] = ggml_cuda_dp4a(v2.y, q8[i], sumi[3]);
    }
    const uint16_t * sl16 = (const uint16_t *)bq3->scales;
    int32_t aux32 = __vsub4(((sl16[0] | (sl16[1] << 16)) >> 4*ib128) & 0x0f0f0f0f, 0x10101010);
    const int8_t * a8 = (const int8_t *)&aux32;
    *result += d * (__low2float(bq8_1[4*ib128+0].ds) * (a8[0] + ((extra << 4) & 0x10)) * sumi[0] +
                    __low2float(bq8_1[4*ib128+1].ds) * (a8[1] + ((extra << 3) & 0x10)) * sumi[1] +
                    __low2float(bq8_1[4*ib128+2].ds) * (a8[2] + ((extra << 2) & 0x10)) * sumi[2] +
                    __low2float(bq8_1[4*ib128+3].ds) * (a8[3] + ((extra << 1) & 0x10)) * sumi[3]);

}

// RAON_IQ3KS_MV (default on): one-token, one-matrix IQ3_KS matvec with lane groups. The stock layout gives a
// 2560-wide row four warps (48 of 128 threads idle, one row in flight per CTA); here LPR lanes own a row (shuffle
// reduction only) and a CTA holds 4*32/LPR*RPS rows. Same vec_dot; only the float summation order changes.
template <int LPR, int RPS, bool fused>
__launch_bounds__(128, 1)
static __global__ void k_iq3_ks_mv_w(const char * __restrict__ vx_u, const char * __restrict__ vx_g, const void * __restrict__ vy,
        float * __restrict__ dst, const int ncols_x, const int nrows, const int64_t row_size,
        const ggml_unary_op unary_op, const float limit, const float limit2) {
    constexpr int qk  = ggml_cuda_type_traits<GGML_TYPE_IQ3_KS>::qk;
    constexpr int qi  = ggml_cuda_type_traits<GGML_TYPE_IQ3_KS>::qi;
    constexpr int vdr = VDR_IQ3_K_Q8_1_MMVQ;
    constexpr int tpb = qi/vdr;
    static_assert(LPR >= tpb && LPR % tpb == 0 && LPR <= WARP_SIZE, "bad lanes per row");
    constexpr int bpi = LPR/tpb;
    constexpr int R   = 4*(WARP_SIZE/LPR)*RPS;

    const int l     = threadIdx.x % LPR;
    const int rbase = blockIdx.x*R + (threadIdx.y*(WARP_SIZE/LPR) + threadIdx.x/LPR)*RPS;
    const int bpr   = ncols_x/qk;
    const block_q8_1 * y = (const block_q8_1 *) vy;

    float tu[RPS] = {0.0f}, tg[RPS] = {0.0f};
    const int kqs = vdr*(l % tpb);
    for (int kbx = l/tpb; kbx < bpr; kbx += bpi) {
        const block_q8_1 * yb = y + kbx*(qk/QK8_1);
#pragma unroll
        for (int r = 0; r < RPS; ++r) {
            const int64_t row = min(rbase + r, nrows - 1);
            vec_dot_iq3_ks_q8_1(vx_u + row*row_size, yb, kbx, kqs, &tu[r]);
            if constexpr (fused) vec_dot_iq3_ks_q8_1(vx_g + row*row_size, yb, kbx, kqs, &tg[r]);
        }
    }
#pragma unroll
    for (int r = 0; r < RPS; ++r) {
#pragma unroll
        for (int off = LPR/2; off > 0; off >>= 1) {
            tu[r] += __shfl_xor_sync(0xFFFFFFFF, tu[r], off, WARP_SIZE);
            if constexpr (fused) tg[r] += __shfl_xor_sync(0xFFFFFFFF, tg[r], off, WARP_SIZE);
        }
    }
    if (l != 0) return;
#pragma unroll
    for (int r = 0; r < RPS; ++r) {
        if (rbase + r >= nrows) break;
        float res = tu[r];
        if constexpr (fused) {
            float u = tu[r], g = tg[r];
            switch (unary_op) {
                case GGML_UNARY_OP_SILU: {
                    g = g/(1 + expf(-g));
                    g = min(g, limit);
                    res = max(-limit, min(limit, u))*g;
                } break;
                case GGML_UNARY_OP_RELU: res = fmaxf(g, 0.0f) * u; break;
                case GGML_UNARY_OP_SITU: res = ggml_cuda_situ(g, u, limit, limit2); break;
                default: {   // GELU (the launcher only sends SILU/RELU/SITU/GELU)
                    constexpr float GELU_COEF_A    = 0.044715f;
                    constexpr float SQRT_2_OVER_PI = 0.79788456080286535587989211986876f;
                    res = 0.5f*g*u*(1.0f + tanhf(SQRT_2_OVER_PI*g*(1.0f + GELU_COEF_A*g*g)));
                } break;
            }
        }
        dst[rbase + r] = res;
    }
}

template <int LPR, int RPS>
static void launch_iq3_ks_mv_w(const mmvq_args & args, bool fused, cudaStream_t stream) {
    constexpr int R = 4*(WARP_SIZE/LPR)*RPS;
    const dim3 grid((args.nrows_x + R - 1)/R, 1, 1), block(WARP_SIZE, 4, 1);
    const int64_t row_size = ggml_row_size(GGML_TYPE_IQ3_KS, args.ncols_x);
    if (fused) {
        k_iq3_ks_mv_w<LPR, RPS, true><<<grid, block, 0, stream>>>((const char *) args.vx_u, (const char *) args.vx_g, args.vy, args.dst,
            args.ncols_x, args.nrows_x, row_size, args.unary_op, args.limit, args.limit2);
    } else {
        k_iq3_ks_mv_w<LPR, RPS, false><<<grid, block, 0, stream>>>((const char *) args.vx_u, nullptr, args.vy, args.dst,
            args.ncols_x, args.nrows_x, row_size, args.unary_op, args.limit, args.limit2);
    }
}

void mul_mat_vec_iq3_ks_q8_1_cuda(const mmvq_args & args, cudaStream_t stream) {
    // RAON_IQ3KS_MV: 0 = stock layout, 1 = auto (default), 2 = 16 lanes x 1 row, 3 = 16 x 2, 4 = 8 x 1, 5 = 32 x 1, 6 = 32 x 2
    static const int mode = [] { const char * e = getenv("RAON_IQ3KS_MV"); return e ? atoi(e) : 1; }();
    const bool fused = args.vx_u && args.vx_g && args.unary_op != GGML_UNARY_OP_COUNT;
    const bool op_ok = !fused || args.unary_op == GGML_UNARY_OP_SILU || args.unary_op == GGML_UNARY_OP_RELU ||
                       args.unary_op == GGML_UNARY_OP_SITU || args.unary_op == GGML_UNARY_OP_GELU;
    if (mode != 0 && args.ncols_y == 1 && args.ne2 == 1 && !args.ids_data && !args.bias_u && !args.bias_g && op_ok &&
        args.nrows_dst == args.nrows_x && args.ncols_x % QK_K == 0) {
        const int bpr = args.ncols_x/QK_K;
        int m = mode;
        if (m == 1) m = bpr <= 16 ? 2 : 5;   // short rows: 16 lanes (2 blocks per step); long rows: a warp per row
        switch (m) {
            case 3:  launch_iq3_ks_mv_w<16, 2>(args, fused, stream); return;
            case 4:  launch_iq3_ks_mv_w< 8, 1>(args, fused, stream); return;
            case 5:  launch_iq3_ks_mv_w<32, 1>(args, fused, stream); return;
            case 6:  launch_iq3_ks_mv_w<32, 2>(args, fused, stream); return;
            default: launch_iq3_ks_mv_w<16, 1>(args, fused, stream); return;
        }
    }
    iqk_mul_mat_vec_q_cuda<GGML_TYPE_IQ3_KS, VDR_IQ3_K_Q8_1_MMVQ, vec_dot_iq3_ks_q8_1>(args, stream);
}

