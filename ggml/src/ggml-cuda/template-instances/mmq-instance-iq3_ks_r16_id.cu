// MoE (mul_mat_id) variant of load_tiles_iq3ks_r16 in mmq.cuh; same band layout and decode.

#include "../mmq_id_common.cuh"

template <int mmq_y, bool need_check> static __device__ __forceinline__ void load_tiles_iq3_ks_r16(
    const char * __restrict__ x, int * __restrict__ x_tile, const int kbx0, const int i_max, const int stride, [[maybe_unused]] int blocks_per_ne00) {

    constexpr int nwarps = mmq_get_nwarps_device();

#ifdef INT8_MMA_AVAILABLE
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + WARP_SIZE*2);
#else
    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_IQ3_KS_R16, mmq_y);
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + txs.qs);
#endif // INT8_MMA_AVAILABLE

    constexpr int qstep = 8;
    const int kqsx = threadIdx.x % qstep;
    const int block_id = (kbx0 + kqsx) % blocks_per_ne00;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * WARP_SIZE/qstep) {
        int i = i0 + threadIdx.y*(WARP_SIZE/qstep) + threadIdx.x/qstep;

        if (need_check) {
            i = min(i, i_max);
        }

        const int ir = i % 16;
        const char * band = x + (i - ir)*stride;
        const float d = ((const float *)band)[ir];
        const block_iq3_ks_r16 * bxi = (const block_iq3_ks_r16 *)(band + 64) + block_id;

        const uint32_t q0 = *(const uint32_t *)(bxi->qs + ir*4);
        const uint32_t q1 = *(const uint32_t *)(bxi->qs + 64 + ir*4);
        const uint32_t h  = *(const uint32_t *)(bxi->qh + ir*4);
        const int page = (bxi->extra >> (16 + ir)) & 1;
        const uint32_t * vtab = (const uint32_t *)iq3nl_values;
        const uint32_t Tl = vtab[2*page + 0];
        const uint32_t Th = vtab[2*page + 1];

#pragma unroll
        for (int m = 0; m < 8; ++m) {
            const uint32_t A = (m < 4 ? ((q0 >> (2*m)) & 0x03030303u) : ((q1 >> (2*(m-4))) & 0x03030303u))
                             | (((h >> m) & 0x01010101u) << 2);
            const uint32_t t = (A | (A >> 4)) & 0x00FF00FFu;
            const uint32_t s = (t | (t >> 8)) & 0x0000FFFFu;
#ifdef INT8_MMA_AVAILABLE
            x_qs[i*MMQ_MMA_TILE_X_K_Q8_0 + 8*kqsx + m] = __byte_perm(Tl, Th, s);
#else
            x_qs[i*(2*WARP_SIZE + 1)     + 8*kqsx + m] = __byte_perm(Tl, Th, s);
#endif // INT8_MMA_AVAILABLE
        }

        const int ul = ((bxi->scales[ir & 7] >> (4*(ir >> 3))) & 0xf) | (((bxi->extra >> ir) & 1) << 4);
#ifdef INT8_MMA_AVAILABLE
        x_df[i*MMQ_MMA_TILE_X_K_Q8_0 + kqsx] = d * (ul - 16);
#else
        x_df[i*(WARP_SIZE/4) + i/4   + kqsx] = d * (ul - 16);
#endif // INT8_MMA_AVAILABLE
    }
}

template <int mmq_x, int mmq_y, bool need_check>
struct mmq_type_traits_id<mmq_x, mmq_y, need_check, GGML_TYPE_IQ3_KS_R16> {
    static constexpr load_tiles_mmq_t load_tiles   = load_tiles_iq3_ks_r16<mmq_y, need_check>;
    static constexpr vec_dot_mmq_t    vec_dot_mma  = vec_dot_q8_0_q8_1_mma<mmq_x, mmq_y, MMQ_Q8_1_DS_LAYOUT_D4>;
    static constexpr vec_dot_mmq_t    vec_dot_dp4a = vec_dot_q8_0_q8_1_dp4a<mmq_x, mmq_y>;
};

DECL_MMQ_CASE(GGML_TYPE_IQ3_KS_R16);
