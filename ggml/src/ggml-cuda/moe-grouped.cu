//
// Grouped small-batch MoE matvec for 2..8 tokens (Raon-1 mini: IQ3_KS gate/up with fused SiTU, IQ4_KS_R16 down).
//
// One launch for the whole batch: the lowest slot (t*n_ids + k) that picked an expert stages the group's q8_1
// activations in shared memory and applies each decoded weight slice to all of them. No host sync (graph safe).
//
// Copyright (C) 2024 Iwan Kawrakow (dot-product math, see moe-grouped-core.h)
// Grouping / shared-memory staging after Strata native_expert_grouped and EXPDEC (MIT, Strata's docs/EXPERTDEC.md).
// MIT license
// SPDX-License-Identifier: MIT
//

#include "moe-grouped.cuh"
#include "moe-grouped-core.h"

#include <cstddef>
#include <cstdlib>

static_assert(sizeof(block_q8_1) == MOEG_Q8_1_BYTES, "q8_1 layout");
static_assert(sizeof(block_iq3_ks) == MOEG_IQ3KS_BYTES, "iq3_ks layout");
static_assert(offsetof(block_iq3_ks, scales) == MOEG_IQ3KS_SCALES, "iq3_ks layout");
static_assert(offsetof(block_iq3_ks, qs) == MOEG_IQ3KS_QS, "iq3_ks layout");
static_assert(offsetof(block_iq3_ks, qh) == MOEG_IQ3KS_QH, "iq3_ks layout");
static_assert(sizeof(block_iq4_ks_r16) == MOEG_IQ4KSR16_BYTES, "iq4_ks_r16 layout");
static_assert(offsetof(block_iq4_ks_r16, qs) == 16, "iq4_ks_r16 layout");

#define MOEG_MAX_SMEM (48*1024)

struct moeg_kargs {
    const char * vx_u;
    const char * vx_g;
    const char * vy;
    char       * dst;
    const char * ids;
    const char * bias;
    int          blocks_per_row;   // weight blocks per row (IQ3_KS: 256 quants, IQ4_KS_R16: 32)
    int          y_blocks;         // q8_1 blocks per activation row to stage
    int          nrows_x;
    int64_t      row_size;
    uint64_t     nb02, ids_nb0, ids_nb1, y_nb_tok, y_nb_k, dst_nb_tok, dst_nb_k, bias_nb1;
    int          n_tok, n_ids;
    int          unary_op;
    int          tpb;              // row tiles per block (RAON_MOEG_TPB)
    moeg_unary   ops;
    float        limit, limit2;
};

// Shared prologue: ids -> group of this block's slot; returns the group size (0: nothing to do, uniform).
static __device__ __forceinline__ int moeg_prologue(const moeg_kargs & a, int * s_ids, int * s_list, int * s_n,
        int8_t * s_tab, const int8_t * tab, int nthreads) {
    const int tid = threadIdx.y*WARP_SIZE + threadIdx.x;
    const int n_slots = a.n_tok*a.n_ids;
    for (int i = tid; i < n_slots; i += nthreads) {
        const int t = i / a.n_ids, k = i % a.n_ids;
        s_ids[i] = *(const int *)(a.ids + t*a.ids_nb1 + k*a.ids_nb0);
    }
    if (tid < 16) s_tab[tid] = tab[tid];
    __syncthreads();
    if (tid == 0) *s_n = moeg_group(s_ids, n_slots, blockIdx.y, s_list);
    __syncthreads();
    return *s_n;
}

// copy nc activation rows (slots s_list[c0..c0+nc)) into shared memory, y_blocks q8_1 blocks each
static __device__ __forceinline__ void moeg_stage(const moeg_kargs & a, const int * s_list, int c0, int nc,
        int * s_y, int nthreads) {
    const int tid = threadIdx.y*WARP_SIZE + threadIdx.x;
    const int wpc = a.y_blocks*MOEG_Q8_1_WORDS;
    for (int i = tid; i < nc*wpc; i += nthreads) {
        const int c = i / wpc, w = i - c*wpc;
        const int s = s_list[c0 + c];
        const int t = s / a.n_ids, k = s - t*a.n_ids;
        s_y[i] = ((const int *)(a.vy + t*a.y_nb_tok + k*a.y_nb_k))[w];
    }
}

static __device__ __forceinline__ float * moeg_dst_row(const moeg_kargs & a, int s) {
    const int t = s / a.n_ids, k = s - t*a.n_ids;
    return (float *)(a.dst + t*a.dst_nb_tok + k*a.dst_nb_k);
}

// IQ3_KS: one row per warp (up and gate when FUSED), 8 lanes per 256-quant block.
template <int NCOL, int NW, bool FUSED>
static __global__ void __launch_bounds__(NW*WARP_SIZE) k_moeg_iq3ks(const moeg_kargs a) {
    extern __shared__ int s_y[];
    __shared__ int s_ids[MOEG_MAX_SLOTS];
    __shared__ int s_list[MOEG_MAX_SLOTS];
    __shared__ int s_n;
    __shared__ __align__(4) int8_t s_tab[16];

    const int n = moeg_prologue(a, s_ids, s_list, &s_n, s_tab, iq3nl_values, NW*WARP_SIZE);
    if (n == 0) return;

    const int e   = s_ids[blockIdx.y];
    const int wpc = a.y_blocks*MOEG_Q8_1_WORDS;
    const int kqs = 4*(threadIdx.x % 8);

    for (int c0 = 0; c0 < n; c0 += NCOL) {
        const int nc = min(NCOL, n - c0);
        if (c0 > 0) __syncthreads();
        moeg_stage(a, s_list, c0, nc, s_y, NW*WARP_SIZE);
        __syncthreads();
      // a.tpb row tiles of NW rows per block share the staged activations
      for (int tile = 0; tile < a.tpb; ++tile) {
        const int row = (blockIdx.x*a.tpb + tile)*NW + threadIdx.y;
        if (row >= a.nrows_x) break;
        const char * xu = a.vx_u + e*a.nb02 + row*a.row_size;
        const char * xg = FUSED ? a.vx_g + e*a.nb02 + row*a.row_size : nullptr;

        float su[NCOL], sg[NCOL];
#pragma unroll
        for (int j = 0; j < NCOL; ++j) su[j] = sg[j] = 0.0f;

        for (int kbx = threadIdx.x/8; kbx < a.blocks_per_row; kbx += WARP_SIZE/8) {
            moeg_iq3ks_slice wu, wg;
            moeg_iq3ks_decode((const uint8_t *)xu, kbx, kqs, s_tab, wu);
            if constexpr (FUSED) moeg_iq3ks_decode((const uint8_t *)xg, kbx, kqs, s_tab, wg);
            const uint8_t * yb = (const uint8_t *)s_y + 8*kbx*MOEG_Q8_1_BYTES;
#pragma unroll
            for (int j = 0; j < NCOL; ++j) {
                if (j < nc) {
                    su[j] += moeg_iq3ks_apply(wu, yb + j*wpc*4, kqs);
                    if constexpr (FUSED) sg[j] += moeg_iq3ks_apply(wg, yb + j*wpc*4, kqs);
                }
            }
        }
#pragma unroll
        for (int j = 0; j < NCOL; ++j) {
            if (j < nc) {
                const float u = warp_reduce_sum(su[j]);
                float r;
                if constexpr (FUSED) {
                    const float g = warp_reduce_sum(sg[j]);
                    r = moeg_epilogue(a.unary_op, a.ops, u, g, a.limit, a.limit2);
                } else {
                    r = a.bias ? u + ((const float *)(a.bias + e*a.bias_nb1))[row] : u;
                }
                if (threadIdx.x == 0) moeg_dst_row(a, s_list[c0 + j])[row] = r;
            }
        }
      }
    }
}

// IQ4_KS_R16: one warp per 4 rows (a quarter of a 16-row group), 2 lanes per 32-column block.
template <int NCOL, int NW>
static __global__ void __launch_bounds__(NW*WARP_SIZE) k_moeg_iq4ksr16(const moeg_kargs a) {
    extern __shared__ int s_y[];
    __shared__ int s_ids[MOEG_MAX_SLOTS];
    __shared__ int s_list[MOEG_MAX_SLOTS];
    __shared__ int s_n;
    __shared__ __align__(4) int8_t s_tab[16];

    const int n = moeg_prologue(a, s_ids, s_list, &s_n, s_tab, iq4k_values, NW*WARP_SIZE);
    if (n == 0) return;

    const int e   = s_ids[blockIdx.y];
    const int wpc = a.y_blocks*MOEG_Q8_1_WORDS;
    const int il = threadIdx.x % 2;

    for (int c0 = 0; c0 < n; c0 += NCOL) {
        const int nc = min(NCOL, n - c0);
        if (c0 > 0) __syncthreads();
        moeg_stage(a, s_list, c0, nc, s_y, NW*WARP_SIZE);
        __syncthreads();
      for (int tile = 0; tile < a.tpb; ++tile) {
        const int g4  = (blockIdx.x*a.tpb + tile)*NW + threadIdx.y;      // 4-row group
        if (4*g4 >= a.nrows_x) break;
        const int ir  = 4*(g4 % 4);
        const char * grp = a.vx_u + e*a.nb02 + (int64_t)(16*(g4/4))*a.row_size;

        float acc[NCOL][4];
#pragma unroll
        for (int j = 0; j < NCOL; ++j) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.0f;

        for (int kbx = threadIdx.x/2; kbx < a.blocks_per_row; kbx += WARP_SIZE/2) {
            moeg_iq4ksr16_slice w;
            moeg_iq4ksr16_decode((const uint8_t *)grp, kbx, il, ir, s_tab, w);
            const uint8_t * yb = (const uint8_t *)s_y + kbx*MOEG_Q8_1_BYTES;
#pragma unroll
            for (int j = 0; j < NCOL; ++j) {
                if (j < nc) moeg_iq4ksr16_apply(w, yb + j*wpc*4, il, acc[j]);
            }
        }
#pragma unroll
        for (int j = 0; j < NCOL; ++j) {
            if (j < nc) {
                float v[4];
#pragma unroll
                for (int r = 0; r < 4; ++r) v[r] = warp_reduce_sum(acc[j][r]);
                if (threadIdx.x < 4) {
                    const int row = 4*g4 + threadIdx.x;
                    float r = v[0];
#pragma unroll
                    for (int q = 1; q < 4; ++q) if ((int)threadIdx.x == q) r = v[q];
                    if (a.bias) r += ((const float *)(a.bias + e*a.bias_nb1))[row];
                    moeg_dst_row(a, s_list[c0 + j])[row] = r;
                }
            }
        }
      }
    }
}

bool ggml_cuda_moe_grouped_enabled() {
    static const bool enabled = [] {
        const char * v = getenv("GGML_CUDA_MOE_GROUPED");
        return !(v && v[0] == '0');
    }();
    return enabled;
}

static int moeg_ncol(int n_tok) { return n_tok <= 4 ? 4 : 8; }

static size_t moeg_smem(int ncol, int64_t ncols_x) {
    return (size_t)ncol*(ncols_x/QK8_1)*MOEG_Q8_1_BYTES;
}

bool ggml_cuda_moe_grouped_supported(ggml_type type, bool fused, int64_t ncols_x, int64_t nrows_x, int n_tok, int n_ids) {
    if (!ggml_cuda_moe_grouped_enabled()) return false;
    if (n_tok < 2 || n_tok > 8 || n_ids < 1 || n_tok*n_ids > MOEG_MAX_SLOTS) return false;
    if (moeg_smem(moeg_ncol(n_tok), ncols_x) > MOEG_MAX_SMEM) return false;
    switch (type) {
        case GGML_TYPE_IQ3_KS:
            return ncols_x % QK_K == 0 && nrows_x > 0;
        case GGML_TYPE_IQ4_KS_R16:
            return !fused && ncols_x % 32 == 0 && nrows_x % 16 == 0;
        default:
            return false;
    }
}

template <int NCOL>
static void moeg_launch(ggml_type type, bool fused, const moeg_kargs & a, int64_t nrows_x, int64_t ncols_x, cudaStream_t stream) {
    const size_t smem = moeg_smem(NCOL, ncols_x);
    const int n_slots = a.n_tok*a.n_ids;
    if (type == GGML_TYPE_IQ3_KS) {
        constexpr int NW = 8;
        const int ntiles = int((nrows_x + NW - 1)/NW);
        const dim3 grid((ntiles + a.tpb - 1)/a.tpb, n_slots, 1);
        const dim3 block(WARP_SIZE, NW, 1);
        if (fused) k_moeg_iq3ks<NCOL, NW, true ><<<grid, block, smem, stream>>>(a);
        else       k_moeg_iq3ks<NCOL, NW, false><<<grid, block, smem, stream>>>(a);
    } else {
        constexpr int NW = 4;   // 4 warps x 4 rows = one 16-row group per tile
        const int ntiles = int(nrows_x/(4*NW));
        const dim3 grid((ntiles + a.tpb - 1)/a.tpb, n_slots, 1);
        const dim3 block(WARP_SIZE, NW, 1);
        k_moeg_iq4ksr16<NCOL, NW><<<grid, block, smem, stream>>>(a);
    }
}

void ggml_cuda_moe_grouped_mmvq(ggml_type type, bool fused, const moeg_params & p, cudaStream_t stream) {
    GGML_ASSERT(ggml_cuda_moe_grouped_supported(type, fused, p.ncols_x, p.nrows_x, p.n_tok, p.n_ids));
    GGML_ASSERT(!fused || (p.vx_g && !p.bias));

    moeg_kargs a;
    a.vx_u = p.vx_u;
    a.vx_g = p.vx_g;
    a.vy   = p.vy;
    a.dst  = (char *)p.dst;
    a.ids  = p.ids;
    a.bias = (const char *)p.bias;
    a.blocks_per_row = int(p.ncols_x / (type == GGML_TYPE_IQ3_KS ? QK_K : 32));
    a.y_blocks = int(p.ncols_x / QK8_1);
    a.nrows_x  = int(p.nrows_x);
    a.row_size = ggml_row_size(type, p.ncols_x);
    a.nb02 = p.nb02; a.ids_nb0 = p.ids_nb0; a.ids_nb1 = p.ids_nb1;
    a.y_nb_tok = p.y_nb_tok; a.y_nb_k = p.y_nb_k;
    a.dst_nb_tok = p.dst_nb_tok; a.dst_nb_k = p.dst_nb_k;
    a.bias_nb1 = p.bias_nb1;
    a.n_tok = p.n_tok; a.n_ids = p.n_ids;
    a.unary_op = int(p.unary_op);
    a.ops = moeg_unary{GGML_UNARY_OP_SILU, GGML_UNARY_OP_RELU, GGML_UNARY_OP_GELU, GGML_UNARY_OP_SITU, GGML_UNARY_OP_SWIGLU_OAI};
    a.limit  = p.limit > 1e-6f ? p.limit : INFINITY;   // as ggml_cuda_op_mul_mat_vec_q_impl
    a.limit2 = p.limit2;
    // RAON_MOEG_TPB=k: k row tiles per block share one staging of the activations (default 4)
    static const int tpb_env = [] { const char * e = getenv("RAON_MOEG_TPB"); return e ? atoi(e) : 4; }();
    a.tpb = tpb_env < 1 ? 1 : tpb_env > 64 ? 64 : tpb_env;

    if (moeg_ncol(p.n_tok) == 4) moeg_launch<4>(type, fused, a, p.nrows_x, p.ncols_x, stream);
    else                         moeg_launch<8>(type, fused, a, p.nrows_x, p.ncols_x, stream);
    CUDA_CHECK(cudaGetLastError());
}
