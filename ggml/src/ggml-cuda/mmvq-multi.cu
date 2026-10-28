// Single-column quantized matvec for decode: one launch for several matrices sharing src1 (GGML_CUDA_MMVQ_MULTI) and
// per-shape thread layouts (GGML_CUDA_MMVQ_TUNE). Same vec_dot_*_q8_1 functions as mmvq.cu; layout 0 is the stock
// mul_mat_vec_q<type, 1, 4> arithmetic (bit-identical), the others only change the order of the float partial sums.
//
// Layouts:
//   A(nw, rpb): stock scheme, nw warps share rpb rows, partial sums reduced through shared memory.
//   W(nw, lpr, rps): each group of lpr lanes owns rps rows on its own (shuffle reduction only, no barrier),
//                    nw*32/lpr*rps rows per CUDA block. For short rows (few blocks per row) and for more
//                    loads in flight per thread.

#include "mmvq-multi.cuh"
#include "mmvq-templates.cuh"
#include "devquants.cuh"

#include <cstdlib>

bool ggml_cuda_mmvq_multi_enabled() {
    static const bool on = [] { const char * e = getenv("GGML_CUDA_MMVQ_MULTI"); return !e || atoi(e) != 0; }();
    return on;
}

static int mmvq_forced_cfg() {
    static const int cfg = [] { const char * e = getenv("GGML_CUDA_MMVQ_CFG"); return e ? atoi(e) : -1; }();
    return cfg;
}

bool ggml_cuda_mmvq_tune_enabled() {
    static const bool on = [] { const char * e = getenv("GGML_CUDA_MMVQ_TUNE"); return !e || atoi(e) != 0; }();
    return on || mmvq_forced_cfg() >= 0;
}

bool ggml_cuda_mmvq_multi_type_supported(ggml_type type) {
    switch (type) {
#if GGML_CUDA_DEVQ_ON(Q4_0)
        case GGML_TYPE_Q4_0:
#endif
#if GGML_CUDA_DEVQ_ON(Q5_0)
        case GGML_TYPE_Q5_0:
#endif
#if GGML_CUDA_DEVQ_ON(Q8_0)
        case GGML_TYPE_Q8_0:
#endif
#if GGML_CUDA_DEVQ_ON(Q4_K)
        case GGML_TYPE_Q4_K:
#endif
#if GGML_CUDA_DEVQ_ON(Q5_K)
        case GGML_TYPE_Q5_K:
#endif
#if GGML_CUDA_DEVQ_ON(Q6_K)
        case GGML_TYPE_Q6_K:
#endif
            return true;
        default:
            return false;
    }
}

// pick matrix of this block; returns local block index
static __device__ __forceinline__ int mmvq_multi_select(const mmvq_multi_set & s, const char * & vx, float * & dst, int & nrows) {
    const int b = blockIdx.x;
    vx = (const char *) s.vx[0]; dst = s.dst[0]; nrows = s.nrows[0];
    int b0 = 0;
#pragma unroll
    for (int i = 1; i < MMVQ_MULTI_MAX; ++i) {
        if (i < s.n && b >= s.blk_end[i-1]) {
            vx = (const char *) s.vx[i]; dst = s.dst[i]; nrows = s.nrows[i]; b0 = s.blk_end[i-1];
        }
    }
    return b - b0;
}

template <ggml_type type, int nwarps, int rpb, int NC = 1>
__launch_bounds__(nwarps*WARP_SIZE, 1)
static __global__ void k_mmvq_multi_a(const mmvq_multi_set s, const void * __restrict__ vy, const int ncols_x) {
    constexpr int qk  = ggml_cuda_type_traits<type>::qk;
    constexpr int qi  = ggml_cuda_type_traits<type>::qi;
    constexpr int vdr = get_vdr_mmvq(type);
    constexpr vec_dot_q_cuda_t vec_dot_q_cuda = get_vec_dot_q_cuda(type);

    const char * vx; float * dst; int nrows;
    const int blk = mmvq_multi_select(s, vx, dst, nrows);
    vx  += blockIdx.y*s.nb02;
    dst  = (float *)((char *) dst + blockIdx.y*s.nb2);
    const block_q8_1 * y = (const block_q8_1 *)((const char *) vy + blockIdx.y*s.nb12);

    const     int tid  = WARP_SIZE*threadIdx.y + threadIdx.x;
    const     int row0 = rpb*blk;
    const     int blocks_per_row_x = ncols_x / qk;
    constexpr int blocks_per_iter = vdr * nwarps*WARP_SIZE / qi;

    int xrow[rpb];
#pragma unroll
    for (int i = 0; i < rpb; ++i) xrow[i] = min(row0 + i, nrows - 1)*blocks_per_row_x;

    float tmp[NC][rpb] = {{0.0f}};
    const int ycb = NC > 1 ? s.y_col_blocks : 0;

    for (int kbx = tid / (qi/vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
        const int kby = kbx * (qk/QK8_1);
        const int kqs = vdr * (tid % (qi/vdr));
#pragma unroll
        for (int j = 0; j < NC; ++j) {
#pragma unroll
            for (int i = 0; i < rpb; ++i) {
                tmp[j][i] += vec_dot_q_cuda(vx, &y[j*ycb + kby], xrow[i] + kbx, kqs);
            }
        }
    }

    __shared__ float tmp_shared[nwarps-1 > 0 ? nwarps-1 : 1][NC][rpb][WARP_SIZE];
    if (threadIdx.y > 0) {
#pragma unroll
        for (int j = 0; j < NC; ++j) {
#pragma unroll
            for (int i = 0; i < rpb; ++i) {
                tmp_shared[threadIdx.y-1][j][i][threadIdx.x] = tmp[j][i];
            }
        }
    }
    if constexpr (nwarps > 1) {
        __syncthreads();
    }
    if (threadIdx.y > 0) {
        return;
    }
#pragma unroll
    for (int j = 0; j < NC; ++j) {
#pragma unroll
        for (int i = 0; i < rpb; ++i) {
#pragma unroll
            for (int l = 0; l < nwarps-1; ++l) {
                tmp[j][i] += tmp_shared[l][j][i][threadIdx.x];
            }
            tmp[j][i] = warp_reduce_sum(tmp[j][i]);
        }
        if (threadIdx.x < rpb) {
            float r = tmp[j][0];
#pragma unroll
            for (int i = 1; i < rpb; ++i) if (threadIdx.x == i) r = tmp[j][i];
            if (row0 + threadIdx.x < nrows) dst[j*nrows + row0 + threadIdx.x] = r;
        }
    }
}

template <ggml_type type, int nwarps, int lpr, int rps, int NC = 1>
__launch_bounds__(nwarps*WARP_SIZE, 1)
static __global__ void k_mmvq_multi_w(const mmvq_multi_set s, const void * __restrict__ vy, const int ncols_x) {
    constexpr int qk  = ggml_cuda_type_traits<type>::qk;
    constexpr int qi  = ggml_cuda_type_traits<type>::qi;
    constexpr int vdr = get_vdr_mmvq(type);
    constexpr vec_dot_q_cuda_t vec_dot_q_cuda = get_vec_dot_q_cuda(type);
    constexpr int tpb = qi/vdr;                 // threads per x block
    static_assert(lpr >= tpb && lpr % tpb == 0 && lpr <= WARP_SIZE, "bad lanes per row");
    constexpr int bpi = lpr/tpb;                // x blocks per iteration of one row group
    constexpr int R   = nwarps*(WARP_SIZE/lpr)*rps;

    const char * vx; float * dst; int nrows;
    const int blk = mmvq_multi_select(s, vx, dst, nrows);
    vx  += blockIdx.y*s.nb02;
    dst  = (float *)((char *) dst + blockIdx.y*s.nb2);
    const block_q8_1 * y = (const block_q8_1 *)((const char *) vy + blockIdx.y*s.nb12);

    const int l     = threadIdx.x % lpr;
    const int rbase = blk*R + (threadIdx.y*(WARP_SIZE/lpr) + threadIdx.x/lpr)*rps;
    const int blocks_per_row_x = ncols_x / qk;

    int xrow[rps];
#pragma unroll
    for (int r = 0; r < rps; ++r) xrow[r] = min(rbase + r, nrows - 1)*blocks_per_row_x;

    float tmp[NC][rps] = {{0.0f}};
    const int ycb = NC > 1 ? s.y_col_blocks : 0;
    const int kqs = vdr*(l % tpb);
#pragma unroll 2
    for (int kbx = l / tpb; kbx < blocks_per_row_x; kbx += bpi) {
        const block_q8_1 * yb = y + kbx*(qk/QK8_1);
#pragma unroll
        for (int j = 0; j < NC; ++j) {
#pragma unroll
            for (int r = 0; r < rps; ++r) {
                tmp[j][r] += vec_dot_q_cuda(vx, yb + j*ycb, xrow[r] + kbx, kqs);
            }
        }
    }
#pragma unroll
    for (int j = 0; j < NC; ++j) {
#pragma unroll
        for (int r = 0; r < rps; ++r) {
#pragma unroll
            for (int off = lpr/2; off > 0; off >>= 1) {
                tmp[j][r] += __shfl_xor_sync(0xFFFFFFFF, tmp[j][r], off, WARP_SIZE);
            }
        }
    }
    if (l == 0) {
#pragma unroll
        for (int j = 0; j < NC; ++j) {
#pragma unroll
            for (int r = 0; r < rps; ++r) {
                if (rbase + r < nrows) dst[j*nrows + rbase + r] = tmp[j][r];
            }
        }
    }
}

static int mmvq_multi_fill(mmvq_multi_set & s, int R) {
    int nb = 0;
    for (int i = 0; i < s.n; ++i) {
        nb += (s.nrows[i] + R - 1)/R;
        s.blk_end[i] = nb;
    }
    for (int i = s.n; i < MMVQ_MULTI_MAX; ++i) s.blk_end[i] = nb;
    return nb;
}

template <ggml_type type, int nw, int rpb, int NC = 1>
static void launch_a(mmvq_multi_set & s, const void * vy, int ncols_x, int ne2, cudaStream_t stream) {
    const int nb = mmvq_multi_fill(s, rpb);
    k_mmvq_multi_a<type, nw, rpb, NC><<<dim3(nb, ne2, 1), dim3(WARP_SIZE, nw, 1), 0, stream>>>(s, vy, ncols_x);
}

template <ggml_type type, int nw, int lpr, int rps, int NC = 1>
static void launch_w(mmvq_multi_set & s, const void * vy, int ncols_x, int ne2, cudaStream_t stream) {
    constexpr int tpb = ggml_cuda_type_traits<type>::qi/get_vdr_mmvq(type);
    if constexpr (lpr < tpb) {
        launch_a<type, 4, 1, NC>(s, vy, ncols_x, ne2, stream);
    } else {
        const int nb = mmvq_multi_fill(s, nw*(WARP_SIZE/lpr)*rps);
        k_mmvq_multi_w<type, nw, lpr, rps, NC><<<dim3(nb, ne2, 1), dim3(WARP_SIZE, nw, 1), 0, stream>>>(s, vy, ncols_x);
    }
}

// several src1 columns: the tuned single-column layouts only (2 = A(4,2), 21 = A(2,2), 22 = A(4,4), 13 = W(4,16,2)),
// with the rows per block halved for many columns to keep the partial sums in registers
template <ggml_type type, int NC>
static void mmvq_multi_launch_nc(int cfg, mmvq_multi_set & s, const void * vy, int ncols_x, int ne2, cudaStream_t stream) {
    switch (cfg) {
        case 21: launch_a<type, 2, 2, NC>(s, vy, ncols_x, ne2, stream); return;
        case 22: if constexpr (NC <= 4) { launch_a<type, 4, 4, NC>(s, vy, ncols_x, ne2, stream); return; }
                 else { launch_a<type, 4, 2, NC>(s, vy, ncols_x, ne2, stream); return; }
        case 13: launch_w<type, 4, 16, 2, NC>(s, vy, ncols_x, ne2, stream); return;
        case  0: launch_a<type, 4, 1, NC>(s, vy, ncols_x, ne2, stream); return;
        // more layouts for several columns (tuning: GGML_CUDA_MMVQ_CFG, or RAON_MMVQ_NC_CFG for the NC > 1 choice only)
        case 23: launch_a<type, 8, 1, NC>(s, vy, ncols_x, ne2, stream); return;
        case 24: launch_a<type, 8, 2, NC>(s, vy, ncols_x, ne2, stream); return;
        case 25: launch_a<type, 2, 4, NC>(s, vy, ncols_x, ne2, stream); return;
        case 31: launch_a<type, 1, 2, NC>(s, vy, ncols_x, ne2, stream); return;
        case 34: launch_a<type, 4, 4, NC>(s, vy, ncols_x, ne2, stream); return;
        case 35: launch_a<type, 1, 4, NC>(s, vy, ncols_x, ne2, stream); return;
        case 11: launch_w<type, 4, 32, 2, NC>(s, vy, ncols_x, ne2, stream); return;
        case 18: launch_w<type, 4, 32, 4, NC>(s, vy, ncols_x, ne2, stream); return;
        default: launch_a<type, 4, 2, NC>(s, vy, ncols_x, ne2, stream); return;
    }
}

// Layout ids (GGML_CUDA_MMVQ_CFG):
//   0 A(4,1) stock   1 A(2,1)   2 A(4,2)   3 A(1,1)
//  10 W(4,32,1)  11 W(4,32,2)  12 W(4,16,1)  13 W(4,16,2)  14 W(4,8,1)  15 W(4,8,2)  16 W(2,32,1)  17 W(8,32,1)
//  18 W(4,32,4)  19 W(4,4,1)   20 W(4,4,2)
template <ggml_type type>
static void mmvq_multi_launch(int cfg, mmvq_multi_set & s, const void * vy, int ncols_x, int ne2, cudaStream_t stream) {
    switch (s.ncols > 1 ? s.ncols : 1) {
        case 2: mmvq_multi_launch_nc<type, 2>(cfg, s, vy, ncols_x, ne2, stream); return;
        case 3: mmvq_multi_launch_nc<type, 3>(cfg, s, vy, ncols_x, ne2, stream); return;
        case 4: mmvq_multi_launch_nc<type, 4>(cfg, s, vy, ncols_x, ne2, stream); return;
        case 5: mmvq_multi_launch_nc<type, 5>(cfg, s, vy, ncols_x, ne2, stream); return;
        case 6: mmvq_multi_launch_nc<type, 6>(cfg, s, vy, ncols_x, ne2, stream); return;
        case 7: mmvq_multi_launch_nc<type, 7>(cfg, s, vy, ncols_x, ne2, stream); return;
        case 8: mmvq_multi_launch_nc<type, 8>(cfg, s, vy, ncols_x, ne2, stream); return;
        default: break;
    }
    if constexpr (type == GGML_TYPE_Q6_K || type == GGML_TYPE_Q8_0) {
        switch (cfg) {
            case  1: launch_a<type, 2, 1>(s, vy, ncols_x, ne2, stream); return;
            case  2: launch_a<type, 4, 2>(s, vy, ncols_x, ne2, stream); return;
            case  3: launch_a<type, 1, 1>(s, vy, ncols_x, ne2, stream); return;
            case 10: launch_w<type, 4, 32, 1>(s, vy, ncols_x, ne2, stream); return;
            case 11: launch_w<type, 4, 32, 2>(s, vy, ncols_x, ne2, stream); return;
            case 12: launch_w<type, 4, 16, 1>(s, vy, ncols_x, ne2, stream); return;
            case 13: launch_w<type, 4, 16, 2>(s, vy, ncols_x, ne2, stream); return;
            case 14: launch_w<type, 4,  8, 1>(s, vy, ncols_x, ne2, stream); return;
            case 15: launch_w<type, 4,  8, 2>(s, vy, ncols_x, ne2, stream); return;
            case 16: launch_w<type, 2, 32, 1>(s, vy, ncols_x, ne2, stream); return;
            case 17: launch_w<type, 8, 32, 1>(s, vy, ncols_x, ne2, stream); return;
            case 18: launch_w<type, 4, 32, 4>(s, vy, ncols_x, ne2, stream); return;
            case 19: launch_w<type, 4,  4, 1>(s, vy, ncols_x, ne2, stream); return;
            case 20: launch_w<type, 4,  4, 2>(s, vy, ncols_x, ne2, stream); return;
            case 21: launch_a<type, 2, 2>(s, vy, ncols_x, ne2, stream); return;
            case 22: launch_a<type, 4, 4>(s, vy, ncols_x, ne2, stream); return;
            case 23: launch_a<type, 8, 1>(s, vy, ncols_x, ne2, stream); return;
            case 24: launch_a<type, 8, 2>(s, vy, ncols_x, ne2, stream); return;
            case 25: launch_a<type, 2, 4>(s, vy, ncols_x, ne2, stream); return;
            case 26: launch_w<type, 2, 32, 2>(s, vy, ncols_x, ne2, stream); return;
            case 27: launch_w<type, 8, 32, 2>(s, vy, ncols_x, ne2, stream); return;
            case 28: launch_w<type, 4, 16, 4>(s, vy, ncols_x, ne2, stream); return;
            case 29: launch_w<type, 4,  8, 4>(s, vy, ncols_x, ne2, stream); return;
            case 30: launch_w<type, 1, 32, 2>(s, vy, ncols_x, ne2, stream); return;
            case 31: launch_a<type, 1, 2>(s, vy, ncols_x, ne2, stream); return;
            case 32: launch_w<type, 4, 16, 3>(s, vy, ncols_x, ne2, stream); return;
            case 33: launch_w<type, 4, 32, 3>(s, vy, ncols_x, ne2, stream); return;
            default: break;
        }
    }
    launch_a<type, 4, 1>(s, vy, ncols_x, ne2, stream);
}

// Tuned layout per shape (on sm_120), GGML_CUDA_MMVQ_TUNE=1.
static int mmvq_multi_choose(ggml_type type, int ncols_x, int nrows, int ne2, int nc = 1) {
    const int forced = mmvq_forced_cfg();
    if (forced >= 0) return forced;
    if (nc > 1) {
        static const int nc_cfg = [] { const char * e = getenv("RAON_MMVQ_NC_CFG"); return e ? atoi(e) : -1; }();
        if (nc_cfg >= 0) return nc_cfg;
        // 2-8 columns: A(1,4) (one warp x 4 rows per block), W(4,16,2) for short Q8_0 rows.
        // RAON_MMVQ_NC_TUNE=0 uses the single-column choice below.
        static const bool nc_tune = [] { const char * e = getenv("RAON_MMVQ_NC_TUNE"); return !e || atoi(e) != 0; }();
        if (nc_tune && ne2 == 1 && (type == GGML_TYPE_Q6_K || type == GGML_TYPE_Q8_0)) {
            return type == GGML_TYPE_Q8_0 && ncols_x <= 1024 ? 13 : 35;
        }
    }
    if (!ggml_cuda_mmvq_tune_enabled()) return 0;
    GGML_UNUSED(nrows);
    if (type == GGML_TYPE_Q6_K) {
        return ne2 > 1 ? 21 : 2;                                  // A(2,2) : A(4,2)
    }
    if (type == GGML_TYPE_Q8_0) {
        return ncols_x >= 1024 ? 2 : ncols_x > 256 ? 22 : 13;     // A(4,2) : A(4,4) : W(4,16,2)
    }
    return 2;                                                     // A(4,2): two rows per block for the other types
}

void ggml_cuda_mmvq_multi(ggml_type type, mmvq_multi_set & s, const void * vy, int ncols_x, int ne2, cudaStream_t stream) {
    GGML_ASSERT(s.n >= 1 && s.n <= MMVQ_MULTI_MAX);
    GGML_ASSERT(s.n == 1 || ne2 == 1);
    GGML_ASSERT(s.ncols <= MMVQ_MULTI_MAX_COLS);
    int nrows = 0;
    for (int i = 0; i < s.n; ++i) nrows += s.nrows[i];
    const int cfg = mmvq_multi_choose(type, ncols_x, nrows, ne2, s.ncols);
    switch (type) {
#if GGML_CUDA_DEVQ_ON(Q4_0)
        case GGML_TYPE_Q4_0: mmvq_multi_launch<GGML_TYPE_Q4_0>(cfg, s, vy, ncols_x, ne2, stream); break;
#endif
#if GGML_CUDA_DEVQ_ON(Q5_0)
        case GGML_TYPE_Q5_0: mmvq_multi_launch<GGML_TYPE_Q5_0>(cfg, s, vy, ncols_x, ne2, stream); break;
#endif
#if GGML_CUDA_DEVQ_ON(Q8_0)
        case GGML_TYPE_Q8_0: mmvq_multi_launch<GGML_TYPE_Q8_0>(cfg, s, vy, ncols_x, ne2, stream); break;
#endif
#if GGML_CUDA_DEVQ_ON(Q4_K)
        case GGML_TYPE_Q4_K: mmvq_multi_launch<GGML_TYPE_Q4_K>(cfg, s, vy, ncols_x, ne2, stream); break;
#endif
#if GGML_CUDA_DEVQ_ON(Q5_K)
        case GGML_TYPE_Q5_K: mmvq_multi_launch<GGML_TYPE_Q5_K>(cfg, s, vy, ncols_x, ne2, stream); break;
#endif
#if GGML_CUDA_DEVQ_ON(Q6_K)
        case GGML_TYPE_Q6_K: mmvq_multi_launch<GGML_TYPE_Q6_K>(cfg, s, vy, ncols_x, ne2, stream); break;
#endif
        default: GGML_ABORT("mmvq_multi: unsupported type %s", ggml_type_name(type));
    }
    CUDA_CHECK(cudaGetLastError());
}
