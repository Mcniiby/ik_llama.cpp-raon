// GGML_OP_LOGITS_TOPM: exact top-M candidates of each logits row.
//
// Two fixed-size launches, no host sync (CUDA-graph safe):
//   1. grid (n_blocks, rows): each warp holds 1,024 logits in registers and runs M warp-argmax rounds in the
//      strict order (value desc, index asc); warp 0 merges the warp lists into the block list.
//   2. grid rows, one warp: merges the n_blocks block lists into the row's first M and writes the result.
// The strict order is total, so the first M of a union are within the first M of each part and the result is
// unique (bit-identical to ggml.c). Split-list scheme after Strata's split top_k sampler (MIT).
//
// NaN and -inf are never selected (they are counted in the non-finite total, as is +inf).

#include "logits-topm.cuh"

#include <climits>

#define TOPM_PER_LANE   32
#define TOPM_WARP_SPAN  (32*TOPM_PER_LANE)
#define TOPM_SMEM_LISTS 32768   // bytes of block lists stage 2 keeps in shared memory

static __device__ __forceinline__ bool topm_before(float av, int ai, float bv, int bi) {
    return av > bv || (av == bv && ai < bi);
}

static __device__ __forceinline__ void topm_warp_first(float & bv, int & bi) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        const float ov = __shfl_xor_sync(0xFFFFFFFF, bv, off, WARP_SIZE);
        const int   oi = __shfl_xor_sync(0xFFFFFFFF, bi, off, WARP_SIZE);
        if (topm_before(ov, oi, bv, bi)) { bv = ov; bi = oi; }
    }
}

// merge nl (<= 32 * TOPM_MAX_HEADS) lists of m entries (each in the strict order, padded with (-inf, INT_MAX)
// sentinels) into their first m; sink(r, v, i) runs in every lane for r = 0..n_out-1. Returns the count written.
#define TOPM_MAX_HEADS 8
template <typename Sink>
static __device__ __forceinline__ int topm_warp_merge(const int2 * lists, int nl, int m, Sink && sink) {
    const int lane = threadIdx.x & 31;
    float hv[TOPM_MAX_HEADS];
    int   hi[TOPM_MAX_HEADS];
    int   pos[TOPM_MAX_HEADS];
#pragma unroll
    for (int h = 0; h < TOPM_MAX_HEADS; ++h) {
        const int L = lane + 32*h;
        pos[h] = 0;
        hv[h]  = -INFINITY;
        hi[h]  = INT_MAX;
        if (L < nl) {
            const int2 c = lists[L*m];
            hv[h] = __int_as_float(c.x);
            hi[h] = c.y;
        }
    }
    int r = 0;
    for (; r < m; ++r) {
        float bv = hv[0];
        int   bi = hi[0];
#pragma unroll
        for (int h = 1; h < TOPM_MAX_HEADS; ++h) {
            if (topm_before(hv[h], hi[h], bv, bi)) { bv = hv[h]; bi = hi[h]; }
        }
        topm_warp_first(bv, bi);
        if (bi == INT_MAX) {
            break;                                  // every head is a sentinel, in every lane
        }
        sink(r, bv, bi);
#pragma unroll
        for (int h = 0; h < TOPM_MAX_HEADS; ++h) {
            if (hi[h] != bi) continue;              // ids are unique across lists: one head in one lane matches
            const int L = lane + 32*h;
            if (++pos[h] < m) {
                const int2 c = lists[L*m + pos[h]];
                hv[h] = __int_as_float(c.x);
                hi[h] = c.y;
            } else {
                hv[h] = -INFINITY;
                hi[h] = INT_MAX;
            }
        }
    }
    return r;
}

static __global__ void k_logits_topm_part(const char * __restrict__ x, const size_t nb1, const int n, const int m,
        const int n_blocks, int2 * __restrict__ cand, int * __restrict__ bad, const int * __restrict__ need) {
    extern __shared__ int2 topm_smem[];                 // [n_warps][m] warp lists
    __shared__ int warp_bad[32];

    const int row  = blockIdx.y;
    if (need && !need[row]) {
        return;                                         // done by the threshold path
    }
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int n_warps = blockDim.x >> 5;
    const float * l = (const float *) (x + (size_t) row*nb1);

    const int lo = (blockIdx.x*n_warps + warp)*TOPM_WARP_SPAN;
    float s[TOPM_PER_LANE];
    int n_bad = 0;
#pragma unroll
    for (int j = 0; j < TOPM_PER_LANE; ++j) {
        const int v = lo + 32*j + lane;
        float xv = NAN;
        if (v < n) {
            xv = l[v];
            n_bad += !isfinite(xv);
            if (!(xv > -INFINITY)) xv = NAN;
        }
        s[j] = xv;
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        n_bad += __shfl_xor_sync(0xFFFFFFFF, n_bad, off, WARP_SIZE);
    }
    if (lane == 0) warp_bad[warp] = n_bad;

    int2 * wl = topm_smem + warp*m;
    float prev_v = INFINITY;                            // round 0 takes everything (+inf included)
    int   prev_i = -1;
    int r = 0;
    for (; r < m; ++r) {
        float b0 = -INFINITY, b1 = -INFINITY;
        int   i0 = INT_MAX,   i1 = INT_MAX;
        // two chains (even/odd j) each walked in ascending index, so a strict `before` keeps the first in order
#pragma unroll
        for (int j = 0; j < TOPM_PER_LANE; j += 2) {
            const int v0 = lo + 32*j + lane, v1 = v0 + 32;
            const float x0 = s[j], x1 = s[j + 1];
            if ((x0 < prev_v || (x0 == prev_v && v0 > prev_i)) && topm_before(x0, v0, b0, i0)) { b0 = x0; i0 = v0; }
            if ((x1 < prev_v || (x1 == prev_v && v1 > prev_i)) && topm_before(x1, v1, b1, i1)) { b1 = x1; i1 = v1; }
        }
        if (topm_before(b1, i1, b0, i0)) { b0 = b1; i0 = i1; }
        topm_warp_first(b0, i0);
        if (i0 == INT_MAX) break;                       // nothing left in this warp's span
        if (lane == 0) wl[r] = make_int2(__float_as_int(b0), i0);
        prev_v = b0;
        prev_i = i0;
    }
    for (int q = r + lane; q < m; q += 32) wl[q] = make_int2(__float_as_int(-INFINITY), INT_MAX);
    __syncthreads();

    if (warp == 0) {
        int2 * dst = cand + ((size_t) row*n_blocks + blockIdx.x)*m;
        const int nw = topm_warp_merge(topm_smem, n_warps, m, [&](int q, float v, int i) {
            if (lane == 0) dst[q] = make_int2(__float_as_int(v), i);
        });
        for (int q = nw + lane; q < m; q += 32) dst[q] = make_int2(__float_as_int(-INFINITY), INT_MAX);
        if (lane == 0) {
            int t = 0;
            for (int w = 0; w < n_warps; ++w) t += warp_bad[w];
            bad[(size_t) row*n_blocks + blockIdx.x] = t;
        }
    }
}

static __global__ void k_logits_topm_merge(const int2 * __restrict__ cand, const int * __restrict__ bad,
        const int n, const int m, const int n_blocks, float * __restrict__ dst, const size_t nb1, const int * __restrict__ need) {
    extern __shared__ int2 topm_smem[];                 // [n_blocks][m]
    const int row  = blockIdx.x;
    if (need && !need[row]) {
        return;
    }
    const int lane = threadIdx.x;
    const int2 * src = cand + (size_t) row*n_blocks*m;
    for (int e = lane; e < n_blocks*m; e += 32) topm_smem[e] = src[e];
    int n_bad = 0;
    for (int b = lane; b < n_blocks; b += 32) n_bad += bad[(size_t) row*n_blocks + b];
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        n_bad += __shfl_xor_sync(0xFFFFFFFF, n_bad, off, WARP_SIZE);
    }
    __syncwarp();

    float * out = (float *) ((char *) dst + (size_t) row*nb1);
    const int nv = topm_warp_merge(topm_smem, n_blocks, m, [&](int q, float v, int i) {
        if (lane == 0) { out[q] = v; out[m + q] = (float) i; }
    });
    for (int q = nv + lane; q < m; q += 32) { out[q] = -INFINITY; out[m + q] = (float) n; }
    if (lane == 0) {
        out[2*m]     = (float) n_bad;
        out[2*m + 1] = (float) nv;
    }
}

// Threshold path (GGML_CUDA_TOPM_FAST=1, default). The M-th largest of the TOPM_FAST_G chunk maxima is a lower
// bound T for the row's M-th entry, so the row's first M are among the entries at or before T; those are
// collected and bitonic-sorted. Rows with more than TOPM_FAST_CAP candidates (ties, e.g. constant rows) are
// flagged and handled by the split-list kernels launched after it. GGML_CUDA_TOPM_FAST=2 forces that fallback
// for every row (tests), =0 runs only the split-list path.
#define TOPM_FAST_G   256
#define TOPM_FAST_NT  256
#define TOPM_FAST_CAP 4096

static __device__ __forceinline__ bool topm_valid(const float v) {
    const unsigned u = __float_as_uint(v);
    return (u & 0x7fffffffu) <= 0x7f800000u && u != 0xff800000u;   // not NaN, not -inf (bit tests: -use_fast_math)
}

static __device__ __forceinline__ int topm_block_sum(int v, int * red) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xFFFFFFFF, v, off, WARP_SIZE);
    if (lane == 0) red[warp] = v;
    __syncthreads();
    int t = 0;
    for (int w = 0; w < (int) (blockDim.x >> 5); ++w) t += red[w];
    __syncthreads();
    return t;
}

static __global__ void __launch_bounds__(TOPM_FAST_NT) k_topm_fast_max(const char * __restrict__ x, const size_t nb1,
        const int n, const int chunk, int2 * __restrict__ bmax, int * __restrict__ bbad, int * __restrict__ cnt) {
    __shared__ float rv[TOPM_FAST_NT/32];
    __shared__ int   ri[TOPM_FAST_NT/32];
    __shared__ int   red[TOPM_FAST_NT/32];
    const int row = blockIdx.y, b = blockIdx.x;
    const float * l = (const float *) (x + (size_t) row*nb1);
    const int lo = b*chunk, hi = min(n, lo + chunk);
    float bv = -INFINITY;
    int   bi = INT_MAX;
    int   nbad = 0;
    for (int i = lo + (int) threadIdx.x; i < hi; i += TOPM_FAST_NT) {
        const float v = l[i];
        nbad += (__float_as_uint(v) & 0x7f800000u) == 0x7f800000u;
        if (topm_valid(v) && topm_before(v, i, bv, bi)) { bv = v; bi = i; }
    }
    topm_warp_first(bv, bi);
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if (lane == 0) { rv[warp] = bv; ri[warp] = bi; }
    nbad = topm_block_sum(nbad, red);
    if (threadIdx.x == 0) {
        for (int w = 1; w < TOPM_FAST_NT/32; ++w) {
            if (topm_before(rv[w], ri[w], bv, bi)) { bv = rv[w]; bi = ri[w]; }
        }
        bmax[(size_t) row*gridDim.x + b] = make_int2(__float_as_int(bv), bi);
        bbad[(size_t) row*gridDim.x + b] = nbad;
        if (b == 0) cnt[row] = 0;
    }
}

// one block per row: T = the M-th of the chunk maxima, or (-inf, INT_MAX) (every valid entry) with fewer valid maxima.
// TOPM_FAST_T_NT/G threads per maximum.
#define TOPM_FAST_T_NT 1024
static __global__ void __launch_bounds__(TOPM_FAST_T_NT) k_topm_fast_t(const int m, const int2 * __restrict__ bmax,
        int2 * __restrict__ tT) {
    static_assert(TOPM_FAST_T_NT % TOPM_FAST_G == 0, "");
    constexpr int parts = TOPM_FAST_T_NT/TOPM_FAST_G, span = TOPM_FAST_G/parts;
    __shared__ int2 smax[TOPM_FAST_G];
    __shared__ int  srank[TOPM_FAST_G];
    const int row = blockIdx.x, tid = threadIdx.x;
    const int e_id = tid % TOPM_FAST_G, part = tid / TOPM_FAST_G;
    if (tid < TOPM_FAST_G) {
        smax[tid]  = bmax[(size_t) row*TOPM_FAST_G + tid];
        srank[tid] = 0;
    }
    __syncthreads();
    const int2  e  = smax[e_id];
    const float ev = __int_as_float(e.x);
    int r = 0;
#pragma unroll 16
    for (int j = part*span; j < (part + 1)*span; ++j) {
        const int2 f = smax[j];
        r += f.y != INT_MAX && topm_before(__int_as_float(f.x), f.y, ev, e.y);
    }
    atomicAdd(srank + e_id, r);
    __syncthreads();
    const bool hit = part == 0 && e.y != INT_MAX && srank[e_id] == m - 1;   // distinct ranks: at most one
    if (hit) tT[row] = e;
    if (__syncthreads_or(hit) == 0 && tid == 0) tT[row] = make_int2(__float_as_int(-INFINITY), INT_MAX);
}

static __global__ void __launch_bounds__(TOPM_FAST_NT) k_topm_fast_sel(const char * __restrict__ x, const size_t nb1,
        const int n, const int chunk, const int2 * __restrict__ tT, int * __restrict__ cnt,
        int2 * __restrict__ cand, const int cap) {
    const int row = blockIdx.y, b = blockIdx.x, tid = threadIdx.x;
    const float * l = (const float *) (x + (size_t) row*nb1);
    const int2  t  = tT[row];
    const float tv = __int_as_float(t.x);
    const int   ti = t.y;

    int2 * rc = cand + (size_t) row*TOPM_FAST_CAP;
    const int lo = b*chunk, hi = min(n, lo + chunk);
    for (int i = lo + tid; i < hi; i += TOPM_FAST_NT) {
        const float v = l[i];
        if (topm_valid(v) && !topm_before(tv, ti, v, i)) {   // v before T, or v is T
            const int p = atomicAdd(cnt + row, 1);
            if (p < cap) rc[p] = make_int2(__float_as_int(v), i);
        }
    }
}

// one block per row: bitonic sort of the candidates (padded to a power of 2 with (-inf, INT_MAX))
#define TOPM_FAST_SORT_NT 1024
static __global__ void __launch_bounds__(TOPM_FAST_SORT_NT) k_topm_fast_sort(const int n, const int m, const int cap,
        const int * __restrict__ bbad, const int * __restrict__ cnt, const int2 * __restrict__ cand,
        int * __restrict__ need, float * __restrict__ dst, const size_t dnb1) {
    __shared__ int2 sbuf[TOPM_FAST_CAP];
    __shared__ int  red[TOPM_FAST_SORT_NT/32];
    const int row = blockIdx.x, tid = threadIdx.x;

    int nbad = 0;
    for (int j = tid; j < TOPM_FAST_G; j += TOPM_FAST_SORT_NT) nbad += bbad[(size_t) row*TOPM_FAST_G + j];
    nbad = topm_block_sum(nbad, red);
    const int c = cnt[row];
    if (c > cap) {
        if (tid == 0) need[row] = 1;                    // the split-list kernels do this row
        return;
    }
    if (tid == 0) need[row] = 0;
    int P = 32;
    while (P < c) P <<= 1;
    const int2 * rc = cand + (size_t) row*TOPM_FAST_CAP;
    for (int j = tid; j < P; j += TOPM_FAST_SORT_NT) {
        sbuf[j] = j < c ? rc[j] : make_int2(__float_as_int(-INFINITY), INT_MAX);
    }
    __syncthreads();
    for (int k = 2; k <= P; k <<= 1) {
        for (int jj = k >> 1; jj > 0; jj >>= 1) {
            for (int q = tid; q < P/2; q += TOPM_FAST_SORT_NT) {
                const int i   = 2*q - (q & (jj - 1));   // lower index of the q-th pair at distance jj
                const int ixj = i + jj;
                const int2 a = sbuf[i], e = sbuf[ixj];
                const bool desc = (i & k) == 0;         // this run is sorted first-before-last
                const bool swap = desc ? topm_before(__int_as_float(e.x), e.y, __int_as_float(a.x), a.y)
                                       : topm_before(__int_as_float(a.x), a.y, __int_as_float(e.x), e.y);
                if (swap) { sbuf[i] = e; sbuf[ixj] = a; }
            }
            __syncthreads();
        }
    }

    const int nv = c < m ? c : m;
    float * out = (float *) ((char *) dst + (size_t) row*dnb1);
    for (int q = tid; q < m; q += TOPM_FAST_SORT_NT) {
        out[q]     = q < nv ? __int_as_float(sbuf[q].x) : -INFINITY;
        out[m + q] = q < nv ? (float) sbuf[q].y : (float) n;
    }
    if (tid == 0) {
        out[2*m]     = (float) nbad;
        out[2*m + 1] = (float) nv;
    }
}

static int ggml_cuda_topm_fast_mode() {
    static const int mode = [] { const char * e = getenv("GGML_CUDA_TOPM_FAST"); return e ? atoi(e) : 1; }();
    return mode;
}

bool ggml_cuda_logits_topm_geometry(int64_t n, int m, int * n_warps, int * n_blocks) {
    if (n <= 0 || n >= (1 << 24) || m < 1 || m > 256) {
        return false;
    }
    for (int nw = 4; nw <= 16; nw *= 2) {
        const int64_t span = (int64_t) nw*TOPM_WARP_SPAN;
        const int64_t nb   = (n + span - 1)/span;
        if (nb <= 32*TOPM_MAX_HEADS && nb*m*(int64_t) sizeof(int2) <= TOPM_SMEM_LISTS) {
            *n_warps  = nw;
            *n_blocks = (int) nb;
            return true;
        }
    }
    return false;
}

void ggml_cuda_op_logits_topm(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    GGML_ASSERT(src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);
    GGML_ASSERT(src0->nb[0] == sizeof(float));

    const int     m    = dst->op_params[0];
    const int64_t n    = src0->ne[0];
    const int64_t rows = src0->ne[1];
    if (rows == 0) {
        return;
    }
    int n_warps = 0, n_blocks = 0;
    GGML_ASSERT(ggml_cuda_logits_topm_geometry(n, m, &n_warps, &n_blocks));
    GGML_ASSERT(rows <= 65535);

    cudaStream_t stream = ctx.stream();
    ggml_cuda_pool_alloc<int2> cand(ctx.pool(), rows*n_blocks*m);
    ggml_cuda_pool_alloc<int>  bad (ctx.pool(), rows*n_blocks);

    const int fast = ggml_cuda_topm_fast_mode();
    ggml_cuda_pool_alloc<int> need(ctx.pool());
    if (fast != 0) {
        need.alloc(rows);                               // before the scoped buffers below (the pool frees LIFO)
        static_assert(TOPM_FAST_G >= 256, "the threshold needs at least M chunk maxima");
        const int chunk = (int) ((n + TOPM_FAST_G - 1)/TOPM_FAST_G);
        ggml_cuda_pool_alloc<int2> bmax(ctx.pool(), rows*TOPM_FAST_G);
        ggml_cuda_pool_alloc<int>  bbad(ctx.pool(), rows*TOPM_FAST_G);
        ggml_cuda_pool_alloc<int>  cnt (ctx.pool(), rows);
        ggml_cuda_pool_alloc<int2> tT  (ctx.pool(), rows);
        ggml_cuda_pool_alloc<int2> fcand(ctx.pool(), rows*TOPM_FAST_CAP);
        const int cap = fast == 2 ? 0 : TOPM_FAST_CAP;
        const dim3 gridf(TOPM_FAST_G, rows, 1);
        k_topm_fast_max<<<gridf, TOPM_FAST_NT, 0, stream>>>((const char *) src0->data, src0->nb[1], (int) n, chunk,
                bmax.get(), bbad.get(), cnt.get());
        k_topm_fast_t<<<rows, TOPM_FAST_T_NT, 0, stream>>>(m, bmax.get(), tT.get());
        k_topm_fast_sel<<<gridf, TOPM_FAST_NT, 0, stream>>>((const char *) src0->data, src0->nb[1], (int) n, chunk,
                tT.get(), cnt.get(), fcand.get(), cap);
        k_topm_fast_sort<<<rows, TOPM_FAST_SORT_NT, 0, stream>>>((int) n, m, cap, bbad.get(), cnt.get(), fcand.get(),
                need.get(), (float *) dst->data, dst->nb[1]);
    }

    const dim3 grid1(n_blocks, rows, 1);
    k_logits_topm_part<<<grid1, n_warps*32, n_warps*m*sizeof(int2), stream>>>(
            (const char *) src0->data, src0->nb[1], (int) n, m, n_blocks, cand.get(), bad.get(), need.get());
    k_logits_topm_merge<<<rows, 32, n_blocks*m*sizeof(int2), stream>>>(
            cand.get(), bad.get(), (int) n, m, n_blocks, (float *) dst->data, dst->nb[1], need.get());
    CUDA_CHECK(cudaGetLastError());
}
