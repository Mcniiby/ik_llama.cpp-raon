#pragma once
// mHC sublayer boundary for a single token, latency-tuned variant of k_raon_hc_boundary (raon-smallfuse.cu).
// Same math, only the summation order differs. No ggml headers; the dispatch is in raon-hcb1.cu.
// Outputs may alias the inputs: every block reads all its inputs before its release-arrive on the counter, and only
// the last block to arrive writes the outputs.

#include <cuda_fp16.h>
#include <stdint.h>
#include "../ggml-raon-hc.h"

#ifndef HCB_STAMP
#define HCB_STAMP(k)
#endif
#ifndef HCB_CLK
#define HCB_CLK(k)
#endif

struct raon_hcb1_args {
    const float * x; const float * post; const float * res; const float * comb; int64_t res_s1;
    float * hc_out;
    const half * fn; int64_t fn_s1;
    const float * scale; const float * base; float rms_eps; float mag;
    float * mixes; float * all;
    const float * nw; float norm_eps; float * normed;
    int E;
    const half * fn_next;   // next boundary's hc_fn (same shape) to prefetch into L2, or nullptr
};

#define RAON_HCB1_MAX_PART 4096
#define RAON_HCB1_MAX_SPART 1280  // partials staged in the last block's shared memory (PC)
static __device__ float    g_hcb1_partials[RAON_HCB1_MAX_PART];
static __device__ unsigned g_hcb1_counter = 0;

static __device__ __forceinline__ unsigned hcb1_atom_add_acq_rel(unsigned * p, unsigned v) {
    unsigned old;
    asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], %2;" : "=r"(old) : "l"(p), "r"(v) : "memory");
    return old;
}

// Reduce N per-lane values over 32 lanes with a transpose-butterfly instead of 5*N shuffles. Returns this lane's
// total and sets idx to the value index it holds (-1 for padding).
template <int N>
static __device__ __forceinline__ float hcb1_warp_transpose_sum(float (&a)[N], int lane, int & idx) {
    constexpr int H1 = (N + 1)/2, H2 = (H1 + 1)/2, H3 = (H2 + 1)/2, H4 = (H3 + 1)/2, H5 = (H4 + 1)/2;
    static_assert(H5 == 1, "N must be <= 32");
    int n = N; idx = 0;
#define HCB1_STEP(H, NPREV, MASK) {                                                     \
        const bool hi = lane & (MASK);                                                  \
        _Pragma("unroll")                                                               \
        for (int k = 0; k < (H); ++k) {                                                 \
            const float lo_v = a[k];                                                    \
            const float hi_v = (H) + k < (NPREV) ? a[(H) + k] : 0.0f;                   \
            const float send = hi ? lo_v : hi_v;                                        \
            const float keep = hi ? hi_v : lo_v;                                        \
            a[k] = keep + __shfl_xor_sync(0xffffffff, send, (MASK), 32);                \
        }                                                                               \
        if (hi) { idx += (H); n = n - (H) > 0 ? n - (H) : 0; } else { n = n < (H) ? n : (H); } }
    HCB1_STEP(H1, N,  16)
    HCB1_STEP(H2, H1,  8)
    HCB1_STEP(H3, H2,  4)
    HCB1_STEP(H4, H3,  2)
    HCB1_STEP(H5, H4,  1)
#undef HCB1_STEP
    if (n < 1) idx = -1;
    return a[0];
}

template <int VEC> struct hcb1_vec;
template <> struct hcb1_vec<1> {
    static __device__ __forceinline__ void ldf(const float * p, float * o) { o[0] = p[0]; }
    static __device__ __forceinline__ void stf(float * p, const float * v) { p[0] = v[0]; }
    static __device__ __forceinline__ void ldh(const half * p, float * o) { o[0] = __half2float(p[0]); }
};
template <> struct hcb1_vec<2> {
    static __device__ __forceinline__ void ldf(const float * p, float * o) { const float2 t = *(const float2 *) p; o[0] = t.x; o[1] = t.y; }
    static __device__ __forceinline__ void stf(float * p, const float * v) { *(float2 *) p = make_float2(v[0], v[1]); }
    static __device__ __forceinline__ void ldh(const half * p, float * o) { const float2 t = __half22float2(*(const half2 *) p); o[0] = t.x; o[1] = t.y; }
};
template <> struct hcb1_vec<4> {
    static __device__ __forceinline__ void ldf(const float * p, float * o) { const float4 t = *(const float4 *) p; o[0] = t.x; o[1] = t.y; o[2] = t.z; o[3] = t.w; }
    static __device__ __forceinline__ void stf(float * p, const float * v) { *(float4 *) p = make_float4(v[0], v[1], v[2], v[3]); }
    static __device__ __forceinline__ void ldh(const half * p, float * o) {
        const uint2 u = *(const uint2 *) p;
        const float2 a = __half22float2(*(const half2 *) &u.x), b = __half22float2(*(const half2 *) &u.y);
        o[0] = a.x; o[1] = a.y; o[2] = b.x; o[3] = b.y;
    }
};

// raon_hc_helmert for a compile-time K: a lane-dependent row index costs a select instead of sqrt + division
template <int K>
static __device__ __forceinline__ float hcb1_helmert(int i) {
    const float d  = sqrtf((float) ((K + 1)*(K + 2)));
    const float hp = 1.0f/d;
    const float hn = -(float) (K + 1)/d;
    return i <= K ? hp : i == K + 1 ? hn : 0.0f/d;
}

// Block b owns columns [b*CPB, (b+1)*CPB) of all S streams, CPB = NT/S*VEC. The last block to arrive does the chart.
template <int S, int NT, int VEC, bool PC>
static __global__ void __launch_bounds__(NT) k_raon_hcb1(const raon_hcb1_args pin, unsigned long long * stamps) {
    // Tail parameters are pulled into registers here: as constant-cache operands, each cold line would stall
    // the in-order chart warp for an L2 round trip.
    raon_hcb1_args p = pin;
    unsigned nbk = gridDim.x;
    float    * gpart = g_hcb1_partials;
    unsigned * gcnt  = &g_hcb1_counter;
    asm volatile("" : "+l"(p.mixes), "+l"(p.all), "+l"(p.normed), "+l"(gpart), "+l"(gcnt));
    asm volatile("" : "+f"(p.rms_eps), "+f"(p.mag), "+f"(p.norm_eps), "+r"(nbk), "+r"(p.E));
    constexpr int M   = 2*S + (S - 1)*(S - 1);
    constexpr int TPS = NT/S;
    constexpr int NW  = NT/32;
    static_assert(TPS % 32 == 0, "a warp must not straddle two streams");
    (void) stamps;
    HCB_STAMP(0);
    const int E    = p.E;
    const int tid  = threadIdx.x;
    const int lane = tid % 32;
    const int warp = tid / 32;
    const int s    = tid / TPS;
    const int i0   = blockIdx.x*(TPS*VEC) + (tid % TPS)*VEC;
    const bool act = i0 < E;            // E % VEC == 0 (checked by the launcher)
    constexpr int NM   = S - 1;
    // PC keeps warp 0 (the chart) out of the z reads so its shared-memory reads do not queue behind them
    constexpr int CST  = PC ? NT - 32 : NT;  // chunk stride
    constexpr int MAXC = (640 + CST - 1)/CST; // chunks per thread; E <= 4*MAXC*CST checked by the launcher
    const int ct = PC ? tid - 32 : tid;      // < 0: no chunks
    const int E4 = E/4;

    // PC: chart operands and norm weights are loaded up front, off the tail's critical path
    float sc[4], b_own = 0.0f, bz[NM*NM];
    float4 wv[MAXC];
    if (PC) {
        if (warp == 0) {
#pragma unroll
            for (int k = 0; k < 4; ++k) sc[k] = __ldg(p.scale + k);
#pragma unroll
            for (int k = 0; k < NM*NM; ++k) bz[k] = __ldg(p.base + 2*S + k);
            if (lane < 2*S) b_own = __ldg(p.base + lane);
        }
#pragma unroll
        for (int c = 0; c < MAXC; ++c) {
            const int i = ct + c*CST;
            if (ct >= 0 && i < E4) wv[c] = __ldg((const float4 *) p.nw + i);
        }
    }

    // all loads first: fn rows, x, res, post, comb
    float f[M][VEC];
    float xv[VEC], r[S][VEC];
    float y[VEC];
    if (act) {
        const half * fp = p.fn + (int64_t) s*E + i0;
#pragma unroll
        for (int m = 0; m < M; ++m) hcb1_vec<VEC>::ldh(fp + m*p.fn_s1, f[m]);
        hcb1_vec<VEC>::ldf(p.x + i0, xv);
#pragma unroll
        for (int j = 0; j < S; ++j) hcb1_vec<VEC>::ldf(p.res + j*p.res_s1 + i0, r[j]);
        const float ps = p.post[s];
        float cj[S];
#pragma unroll
        for (int j = 0; j < S; ++j) cj[j] = p.comb[j*S + s];
        // hc_post, same expression order as k_hc_post
#pragma unroll
        for (int v = 0; v < VEC; ++v) {
            float sum = xv[v]*ps;
#pragma unroll
            for (int j = 0; j < S; ++j) sum += cj[j]*r[j][v];
            y[v] = sum;
        }
        hcb1_vec<VEC>::stf(p.hc_out + (int64_t) s*E + i0, y);
    } else {
#pragma unroll
        for (int v = 0; v < VEC; ++v) y[v] = 0.0f;
#pragma unroll
        for (int m = 0; m < M; ++m) for (int v = 0; v < VEC; ++v) f[m][v] = 0.0f;
    }
    if (p.fn_next) {
        // warm L2 with the next boundary's weights (evict_last) so they are usually still resident when it runs
        const char * nb = (const char *) p.fn_next;
        const int64_t nlines = ((int64_t) M*p.fn_s1*2 + 127)/128;
        for (int64_t k = (int64_t) blockIdx.x*NT + tid; k < nlines; k += (int64_t) nbk*NT) {
            asm volatile("prefetch.global.L2::evict_last [%0];" :: "l"(nb + k*128));
        }
    }
    HCB_STAMP(1);

    float acc[M + 1];
#pragma unroll
    for (int m = 0; m < M; ++m) {
        float t = 0.0f;
#pragma unroll
        for (int v = 0; v < VEC; ++v) t += f[m][v]*y[v];
        acc[m] = t;
    }
    {
        float t = 0.0f;
#pragma unroll
        for (int v = 0; v < VEC; ++v) t += y[v]*y[v];
        acc[M] = t;
    }
    int idx;
    const float wt = hcb1_warp_transpose_sum<M + 1>(acc, lane, idx);

    __shared__ float s_red[NW][M + 1];
    __shared__ int   s_last;
    if (idx >= 0) s_red[warp][idx] = wt;
    __syncthreads();
    if (tid <= M) {
        float t = 0.0f;
#pragma unroll
        for (int w = 0; w < NW; ++w) t += s_red[w][tid];
        __stcg(&gpart[blockIdx.x*(M + 1) + tid], t);
    }
    // chart lane constants (warp 0, lane = p + i*S): Helmert rows, C entry, identity, gamma, sigmoid scale
    static_assert(NM <= 3, "the Helmert rows below cover S <= 4");
    const int pp = lane % S, ii = lane / S;
    float hi[3], hq[3], cc = 0.0f, idv = 0.0f, gam = 0.0f, sca = 0.0f;
    if (PC) {
        // consume the up-front loads here so the tail does not wait on them
        if (warp == 0) {
#pragma unroll
            for (int k = 0; k < 4; ++k) asm volatile("" :: "f"(sc[k]));
#pragma unroll
            for (int k = 0; k < NM*NM; ++k) asm volatile("" :: "f"(bz[k]));
            asm volatile("" :: "f"(b_own));
            hi[0] = hcb1_helmert<0>(ii); hi[1] = hcb1_helmert<1>(ii); hi[2] = hcb1_helmert<2>(ii);
            hq[0] = hcb1_helmert<0>(pp); hq[1] = hcb1_helmert<1>(pp); hq[2] = hcb1_helmert<2>(pp);
            cc  = (pp == ii ? 0.5f : 0.0f) + 0.5f/S;
            idv = pp == ii ? 1.0f : 0.0f;
            gam = raon_hc_sigmoid(sc[3]);
            sca = lane < S ? sc[0] : sc[1];
            asm volatile("" : "+f"(hi[0]), "+f"(hi[1]), "+f"(hi[2]), "+f"(hq[0]), "+f"(hq[1]), "+f"(hq[2]));
            asm volatile("" : "+f"(cc), "+f"(idv), "+f"(gam), "+f"(sca));
        }
#pragma unroll
        for (int c = 0; c < MAXC; ++c) {
            if (ct >= 0 && ct + c*CST < E4) asm volatile("" :: "f"(wv[c].x), "f"(wv[c].y), "f"(wv[c].z), "f"(wv[c].w));
        }
    }
    HCB_STAMP(2);
    __syncthreads();
    if (tid == 0) {
        s_last = hcb1_atom_add_acq_rel(gcnt, 1u) == nbk - 1;
    }
    __syncthreads();
    HCB_STAMP(3);
    if (!s_last) return;
    HCB_CLK(0);

    // last block. PC: the partials are read first, then the stream reads for z are issued so they overlap the
    // totals and the chart.
    __shared__ float s_part[RAON_HCB1_MAX_SPART];
    const int npart = nbk*(M + 1);
    if (PC) {
        for (int k = tid; k < npart; k += NT) s_part[k] = __ldcg(&gpart[k]);
    }
    float4 yv[MAXC][S];
    auto load_yv = [&]() {
#pragma unroll
        for (int c = 0; c < MAXC; ++c) {
            const int i = ct + c*CST;
            if (ct >= 0 && i < E4) {
#ifndef HCB1_DIAG_NO_YV
#pragma unroll
                for (int k = 0; k < S; ++k) yv[c][k] = __ldcg((const float4 *) (p.hc_out + (int64_t) k*E) + i);
#else
#pragma unroll
                for (int k = 0; k < S; ++k) yv[c][k] = make_float4(0.f, 0.f, 0.f, 0.f);
#endif
                if (!PC) wv[c] = __ldg((const float4 *) p.nw + i);
            }
        }
    };
#ifndef HCB1_DIAG_YV_AFTER
    load_yv();
#endif
    HCB_CLK(1);
    __shared__ float s_tot[M + 1];
    __shared__ float s_pre[S];
    __shared__ float s_sum[NW];
    if (PC) {
        // warp 0: totals in block order, then the chart spread over S*S lanes, with the same per-element
        // operation order as raon_hc_coeffs
        HCB_CLK(2);
        __syncthreads();
        HCB_CLK(3);
        if (warp == 0) {
            // totals in a fixed order (4 interleaved partial sums), broadcast by shuffles
            float t = 0.0f;
            if (lane <= M) {
                float t0 = 0.0f, t1 = 0.0f, t2 = 0.0f, t3 = 0.0f;
                int b = 0;
                for (; b + 4 <= (int) nbk; b += 4) {
                    t0 += s_part[(b + 0)*(M + 1) + lane]; t1 += s_part[(b + 1)*(M + 1) + lane];
                    t2 += s_part[(b + 2)*(M + 1) + lane]; t3 += s_part[(b + 3)*(M + 1) + lane];
                }
                for (; b < (int) nbk; ++b) t0 += s_part[b*(M + 1) + lane];
                t = (t0 + t1) + (t2 + t3);
                if (lane < M) p.mixes[lane] = t;
            }
            if (lane == 0) *gcnt = 0;
            HCB_CLK(4);
            HCB_STAMP(4);
            const float ssq = __shfl_sync(0xffffffff, t, M, 32);
            const float own = __shfl_sync(0xffffffff, t, lane < 2*S ? lane : 0, 32);
            float zc[NM*NM];
#pragma unroll
            for (int k = 0; k < NM*NM; ++k) zc[k] = __shfl_sync(0xffffffff, t, 2*S + k, 32);
            const float rr = p.rms_eps >= 0.0f ? rsqrtf(ssq/(E*S) + p.rms_eps) : 1.0f;
            // pre (lanes < S) and post (S <= lane < 2S) share one sigmoid evaluation
            const float sg = raon_hc_sigmoid((own*rr)*sca + b_own);
#pragma unroll
            for (int k = 0; k < NM*NM; ++k) zc[k] = (zc[k]*rr)*sc[2] + bz[k];
            float e1[NM];
#pragma unroll
            for (int n = 0; n < NM; ++n) {
                float sum = 0.0f;
#pragma unroll
                for (int k = 0; k < NM; ++k) sum += hi[k]*zc[k + n*NM];
                e1[n] = sum;
            }
            float e = 0.0f;
#pragma unroll
            for (int k = 0; k < NM; ++k) e += hq[k]*e1[k];
            float rho = lane < S*S ? -(e/cc) : -INFINITY;
#pragma unroll
            for (int m = 16; m > 0; m >>= 1) {
                const float o = __shfl_xor_sync(0xffffffff, rho, m, 32);
                rho = o > rho ? o : rho;
            }
            rho = (rho > 0.0f ? rho : 0.0f) + 1.0f;
            if (lane < S) {
                s_pre[lane] = sg;
            } else if (lane < 2*S) {
                p.all[E + lane - S] = p.mag*sg;
            }
            if (lane < S*S) {
                const float phi = e/rho + cc;
                p.all[E + S + pp + ii*S] = (phi - idv)*gam + idv;
            }
            HCB_CLK(5);
        }
        __syncthreads();
        HCB_CLK(6);
    } else {
        if (tid <= M) {
            // totals in block order
            float t = 0.0f;
            for (int b = 0; b < (int) nbk; ++b) t += __ldcg(&gpart[b*(M + 1) + tid]);
            s_tot[tid] = t;
            if (tid < M) p.mixes[tid] = t;
        }
        if (tid == 0) *gcnt = 0;
        __syncthreads();
        HCB_STAMP(4);
        if (tid == 0) {
            const float rr = p.rms_eps >= 0.0f ? rsqrtf(s_tot[M]/(E*S) + p.rms_eps) : 1.0f;
            float pre[S], pst[S], cmb[S*S];
            raon_hc_coeffs(S, s_tot, rr, p.scale, p.base, p.mag, pre, pst, cmb);
    #pragma unroll
            for (int k = 0; k < S; ++k) { s_pre[k] = pre[k]; p.all[E + k] = pst[k]; }
    #pragma unroll
            for (int k = 0; k < S*S; ++k) p.all[E + S + k] = cmb[k];
        }
        __syncthreads();
    }
#ifdef HCB1_DIAG_YV_AFTER
    load_yv();
#endif
    HCB_STAMP(5);
    float w[S];
#pragma unroll
    for (int k = 0; k < S; ++k) w[k] = s_pre[k];
    float4 z[MAXC];
    float ss = 0.0f;
#pragma unroll
    for (int c = 0; c < MAXC; ++c) {
        const int i = ct + c*CST;
        z[c] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        if (ct >= 0 && i < E4) {
            float4 t = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
#pragma unroll
            for (int k = 0; k < S; ++k) {
                t.x += yv[c][k].x*w[k]; t.y += yv[c][k].y*w[k]; t.z += yv[c][k].z*w[k]; t.w += yv[c][k].w*w[k];
            }
            z[c] = t;
            ((float4 *) p.all)[i] = t;
            ss += t.x*t.x + t.y*t.y + t.z*t.z + t.w*t.w;
        }
    }
#pragma unroll
    for (int m = 16; m > 0; m >>= 1) ss += __shfl_xor_sync(0xffffffff, ss, m, 32);
    if (lane == 0) s_sum[warp] = ss;
    __syncthreads();
    ss = 0.0f;
#pragma unroll
    for (int k = 0; k < NW; ++k) ss += s_sum[k];
    const float rs = rsqrtf(ss/E + p.norm_eps);
    HCB_CLK(7);
    HCB_STAMP(6);
#pragma unroll
    for (int c = 0; c < MAXC; ++c) {
        const int i = ct + c*CST;
        if (ct >= 0 && i < E4) {
            ((float4 *) p.normed)[i] = make_float4(rs*wv[c].x*z[c].x, rs*wv[c].y*z[c].y, rs*wv[c].z*z[c].z, rs*wv[c].w*z[c].w);
        }
    }
    HCB_STAMP(7);
}

template <int S, int NT, int VEC, bool PC>
static bool raon_hcb1_launch_t(const raon_hcb1_args & a, cudaStream_t st, unsigned long long * stamps) {
    constexpr int M = 2*S + (S - 1)*(S - 1);
    constexpr int CPB = NT/S*VEC;
    const int nb = (a.E + CPB - 1)/CPB;
    if (a.E <= 0 || a.E % VEC != 0 || a.E % 4 != 0 || a.E > 4*((640 + (PC ? NT - 32 : NT) - 1)/(PC ? NT - 32 : NT))*(PC ? NT - 32 : NT) || nb*(M + 1) > (PC ? RAON_HCB1_MAX_SPART : RAON_HCB1_MAX_PART)) return false;
    k_raon_hcb1<S, NT, VEC, PC><<<nb, NT, 0, st>>>(a, stamps);
    return true;
}

// variant: threads x columns per thread, 1 = 512x1, 2 = 256x1, 3 = 128x1, 4 = 256x2, 5 = 512x2, 6 = 128x2, 7 = 256x4
template <bool PCV>
static bool raon_hcb1_launch_v(const raon_hcb1_args & a, int variant, cudaStream_t st, unsigned long long * stamps) {
    switch (variant) {
        case 1: return raon_hcb1_launch_t<4, 512, 1, PCV>(a, st, stamps);
        case 2: return raon_hcb1_launch_t<4, 256, 1, PCV>(a, st, stamps);
        case 3: return raon_hcb1_launch_t<4, 128, 1, PCV>(a, st, stamps);
        case 4: return raon_hcb1_launch_t<4, 256, 2, PCV>(a, st, stamps);
        case 5: return raon_hcb1_launch_t<4, 512, 2, PCV>(a, st, stamps);
        case 6: return raon_hcb1_launch_t<4, 128, 2, PCV>(a, st, stamps);
        case 7: return raon_hcb1_launch_t<4, 256, 4, PCV>(a, st, stamps);
        default: return false;
    }
}

// variant + 10: same layout with the warp-parallel chart and up-front operand loads (PC)
static bool raon_hcb1_launch(const raon_hcb1_args & a, int variant, cudaStream_t st, unsigned long long * stamps) {
    if (variant > 20) return false;
    return variant > 10 ? raon_hcb1_launch_v<true>(a, variant - 10, st, stamps) : raon_hcb1_launch_v<false>(a, variant, st, stamps);
}
