// Persistent MoE-block kernel for one Raon-1 MoE layer at batch 1.
// One normal launch (graph capturable). Blocks pull tasks from an atomic work queue; each task only waits on
// tasks with a smaller index, already claimed by running blocks: no grid barrier, no deadlock when the grid
// exceeds what is co-resident.
//
//   task range   work                                                              waits on
//   [0]          q8_1-quantize x                                                   -
//   [1, 17)      router, 8 rows each; the last one does top-6 and publishes ids/w  -
//   [17, 51)     shared gate+up (Q6_K), 32-row tiles, SiTU, q8_1 of the outputs    xq
//   [51, 255)    routed gate+up (IQ3_KS), 6 experts x 34 tiles                     xq, route
//   [255, 415)   down: 16 rows x (6 IQ4_KS_R16 + 1 Q8_0), weighted sum, writes y   all gate+up
// Readiness counters live in p.sync; the last block to exit resets them, so launches need no memset.
#pragma once
#include "mk_dot.h"

constexpr int MK_THREADS     = 256;
constexpr int MK_WARPS       = MK_THREADS / 32;
constexpr int MK_RT_ROWS     = 8;                       // router rows per task (1 per warp)
constexpr int MK_NT_QX       = 1;
constexpr int MK_NT_ROUTER   = MK_E / MK_RT_ROWS;
constexpr int MK_GU_TILE     = 32;                      // rows per gate/up task (= one q8_1 block of h)
constexpr int MK_GU_TILES    = MK_F / MK_GU_TILE;
constexpr int MK_NT_GU       = MK_NX * MK_GU_TILES;
constexpr int MK_DN_ROWS     = 16;                      // rows per down task (= one R16 group)
constexpr int MK_NT_DOWN     = MK_H / MK_DN_ROWS;
constexpr int MK_T_ROUTER    = MK_NT_QX;
constexpr int MK_T_GU_SH     = MK_T_ROUTER + MK_NT_ROUTER;
constexpr int MK_T_GU_RT     = MK_T_GU_SH + MK_GU_TILES;
constexpr int MK_T_DOWN      = MK_T_GU_SH + MK_NT_GU;
constexpr int MK_NT_TOTAL    = MK_T_DOWN + MK_NT_DOWN;

constexpr int MK_XQ_BLOCKS   = MK_H / 32;
constexpr int MK_HQ_BLOCKS   = MK_F / 32;               // per slot

// staged mode smem: gate+up tile or the down task's 7 weight slices
constexpr int MK_STAGE_GU    = 2 * MK_GU_TILE * IQ3KS_ROW;               // 65408
constexpr int MK_STAGE_DN    = MK_K * IQ4R16_GROUP + MK_DN_ROWS * Q80_ROW; // 74368
constexpr int MK_STAGE_BYTES = MK_STAGE_GU > MK_STAGE_DN ? MK_STAGE_GU : MK_STAGE_DN;

// pipe variant: SY_GUS + slot = gate/up tiles done per slot; SY_EPOCH = launch counter (never reset)
enum { SY_HEAD = 0, SY_XQ, SY_ROUTER, SY_ROUTE, SY_GU, SY_DONE, SY_GUS = 8, SY_EPOCH = 15, SY_COUNT = 16 };

struct MkParams {
    const float   * x;          // [H]
    const float   * router_w;   // [E][H]
    const float   * router_b;   // [E]
    const uint8_t * gate_exps;  // IQ3_KS
    const uint8_t * up_exps;    // IQ3_KS
    const uint8_t * down_exps;  // IQ4_KS_R16
    const uint8_t * gate_sh;    // Q6_K
    const uint8_t * up_sh;      // Q6_K
    const uint8_t * down_sh;    // Q8_0
    size_t gu_exp;              // byte stride between experts, gate/up (tensor nb[2])
    size_t dn_exp;              // byte stride between experts, down (tensor nb[2])
    size_t router_nb1;          // byte stride between router rows
    int   n_exp;                // experts allocated (ids are taken mod n_exp)
    int   norm_w;               // divide the selected weights by their sum
    float beta, lb, w_scale;
    // scratch / outputs
    mk_q8_1  * xq;              // [80]
    float    * logits;          // [E]
    int      * sel_ids;         // [K]
    float    * sel_w;           // [K]
    mk_q8_1  * hq;              // [NX][34]
    float    * y;               // [H]
    unsigned * sync;            // [SY_COUNT], zero before the first launch
    unsigned * diag;            // [MK_DIAG]: first spin-wait timeout (0 = none), cleared by the host
    unsigned long long * lg64;  // pipe variant: [E] router logits as (epoch << 32 | float bits); zero before first use
    unsigned long long * hsh64; // pipe MQ_SHQ: [MK_F] shared-expert SiTU outputs as (epoch << 32 | float bits)
    unsigned long long * ya64;  // pipe MQ_DNSPLIT: [MK_H] down A partials as (epoch << 32 | float bits)
    unsigned long long wait_ns; // spin-wait watchdog (RAON_MK_WAIT_MS overrides MK_WAIT_TIMEOUT)
#ifdef MK_TIMELINE
    unsigned long long * tl;    // optional timeline: [MK_NT_TOTAL][4] = claim ns, wait ns, end ns, block
#endif
};
#ifdef MK_TIMELINE
static __shared__ unsigned long long mk_tl_wait;
constexpr int MP_NT_TOTAL_TL = 462;   // timeline: per-block slots start after the pipe variant's task slots
#endif

// Poll a readiness counter. A volatile load goes to L2 (no stale L1); the __threadfence in mk_wait_ge orders
// the following reads. No .global inline asm: a generic pointer is not a valid .global operand.
__device__ __forceinline__ unsigned mk_ld_acquire(const unsigned * p) {
    const unsigned v = *(const volatile unsigned *) p;
    __threadfence();
    return v;
}
constexpr int      MK_DIAG         = 8;
constexpr uint64_t MK_WAIT_TIMEOUT = 50ull * 1000 * 1000; // ns, far above any legitimate wait
__device__ __forceinline__ uint64_t mk_globaltimer() {
    uint64_t t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}
// Spin until sync[which] >= target. After MK_WAIT_TIMEOUT, record the first timeout in diag and return, so a
// protocol bug gives a wrong result instead of a GPU that spins forever.
__device__ __forceinline__ void mk_wait_ge(const MkParams & p, int which, unsigned target) {
    if (threadIdx.x == 0) {
        const unsigned * c = &p.sync[which];
        unsigned v = mk_ld_acquire(c);
        if (v < target) {
            const uint64_t t0 = mk_globaltimer();
            while ((v = mk_ld_acquire(c)) < target) {
                __nanosleep(64);
                if (mk_globaltimer() - t0 > p.wait_ns) {
                    if (atomicCAS(&p.diag[0], 0u, 1u) == 0u) {
                        p.diag[1] = (unsigned) which; p.diag[2] = v; p.diag[3] = target; p.diag[4] = blockIdx.x;
                        p.diag[5] = mk_ld_acquire(&p.sync[SY_HEAD]); p.diag[6] = mk_ld_acquire(&p.sync[SY_GU]);
                        p.diag[7] = mk_ld_acquire(&p.sync[SY_DONE]);
                    }
                    break;
                }
            }
#ifdef MK_TIMELINE
            mk_tl_wait += mk_globaltimer() - t0;
#endif
        }
        __threadfence();
    }
    __syncthreads();
}
__device__ __forceinline__ void mk_signal(unsigned * p) {
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) { __threadfence(); atomicAdd(p, 1u); }
}
__device__ __forceinline__ float mk_warp_sum(float v) {
#pragma unroll
    for (int m = 16; m > 0; m >>= 1) v += __shfl_xor_sync(0xffffffff, v, m);
    return v;
}
__device__ __forceinline__ float mk_warp_max(float v) {
#pragma unroll
    for (int m = 16; m > 0; m >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, m));
    return v;
}
// one warp quantizes 32 values (lane i holds value i) into one q8_1 block, same math as ik quantize_q8_1
__device__ __forceinline__ void mk_warp_quantize_q8_1(float v, mk_q8_1 * dst) {
    const int lane = threadIdx.x & 31;
    const float amax = mk_warp_max(fabsf(v));
    const float sum  = mk_warp_sum(v);
    const float d = MK_DIV(amax, 127.0f);
    dst->qs[lane] = amax == 0.0f ? 0 : (int8_t)roundf(MK_DIV(v, d));
    if (lane == 0) { dst->d = mk_f2h(d); dst->s = mk_f2h(sum); }
}
// cp.async a contiguous, 16B-aligned byte range into shared memory (all threads participate)
__device__ __forceinline__ void mk_stage(void * s_dst, const void * g_src, int bytes) {
    const uint32_t s0 = (uint32_t)__cvta_generic_to_shared(s_dst);
    // The cp.async source must be a .global address; passing the generic pointer is undefined. Convert explicitly.
    const size_t   g0 = __cvta_generic_to_global(g_src);
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800   // cp.async: sm_80+; the host uses the direct variant below
    for (int off = 16 * threadIdx.x; off < bytes; off += 16 * MK_THREADS) {
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(s0 + off), "l"(g0 + off) : "memory");
    }
    asm volatile("cp.async.commit_group;" ::: "memory");
#else
    (void)s0; (void)g0; (void)bytes; __trap();
#endif
}
__device__ __forceinline__ void mk_stage_wait() {
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
    asm volatile("cp.async.wait_all;" ::: "memory");
#endif
    __syncthreads();
}

struct MkSmem {
    mk_q8_1 xq[MK_XQ_BLOCKS];
    mk_q8_1 hq[MK_NX * MK_HQ_BLOCKS];
    float   g[MK_GU_TILE], u[MK_GU_TILE];
    float   part[MK_NX][MK_DN_ROWS];
    int     ids[MK_K];
    float   w[MK_K];
    int     task;
    int     have;   // bit0 xq, bit1 route, bit2 hq
    int     last;
};

static __device__ void mk_task_qx(const MkParams & p) {
    const int warp = threadIdx.x / 32, lane = threadIdx.x & 31;
    for (int ib = warp; ib < MK_XQ_BLOCKS; ib += MK_WARPS) mk_warp_quantize_q8_1(p.x[32*ib + lane], &p.xq[ib]);
    mk_signal(&p.sync[SY_XQ]);
}

// top-6 from the 128 logits (lane holds logits lane + 32*j): sigmoid + bias, top-6 on the biased scores,
// weights from the unbiased sigmoid, optional normalisation. Lanes < MK_K write ids/w[lane].
__device__ __forceinline__ void mk_topk_core(const MkParams & p, const float (&lg)[MK_E/32], int * ids, float * w) {
    const int lane = threadIdx.x & 31;
    float prob[MK_E/32], sel[MK_E/32];
#pragma unroll
    for (int j = 0; j < MK_E/32; ++j) {
        const int e = lane + 32*j;
        prob[j] = MK_DIV(1.0f, 1.0f + expf(-lg[j]));
        sel[j]  = prob[j] + p.router_b[e];
    }
    int my_id = -1; float my_p = 0.0f, wsum = 0.0f;
    for (int k = 0; k < MK_K; ++k) {
        float bv = -INFINITY; int bi = 1 << 30;
#pragma unroll
        for (int j = 0; j < MK_E/32; ++j) if (sel[j] > bv) { bv = sel[j]; bi = lane + 32*j; }
#pragma unroll
        for (int m = 16; m > 0; m >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffff, bv, m);
            const int   oi = __shfl_xor_sync(0xffffffff, bi, m);
            if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
        }
        const int owner = bi & 31, jj = bi >> 5;
        float pw = 0.0f;
#pragma unroll
        for (int j = 0; j < MK_E/32; ++j) if (lane == owner && j == jj) { pw = prob[j]; sel[j] = -INFINITY; }
        pw = __shfl_sync(0xffffffff, pw, owner);
        wsum += pw;
        if (lane == k) { my_id = bi; my_p = pw; }
    }
    if (lane < MK_K) {
        ids[lane] = my_id % p.n_exp;
        w[lane]   = (p.norm_w ? MK_DIV(my_p, wsum) : my_p) * p.w_scale;
    }
}
// last router task: top-6 from the published logits into p.sel_ids / p.sel_w (warp 0)
__device__ __forceinline__ void mk_router_topk(const MkParams & p) {
    if (threadIdx.x < 32) {
        float lg[MK_E/32];
#pragma unroll
        for (int j = 0; j < MK_E/32; ++j) lg[j] = __ldcg(&p.logits[threadIdx.x + 32*j]);
        mk_topk_core(p, lg, p.sel_ids, p.sel_w);
    }
}

static __device__ void mk_task_router(const MkParams & p, MkSmem & s, int rt) {
    const int warp = threadIdx.x / 32, lane = threadIdx.x & 31;
    const int row = rt * MK_RT_ROWS + warp;
    const float4 * w4 = (const float4 *)((const char *)p.router_w + (size_t)row * p.router_nb1);
    const float4 * x4 = (const float4 *)p.x;
    float acc = 0.0f;
#pragma unroll 4
    for (int i = lane; i < MK_H / 4; i += 32) {
        const float4 a = __ldg(w4 + i), b = __ldg(x4 + i);
        acc += a.x*b.x + a.y*b.y + a.z*b.z + a.w*b.w;
    }
    acc = mk_warp_sum(acc);
    if (lane == 0) p.logits[row] = acc;
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) {
        __threadfence();
        s.last = atomicAdd(&p.sync[SY_ROUTER], 1u) == MK_NT_ROUTER - 1;
        if (s.last) __threadfence();
    }
    __syncthreads();
    if (!s.last) return;
    mk_router_topk(p);
    mk_signal(&p.sync[SY_ROUTE]);
}

__device__ __forceinline__ void mk_load_xq(const MkParams & p, MkSmem & s) {
    if (s.have & 1) return;
    mk_wait_ge(p, SY_XQ, 1);
    const int * src = (const int *)p.xq; int * dst = (int *)s.xq;
    for (int i = threadIdx.x; i < (int)(sizeof(s.xq)/4); i += MK_THREADS) dst[i] = __ldcg(src + i);
    __syncthreads();
    if (threadIdx.x == 0) s.have |= 1;
    __syncthreads();
}
__device__ __forceinline__ void mk_load_route(const MkParams & p, MkSmem & s) {
    if (s.have & 2) return;
    mk_wait_ge(p, SY_ROUTE, 1);
    if (threadIdx.x < MK_K) { s.ids[threadIdx.x] = __ldcg(&p.sel_ids[threadIdx.x]); s.w[threadIdx.x] = __ldcg(&p.sel_w[threadIdx.x]); }
    __syncthreads();
    if (threadIdx.x == 0) s.have |= 2;
    __syncthreads();
}

template <bool STAGED>
static __device__ void mk_task_gu(const MkParams & p, MkSmem & s, uint8_t * stage, int slot, int tile) {
    const int warp = threadIdx.x / 32, lane = threadIdx.x & 31;
    const bool shared = slot == MK_K;
    mk_load_xq(p, s);
    const int r0 = tile * MK_GU_TILE;
    float g[4] = {0, 0, 0, 0}, u[4] = {0, 0, 0, 0};
    if (shared) {
        // Q6_K, 2100-byte rows: kbx = super-block, iqs = lane (ik: qi 32, vdr 1)
        const uint8_t * gb = p.gate_sh + (size_t)(r0 + 4*warp) * Q6K_ROW;
        const uint8_t * ub = p.up_sh   + (size_t)(r0 + 4*warp) * Q6K_ROW;
        for (int kbx = 0; kbx < MK_H / QK_K_; ++kbx) {
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                g[i] += mk_vec_dot_q6_K_q8_1(gb + i*Q6K_ROW, &s.xq[8*kbx], kbx, lane);
                u[i] += mk_vec_dot_q6_K_q8_1(ub + i*Q6K_ROW, &s.xq[8*kbx], kbx, lane);
            }
        }
    } else {
        mk_load_route(p, s);
        const int eid = s.ids[slot];
        const uint8_t * gt = p.gate_exps + (size_t)eid * p.gu_exp + (size_t)r0 * IQ3KS_ROW;
        const uint8_t * ut = p.up_exps   + (size_t)eid * p.gu_exp + (size_t)r0 * IQ3KS_ROW;
        if constexpr (STAGED) {
            mk_stage(stage, gt, MK_GU_TILE * IQ3KS_ROW);
            mk_stage(stage + MK_GU_TILE * IQ3KS_ROW, ut, MK_GU_TILE * IQ3KS_ROW);
            mk_stage_wait();
            gt = stage; ut = stage + MK_GU_TILE * IQ3KS_ROW;
        }
        const uint8_t * gb = gt + (size_t)(4*warp) * IQ3KS_ROW;
        const uint8_t * ub = ut + (size_t)(4*warp) * IQ3KS_ROW;
        // ik mapping: 8 lanes per 256-quant super-block, iqs = 4*(lane%8)
        for (int kbx = lane >> 3; kbx < MK_H / QK_K_; kbx += 4) {
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                mk_vec_dot_iq3_ks_q8_1(gb + i*IQ3KS_ROW, &s.xq[8*kbx], kbx, 4*(lane & 7), &g[i]);
                mk_vec_dot_iq3_ks_q8_1(ub + i*IQ3KS_ROW, &s.xq[8*kbx], kbx, 4*(lane & 7), &u[i]);
            }
        }
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float gs = mk_warp_sum(g[i]), us = mk_warp_sum(u[i]);
        if (lane == 0) { s.g[4*warp + i] = gs; s.u[4*warp + i] = us; }
    }
    __syncthreads();
    if (warp == 0) {
        const float h = mk_situ(s.g[lane], s.u[lane], p.beta, p.lb);
        mk_warp_quantize_q8_1(h, &p.hq[slot * MK_HQ_BLOCKS + tile]);
    }
    mk_signal(&p.sync[SY_GU]);
}

template <bool STAGED>
static __device__ void mk_task_down(const MkParams & p, MkSmem & s, uint8_t * stage, int t) {
    const int warp = threadIdx.x / 32, lane = threadIdx.x & 31;
    if (!(s.have & 4)) {
        mk_wait_ge(p, SY_GU, MK_NT_GU);
        const int * src = (const int *)p.hq; int * dst = (int *)s.hq;
        for (int i = threadIdx.x; i < (int)(sizeof(s.hq)/4); i += MK_THREADS) dst[i] = __ldcg(src + i);
        __syncthreads();
        if (threadIdx.x == 0) s.have |= 4;
        __syncthreads();
    }
    mk_load_route(p, s);   // already published (gate/up waited on it); loads ids/w into this block
    const int r0 = t * MK_DN_ROWS;
    // this warp's weight slice: routed expert `warp` (warps 0..5) or the shared Q8_0 rows (warps 6, 7)
    const uint8_t * gp  = p.down_exps + (size_t)s.ids[warp < MK_K ? warp : 0]*p.dn_exp + (size_t)t*IQ4R16_GROUP;
    const uint8_t * shr = p.down_sh + (size_t)r0 * Q80_ROW;
    if constexpr (STAGED) {
#pragma unroll
        for (int k = 0; k < MK_K; ++k) mk_stage(stage + k*IQ4R16_GROUP, p.down_exps + (size_t)s.ids[k]*p.dn_exp + (size_t)t*IQ4R16_GROUP, IQ4R16_GROUP);
        mk_stage(stage + MK_K*IQ4R16_GROUP, shr, MK_DN_ROWS * Q80_ROW);
        mk_stage_wait();
        gp  = stage + (warp < MK_K ? warp : 0)*IQ4R16_GROUP;
        shr = stage + MK_K*IQ4R16_GROUP;
    }
    if (warp < MK_K) {
        // one warp per routed expert: lane -> (4-row sub-group ir, column half il), stride 4 over the 34 blocks
        const mk_q8_1 * hq = &s.hq[warp * MK_HQ_BLOCKS];
        const int combo = lane & 7, ir = 4 * (combo >> 1), il = combo & 1;
        float res[4] = {0, 0, 0, 0};
        for (int kbx = lane >> 3; kbx < MK_F / 32; kbx += 4) mk_vec_dot_iq4_ks_r16_q8_1(gp, &hq[kbx], kbx, 8*il, ir, res);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            res[j] += __shfl_xor_sync(0xffffffff, res[j], 1);
            res[j] += __shfl_xor_sync(0xffffffff, res[j], 8);
            res[j] += __shfl_xor_sync(0xffffffff, res[j], 16);
        }
        if ((lane & 0x19) == 0) {
#pragma unroll
            for (int j = 0; j < 4; ++j) s.part[warp][ir + j] = res[j];
        }
    } else {
        // shared expert Q8_0: 2 warps x 8 rows, ik mapping iqs = 2*(lane%4), 8 blocks per pass
        const int rl0 = (warp - MK_K) * 8;
        const mk_q8_1 * hq = &s.hq[MK_K * MK_HQ_BLOCKS];
        float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        for (int kbx = lane >> 2; kbx < MK_F / 32; kbx += 8) {
#pragma unroll
            for (int i = 0; i < 8; ++i) acc[i] += mk_vec_dot_q8_0_q8_1(shr + (size_t)(rl0 + i) * Q80_ROW, &hq[kbx], kbx, 2*(lane & 3));
        }
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const float v = mk_warp_sum(acc[i]);
            if (lane == 0) s.part[MK_K][rl0 + i] = v;
        }
    }
    __syncthreads();
    if (threadIdx.x < MK_DN_ROWS) {
        float acc = s.part[MK_K][threadIdx.x];
#pragma unroll
        for (int k = 0; k < MK_K; ++k) acc += s.w[k] * s.part[k][threadIdx.x];
        p.y[r0 + threadIdx.x] = acc;
    }
    // no __syncthreads needed here: the task loop starts with one before smem is reused
}

// MINB: min resident blocks per SM requested from ptxas (caps registers at 65536/(256*MINB))
template <bool STAGED, int MINB>
__global__ void __launch_bounds__(MK_THREADS, MINB) mk_moe_layer_kernel(const MkParams p) {
    extern __shared__ __align__(16) uint8_t mk_stage_buf[];
    __shared__ __align__(16) MkSmem s;
#ifdef MK_TIMELINE
    if (threadIdx.x == 0) p.tl[4*MP_NT_TOTAL_TL + 8*blockIdx.x + 2] = mk_globaltimer();
#endif
    if (threadIdx.x == 0) s.have = 0;
    for (;;) {
        __syncthreads();
        if (threadIdx.x == 0) s.task = (int)atomicAdd(&p.sync[SY_HEAD], 1u);
        __syncthreads();
        const int t = s.task;
        if (t >= MK_NT_TOTAL) break;
#ifdef MK_TIMELINE
        const unsigned long long tl0 = mk_globaltimer();
        if (threadIdx.x == 0) mk_tl_wait = 0;
#endif
        if (t < MK_T_ROUTER)      mk_task_qx(p);
        else if (t < MK_T_GU_SH)  mk_task_router(p, s, t - MK_T_ROUTER);
        else if (t < MK_T_GU_RT)  mk_task_gu<STAGED>(p, s, mk_stage_buf, MK_K, t - MK_T_GU_SH);
        else if (t < MK_T_DOWN)   { const int i = t - MK_T_GU_RT; mk_task_gu<STAGED>(p, s, mk_stage_buf, i / MK_GU_TILES, i % MK_GU_TILES); }
        else                      mk_task_down<STAGED>(p, s, mk_stage_buf, t - MK_T_DOWN);
#ifdef MK_TIMELINE
        __syncthreads();
        if (threadIdx.x == 0) { p.tl[4*t] = tl0; p.tl[4*t+1] = mk_tl_wait; p.tl[4*t+2] = mk_globaltimer(); p.tl[4*t+3] = blockIdx.x; }
#endif
    }
    if (threadIdx.x == 0) {
        __threadfence();
        if (atomicAdd(&p.sync[SY_DONE], 1u) == gridDim.x - 1) {
            for (int i = 0; i < SY_GUS; ++i) atomicExch(&p.sync[i], 0u);   // (SY_GUS.. belong to the pipe variant)
            __threadfence();
        }
    }
#ifdef MK_TIMELINE
    if (threadIdx.x == 0) p.tl[4*MP_NT_TOTAL_TL + 8*blockIdx.x + 3] = mk_globaltimer();
#endif
}
