// Pipelined, warp-specialised variant of the persistent MoE-block kernel (RAON_MK_MOE_VARIANT=pipe, the default).
// Same math as mk_kernel.cuh; some float partial sums are grouped differently, in a fixed order (deterministic).
//
// Per block, a producer warp claims tasks and streams their weights in chunks (<= MP_CH bytes) into a ring of
// MP_NST smem stages with cp.async.bulk ("full" mbarrier); MP_WARPS consumer warps compute and arrive on "empty",
// syncing with each other only at task boundaries (named barrier 1).
//
//   task range   chunks  work                                                   waits on
//   [0, 64)      1       router: 2 rows of the F32 2560x128 matvec              -
//   [64, 98)     8       shared gate+up (Q6_K), 32-row tile                     -
//   [98, 302)    4       routed gate+up (IQ3_KS), 6 x 34 tiles                  route
//   [302, 462)   4       down, 16 rows: shared Q8_0, experts {0,1},{2,3},{4,5}  route, gate/up of its slots
// Router logits are published epoch-tagged and every block's producer runs top-6 itself. Dependencies only point
// to smaller task indices and blocks finish tasks in claim order, so the queue cannot deadlock even if the grid is
// not co-resident. Every wait has the 50 ms watchdog (diag + abort flag).
#pragma once
#include "mk_kernel.cuh"

// cp.async.bulk / mbarrier try_wait.parity / expect_tx / fence.mbarrier_init need sm_90+. Older device passes compile
// a stub that traps; raon-moe.cu falls back to the staged variant on 8.x and the direct one below that.
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 900
#define MP_ARCH_OK 1
#else
#define MP_ARCH_OK 0
#endif

#ifndef MP_WARPS_N
#define MP_WARPS_N 16
#endif
constexpr int MP_WARPS      = MP_WARPS_N;              // consumer warps
constexpr int MP_CTHREADS   = 32 * MP_WARPS;
constexpr int MP_THREADS    = 64 + MP_CTHREADS;        // + producer warp + finisher warp
static_assert(MP_WARPS == 16, "the chunk mappings below assume 16 consumer warps");
constexpr int MP_NST        = 4;
constexpr int MP_CH         = 20480;
constexpr int MP_SMEM       = MP_NST * MP_CH;          // 81920 B dynamic
constexpr int MP_RT_ROWS    = 2;
constexpr int MP_NT_ROUTER  = MK_E / MP_RT_ROWS;
constexpr int MP_T_GU_SH    = MP_NT_ROUTER;
constexpr int MP_T_GU_RT    = MP_T_GU_SH + MK_GU_TILES;
constexpr int MP_T_DOWN     = MP_T_GU_RT + MK_K * MK_GU_TILES;
constexpr int MP_NT_TOTAL   = MP_T_DOWN + MK_NT_DOWN;  // 462
constexpr int MP_SH_ROWS    = 4;                       // shared gate+up rows per chunk (x2 matrices)
constexpr int MP_RT_GROWS   = 8;                       // routed gate+up rows per chunk (x2 matrices)
static_assert(MP_RT_ROWS * MK_H * 4 <= MP_CH, "router chunk");
static_assert(2 * MP_SH_ROWS * Q6K_ROW <= MP_CH && (MP_SH_ROWS * Q6K_ROW) % 16 == 0, "shared gu chunk");
static_assert(2 * MP_RT_GROWS * IQ3KS_ROW <= MP_CH && (MP_RT_GROWS * IQ3KS_ROW) % 16 == 0, "routed gu chunk");
static_assert(2 * IQ4R16_GROUP <= MP_CH && IQ4R16_GROUP % 16 == 0 && (MK_DN_ROWS * Q80_ROW) % 16 == 0, "down chunk");
static_assert(MP_CH % 128 == 0, "stage alignment");
constexpr int MP_SH_PARTS   = MP_WARPS / (2 * MP_SH_ROWS);   // 2 warps per shared gate/up row (kbx split)
constexpr int MP_RT_PARTS   = MP_WARPS / (2 * MP_RT_GROWS);  // 1 warp per routed gate/up row
constexpr int MP_GU_PARTS   = MP_SH_PARTS > MP_RT_PARTS ? MP_SH_PARTS : MP_RT_PARTS;
constexpr int MP_DN_SUB     = MP_WARPS / 2;                  // 8 warps per routed expert in a down chunk
constexpr int MP_SHD_PARTS  = MP_WARPS / MK_DN_ROWS;         // 1 warp per shared down row

struct MpSmem {
    unsigned long long full[MP_NST], empty[MP_NST];          // mbarriers
    mk_q8_1 xq[MK_XQ_BLOCKS];
    mk_q8_1 hq[MK_NX * MK_HQ_BLOCKS];
    // per-task partial sums, double-buffered by consumer task parity (dpart is single: see mp_consumers)
    float   g[2][MP_GU_PARTS][MK_GU_TILE], u[2][MP_GU_PARTS][MK_GU_TILE];
    float   rpart[2][MP_WARPS];
    float   dpart[MK_K][MP_DN_SUB][MK_DN_ROWS];
    float   shpart[2][MP_SHD_PARTS][MK_DN_ROWS];
    float   dpartB[2][2][MP_DN_SUB][MK_DN_ROWS];   // MQ_DNSPLIT B tasks (experts 4, 5), double-buffered
    float   apart[2][MK_DN_ROWS];                  // MQ_DNSPLIT B tasks: the A partial of the row group
    unsigned long long done[2];  // mbarriers: consumers -> finisher, one task each
    int     fin_task[2];
    int     fin_read;            // tasks whose partials the finisher has read (volatile)
    int     ids[MK_K];
    float   w[MK_K];
    int     st_task[MP_NST];    // task of the chunk in each stage (-1: no more work)
    int     st_chunk[MP_NST];
    int     abort;              // a watchdog fired: everybody stops
};

// Kernel config bits (template CFG; host switches in raon-moe.cu):
//   MQ_SAFE     abort path that cannot hang: warp-uniform waits, bar.red exit loop, the producer drains its bulk
//               copies before exiting (otherwise they land in smem the next block owns)
//   MQ_SHQ      shared gate/up as 136 tasks of 8 rows; SiTU outputs published epoch-tagged (hsh64)
//   MQ_DNSPLIT  down as A (shared + experts 0-3) then B (experts 4, 5) tasks; y = B + A (ya64)
//   MQ_LANEARR  every consumer lane arrives on "empty" (a form racecheck can follow)
//   MQ_RTL2     router rows evict_last, other weights evict_first, so router weights stay in L2
constexpr int MQ_SAFE = 1, MQ_SHQ = 2, MQ_DNSPLIT = 4, MQ_LANEARR = 8, MQ_RTL2 = 16;
template <int CFG> struct MpL {
    static constexpr bool SHQ = (CFG & MQ_SHQ) != 0, DNS = (CFG & MQ_DNSPLIT) != 0;
    static constexpr int SH_TROWS = SHQ ? 2 * MP_SH_ROWS : MK_GU_TILE;    // rows per shared gate/up task
    static constexpr int NT_SH = MK_F / SH_TROWS;
    static constexpr int T_SH  = MP_NT_ROUTER;
    static constexpr int T_GU  = T_SH + NT_SH;
    static constexpr int T_DN  = T_GU + MK_K * MK_GU_TILES;
    static constexpr int T_DNB = T_DN + MK_NT_DOWN;                         // DNS: first B task
    static constexpr int NT    = T_DN + (DNS ? 2 : 1) * MK_NT_DOWN;
};
static_assert(MpL<0>::T_GU == MP_T_GU_RT && MpL<0>::T_DN == MP_T_DOWN && MpL<0>::NT == MP_NT_TOTAL, "cfg 0 layout");
static_assert(MK_F % (2 * MP_SH_ROWS) == 0, "shared eighths");
constexpr int MP_TL_TASKS = 1024;   // MK_TIMELINE: task slots before the per-block slots

template <int CFG> __device__ __forceinline__ int mp_nchunks(int t) {
    using L = MpL<CFG>;
    return t < L::T_SH ? 1 : t < L::T_GU ? L::SH_TROWS / MP_SH_ROWS : t < L::T_DN ? MK_GU_TILE / MP_RT_GROWS
         : !L::DNS ? 4 : t < L::T_DNB ? 3 : 1;
}
// down chunk kind: 0 shared Q8_0, 1..3 routed experts {0,1}, {2,3}, {4,5}
template <int CFG> __device__ __forceinline__ int mp_dk(int t, int c) { return MpL<CFG>::DNS && t >= MpL<CFG>::T_DNB ? 3 : c; }
template <int CFG> __device__ __forceinline__ int mp_dgroup(int t) { return (t - MpL<CFG>::T_DN) % MK_NT_DOWN; }

__device__ __forceinline__ uint32_t mp_sa(const void * p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void mp_mbar_init(unsigned long long * b, unsigned count) {
#if MP_ARCH_OK
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(mp_sa(b)), "r"(count) : "memory");
#else
    __trap();
#endif
}
__device__ __forceinline__ void mp_mbar_arrive(unsigned long long * b) {
#if MP_ARCH_OK
    asm volatile("{\n\t.reg .b64 st;\n\tmbarrier.arrive.shared::cta.b64 st, [%0];\n\t}" :: "r"(mp_sa(b)) : "memory");
#else
    __trap();
#endif
}
__device__ __forceinline__ void mp_mbar_arrive_tx(unsigned long long * b, unsigned tx) {
#if MP_ARCH_OK
    asm volatile("{\n\t.reg .b64 st;\n\tmbarrier.arrive.expect_tx.shared::cta.b64 st, [%0], %1;\n\t}" :: "r"(mp_sa(b)), "r"(tx) : "memory");
#else
    __trap();
#endif
}
__device__ __forceinline__ bool mp_mbar_try(unsigned long long * b, unsigned parity) {
#if MP_ARCH_OK
    uint32_t ok;
    asm volatile("{\n\t.reg .pred p;\n\tmbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n\tselp.u32 %0, 1, 0, p;\n\t}"
                 : "=r"(ok) : "r"(mp_sa(b)), "r"(parity) : "memory");
    return ok != 0;
#else
    __trap(); return false;
#endif
}
// bulk copy global -> shared, completion counted on mbarrier b (16-byte aligned). The .shared::cta form on purpose:
// the .shared::cluster form makes the driver reserve ~1.4 GB of device memory per context; we launch no clusters.
__device__ __forceinline__ void mp_bulk(uint8_t * s_dst, const void * g_src, unsigned bytes, unsigned long long * b) {
#if MP_ARCH_OK
    asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                 :: "r"(mp_sa(s_dst)), "l"(__cvta_generic_to_global(g_src)), "r"(bytes), "r"(mp_sa(b)) : "memory");
#else
    __trap();
#endif
}
// the same copy with an L2 eviction-priority policy (createpolicy)
__device__ __forceinline__ void mp_bulk_pol(uint8_t * s_dst, const void * g_src, unsigned bytes, unsigned long long * b, uint64_t pol) {
#if MP_ARCH_OK
    asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;"
                 :: "r"(mp_sa(s_dst)), "l"(__cvta_generic_to_global(g_src)), "r"(bytes), "r"(mp_sa(b)), "l"(pol) : "memory");
#else
    __trap();
#endif
}
__device__ __forceinline__ uint64_t mp_policy(bool last) {
    uint64_t pol = 0;
#if MP_ARCH_OK
    if (last) asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;"  : "=l"(pol));
    else      asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pol));
#endif
    return pol;
}
// consumer-only barriers (the producer warp never takes part)
__device__ __forceinline__ void mp_csync()   { asm volatile("bar.sync 1, %0;"   :: "n"(MP_CTHREADS) : "memory"); }
// bar.red.and over the consumer threads: true iff every consumer thread passed v = true
__device__ __forceinline__ bool mp_csync_and(bool v) {
    uint32_t r;
    asm volatile("{\n\t.reg .pred q, p;\n\tsetp.ne.u32 q, %1, 0;\n\tbar.red.and.pred p, 1, %2, q;\n\tselp.u32 %0, 1, 0, p;\n\t}"
                 : "=r"(r) : "r"((uint32_t)v), "n"(MP_CTHREADS) : "memory");
    return r != 0;
}
template <int CFG> __device__ __forceinline__ void mp_csync_c() {
    if constexpr (CFG & MQ_SAFE) (void)mp_csync_and(false); else mp_csync();
}
// a wait result the whole warp agrees on (MQ_SAFE): if any lane gave up, every lane does
template <int CFG> __device__ __forceinline__ bool mp_uni(bool ok) {
    if constexpr (CFG & MQ_SAFE) return __all_sync(0xffffffff, ok); else return ok;
}
__device__ __forceinline__ void mp_fsync()   { asm volatile("bar.sync 2, %0;"   :: "n"(MP_CTHREADS) : "memory"); }
__device__ __forceinline__ void mp_farrive() { asm volatile("bar.arrive 2, %0;" :: "n"(MP_CTHREADS) : "memory"); }

// record a watchdog timeout (first one only) and make the whole block stop
__device__ __forceinline__ void mp_timeout(const MkParams & p, MpSmem & s, unsigned which, unsigned seen, unsigned target) {
    if (atomicCAS(&p.diag[0], 0u, 1u) == 0u) {
        p.diag[1] = which; p.diag[2] = seen; p.diag[3] = target; p.diag[4] = blockIdx.x;
        p.diag[5] = *(const volatile unsigned *)&p.sync[SY_HEAD]; p.diag[6] = 0; p.diag[7] = 0;
    }
    *(volatile int *)&s.abort = 1;
}
// wait for an mbarrier phase with the watchdog; returns false if the block is aborting
__device__ __forceinline__ bool mp_mbar_wait(const MkParams & p, MpSmem & s, unsigned long long * b, unsigned parity, unsigned which) {
    if (mp_mbar_try(b, parity)) return true;
    const uint64_t t0 = mk_globaltimer();
    while (!mp_mbar_try(b, parity)) {
        if (*(volatile int *)&s.abort) return false;
        if (mk_globaltimer() - t0 > p.wait_ns) { mp_timeout(p, s, which, 0, parity); return false; }
    }
    return true;
}

__device__ __forceinline__ void mp_router_chunk(MpSmem & s, const MkParams & p, const uint8_t * st, int cw, int lane, int q) {
    constexpr int WPR = MP_WARPS / MP_RT_ROWS;          // 8 warps per row
    constexpr int F4  = MK_H / 4 / WPR;
    const float4 * w4 = (const float4 *)(st + (cw / WPR) * MK_H * 4) + (cw % WPR) * F4;
    const float4 * x4 = (const float4 *)p.x + (cw % WPR) * F4;
    float acc = 0.0f;
#pragma unroll
    for (int i = lane; i < F4; i += 32) {
        const float4 a = w4[i], b = __ldg(x4 + i);
        acc += a.x*b.x + a.y*b.y + a.z*b.z + a.w*b.w;
    }
    acc = mk_warp_sum(acc);
    if (lane == 0) s.rpart[q][cw] = acc;
}

__device__ __forceinline__ void mp_gu_sh_chunk(MpSmem & s, const uint8_t * st, int c, int cw, int lane, int q) {
    const int rm = cw / MP_SH_PARTS, part = cw % MP_SH_PARTS;      // rm: 2 matrices x 4 rows
    const int mat = rm / MP_SH_ROWS, r = rm % MP_SH_ROWS;
    const uint8_t * row = st + mat * MP_SH_ROWS * Q6K_ROW + r * Q6K_ROW;
    float acc = 0.0f;
#pragma unroll
    for (int j = 0; j < (MK_H / QK_K_ + MP_SH_PARTS - 1) / MP_SH_PARTS; ++j) {
        const int kbx = part + j * MP_SH_PARTS;
        if (kbx < MK_H / QK_K_) acc += mk_vec_dot_q6_K_q8_1(row, &s.xq[8*kbx], kbx, lane);
    }
    acc = mk_warp_sum(acc);
    if (lane == 0) (mat ? s.u : s.g)[q][part][c * MP_SH_ROWS + r] = acc;
}

__device__ __forceinline__ void mp_gu_rt_chunk(MpSmem & s, const uint8_t * st, int c, int cw, int lane, int q) {
    const int rm = cw / MP_RT_PARTS, part = cw % MP_RT_PARTS;      // rm: 2 matrices x 8 rows
    const int mat = rm / MP_RT_GROWS, r = rm % MP_RT_GROWS;
    const uint8_t * row = st + (mat * MP_RT_GROWS + r) * IQ3KS_ROW;
    float acc = 0.0f;
    // ik mapping: 8 lanes per 256-quant super-block, iqs = 4*(lane%8)
#pragma unroll
    for (int j = 0; j < (MK_H / QK_K_ + 4*MP_RT_PARTS - 1) / (4*MP_RT_PARTS); ++j) {
        const int kbx = (lane >> 3) + 4*part + j * 4*MP_RT_PARTS;
        if (kbx < MK_H / QK_K_) mk_vec_dot_iq3_ks_q8_1(row, &s.xq[8*kbx], kbx, 4*(lane & 7), &acc);
    }
    acc = mk_warp_sum(acc);
    if (lane == 0) (mat ? s.u : s.g)[q][part][c * MP_RT_GROWS + r] = acc;
}

// c: down chunk kind (mp_dk). bq: B task of MQ_DNSPLIT (experts 4, 5 into dpartB[q])
__device__ __forceinline__ void mp_down_chunk(MpSmem & s, const uint8_t * st, int c, int cw, int lane, int q, bool bq = false) {
    if (c > 0) {
        // MP_DN_SUB warps per routed expert; within a warp lane -> (4-row sub-group ir, column half il)
        const int e = 2*(c-1) + cw / MP_DN_SUB, sub = cw % MP_DN_SUB;
        const uint8_t * gp = st + (cw / MP_DN_SUB) * IQ4R16_GROUP;
        const mk_q8_1 * hq = &s.hq[e * MK_HQ_BLOCKS];
        const int combo = lane & 7, ir = 4 * (combo >> 1), il = combo & 1;
        float res[4] = {0, 0, 0, 0};
#pragma unroll
        for (int j = 0; j < (MK_F / 32 + 4*MP_DN_SUB - 1) / (4*MP_DN_SUB); ++j) {
            const int kbx = (lane >> 3) + 4*sub + j * 4*MP_DN_SUB;
            if (kbx < MK_F / 32) mk_vec_dot_iq4_ks_r16_q8_1(gp, &hq[kbx], kbx, 8*il, ir, res);
        }
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            res[j] += __shfl_xor_sync(0xffffffff, res[j], 1);
            res[j] += __shfl_xor_sync(0xffffffff, res[j], 8);
            res[j] += __shfl_xor_sync(0xffffffff, res[j], 16);
        }
        if ((lane & 0x19) == 0) {
#pragma unroll
            for (int j = 0; j < 4; ++j) (bq ? s.dpartB[q][e - 4] : s.dpart[e])[sub][ir + j] = res[j];
        }
    } else {
        // shared expert Q8_0: one warp per row, ik mapping iqs = 2*(lane%4), 8 blocks per pass
        const mk_q8_1 * hq = &s.hq[MK_K * MK_HQ_BLOCKS];
        const int r = cw / MP_SHD_PARTS, part = cw % MP_SHD_PARTS;
        float acc = 0.0f;
#pragma unroll
        for (int j = 0; j < (MK_F / 32 + 8*MP_SHD_PARTS - 1) / (8*MP_SHD_PARTS); ++j) {
            const int kbx = (lane >> 2) + 8*part + j * 8*MP_SHD_PARTS;
            if (kbx < MK_F / 32) acc += mk_vec_dot_q8_0_q8_1(st + (size_t)r * Q80_ROW, &hq[kbx], kbx, 2*(lane & 3));
        }
        acc = mk_warp_sum(acc);
        if (lane == 0) s.shpart[q][part][r] = acc;
    }
}

// returns the bytes of chunk c of task t and issues its bulk copies (lane 0 only)
template <int CFG> __device__ __forceinline__ unsigned mp_chunk_bytes(int t, int c) {
    using L = MpL<CFG>;
    if (t < L::T_SH)  return MP_RT_ROWS * MK_H * 4;
    if (t < L::T_GU)  return 2 * MP_SH_ROWS * Q6K_ROW;
    if (t < L::T_DN)  return 2 * MP_RT_GROWS * IQ3KS_ROW;
    return mp_dk<CFG>(t, c) > 0 ? 2 * IQ4R16_GROUP : MK_DN_ROWS * Q80_ROW;
}
template <int CFG>
__device__ __forceinline__ void mp_issue_chunk(const MkParams & p, const MpSmem & s, uint8_t * st, unsigned long long * bar, int t, int c) {
    using L = MpL<CFG>;
    auto mp_bulk = [&](uint8_t * d, const void * g, unsigned n, unsigned long long * b) {
        if constexpr (CFG & MQ_RTL2) mp_bulk_pol(d, g, n, b, mp_policy(t < L::T_SH));
        else ::mp_bulk(d, g, n, b);
    };
    if (t < L::T_SH) {
        const int row = t * MP_RT_ROWS;
        for (int r = 0; r < MP_RT_ROWS; ++r) mp_bulk(st + r * MK_H * 4, (const char *)p.router_w + (size_t)(row + r) * p.router_nb1, MK_H * 4, bar);
    } else if (t < L::T_GU) {
        const size_t off = (size_t)((t - L::T_SH) * L::SH_TROWS + c * MP_SH_ROWS) * Q6K_ROW;
        mp_bulk(st,                        p.gate_sh + off, MP_SH_ROWS * Q6K_ROW, bar);
        mp_bulk(st + MP_SH_ROWS * Q6K_ROW, p.up_sh   + off, MP_SH_ROWS * Q6K_ROW, bar);
    } else if (t < L::T_DN) {
        const int i = t - L::T_GU, slot = i / MK_GU_TILES, tile = i % MK_GU_TILES;
        const size_t off = (size_t)s.ids[slot] * p.gu_exp + (size_t)(tile * MK_GU_TILE + c * MP_RT_GROWS) * IQ3KS_ROW;
        mp_bulk(st,                           p.gate_exps + off, MP_RT_GROWS * IQ3KS_ROW, bar);
        mp_bulk(st + MP_RT_GROWS * IQ3KS_ROW, p.up_exps   + off, MP_RT_GROWS * IQ3KS_ROW, bar);
    } else {
        const int d = mp_dgroup<CFG>(t), dk = mp_dk<CFG>(t, c);
        if (dk > 0) {
            for (int e = 0; e < 2; ++e) mp_bulk(st + e * IQ4R16_GROUP, p.down_exps + (size_t)s.ids[2*(dk-1) + e] * p.dn_exp + (size_t)d * IQ4R16_GROUP, IQ4R16_GROUP, bar);
        } else {
            mp_bulk(st, p.down_sh + (size_t)d * MK_DN_ROWS * Q80_ROW, MK_DN_ROWS * Q80_ROW, bar);
        }
    }
}

// warp 0: top-6 from the epoch-tagged logits into s.ids / s.w (spins with the watchdog)
template <int CFG>
__device__ __forceinline__ bool mp_get_route(const MkParams & p, MpSmem & s, unsigned epoch) {
    const int lane = threadIdx.x & 31;
    float lg[MK_E/32];
    uint64_t t0 = 0;
    for (;;) {
        bool mine = true;
#pragma unroll
        for (int j = 0; j < MK_E/32; ++j) {
            const unsigned long long v = *(const volatile unsigned long long *)&p.lg64[lane + 32*j];
            mine = mine && (unsigned)(v >> 32) == epoch;
            lg[j] = __uint_as_float((unsigned)v);
        }
        if (__all_sync(0xffffffff, mine)) break;
        if (t0 == 0) t0 = mk_globaltimer();
        __nanosleep(32);
        if (!mp_uni<CFG>(!(mk_globaltimer() - t0 > p.wait_ns || *(volatile int *)&s.abort))) {
            if (lane == 0) mp_timeout(p, s, SY_EPOCH, 0, epoch);
            return false;
        }
    }
#ifdef MK_TIMELINE
    if (lane == 0 && t0) mk_tl_wait += mk_globaltimer() - t0;
#endif
    mk_topk_core(p, lg, s.ids, s.w);
    if (lane < MK_K) { p.sel_ids[lane] = s.ids[lane]; p.sel_w[lane] = s.w[lane]; }   // RAON_MK_MOE_DEBUG only
    __syncwarp();
    return true;
}

// MQ_SAFE abort: wait until the bulk copies of chunks [0, issued) have landed (1 s bound against a broken barrier)
__device__ __forceinline__ void mp_drain(MpSmem & s, int issued) {
    for (int i = issued > MP_NST ? issued - MP_NST : 0; i < issued; ++i) {
        const uint64_t t0 = mk_globaltimer();
        while (!mp_mbar_try(&s.full[i % MP_NST], (i / MP_NST) & 1) && mk_globaltimer() - t0 < 1000000000ull) { }
    }
    __syncwarp();
}

template <int CFG>
__device__ __forceinline__ void mp_producer(const MkParams & p, MpSmem & s, uint8_t * ring) {
    const int lane = threadIdx.x & 31;
    const unsigned epoch = *(const volatile unsigned *)&p.sync[SY_EPOCH] + 1u;
    bool have_route = false;
    int seq = 0;
    for (;;) {
        int t = 0;
        if (lane == 0) t = (int)atomicAdd(&p.sync[SY_HEAD], 1u);
        t = __shfl_sync(0xffffffff, t, 0);
#ifdef MK_TIMELINE
        if (lane == 0 && seq == 0) p.tl[4*MP_TL_TASKS + 8*blockIdx.x + 5] = mk_globaltimer();
#endif
        if (t >= MpL<CFG>::NT) break;
        if (t >= MpL<CFG>::T_GU && !have_route) {
            if (!mp_get_route<CFG>(p, s, epoch)) {   // aborting
                if constexpr (CFG & MQ_SAFE) mp_drain(s, seq);
                return;
            }
            have_route = true;
        }
        const int nc = mp_nchunks<CFG>(t);
        for (int c = 0; c < nc; ++c, ++seq) {
            const int stg = seq % MP_NST;
#ifdef MK_TIMELINE
            const unsigned long long tw0 = mk_globaltimer();
#endif
            if (seq >= MP_NST && !mp_uni<CFG>(mp_mbar_wait(p, s, &s.empty[stg], ((seq / MP_NST) - 1) & 1, 100 + stg))) {
                if constexpr (CFG & MQ_SAFE) mp_drain(s, seq);
                return;
            }
#ifdef MK_TIMELINE
            if (lane == 0) p.tl[4*MP_TL_TASKS + 8*blockIdx.x + 1] += mk_globaltimer() - tw0;
#endif
            if (lane == 0) {
                s.st_task[stg] = t; s.st_chunk[stg] = c;
                mp_mbar_arrive_tx(&s.full[stg], mp_chunk_bytes<CFG>(t, c));
                mp_issue_chunk<CFG>(p, s, ring + stg * MP_CH, &s.full[stg], t, c);
#ifdef MK_TIMELINE
                if (seq == 0) p.tl[4*MP_TL_TASKS + 8*blockIdx.x + 6] = mk_globaltimer();
#endif
            }
            __syncwarp();
        }
    }
    // end marker
    const int stg = seq % MP_NST;
    if (seq >= MP_NST && !mp_uni<CFG>(mp_mbar_wait(p, s, &s.empty[stg], ((seq / MP_NST) - 1) & 1, 100 + stg))) {
        if constexpr (CFG & MQ_SAFE) mp_drain(s, seq);
        return;
    }
    if (lane == 0) { s.st_task[stg] = -1; mp_mbar_arrive(&s.full[stg]); }
    // routed-epoch consumers write lg64 with this epoch: keep it for them
    (void)epoch;
}

// consumer thread 0: hand task t (or -1: stop) to the finisher in slot k & 1. The slot is free: the finisher
// waited on done[k & 1] for task k - 2 before the consumers could get here (fin_read >= k - 1 was checked).
__device__ __forceinline__ void mp_post_finish(MpSmem & s, int k, int t) {
    if (threadIdx.x != 64) return;
    s.fin_task[k & 1] = t;
    mp_mbar_arrive(&s.done[k & 1]);
}
// finisher: the partials of its k-th task are in registers; the consumers may overwrite them
__device__ __forceinline__ void mp_fin_read_done(MpSmem & s, int k) {
    __syncwarp();
    __threadfence_block();
    if ((threadIdx.x & 31) == 0) *(volatile int *)&s.fin_read = k + 1;
}
// all consumer threads: wait until the finisher has read the partials of tasks < n
__device__ __forceinline__ bool mp_wait_fin_read(const MkParams & p, MpSmem & s, int n) {
    if (*(volatile int *)&s.fin_read >= n) { __threadfence_block(); return true; }
    const uint64_t t0 = mk_globaltimer();
    while (*(volatile int *)&s.fin_read < n) {
        __nanosleep(32);
        if (*(volatile int *)&s.abort) return false;
        if (mk_globaltimer() - t0 > p.wait_ns) { mp_timeout(p, s, 400, *(volatile int *)&s.fin_read, n); return false; }
    }
    __threadfence_block();
    return true;
}

// spin on epoch-tagged 8-byte words (whole warp, watchdog); returns the value bits
template <int CFG>
__device__ __forceinline__ unsigned mp_epoch_ld_warp(const MkParams & p, MpSmem & s, const unsigned long long * w, unsigned epoch, unsigned which) {
    unsigned long long v = *(const volatile unsigned long long *)w;
    if (!__all_sync(0xffffffff, (unsigned)(v >> 32) == epoch)) {
        const uint64_t t0 = mk_globaltimer();
        for (;;) {
            __nanosleep(32);
            v = *(const volatile unsigned long long *)w;
            if (__all_sync(0xffffffff, (unsigned)(v >> 32) == epoch)) break;
            if (__any_sync(0xffffffff, mk_globaltimer() - t0 > p.wait_ns || *(volatile int *)&s.abort)) {
                if ((threadIdx.x & 31) == 0) mp_timeout(p, s, which, 0, epoch);
                break;
            }
        }
    }
    return (unsigned)v;
}
// MQ_SHQ: the shared expert's hq from the published SiTU outputs (the finisher's quantisation, same in every block)
template <int CFG>
__device__ __forceinline__ void mp_shared_hq(const MkParams & p, MpSmem & s, unsigned epoch, int cw, int lane) {
    constexpr int NJ = (MK_HQ_BLOCKS + MP_WARPS - 1) / MP_WARPS;
    unsigned long long v[NJ];
#pragma unroll
    for (int j = 0; j < NJ; ++j) {   // all loads in flight at once (one L2 round trip, not NJ)
        const int b = cw + MP_WARPS * j;
        v[j] = b < MK_HQ_BLOCKS ? *(const volatile unsigned long long *)&p.hsh64[32 * b + lane] : 0ull;
    }
#pragma unroll
    for (int j = 0; j < NJ; ++j) {
        const int b = cw + MP_WARPS * j;
        if (b >= MK_HQ_BLOCKS) break;
        const unsigned bits = __all_sync(0xffffffff, (unsigned)(v[j] >> 32) == epoch) ? (unsigned)v[j]
                            : mp_epoch_ld_warp<CFG>(p, s, &p.hsh64[32 * b + lane], epoch, 500);
        mk_warp_quantize_q8_1(__uint_as_float(bits), &s.hq[MK_K * MK_HQ_BLOCKS + b]);
    }
}
template <int CFG>
__device__ __forceinline__ void mp_consumers(const MkParams & p, MpSmem & s, const uint8_t * ring) {
    using L = MpL<CFG>;
    const int cw = threadIdx.x / 32 - 2, lane = threadIdx.x & 31, ct = threadIdx.x - 64;
    const unsigned epoch = *(const volatile unsigned *)&p.sync[SY_EPOCH] + 1u;
    bool have_xq = false;
    int k = 0;              // consumer task count; partial buffers are indexed by k & 1
    unsigned hq_mask = 0;
    for (int seq = 0;; ++seq) {
        const int stg = seq % MP_NST;
#ifdef MK_TIMELINE
        const unsigned long long tw0 = mk_globaltimer();
#endif
        if (!mp_uni<CFG>(mp_mbar_wait(p, s, &s.full[stg], (seq / MP_NST) & 1, 200 + stg))) return;   // aborting (the finisher sees s.abort)
#ifdef MK_TIMELINE
        if (ct == 0 && seq == 0) p.tl[4*MP_TL_TASKS + 8*blockIdx.x + 7] = mk_globaltimer();
        if (ct == 0) { p.tl[4*MP_TL_TASKS + 8*blockIdx.x + 0] += mk_globaltimer() - tw0; p.tl[4*MP_TL_TASKS + 8*blockIdx.x + 4] += 1; }
#endif
        const int t = s.st_task[stg], c = s.st_chunk[stg];
        if (t < 0) {   // end: hand the finisher a stop marker once it has drained slot k & 1
            if (k < 2 || mp_uni<CFG>(mp_wait_fin_read(p, s, k - 1))) mp_post_finish(s, k, -1);
            return;
        }
        const int q = k & 1;
        // before writing partials: buffer q was last used by task k-2, dpart (down, from chunk 1 on) by any earlier task
        if (c == 0 && k >= 2 && !mp_uni<CFG>(mp_wait_fin_read(p, s, k - 1))) return;
        if (t >= L::T_DN && c == 1 && k >= 1 && !mp_uni<CFG>(mp_wait_fin_read(p, s, k))) return;
        const uint8_t * st = ring + stg * MP_CH;
#ifdef MK_TIMELINE
        if (c == 0 && ct == 0) { p.tl[4*t] = mk_globaltimer(); p.tl[4*t+3] = blockIdx.x; }
#endif
        if (t >= L::T_SH && !have_xq) {
            constexpr int NB = MK_XQ_BLOCKS / MP_WARPS;
            float v[NB];
#pragma unroll
            for (int j = 0; j < NB; ++j) v[j] = __ldg(p.x + 32 * (cw + MP_WARPS * j) + lane);
#pragma unroll
            for (int j = 0; j < NB; ++j) mk_warp_quantize_q8_1(v[j], &s.xq[cw + MP_WARPS * j]);
            mp_csync_c<CFG>();
            have_xq = true;
            if (blockIdx.x == 0) {   // RAON_MK_MOE_DEBUG=2 only
                const int * src = (const int *)s.xq; int * dst = (int *)p.xq;
                for (int i = ct; i < (int)(sizeof(s.xq)/4); i += MP_CTHREADS) dst[i] = src[i];
            }
        }
        const int dk = mp_dk<CFG>(t, c);
        if (t >= L::T_DN) {
            const unsigned need = (dk == 0 ? 1u << MK_K : 3u << (2*(dk - 1))) & ~hq_mask;
            // MQ_SHQ: the shared slot comes from hsh64 (no counter, no p.hq)
            const unsigned need_g = L::SHQ ? need & ~(1u << MK_K) : need;
            if (need) {
                if (ct == 0) {
                    for (int slot = 0; slot < MK_NX; ++slot) {
                        if (!(need_g >> slot & 1)) continue;
                        const unsigned * cnt = &p.sync[SY_GUS + slot];
                        unsigned v = mk_ld_acquire(cnt);
                        if (v < MK_GU_TILES) {
                            const uint64_t t0 = mk_globaltimer();
                            while ((v = mk_ld_acquire(cnt)) < MK_GU_TILES) {
                                __nanosleep(32);
                                if (mk_globaltimer() - t0 > p.wait_ns || *(volatile int *)&s.abort) { mp_timeout(p, s, SY_GUS + slot, v, MK_GU_TILES); break; }
                            }
#ifdef MK_TIMELINE
                            mk_tl_wait += mk_globaltimer() - t0;
#endif
                        }
                    }
                    __threadfence();
                }
                mp_csync_c<CFG>();
                for (int slot = 0; slot < MK_NX; ++slot) {
                    if (!(need_g >> slot & 1)) continue;
                    const int * src = (const int *)&p.hq[slot * MK_HQ_BLOCKS]; int * dst = (int *)&s.hq[slot * MK_HQ_BLOCKS];
                    for (int i = ct; i < MK_HQ_BLOCKS * (int)sizeof(mk_q8_1) / 4; i += MP_CTHREADS) dst[i] = __ldcg(src + i);
                }
                if (L::SHQ && need != need_g) mp_shared_hq<CFG>(p, s, epoch, cw, lane);
                hq_mask |= need;
                mp_csync_c<CFG>();
                if (L::SHQ && need != need_g && blockIdx.x == 0) {   // RAON_MK_MOE_DEBUG=2 reads p.hq
                    const int * src = (const int *)&s.hq[MK_K * MK_HQ_BLOCKS]; int * dst = (int *)&p.hq[MK_K * MK_HQ_BLOCKS];
                    for (int i = ct; i < MK_HQ_BLOCKS * (int)sizeof(mk_q8_1) / 4; i += MP_CTHREADS) dst[i] = src[i];
                }
                if constexpr (CFG & MQ_SAFE) { if (__any_sync(0xffffffff, *(volatile int *)&s.abort)) return; }
                else if (*(volatile int *)&s.abort) return;
            }
        }
        // MQ_DNSPLIT B task: fetch the A partial of the row group early, use it after the dot products
        const bool isB = L::DNS && t >= L::T_DNB;
        unsigned long long av = 0;
        if (isB && ct < MK_DN_ROWS) av = *(const volatile unsigned long long *)&p.ya64[mp_dgroup<CFG>(t) * MK_DN_ROWS + ct];
        if (t < L::T_SH)      mp_router_chunk(s, p, st, cw, lane, q);
#ifndef MP_ABL_NOCOMPUTE
        else if (t < L::T_GU) mp_gu_sh_chunk(s, st, c, cw, lane, q);
        else if (t < L::T_DN) mp_gu_rt_chunk(s, st, c, cw, lane, q);
        else                  mp_down_chunk(s, st, dk, cw, lane, q, isB);
#endif
        if constexpr (CFG & MQ_LANEARR) {
            mp_mbar_arrive(&s.empty[stg]);              // stage free (every lane: count MP_CTHREADS)
        } else {
            __syncwarp();
            if (lane == 0) mp_mbar_arrive(&s.empty[stg]);   // stage free
        }
        if (isB && ct < MK_DN_ROWS) {
            if ((unsigned)(av >> 32) != epoch) {
                const uint64_t t0 = mk_globaltimer();
                while ((unsigned)((av = *(const volatile unsigned long long *)&p.ya64[mp_dgroup<CFG>(t) * MK_DN_ROWS + ct]) >> 32) != epoch) {
                    __nanosleep(32);
                    if (mk_globaltimer() - t0 > p.wait_ns || *(volatile int *)&s.abort) { mp_timeout(p, s, 501, 0, epoch); break; }
                }
            }
            s.apart[q][ct] = __uint_as_float((unsigned)av);
        }
        if (c + 1 < mp_nchunks<CFG>(t)) continue;
        // task done: partials complete after this barrier, then the finisher warp takes over
        mp_csync_c<CFG>();
        if (ct == 0) mp_post_finish(s, k, t);
        ++k;
    }
}

// Consumes finished tasks in order: reads the partials, tells the consumers (fin_read), then does the global
// writes, fences and signals, so no compute warp ever waits for a fence or a store.
template <int CFG>
__device__ __forceinline__ void mp_finisher(const MkParams & p, MpSmem & s) {
    const int lane = threadIdx.x & 31;
    using L = MpL<CFG>;
    const unsigned epoch = *(const volatile unsigned *)&p.sync[SY_EPOCH] + 1u;
    for (int k = 0;; ++k) {
        const int q = k & 1;
        if (!mp_uni<CFG>(mp_mbar_wait(p, s, &s.done[q], (k >> 1) & 1, 300 + q))) return;
        const int t = s.fin_task[q];
        if (t < 0) return;
        if (t < L::T_SH) {
            float v = 0.0f;
            if (lane < MP_RT_ROWS) {
                constexpr int WPR = MP_WARPS / MP_RT_ROWS;
#pragma unroll
                for (int i = 0; i < WPR; ++i) v += s.rpart[q][WPR * lane + i];
            }
            mp_fin_read_done(s, k);
            // value and launch epoch in one 8-byte word: a reader that sees the epoch sees the value
            if (lane < MP_RT_ROWS) *(volatile unsigned long long *)&p.lg64[t * MP_RT_ROWS + lane] = (unsigned long long)epoch << 32 | __float_as_uint(v);
        } else if (L::SHQ && t < L::T_GU) {
            // shared eighth-tile: publish the SiTU outputs epoch-tagged (no fence, no counter)
            float g = 0.0f, u = 0.0f;
            if (lane < L::SH_TROWS) {
#pragma unroll
                for (int j = 0; j < MP_SH_PARTS; ++j) { g += s.g[q][j][lane]; u += s.u[q][j][lane]; }
            }
            mp_fin_read_done(s, k);
            if (lane < L::SH_TROWS) {
                const float h = mk_situ(g, u, p.beta, p.lb);
                *(volatile unsigned long long *)&p.hsh64[(t - L::T_SH) * L::SH_TROWS + lane] = (unsigned long long)epoch << 32 | __float_as_uint(h);
            }
        } else if (t < L::T_DN) {
            const bool sh = t < L::T_GU;
            const int i = t - L::T_GU, slot = sh ? MK_K : i / MK_GU_TILES, tile = sh ? t - L::T_SH : i % MK_GU_TILES;
            float g = 0.0f, u = 0.0f;
            if (sh) {
#pragma unroll
                for (int j = 0; j < MP_SH_PARTS; ++j) { g += s.g[q][j][lane]; u += s.u[q][j][lane]; }
            } else {
#pragma unroll
                for (int j = 0; j < MP_RT_PARTS; ++j) { g += s.g[q][j][lane]; u += s.u[q][j][lane]; }
            }
            mp_fin_read_done(s, k);
            const float h = mk_situ(g, u, p.beta, p.lb);
            mk_warp_quantize_q8_1(h, &p.hq[slot * MK_HQ_BLOCKS + tile]);
            __threadfence();
            __syncwarp();
            if (lane == 0) atomicAdd(&p.sync[SY_GUS + slot], 1u);
        } else if (L::DNS) {
            const int d = mp_dgroup<CFG>(t);
            const bool b = t >= L::T_DNB;
            float acc = 0.0f;
            if (lane < MK_DN_ROWS) {
                if (!b) {
#pragma unroll
                    for (int i = 0; i < MP_SHD_PARTS; ++i) acc += s.shpart[q][i][lane];
#pragma unroll
                    for (int e = 0; e < 4; ++e) {
                        float v = 0.0f;
#pragma unroll
                        for (int i = 0; i < MP_DN_SUB; ++i) v += s.dpart[e][i][lane];
                        acc += s.w[e] * v;
                    }
                } else {
#pragma unroll
                    for (int e = 4; e < MK_K; ++e) {
                        float v = 0.0f;
#pragma unroll
                        for (int i = 0; i < MP_DN_SUB; ++i) v += s.dpartB[q][e - 4][i][lane];
                        acc += s.w[e] * v;
                    }
                    acc += s.apart[q][lane];   // B + A: two terms, so the same bits whichever finished first
                }
            }
            mp_fin_read_done(s, k);
            if (lane < MK_DN_ROWS) {
                if (!b) *(volatile unsigned long long *)&p.ya64[d * MK_DN_ROWS + lane] = (unsigned long long)epoch << 32 | __float_as_uint(acc);
                else    p.y[d * MK_DN_ROWS + lane] = acc;
            }
        } else {
            float acc = 0.0f;
            if (lane < MK_DN_ROWS) {
#pragma unroll
                for (int i = 0; i < MP_SHD_PARTS; ++i) acc += s.shpart[q][i][lane];
#pragma unroll
                for (int e = 0; e < MK_K; ++e) {
                    float v = 0.0f;
#pragma unroll
                    for (int i = 0; i < MP_DN_SUB; ++i) v += s.dpart[e][i][lane];
                    acc += s.w[e] * v;
                }
            }
            mp_fin_read_done(s, k);
            if (lane < MK_DN_ROWS) p.y[(t - L::T_DN) * MK_DN_ROWS + lane] = acc;
        }
#ifdef MK_TIMELINE
        if (lane == 0) { p.tl[4*t+1] = 0; p.tl[4*t+2] = mk_globaltimer(); }
#endif
    }
}

template <int MINB, int CFG = 0>
__global__ void __launch_bounds__(MP_THREADS, MINB) mk_moe_pipe_kernel(const MkParams p) {
#if !MP_ARCH_OK
    __trap();   // never launched below sm_90 (see MP_ARCH_OK)
#else
    extern __shared__ __align__(128) uint8_t mp_ring[];
    __shared__ __align__(16) MpSmem s;
#ifdef MK_TIMELINE
    if (threadIdx.x == 0) p.tl[4*MP_TL_TASKS + 8*blockIdx.x + 2] = mk_globaltimer();
#endif
    if (threadIdx.x == 0) {
        for (int i = 0; i < MP_NST; ++i) { mp_mbar_init(&s.full[i], 1); mp_mbar_init(&s.empty[i], (CFG & MQ_LANEARR) ? MP_CTHREADS : MP_WARPS); }
        for (int i = 0; i < 2; ++i) mp_mbar_init(&s.done[i], 1);
        s.abort = 0; s.fin_read = 0;
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    __syncthreads();
    if (threadIdx.x < 32)      mp_producer<CFG>(p, s, mp_ring);
    else if (threadIdx.x < 64) mp_finisher<CFG>(p, s);
    else {
        mp_consumers<CFG>(p, s, mp_ring);
        // MQ_SAFE: leave together; a warp that is out keeps arriving until every consumer warp is
        if constexpr (CFG & MQ_SAFE) { while (!mp_csync_and(true)) { } }
    }
    __syncthreads();
    // the last block out resets the counters for the next launch and advances the epoch
    if (threadIdx.x == 0) {
        __threadfence();
        s.abort = atomicAdd(&p.sync[SY_DONE], 1u) == gridDim.x - 1;   // reused as "last block" flag
    }
    __syncthreads();
    if (s.abort && threadIdx.x < SY_COUNT) {   // plain stores: the next launch starts after this one ends
        const unsigned v = threadIdx.x == SY_EPOCH ? *(volatile unsigned *)&p.sync[SY_EPOCH] + 1u : 0u;
        *(volatile unsigned *)&p.sync[threadIdx.x] = v;
    }
#ifdef MK_TIMELINE
    if (threadIdx.x == 0) p.tl[4*MP_TL_TASKS + 8*blockIdx.x + 3] = mk_globaltimer();
#endif
#endif // MP_ARCH_OK
}
