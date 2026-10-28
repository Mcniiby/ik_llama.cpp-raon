// GGML_OP_RAON_MOE_BLOCK: Raon-1 mini MoE layer (router + top-6 + 6 routed SiTU experts + shared expert +
// weighted sum) for one token in a single persistent launch. See raon-mk/mk_kernel.cuh for the task layout.
//
// Variants (env, read once):
//   RAON_MK_MOE_VARIANT=pipe (default)    warp-specialised producer/consumer ring (raon-mk/mk_pipe.cuh)
//   RAON_MK_MOE_VARIANT=staged            cp.async-staged weight tiles, 1 block/SM
//   RAON_MK_MOE_VARIANT=direct            direct loads, 128 regs, 2 blocks/SM
//   RAON_MK_MOE_GRID=N                    override the grid size (default: resident blocks x SMs)
// RAON_MK_MOE=0 turns the op off at graph level. pipe needs compute capability >= 9.0, staged >= 8.0, else direct.
//
// Counters and scratch live in one persistent allocation per (device, stream); the kernel's last block resets the
// counters, so launches and graph replays need no memset. Launches on one stream serialise, so layers share scratch.
#include "raon-moe.cuh"
#include "raon-mk/mk_pipe.cuh"

#include <map>
#include <utility>
#include <string>
#include <vector>
#include <mutex>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <cmath>

namespace {

struct mk_state {
    unsigned * sync   = nullptr;
    mk_q8_1  * xq     = nullptr;
    float    * logits = nullptr;
    int      * ids    = nullptr;
    float    * w      = nullptr;
    mk_q8_1  * hq     = nullptr;
    unsigned * diag   = nullptr;
    unsigned long long * lg64 = nullptr;
    unsigned long long * hsh64 = nullptr;
    unsigned long long * ya64  = nullptr;
};

struct mk_device_info {
    bool init   = false;
    int  n_sm   = 0;
    int  occ_staged = 0;
    int  occ_direct = 0;
    int  occ_pipe   = 0;
    int  cc         = 0;   // compute capability, major*10 + minor
};

std::mutex                                  mk_mutex;
std::map<std::pair<int, cudaStream_t>, mk_state> mk_states;
mk_device_info                              mk_dev[GGML_CUDA_MAX_DEVICES];

static int mk_variant() { // 0 pipe, 1 staged, 2 direct
    static const int v = [] {
        const char * e = getenv("RAON_MK_MOE_VARIANT");
        return !e ? 0 : strcmp(e, "staged") == 0 ? 1 : strcmp(e, "direct") == 0 ? 2 : 0;
    }();
    return v;
}

static int mk_grid_override() {
    static const int g = [] {
        const char * e = getenv("RAON_MK_MOE_GRID");
        return e ? atoi(e) : 0;
    }();
    return g;
}

// pipe kernel config (mk_pipe.cuh MQ_* bits):
//   RAON_MK_SAFE_ABORT=0   disable MQ_SAFE
//   RAON_MK_SHQ=1          shared-expert gate/up as 136 eighth-tiles (MQ_SHQ)
//   RAON_MK_DNSPLIT=1      down as A (shared + experts 0-3) then B (experts 4, 5) tasks (MQ_DNSPLIT)
//   RAON_MK_RTL2=1         router weights evict_last in L2 + persisting set-aside (MQ_RTL2)
//   RAON_MK_LANE_ARRIVE=1  every consumer lane releases ring stages (MQ_LANEARR; racecheck-visible form)
static int mk_pipe_cfg() {
    static const int c = [] {
        auto on = [](const char * name, int def) { const char * e = getenv(name); return e ? atoi(e) != 0 : def != 0; };
        int cfg = 0;
        if (on("RAON_MK_SAFE_ABORT", 1))  cfg |= MQ_SAFE;
        if (on("RAON_MK_SHQ", 0))         cfg |= MQ_SHQ;
        if (on("RAON_MK_DNSPLIT", 0))     cfg |= MQ_DNSPLIT;
        if (on("RAON_MK_LANE_ARRIVE", 0)) cfg |= MQ_LANEARR;
        if (on("RAON_MK_RTL2", 0))        cfg |= MQ_RTL2;
        return cfg;
    }();
    return c;
}
template <int C> static bool mk_pipe_launch_one(int cfg, int grid, cudaStream_t stream, const MkParams & p) {
    if (cfg != C) return false;
    static bool attr_set[GGML_CUDA_MAX_DEVICES] = {};
    int dev; CUDA_CHECK(cudaGetDevice(&dev));
    if (!attr_set[dev]) {
        CUDA_CHECK(cudaFuncSetAttribute((const void *) mk_moe_pipe_kernel<1, C>, cudaFuncAttributeMaxDynamicSharedMemorySize, MP_SMEM));
        attr_set[dev] = true;
    }
    mk_moe_pipe_kernel<1, C><<<grid, MP_THREADS, MP_SMEM, stream>>>(p);
    return true;
}
// every combination of the MQ_* bits is instantiated (32 kernels)
template <int... Cs> static void mk_pipe_launch_list(int cfg, int grid, cudaStream_t stream, const MkParams & p, std::integer_sequence<int, Cs...>) {
    if (!(mk_pipe_launch_one<Cs>(cfg, grid, stream, p) || ...)) GGML_ABORT("raon-mk: pipe config %d not instantiated", cfg);
}
static void mk_pipe_launch(int cfg, int grid, cudaStream_t stream, const MkParams & p) {
    static_assert(MQ_RTL2 == 16, "update the instantiation count");
    mk_pipe_launch_list(cfg, grid, stream, p, std::make_integer_sequence<int, 32>{});
}

static unsigned long long mk_wait_ns() { // RAON_MK_WAIT_MS: watchdog per spin-wait (default 50 ms)
    static const unsigned long long w = [] {
        const char * e = getenv("RAON_MK_WAIT_MS");
        return e && atoll(e) > 0 ? (unsigned long long) atoll(e) * 1000000ull : (unsigned long long) MK_WAIT_TIMEOUT;
    }();
    return w;
}

static const mk_device_info & mk_device(int device) {
    mk_device_info & d = mk_dev[device];
    if (!d.init) {
        CUDA_CHECK(cudaFuncSetAttribute((const void *) mk_moe_layer_kernel<true, 1>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, MK_STAGE_BYTES));
        CUDA_CHECK(cudaDeviceGetAttribute(&d.n_sm, cudaDevAttrMultiProcessorCount, device));
        int cc_major = 0, cc_minor = 0;
        CUDA_CHECK(cudaDeviceGetAttribute(&cc_major, cudaDevAttrComputeCapabilityMajor, device));
        CUDA_CHECK(cudaDeviceGetAttribute(&cc_minor, cudaDevAttrComputeCapabilityMinor, device));
        d.cc = 10*cc_major + cc_minor;
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&d.occ_staged, mk_moe_layer_kernel<true, 1>,
                MK_THREADS, MK_STAGE_BYTES));
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&d.occ_direct, mk_moe_layer_kernel<false, 2>,
                MK_THREADS, 0));
        if (d.cc >= 90) {   // the pipe kernel is a stub below sm_90 (mk_pipe.cuh MP_ARCH_OK)
            CUDA_CHECK(cudaFuncSetAttribute((const void *) mk_moe_pipe_kernel<1>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize, MP_SMEM));
            CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&d.occ_pipe, mk_moe_pipe_kernel<1>,
                    MP_THREADS, MP_SMEM));
        }
        d.occ_pipe   = std::max(d.occ_pipe, 1);
        d.occ_staged = std::max(d.occ_staged, 1);
        d.occ_direct = std::max(d.occ_direct, 1);
        d.init = true;
    }
    return d;
}

static const mk_state & mk_get_state(int device, cudaStream_t stream) {
    auto it = mk_states.find({device, stream});
    if (it != mk_states.end()) {
        return it->second;
    }
    // layout: sync | xq | logits | ids | w | hq | diag | lg64 | hsh64 | ya64, every part 256-byte aligned
    const size_t sz[10] = {
        SY_COUNT*sizeof(unsigned), MK_XQ_BLOCKS*sizeof(mk_q8_1), MK_E*sizeof(float),
        MK_K*sizeof(int), MK_K*sizeof(float), (size_t) MK_NX*MK_HQ_BLOCKS*sizeof(mk_q8_1), MK_DIAG*sizeof(unsigned),
        MK_E*sizeof(unsigned long long), MK_F*sizeof(unsigned long long), MK_H*sizeof(unsigned long long),
    };
    size_t off[10], total = 0;
    for (int i = 0; i < 10; ++i) {
        off[i] = total;
        total += (sz[i] + 255) & ~(size_t) 255;
    }
    char * base = nullptr;
    // cudaMalloc is legal under ggml's relaxed capture mode; if the zeroing gets captured, it re-zeroes the
    // already self-reset counters on every replay, which is harmless.
    CUDA_CHECK(cudaMalloc((void **) &base, total));
    CUDA_CHECK(cudaMemsetAsync(base, 0, total, stream));
    mk_state st;
    st.sync   = (unsigned *) (base + off[0]);
    st.xq     = (mk_q8_1  *) (base + off[1]);
    st.logits = (float    *) (base + off[2]);
    st.ids    = (int      *) (base + off[3]);
    st.w      = (float    *) (base + off[4]);
    st.hq     = (mk_q8_1  *) (base + off[5]);
    st.diag   = (unsigned *) (base + off[6]);
    st.lg64   = (unsigned long long *) (base + off[7]);
    st.hsh64  = (unsigned long long *) (base + off[8]);
    st.ya64   = (unsigned long long *) (base + off[9]);
    return mk_states.emplace(std::make_pair(device, stream), st).first->second;
}

static bool mk_aligned16(const void * p) {
    return ((uintptr_t) p & 15) == 0;
}

// RAON_MK_MOE_DEBUG=1: check before every launch that each kernel pointer range lies inside one device
// allocation (abort instead of a GPU fault); after the launch, when not capturing, report the selected experts.
static int mk_debug() {
    static const int d = [] { const char * e = getenv("RAON_MK_MOE_DEBUG"); return e ? atoi(e) : 0; }();
    return d;
}

static void mk_check_range(const char * what, const char * name, const void * p, size_t bytes) {
    cudaPointerAttributes attr;
    const cudaError_t err = cudaPointerGetAttributes(&attr, p);
    if (err != cudaSuccess || attr.type != cudaMemoryTypeDevice) {
        (void) cudaGetLastError();
        GGML_ABORT("RAON_MK_MOE_DEBUG: %s (%s) = %p is not device memory (err %d, type %d)",
                what, name, p, (int) err, err == cudaSuccess ? (int) attr.type : -1);
    }
    CUdeviceptr base = 0;
    size_t size = 0;
    if (cuMemGetAddressRange(&base, &size, (CUdeviceptr) p) != CUDA_SUCCESS) {
        GGML_ABORT("RAON_MK_MOE_DEBUG: %s (%s) = %p: cuMemGetAddressRange failed", what, name, p);
    }
    if ((uintptr_t) p + bytes > (uintptr_t) base + size) {
        GGML_ABORT("RAON_MK_MOE_DEBUG: %s (%s) = %p + %zu runs past its allocation [%p, +%zu)",
                what, name, p, bytes, (void *) base, size);
    }
}

// RAON_MK_MOE_DEBUG=2: after a synchronised launch, recompute the layer on the host from the device's own
// intermediates (same mk_dot.h code) and report whether gate/up + SiTU or down + weighted sum disagrees.
static void mk_debug_reference(const MkParams & p, const char * name) {
    std::vector<mk_q8_1> xq(MK_XQ_BLOCKS), hq((size_t) MK_NX*MK_HQ_BLOCKS);
    int ids[MK_K]; float w[MK_K];
    std::vector<float> y(MK_H);
    CUDA_CHECK(cudaMemcpy(xq.data(), p.xq, xq.size()*sizeof(mk_q8_1), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hq.data(), p.hq, hq.size()*sizeof(mk_q8_1), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(ids, p.sel_ids, sizeof(ids), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(w,   p.sel_w,   sizeof(w),   cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(y.data(), p.y, MK_H*sizeof(float), cudaMemcpyDeviceToHost));

    // gate/up + SiTU
    std::vector<uint8_t> gw, uw;
    for (int slot = 0; slot < MK_NX; ++slot) {
        const bool sh = slot == MK_K;
        const size_t row = sh ? Q6K_ROW : IQ3KS_ROW, mat = (size_t) MK_F*row;
        const uint8_t * gsrc = sh ? p.gate_sh : p.gate_exps + (size_t) ids[slot]*p.gu_exp;
        const uint8_t * usrc = sh ? p.up_sh   : p.up_exps   + (size_t) ids[slot]*p.gu_exp;
        gw.resize(mat); uw.resize(mat);
        CUDA_CHECK(cudaMemcpy(gw.data(), gsrc, mat, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(uw.data(), usrc, mat, cudaMemcpyDeviceToHost));
        double max_err = 0, max_ref = 0; int nonfin = 0;
        for (int r = 0; r < MK_F; ++r) {
            float g = 0, u = 0;
            for (int kbx = 0; kbx < MK_H/QK_K_; ++kbx) {
                if (sh) {
                    for (int iqs = 0; iqs < 32; ++iqs) {
                        g += mk_vec_dot_q6_K_q8_1(gw.data() + r*row, &xq[8*kbx], kbx, iqs);
                        u += mk_vec_dot_q6_K_q8_1(uw.data() + r*row, &xq[8*kbx], kbx, iqs);
                    }
                } else {
                    for (int iqs = 0; iqs < 32; iqs += 4) {
                        mk_vec_dot_iq3_ks_q8_1(gw.data() + r*row, &xq[8*kbx], kbx, iqs, &g);
                        mk_vec_dot_iq3_ks_q8_1(uw.data() + r*row, &xq[8*kbx], kbx, iqs, &u);
                    }
                }
            }
            const float h = mk_situ(g, u, p.beta, p.lb);
            const mk_q8_1 & b = hq[(size_t) slot*MK_HQ_BLOCKS + r/32];
            const float hd = mk_h2f_host(b.d) * b.qs[r % 32];
            if (!std::isfinite(hd)) ++nonfin;
            if (r < 6 || r == 32 || r == 1087) fprintf(stderr, "    slot %d row %4d: g %9.4f u %9.4f h_ref %9.4f h_dev %9.4f (d %.4g q %d)\n",
                    slot, r, g, u, h, hd, mk_h2f_host(b.d), (int) b.qs[r % 32]);
            max_err = std::max(max_err, (double) std::fabs(hd - h));
            max_ref = std::max(max_ref, (double) std::fabs(h));
        }
        fprintf(stderr, "RAON_MK_MOE_DEBUG: %s: ref gu slot %d (%s %d): max|h_ref| %.4g max|h_dev - h_ref| %.4g non-finite %d\n",
                name, slot, sh ? "shared" : "expert", sh ? -1 : ids[slot], max_ref, max_err, nonfin);
    }

    // down + weighted sum, from the device hq
    std::vector<uint8_t> dw((size_t) MK_K*IQ4R16_EXP), dsh(Q80_MAT);
    for (int k = 0; k < MK_K; ++k) {
        CUDA_CHECK(cudaMemcpy(dw.data() + (size_t) k*IQ4R16_EXP, p.down_exps + (size_t) ids[k]*p.dn_exp, IQ4R16_EXP, cudaMemcpyDeviceToHost));
    }
    CUDA_CHECK(cudaMemcpy(dsh.data(), p.down_sh, Q80_MAT, cudaMemcpyDeviceToHost));
    double max_err = 0, ss = 0; int nonfin_dev = 0, nonfin_ref = 0;
    for (int t = 0; t < MK_NT_DOWN; ++t) {
        float part[MK_NX][MK_DN_ROWS] = {};
        for (int k = 0; k < MK_K; ++k) {
            const uint8_t * grp = dw.data() + (size_t) k*IQ4R16_EXP + (size_t) t*IQ4R16_GROUP;
            for (int ir = 0; ir < 16; ir += 4) for (int il = 0; il < 2; ++il) for (int kbx = 0; kbx < MK_F/32; ++kbx) {
                mk_vec_dot_iq4_ks_r16_q8_1(grp, &hq[(size_t) k*MK_HQ_BLOCKS + kbx], kbx, 8*il, ir, &part[k][ir]);
            }
        }
        for (int r = 0; r < MK_DN_ROWS; ++r) {
            float acc = 0;
            for (int kbx = 0; kbx < MK_F/32; ++kbx) for (int iqs = 0; iqs < 8; iqs += 2) {
                acc += mk_vec_dot_q8_0_q8_1(dsh.data() + (size_t) (16*t + r)*Q80_ROW, &hq[(size_t) MK_K*MK_HQ_BLOCKS + kbx], kbx, iqs);
            }
            part[MK_K][r] = acc;
        }
        for (int r = 0; r < MK_DN_ROWS; ++r) {
            float yr = part[MK_K][r];
            for (int k = 0; k < MK_K; ++k) yr += w[k]*part[k][r];
            const float yd = y[16*t + r];
            if (!std::isfinite(yr)) ++nonfin_ref;
            if (!std::isfinite(yd)) ++nonfin_dev; else max_err = std::max(max_err, (double) std::fabs(yd - yr));
            ss += (double) yr*yr;
        }
    }
    fprintf(stderr, "RAON_MK_MOE_DEBUG: %s: ref down: y_ref rms %.4g (non-finite %d) | y_dev non-finite %d, max|y_dev - y_ref| %.4g\n",
            name, std::sqrt(ss/MK_H), nonfin_ref, nonfin_dev, max_err);
}

} // namespace

bool ggml_cuda_raon_moe_block_supported(const ggml_tensor * op) {
    const ggml_tensor * x    = op->src[0];
    const ggml_tensor * rw   = op->src[1];
    const ggml_tensor * rb   = op->src[2];
    const ggml_tensor * up_e = op->src[3];
    const ggml_tensor * gt_e = op->src[4];
    const ggml_tensor * dn_e = op->src[5];
    const ggml_tensor * up_s = op->src[6];
    const ggml_tensor * gt_s = op->src[7];
    const ggml_tensor * dn_s = op->src[8];
    if (!x || !rw || !rb || !up_e || !gt_e || !dn_e || !up_s || !gt_s || !dn_s) return false;
    if (op->op_params[0] != MK_K) return false;
    if (x->type != GGML_TYPE_F32 || x->ne[0] != MK_H || ggml_nrows(x) != 1 || !ggml_is_contiguous(x)) return false;
    if (rw->type != GGML_TYPE_F32 || rw->ne[0] != MK_H || rw->ne[1] != MK_E || rw->nb[1] % 16 != 0) return false;
    if (rb->type != GGML_TYPE_F32 || ggml_nelements(rb) != MK_E || !ggml_is_contiguous(rb)) return false;
    if (up_e->type != GGML_TYPE_IQ3_KS || gt_e->type != GGML_TYPE_IQ3_KS) return false;
    if (up_e->ne[0] != MK_H || up_e->ne[1] != MK_F || up_e->ne[2] != MK_E) return false;
    if (up_e->nb[1] != (size_t) IQ3KS_ROW || gt_e->nb[1] != (size_t) IQ3KS_ROW || up_e->nb[2] != gt_e->nb[2]) return false;
    if (dn_e->type != GGML_TYPE_IQ4_KS_R16 || dn_e->ne[0] != MK_F || dn_e->ne[1] != MK_H || dn_e->ne[2] != MK_E) return false;
    if (16*dn_e->nb[1] != (size_t) IQ4R16_GROUP) return false;
    if (up_s->type != GGML_TYPE_Q6_K || gt_s->type != GGML_TYPE_Q6_K) return false;
    if (up_s->ne[0] != MK_H || up_s->ne[1] != MK_F || ggml_nrows(up_s) != MK_F) return false;
    if (up_s->nb[1] != (size_t) Q6K_ROW || gt_s->nb[1] != (size_t) Q6K_ROW) return false;
    if (dn_s->type != GGML_TYPE_Q8_0 || dn_s->ne[0] != MK_F || ggml_nrows(dn_s) != MK_H || dn_s->nb[1] != (size_t) Q80_ROW) return false;
    return true;
}

void ggml_cuda_op_raon_moe_block(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(ggml_cuda_raon_moe_block_supported(dst));
    const ggml_tensor * up_e = dst->src[3];
    const ggml_tensor * dn_e = dst->src[5];

    cudaStream_t stream = ctx.stream();
    const int device = ctx.device;

    MkParams p;
    p.x         = (const float   *) dst->src[0]->data;
    p.router_w  = (const float   *) dst->src[1]->data;
    p.router_b  = (const float   *) dst->src[2]->data;
    p.up_exps   = (const uint8_t *) dst->src[3]->data;
    p.gate_exps = (const uint8_t *) dst->src[4]->data;
    p.down_exps = (const uint8_t *) dst->src[5]->data;
    p.up_sh     = (const uint8_t *) dst->src[6]->data;
    p.gate_sh   = (const uint8_t *) dst->src[7]->data;
    p.down_sh   = (const uint8_t *) dst->src[8]->data;
    p.gu_exp     = up_e->nb[2];
    p.dn_exp     = dn_e->nb[2];
    p.router_nb1 = dst->src[1]->nb[1];
    p.n_exp      = MK_E;
    p.norm_w     = dst->op_params[4] != 0;
    memcpy(&p.beta, &dst->op_params[1], sizeof(float));
    memcpy(&p.lb,   &dst->op_params[2], sizeof(float));
    float w_scale;
    memcpy(&w_scale, &dst->op_params[3], sizeof(float));
    p.w_scale = std::fabs(w_scale - 1.0f) > 1e-5f ? w_scale : 1.0f; // llm_build_moe_ffn skips the scale
    p.y = (float *) dst->data;
    p.wait_ns = mk_wait_ns();

    mk_device_info dinfo;
    {
        std::lock_guard<std::mutex> lock(mk_mutex);
        dinfo = mk_device(device);
        const mk_state & st = mk_get_state(device, stream);
        p.xq = st.xq; p.logits = st.logits; p.sel_ids = st.ids; p.sel_w = st.w; p.hq = st.hq; p.sync = st.sync; p.diag = st.diag; p.lg64 = st.lg64;
        p.hsh64 = st.hsh64; p.ya64 = st.ya64;
    }

    // cp.async / cp.async.bulk need 16-byte aligned sources (and router rows): true for whole GGUF tensors,
    // checked for views; otherwise fall back to the direct-load variant
    int variant = mk_variant();
    if (variant == 0 && dinfo.cc < 90) variant = 1;
    if (variant == 1 && dinfo.cc < 80) variant = 2;
    if (variant != 2) {
        bool al = p.gu_exp % 16 == 0 && p.dn_exp % 16 == 0 && p.router_nb1 % 16 == 0;
        for (int i = 1; i <= 8; ++i) al = al && (i == 2 || mk_aligned16(dst->src[i]->data));
        if (!al) variant = 2;
    }
    const bool staged = variant == 1;
    int grid = mk_grid_override();
    if (grid <= 0) {
        grid = (variant == 0 ? dinfo.occ_pipe : staged ? dinfo.occ_staged : std::min(dinfo.occ_direct, 2)) * dinfo.n_sm;
    }
    grid = std::max(1, std::min(grid, variant == 0 ? MP_NT_TOTAL : MK_NT_TOTAL));

    const bool debug = mk_debug() > 0;
    if (debug) {
        static const char * srcn[9] = { "x", "router_w", "router_b", "up_exps", "gate_exps", "down_exps",
                                        "up_shexp", "gate_shexp", "down_shexp" };
        for (int i = 0; i < 9; ++i) {
            const ggml_tensor * s = dst->src[i];
            mk_check_range(srcn[i], s->name, s->data, ggml_nbytes(s));
        }
        mk_check_range("dst", dst->name, dst->data, ggml_nbytes(dst));
        mk_check_range("sync", "scratch", p.sync, SY_COUNT*sizeof(unsigned));
        mk_check_range("hq", "scratch", p.hq, (size_t) MK_NX*MK_HQ_BLOCKS*sizeof(mk_q8_1));
        {
            unsigned pre[SY_COUNT];
            CUDA_CHECK(cudaStreamSynchronize(stream));
            CUDA_CHECK(cudaMemcpy(pre, p.sync, sizeof(pre), cudaMemcpyDeviceToHost));
            fprintf(stderr, "RAON_MK_MOE_DEBUG: %s: counters before launch: %u %u %u %u %u %u %u %u\n", dst->name,
                    pre[0], pre[1], pre[2], pre[3], pre[4], pre[5], pre[6], pre[7]);
        }
        fprintf(stderr, "RAON_MK_MOE_DEBUG: %s: launch %s grid %d dev %d stream %p | gate %p up %p down %p down_sh %p gu_exp %zu dn_exp %zu router_nb1 %zu n_exp %d norm %d w_scale %g beta %g lb %g\n",
                dst->name, variant == 0 ? "pipe" : staged ? "staged" : "direct", grid, device, (void *) stream,
                (const void *) p.gate_exps, (const void *) p.up_exps, (const void *) p.down_exps, (const void *) p.down_sh,
                p.gu_exp, p.dn_exp, p.router_nb1, p.n_exp, p.norm_w, p.w_scale, p.beta, p.lb);
    }

    if (variant == 0) {
        // MQ_RTL2: evict_last lines only stay in L2 inside the persisting set-aside, so raise it once to
        // RAON_MK_RTL2_MB (default: the device maximum). Not during stream capture (the first eager evaluation does it).
        static bool l2_set[GGML_CUDA_MAX_DEVICES] = {};
        if ((mk_pipe_cfg() & MQ_RTL2) && !l2_set[device]) {
            cudaStreamCaptureStatus cs;
            CUDA_CHECK(cudaStreamIsCapturing(stream, &cs));
            if (cs == cudaStreamCaptureStatusNone) {
                int max_persist = 0;
                CUDA_CHECK(cudaDeviceGetAttribute(&max_persist, cudaDevAttrMaxPersistingL2CacheSize, device));
                const char * e = getenv("RAON_MK_RTL2_MB");
                size_t want = e ? (size_t) atoll(e) << 20 : (size_t) max_persist;
                want = std::min(want, (size_t) max_persist);
                size_t cur = 0;
                CUDA_CHECK(cudaDeviceGetLimit(&cur, cudaLimitPersistingL2CacheSize));
                if (want > cur) {
                    CUDA_CHECK(cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, want));
                }
                l2_set[device] = true;
            }
        }
        mk_pipe_launch(mk_pipe_cfg(), grid, stream, p);
    } else if (staged) {
        mk_moe_layer_kernel<true, 1><<<grid, MK_THREADS, MK_STAGE_BYTES, stream>>>(p);
    } else {
        mk_moe_layer_kernel<false, 2><<<grid, MK_THREADS, 0, stream>>>(p);
    }
    CUDA_CHECK(cudaGetLastError());

    if (debug) {
        cudaStreamCaptureStatus cs;
        CUDA_CHECK(cudaStreamIsCapturing(stream, &cs));
        if (cs == cudaStreamCaptureStatusNone) {
            CUDA_CHECK(cudaStreamSynchronize(stream));
            int ids[MK_K]; float w[MK_K]; unsigned sy[SY_COUNT];
            CUDA_CHECK(cudaMemcpy(ids, p.sel_ids, sizeof(ids), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(w,   p.sel_w,   sizeof(w),   cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(sy,  p.sync,    sizeof(sy),  cudaMemcpyDeviceToHost));
            unsigned dg[MK_DIAG];
            CUDA_CHECK(cudaMemcpy(dg, p.diag, sizeof(dg), cudaMemcpyDeviceToHost));
            if (dg[0]) {
                fprintf(stderr, "RAON_MK_MOE_DEBUG: %s: WAIT TIMEOUT on counter %u: saw %u, target %u, block %u | head %u gu %u done %u\n",
                        dst->name, dg[1], dg[2], dg[3], dg[4], dg[5], dg[6], dg[7]);
                CUDA_CHECK(cudaMemset(p.diag, 0, sizeof(dg)));
            }
            {
                std::vector<float> hx(MK_H), hy(MK_H);
                CUDA_CHECK(cudaMemcpy(hx.data(), p.x, MK_H*sizeof(float), cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(hy.data(), p.y, MK_H*sizeof(float), cudaMemcpyDeviceToHost));
                double sx = 0, sy = 0; int bx = 0, by = 0;
                for (int i = 0; i < MK_H; ++i) {
                    if (std::isfinite(hx[i])) sx += (double) hx[i]*hx[i]; else ++bx;
                    if (std::isfinite(hy[i])) sy += (double) hy[i]*hy[i]; else ++by;
                }
                fprintf(stderr, "RAON_MK_MOE_DEBUG: %s: x rms %.4g (%d non-finite) y rms %.4g (%d non-finite) x@%p y@%p\n",
                        dst->name, std::sqrt(sx/MK_H), bx, std::sqrt(sy/MK_H), by, (const void *) p.x, (void *) p.y);
                // which 16-row down groups are bad, and are the q8_1 scales of xq / hq finite?
                std::string groups;
                for (int g = 0; g < MK_NT_DOWN; ++g) {
                    bool bad = false;
                    for (int r = 0; r < MK_DN_ROWS; ++r) bad |= !std::isfinite(hy[g*MK_DN_ROWS + r]) || std::fabs(hy[g*MK_DN_ROWS + r]) > 1e4f;
                    groups += bad ? 'X' : '.';
                }
                std::vector<mk_q8_1> hxq(MK_XQ_BLOCKS), hhq(MK_NX*MK_HQ_BLOCKS);
                CUDA_CHECK(cudaMemcpy(hxq.data(), p.xq, hxq.size()*sizeof(mk_q8_1), cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(hhq.data(), p.hq, hhq.size()*sizeof(mk_q8_1), cudaMemcpyDeviceToHost));
                int bxq = 0; std::string slots;
                for (auto & b : hxq) bxq += !std::isfinite(mk_h2f_host(b.d));
                for (int sl = 0; sl < MK_NX; ++sl) {
                    int bad = 0; float mx = 0;
                    for (int t = 0; t < MK_HQ_BLOCKS; ++t) {
                        const float d = mk_h2f_host(hhq[sl*MK_HQ_BLOCKS + t].d);
                        if (!std::isfinite(d)) ++bad; else mx = std::max(mx, std::fabs(d));
                    }
                    char buf[48]; snprintf(buf, sizeof buf, " slot%d bad %d max_d %.3g", sl, bad, mx); slots += buf;
                }
                fprintf(stderr, "RAON_MK_MOE_DEBUG: %s: y groups %s\nRAON_MK_MOE_DEBUG: %s: xq bad d %d |%s\n",
                        dst->name, groups.c_str(), dst->name, bxq, slots.c_str());
            }
            if (mk_debug() >= 2) mk_debug_reference(p, dst->name);
            fprintf(stderr, "RAON_MK_MOE_DEBUG: %s: ok, ids %d %d %d %d %d %d, w %.3f %.3f %.3f %.3f %.3f %.3f, counters %u %u %u %u %u %u\n",
                    dst->name, ids[0], ids[1], ids[2], ids[3], ids[4], ids[5], w[0], w[1], w[2], w[3], w[4], w[5],
                    sy[0], sy[1], sy[2], sy[3], sy[4], sy[5]);
        }
    }
}
