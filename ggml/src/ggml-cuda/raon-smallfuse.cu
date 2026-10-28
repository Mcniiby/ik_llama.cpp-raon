// Decode small-kernel fusions; same math as the unfused kernels, only the reduction order changes.

#include "raon-smallfuse.cuh"
#include "raon-multitoken.cuh"
#include "raon-leftover.cuh"
#include "../ggml-raon-hc.h"
#include "../ggml-rda.h"

int64_t g_raon_fused_hc_boundary = 0;
int64_t g_raon_fused_hc_collapse = 0;
int64_t g_raon_fused_f16_mv = 0;

static bool raon_env_on(const char * name) {
    const char * e = getenv(name);
    return !e || atoi(e) != 0;
}

// next node after i that launches something, or n_nodes
static int raon_next_node(const ggml_cgraph * cgraph, int i) {
    int j = i + 1;
    while (j < cgraph->n_nodes && ggml_is_noop(cgraph->nodes[j])) ++j;
    return j;
}

static bool raon_aligned16(const void * p) { return ((uintptr_t) p) % 16 == 0; }

// mHC sublayer boundary: hc_post -> mixes = fn @ y -> hc_pre (Helmert chart) -> rms_norm(z)*w, one launch.
// Blocks write partial dot products; the last to arrive (threadfence + counter) reduces them in block order.
// Outputs may alias inputs, so they are only written after every block has read its inputs.

#define RAON_HCB_MAX_BLOCKS 64
#define RAON_HCB_MAX_M      72
#define RAON_HCB_MAX_T      8   // tokens (spec verify, multi-slot), blockIdx.y
static __device__ float    g_hcb_partials[RAON_HCB_MAX_T*RAON_HCB_MAX_BLOCKS*RAON_HCB_MAX_M];
static __device__ unsigned g_hcb_counter[RAON_HCB_MAX_T] = {0};
// T > 1: blocks of all tokens that have read their inputs, and finalizers past the wait (both reset by the last one)
static __device__ unsigned g_hcb_read = 0;
static __device__ unsigned g_hcb_fin  = 0;

// per-token strides of the boundary's tensors, in floats (token t = blockIdx.y)
struct raon_hcb_strides {
    int64_t x, post, res, comb;
    int64_t z, pst, cmb;            // RAON_HC_PRE output parts: z [E, T], post [S, T], comb [S*S, T]
    int64_t pst0, cmb0;             // offsets of post and comb in the output
};

template <int S, int NT>
static __global__ void __launch_bounds__(NT) k_raon_hc_boundary(
        const float * __restrict__ x, const float * __restrict__ post_in, const float * __restrict__ res,
        const float * __restrict__ comb_in, const int64_t res_s1, float * __restrict__ hc_out,
        const half * __restrict__ fn, const int64_t fn_s1, const float * __restrict__ scale, const float * __restrict__ base,
        const float rms_eps, const float mag, float * __restrict__ mixes_out, float * __restrict__ pre_all,
        const float * __restrict__ nw, const float norm_eps, float * __restrict__ normed, const int E,
        const raon_hcb_strides st, const float * __restrict__ x2, float * __restrict__ add_out) {
    constexpr int M   = 2*S + (S - 1)*(S - 1);
    constexpr int CPB = NT/S;
    constexpr int NW  = NT/WARP_SIZE;
    static_assert(M + 1 <= RAON_HCB_MAX_M, "too many mixes");

    // token t: shift every per-token pointer
    const int tok = blockIdx.y;
    x       += tok*st.x;
    if (x2) { x2 += tok*st.x; add_out += tok*st.x; }
    post_in += tok*st.post;
    res     += tok*st.res;
    comb_in += tok*st.comb;
    hc_out  += (int64_t) tok*S*E;
    mixes_out += (int64_t) tok*M;
    normed  += (int64_t) tok*E;
    float * z_out   = pre_all + tok*st.z;
    float * pst_out = pre_all + st.pst0 + tok*st.pst;
    float * cmb_out = pre_all + st.cmb0 + tok*st.cmb;
    float    * partials = g_hcb_partials + (int64_t) tok*RAON_HCB_MAX_BLOCKS*RAON_HCB_MAX_M;
    unsigned * counter  = g_hcb_counter + tok;

    const int tid  = threadIdx.x;
    const int s    = tid / CPB;
    const int i0   = blockIdx.x*CPB + tid % CPB;
    const int lane = tid % WARP_SIZE;
    const int warp = tid / WARP_SIZE;

    // hc_post, same expression order as k_hc_post
    float v = 0.0f;
    if (i0 < E) {
        // x2: fused preceding ADD (x + x2), its result written once (stream 0)
        float xv = x[i0];
        if (x2) {
            xv += x2[i0];
            if (s == 0) add_out[i0] = xv;
        }
        float sum = xv*post_in[s];
#pragma unroll
        for (int j = 0; j < S; ++j) {
            sum += comb_in[j*S + s]*res[j*res_s1 + i0];
        }
        v = sum;
        hc_out[s*E + i0] = v;
    }

    // partial mixes + sum of squares over this block's elements
    __shared__ float s_red[NW][M + 1];
    __shared__ bool  s_last;
    {
        const int64_t idx = (int64_t) s*E + i0;
        float acc[M + 1];
#pragma unroll
        for (int r = 0; r < M; ++r) {
            acc[r] = i0 < E ? __half2float(fn[r*fn_s1 + idx])*v : 0.0f;
        }
        acc[M] = v*v;
#pragma unroll
        for (int r = 0; r <= M; ++r) {
            const float t = warp_reduce_sum(acc[r]);
            if (lane == 0) s_red[warp][r] = t;
        }
    }
    __syncthreads();
    if (tid <= M) {
        float t = 0.0f;
#pragma unroll
        for (int w = 0; w < NW; ++w) t += s_red[w][tid];
        partials[blockIdx.x*(M + 1) + tid] = t;
    }
    // grid.sync-style arrival: the barrier orders the block's writes before thread 0's fence + atomic
    __syncthreads();
    if (tid == 0) {
        __threadfence();
        if (gridDim.y > 1) atomicAdd(&g_hcb_read, 1u);
        s_last = atomicAdd(counter, 1u) == gridDim.x - 1;
        if (s_last) __threadfence();
    }
    __syncthreads();
    if (!s_last) return;
    // T > 1: outputs may alias another token's inputs, so no token is finalized until every block of every token
    // has read its inputs. At most T (<= 8) blocks wait here, so the blocks they wait for can always be scheduled.
    if (gridDim.y > 1) {
        if (tid == 0) {
            const unsigned total = gridDim.x*gridDim.y;
            while (*(volatile unsigned *) &g_hcb_read < total) __nanosleep(32);
            __threadfence();
            if (atomicAdd(&g_hcb_fin, 1u) == gridDim.y - 1) { g_hcb_read = 0; g_hcb_fin = 0; }
        }
        __syncthreads();
    }

    // last block: totals in block order, chart, pre-weighted stream sum, rms_norm
    __shared__ float s_part[RAON_HCB_MAX_BLOCKS*(M + 1)];
    __shared__ float s_tot[M + 1];
    __shared__ float s_pre[S];
    __shared__ float s_sum[NW];
    const int npart = gridDim.x*(M + 1);
    for (int k = tid; k < npart; k += NT) s_part[k] = __ldcg(&partials[k]);
    if (tid == 0) *counter = 0;
    __syncthreads();
    if (tid <= M) {
        float t = 0.0f;
        for (int b = 0; b < (int) gridDim.x; ++b) t += s_part[b*(M + 1) + tid];
        s_tot[tid] = t;
        if (tid < M) mixes_out[tid] = t;
    }
    __syncthreads();
    if (tid == 0) {
        const float r = rms_eps >= 0.0f ? rsqrtf(s_tot[M]/(E*S) + rms_eps) : 1.0f;
        float pre[S], pst[S], cmb[S*S];
        raon_hc_coeffs(S, s_tot, r, scale, base, mag, pre, pst, cmb);
#pragma unroll
        for (int k = 0; k < S; ++k) { s_pre[k] = pre[k]; pst_out[k] = pst[k]; }
#pragma unroll
        for (int k = 0; k < S*S; ++k) cmb_out[k] = cmb[k];
    }
    __syncthreads();

    float w[S];
#pragma unroll
    for (int k = 0; k < S; ++k) w[k] = s_pre[k];
    constexpr int MAXC = 8;
    float z[MAXC];
    float ss = 0.0f;
#pragma unroll
    for (int c = 0; c < MAXC; ++c) {
        const int i = tid + c*NT;
        z[c] = 0.0f;
        if (i < E) {
            float sum = 0.0f;
#pragma unroll
            for (int k = 0; k < S; ++k) sum += __ldcg(&hc_out[k*E + i])*w[k];
            z[c] = sum;
            z_out[i] = sum;
            ss += sum*sum;
        }
    }
    ss = warp_reduce_sum(ss);
    if (lane == 0) s_sum[warp] = ss;
    __syncthreads();
    ss = lane < NW ? s_sum[lane] : 0.0f;
    ss = warp_reduce_sum(ss);
    const float rs = rsqrtf(ss/E + norm_eps);
#pragma unroll
    for (int c = 0; c < MAXC; ++c) {
        const int i = tid + c*NT;
        if (i < E) normed[i] = rs*nw[i]*z[c];
    }
}

static int raon_hc_boundary_impl(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int i, const ggml_tensor * add);

int ggml_cuda_raon_hc_boundary(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int i) {
    return raon_hc_boundary_impl(ctx, cgraph, i, nullptr);
}

// ADD feeding the HC_POST of a boundary (MoE combine with 2-8 tokens): one launch for ADD + boundary.
// RAON_HC_BOUNDARY_ADD=0 keeps the ADD separate.
int ggml_cuda_raon_hc_boundary_add(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int ia) {
    static const bool enabled = raon_env_on("RAON_HC_BOUNDARY_ADD");
    if (!enabled || !ctx.fusion) return -1;
    const ggml_tensor * add = cgraph->nodes[ia];
    const int j = raon_next_node(cgraph, ia);
    if (j >= cgraph->n_nodes || cgraph->nodes[j]->op != GGML_OP_HC_POST || cgraph->nodes[j]->src[0] != add) return -1;
    const ggml_tensor * a = add->src[0], * b = add->src[1];
    if (add->type != GGML_TYPE_F32 || a->type != GGML_TYPE_F32 || b->type != GGML_TYPE_F32) return -1;
    if (!ggml_is_contiguous(add) || !ggml_is_contiguous(a) || !ggml_is_contiguous(b)) return -1;
    if (!ggml_are_same_shape(a, b) || !ggml_are_same_shape(add, a) || add->ne[2] != 1 || add->ne[3] != 1) return -1;
    if (add->ne[1] < 2) return -1;   // one token: handled by RAON_HCB1
    return raon_hc_boundary_impl(ctx, cgraph, j, add);
}

static int raon_hc_boundary_impl(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int i, const ggml_tensor * add) {
    static const bool enabled = raon_env_on("RAON_HC_BOUNDARY");
    if (!enabled || !ctx.fusion) return -1;

    constexpr int S  = 4;
    constexpr int NT = 512;
    constexpr int M  = 2*S + (S - 1)*(S - 1);

    const ggml_tensor * hp   = cgraph->nodes[i];
    const ggml_tensor * x    = add ? add->src[0] : hp->src[0];
    const ggml_tensor * x2   = add ? add->src[1] : nullptr;
    const ggml_tensor * post = hp->src[1];
    const ggml_tensor * res  = hp->src[2];
    const ggml_tensor * comb = hp->src[3];
    if (hp->type != GGML_TYPE_F32 || x->type != GGML_TYPE_F32 || post->type != GGML_TYPE_F32 ||
        res->type != GGML_TYPE_F32 || comb->type != GGML_TYPE_F32) return -1;
    if (add && (x2->nb[1] != x->nb[1] || add->nb[1] != x->nb[1])) return -1;
    const int64_t E = x->ne[0];
    // T = 1 decode, 2..8 spec verify / multi-slot (RAON_HC_BOUNDARY_NT=0 disables T > 1)
    const int64_t T = hp->ne[2];
    static const bool multi_t = raon_env_on("RAON_HC_BOUNDARY_NT");
    if (T < 1 || T > RAON_HCB_MAX_T || (T > 1 && !multi_t)) return -1;
    if (ggml_nrows(x) != T || x->ne[1] != T || x->nb[0] != sizeof(float) || x->nb[1] % sizeof(float) != 0) return -1;
    if (post->ne[0] != S || ggml_nrows(post) != T || post->ne[1] != T || post->nb[0] != sizeof(float) || post->nb[1] % sizeof(float) != 0) return -1;
    if (res->ne[0] != E || res->ne[1] != S || res->ne[2] != T || res->ne[3] != 1 || res->nb[0] != sizeof(float) ||
        res->nb[1] % sizeof(float) != 0 || res->nb[2] % sizeof(float) != 0) return -1;
    if (comb->ne[0] != S || comb->ne[1] != S || comb->ne[2] != T || comb->ne[3] != 1 || comb->nb[0] != sizeof(float) ||
        comb->nb[1] != S*sizeof(float) || comb->nb[2] % sizeof(float) != 0) return -1;
    if (!ggml_is_contiguous(hp) || hp->ne[0] != E || hp->ne[1] != S || hp->ne[3] != 1) return -1;
    if (E > 8*NT || (E + NT/S - 1)/(NT/S) > RAON_HCB_MAX_BLOCKS) return -1;

    const int jm = raon_next_node(cgraph, i);
    if (jm >= cgraph->n_nodes) return -1;
    const ggml_tensor * mm = cgraph->nodes[jm];
    if (mm->op != GGML_OP_MUL_MAT) return -1;
    const ggml_tensor * fn = mm->src[0];
    const ggml_tensor * y1 = mm->src[1];
    if (y1->view_src != hp && y1 != hp) return -1;
    if (y1->data != hp->data || ggml_nelements(y1) != E*S*T || y1->ne[0] != E*S || !ggml_is_contiguous(y1)) return -1;
    if (fn->type != GGML_TYPE_F16 || fn->ne[0] != E*S || fn->ne[1] != M || fn->ne[2] != 1 || fn->ne[3] != 1 ||
        fn->nb[0] != sizeof(half) || fn->nb[1] % sizeof(half) != 0) return -1;
    if (mm->type != GGML_TYPE_F32 || !ggml_is_contiguous(mm) || mm->ne[0] != M || ggml_nrows(mm) != T) return -1;

    const int jp = raon_next_node(cgraph, jm);
    if (jp >= cgraph->n_nodes) return -1;
    const ggml_tensor * pre = cgraph->nodes[jp];
    if (pre->op != GGML_OP_RAON_HC_PRE || pre->src[0] != hp || pre->src[1] != mm) return -1;
    const ggml_tensor * scale = pre->src[2];
    const ggml_tensor * base  = pre->src[3];
    if (pre->type != GGML_TYPE_F32 || !ggml_is_contiguous(pre) || ggml_nelements(pre) != T*(E + S + S*S)) return -1;
    if (scale->type != GGML_TYPE_F32 || base->type != GGML_TYPE_F32 || !ggml_is_contiguous(scale) || !ggml_is_contiguous(base)) return -1;

    const int jn = raon_next_node(cgraph, jp);
    if (jn >= cgraph->n_nodes) return -1;
    const ggml_tensor * nrm = cgraph->nodes[jn];
    if (nrm->op != GGML_OP_FUSED_RMS_NORM || nrm->op_params[1] > 1) return -1;
    const ggml_tensor * zin = nrm->src[0];
    const ggml_tensor * nw  = nrm->src[1];
    if (zin->data != pre->data || zin->ne[0] != E || ggml_nrows(zin) != T || zin->type != GGML_TYPE_F32 ||
        (T > 1 && zin->nb[1] != E*sizeof(float))) return -1;
    if (!nw || nw->type != GGML_TYPE_F32 || nw->ne[0] != E || ggml_nrows(nw) != 1) return -1;
    if (nrm->type != GGML_TYPE_F32 || !ggml_is_contiguous(nrm) || nrm->ne[0] != E || ggml_nrows(nrm) != T) return -1;

    float rms_eps, mag, norm_eps;
    memcpy(&rms_eps,  &pre->op_params[0], sizeof(float));
    memcpy(&mag,      &pre->op_params[1], sizeof(float));
    memcpy(&norm_eps, &nrm->op_params[0], sizeof(float));

    // RAON_HCB_DEBUG=1: report outputs placed over inputs
    static const bool dbg = getenv("RAON_HCB_DEBUG") && atoi(getenv("RAON_HCB_DEBUG")) != 0;
    if (dbg && T > 1) {
        auto ov = [](const ggml_tensor * a, const ggml_tensor * b) {
            const char * a0 = (const char *) a->data, * b0 = (const char *) b->data;
            return a0 < b0 + ggml_nbytes(b) && b0 < a0 + ggml_nbytes(a);
        };
        const ggml_tensor * outs[3] = { mm, pre, nrm };
        const ggml_tensor * ins[5]  = { x, post, res, comb, x2 ? x2 : x };
        for (const ggml_tensor * o : outs) for (const ggml_tensor * in : ins) {
            if (ov(o, in)) fprintf(stderr, "raon_hcb T=%d: %s (%s) overlaps input %s\n", (int) T, o->name, ggml_op_desc(o), in->name);
        }
    }

    float * pre_d = (float *) pre->data;
    const int nb = (int) ((E + NT/S - 1)/(NT/S));
    raon_hcb_strides st = {};
    st.x    = (int64_t) (x->nb[1]/sizeof(float));
    st.post = (int64_t) (post->nb[1]/sizeof(float));
    st.res  = (int64_t) (res->nb[2]/sizeof(float));
    st.comb = (int64_t) (comb->nb[2]/sizeof(float));
    st.z = E; st.pst = S; st.cmb = S*S; st.pst0 = E*T; st.cmb0 = (E + S)*T;
    k_raon_hc_boundary<S, NT><<<dim3(nb, (unsigned) T, 1), NT, 0, ctx.stream()>>>(
        (const float *) x->data, (const float *) post->data, (const float *) res->data, (const float *) comb->data,
        (int64_t) (res->nb[1]/sizeof(float)), (float *) hp->data,
        (const half *) fn->data, (int64_t) (fn->nb[1]/sizeof(half)), (const float *) scale->data, (const float *) base->data,
        rms_eps, mag, (float *) mm->data, pre_d, (const float *) nw->data, norm_eps, (float *) nrm->data, (int) E, st,
        x2 ? (const float *) x2->data : nullptr, add ? (float *) add->data : nullptr);
    ++g_raon_fused_hc_boundary;
    return jn;
}

// Final mHC collapse before the output head: hc_post -> ((y0 + y1) + y2) + y3 -> *1/S -> rms_norm * w, one block.

template <int S, int NT>
static __global__ void __launch_bounds__(NT) k_raon_hc_collapse(
        const float * __restrict__ x, const float * __restrict__ post_in, const float * __restrict__ res,
        const float * __restrict__ comb_in, const int64_t res_s1, float * __restrict__ hc_out,
        const float sc, float * __restrict__ coll, const float * __restrict__ nw, const float norm_eps,
        float * __restrict__ normed, const int E) {
    constexpr int NW = NT/WARP_SIZE;
    constexpr int MAXC = 8;
    const int tid = threadIdx.x;
    float c[MAXC];
    float ss = 0.0f;
#pragma unroll
    for (int k = 0; k < MAXC; ++k) {
        const int i0 = tid + k*NT;
        c[k] = 0.0f;
        if (i0 < E) {
            const float xv = x[i0];
            float r[S];
#pragma unroll
            for (int j = 0; j < S; ++j) r[j] = res[j*res_s1 + i0];
            float acc = 0.0f;
#pragma unroll
            for (int s = 0; s < S; ++s) {
                float sum = xv*post_in[s];
#pragma unroll
                for (int j = 0; j < S; ++j) sum += comb_in[j*S + s]*r[j];
                hc_out[s*E + i0] = sum;
                acc = s == 0 ? sum : acc + sum;
            }
            acc = sc*acc;
            coll[i0] = acc;
            c[k] = acc;
            ss += acc*acc;
        }
    }
    __shared__ float s_sum[NW];
    ss = warp_reduce_sum(ss);
    if (tid % WARP_SIZE == 0) s_sum[tid / WARP_SIZE] = ss;
    __syncthreads();
    ss = tid % WARP_SIZE < NW ? s_sum[tid % WARP_SIZE] : 0.0f;
    ss = warp_reduce_sum(ss);
    const float rs = rsqrtf(ss/E + norm_eps);
#pragma unroll
    for (int k = 0; k < MAXC; ++k) {
        const int i0 = tid + k*NT;
        if (i0 < E) normed[i0] = rs*nw[i0]*c[k];
    }
}

// view of hp's data at byte offset offs, E floats, one row
static bool raon_is_stream_view(const ggml_tensor * v, const ggml_tensor * hp, int64_t E, size_t offs) {
    return v->type == GGML_TYPE_F32 && (v->view_src == hp || v == hp) && (const char *) v->data == (const char *) hp->data + offs &&
           v->ne[0] == E && ggml_nrows(v) == 1 && v->nb[0] == sizeof(float);
}

int ggml_cuda_raon_hc_collapse(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int i) {
    static const bool enabled = raon_env_on("RAON_HC_COLLAPSE");
    if (!enabled || !ctx.fusion) return -1;

    constexpr int S  = 4;
    constexpr int NT = 1024;

    const ggml_tensor * hp   = cgraph->nodes[i];
    const ggml_tensor * x    = hp->src[0];
    const ggml_tensor * post = hp->src[1];
    const ggml_tensor * res  = hp->src[2];
    const ggml_tensor * comb = hp->src[3];
    if (hp->type != GGML_TYPE_F32 || x->type != GGML_TYPE_F32 || post->type != GGML_TYPE_F32 ||
        res->type != GGML_TYPE_F32 || comb->type != GGML_TYPE_F32) return -1;
    const int64_t E = x->ne[0];
    if (E > 8*NT || ggml_nrows(x) != 1 || x->nb[0] != sizeof(float) || post->ne[0] != S || ggml_nrows(post) != 1 || post->nb[0] != sizeof(float)) return -1;
    if (res->ne[0] != E || res->ne[1] != S || res->ne[2] != 1 || res->ne[3] != 1 || res->nb[0] != sizeof(float) ||
        res->nb[1] % sizeof(float) != 0) return -1;
    if (comb->ne[0] != S || comb->ne[1] != S || comb->ne[2] != 1 || comb->ne[3] != 1 || comb->nb[0] != sizeof(float) ||
        comb->nb[1] != S*sizeof(float)) return -1;
    if (!ggml_is_contiguous(hp) || hp->ne[0] != E || hp->ne[1] != S || hp->ne[2] != 1 || hp->ne[3] != 1) return -1;

    // ADD(view0, view1), ADD(prev, view2), ..., SCALE(prev, no bias), FUSED_RMS_NORM(scale, w)
    int j = i;
    const ggml_tensor * prev = nullptr;
    for (int s = 1; s < S; ++s) {
        j = raon_next_node(cgraph, j);
        if (j >= cgraph->n_nodes) return -1;
        const ggml_tensor * a = cgraph->nodes[j];
        if (a->op != GGML_OP_ADD || a->type != GGML_TYPE_F32 || !ggml_is_contiguous(a) || a->ne[0] != E || ggml_nrows(a) != 1) return -1;
        if (s == 1) {
            if (!raon_is_stream_view(a->src[0], hp, E, 0)) return -1;
        } else if (a->src[0] != prev) return -1;
        if (!raon_is_stream_view(a->src[1], hp, E, s*E*sizeof(float))) return -1;
        prev = a;
    }
    j = raon_next_node(cgraph, j);
    if (j >= cgraph->n_nodes) return -1;
    const ggml_tensor * scn = cgraph->nodes[j];
    if (scn->op != GGML_OP_SCALE || scn->src[0] != prev || scn->type != GGML_TYPE_F32 || !ggml_is_contiguous(scn)) return -1;
    float sc, bias;
    memcpy(&sc,   &scn->op_params[0], sizeof(float));
    memcpy(&bias, &scn->op_params[1], sizeof(float));
    if (bias != 0.0f) return -1;
    j = raon_next_node(cgraph, j);
    if (j >= cgraph->n_nodes) return -1;
    const ggml_tensor * nrm = cgraph->nodes[j];
    if (nrm->op != GGML_OP_FUSED_RMS_NORM || nrm->src[0] != scn || nrm->op_params[1] > 1) return -1;
    const ggml_tensor * nw = nrm->src[1];
    if (!nw || nw->type != GGML_TYPE_F32 || nw->ne[0] != E || ggml_nrows(nw) != 1) return -1;
    if (nrm->type != GGML_TYPE_F32 || !ggml_is_contiguous(nrm) || nrm->ne[0] != E || ggml_nrows(nrm) != 1) return -1;
    float norm_eps;
    memcpy(&norm_eps, &nrm->op_params[0], sizeof(float));

    k_raon_hc_collapse<S, NT><<<1, NT, 0, ctx.stream()>>>(
        (const float *) x->data, (const float *) post->data, (const float *) res->data, (const float *) comb->data,
        (int64_t) (res->nb[1]/sizeof(float)), (float *) hp->data, sc, (float *) scn->data,
        (const float *) nw->data, norm_eps, (float *) nrm->data, (int) E);
    ++g_raon_fused_hc_collapse;
    return j;
}

// Small F16 matvecs (one token): up to 4 MUL_MATs sharing one F32 input in one launch, and the short-row ssm_f_b.
// L lanes per row, 16-byte loads.

#define RAON_MV_MAX_JOBS 4
struct raon_mv_params {
    const half * w[RAON_MV_MAX_JOBS];
    float      * dst[RAON_MV_MAX_JOBS];
    int64_t      s1[RAON_MV_MAX_JOBS];      // row stride in halves
    int          row0[RAON_MV_MAX_JOBS + 1]; // first global row of each job
    int          njob;
    int64_t      x_s1;                       // x column stride in floats (several tokens)
};

// U > 1 (RAON_F16_MV_U): each lane issues U loads before accumulating, in the same k order as U = 1.
template <int L, int U>
static __global__ void __launch_bounds__(256) k_raon_f16_mv(const __grid_constant__ raon_mv_params p, const float * __restrict__ x,
        const int ne0, const int nrows) {
    const int lane = threadIdx.x % L;
    const int grow = (blockIdx.x*blockDim.x + threadIdx.x) / L;
    const bool valid = grow < nrows;
    // select the job without dynamic indexing into the parameter struct (that would copy it to local memory)
    const half * w = p.w[0]; float * dst = p.dst[0]; int64_t s1 = p.s1[0]; int r0 = 0;
#pragma unroll
    for (int k = 1; k < RAON_MV_MAX_JOBS; ++k) {
        if (k < p.njob && grow >= p.row0[k]) { w = p.w[k]; dst = p.dst[k]; s1 = p.s1[k]; r0 = p.row0[k]; }
    }
    const int row = grow - r0;
    const uint4  * wr = (const uint4 *) (w + (int64_t) row*s1);
    const float4 * xr = (const float4 *) x;
    const int n8 = valid ? ne0/8 : 0;
    float sum = 0.0f;
    for (int k0 = lane; k0 < n8; k0 += L*U) {
        uint4 wv[U]; float4 x0[U], x1[U];
#pragma unroll
        for (int u = 0; u < U; ++u) {
            const int k = k0 + u*L;
            if (k < n8) { wv[u] = wr[k]; x0[u] = xr[2*k + 0]; x1[u] = xr[2*k + 1]; }
        }
#pragma unroll
        for (int u = 0; u < U; ++u) {
            if (k0 + u*L < n8) {
                const half2 * h = (const half2 *) &wv[u];
                const float2 f0 = __half22float2(h[0]);
                const float2 f1 = __half22float2(h[1]);
                const float2 f2 = __half22float2(h[2]);
                const float2 f3 = __half22float2(h[3]);
                sum += f0.x*x0[u].x + f0.y*x0[u].y + f1.x*x0[u].z + f1.y*x0[u].w + f2.x*x1[u].x + f2.y*x1[u].y + f3.x*x1[u].z + f3.y*x1[u].w;
            }
        }
    }
#pragma unroll
    for (int off = L/2; off > 0; off >>= 1) {
        sum += __shfl_xor_sync(0xffffffff, sum, off, WARP_SIZE);
    }
    if (valid && lane == 0) {
        dst[row] = sum;
    }
}

// several tokens (2-8 columns, spec verify windows): one pass, NC partial sums per lane
template <int L, int NC>
static __global__ void __launch_bounds__(256) k_raon_f16_mv_nc(const __grid_constant__ raon_mv_params p, const float * __restrict__ x,
        const int ne0, const int nrows) {
    const int lane = threadIdx.x % L;
    const int grow = (blockIdx.x*blockDim.x + threadIdx.x) / L;
    const bool valid = grow < nrows;
    // select the job without dynamic indexing into the parameter struct (that would copy it to local memory)
    const half * w = p.w[0]; float * dst = p.dst[0]; int64_t s1 = p.s1[0]; int r0 = 0;
#pragma unroll
    for (int k = 1; k < RAON_MV_MAX_JOBS; ++k) {
        if (k < p.njob && grow >= p.row0[k]) { w = p.w[k]; dst = p.dst[k]; s1 = p.s1[k]; r0 = p.row0[k]; }
    }
    const int row = grow - r0;
    // rows of this job, static indices only
    int nr = p.row0[1];
#pragma unroll
    for (int k = 1; k < RAON_MV_MAX_JOBS; ++k) {
        if (k < p.njob && grow >= p.row0[k]) { nr = p.row0[k + 1] - p.row0[k]; }
    }
    const uint4  * wr = (const uint4 *) (w + (int64_t) row*s1);
    const int n8 = valid ? ne0/8 : 0;
    float sum[NC];
#pragma unroll
    for (int c = 0; c < NC; ++c) sum[c] = 0.0f;
    for (int k = lane; k < n8; k += L) {
        const uint4 wv = wr[k];
        const half2 * h = (const half2 *) &wv;
        const float2 f0 = __half22float2(h[0]);
        const float2 f1 = __half22float2(h[1]);
        const float2 f2 = __half22float2(h[2]);
        const float2 f3 = __half22float2(h[3]);
#pragma unroll
        for (int c = 0; c < NC; ++c) {
            const float4 * xr = (const float4 *) (x + c*p.x_s1);
            const float4 x0 = xr[2*k + 0];
            const float4 x1 = xr[2*k + 1];
            sum[c] += f0.x*x0.x + f0.y*x0.y + f1.x*x0.z + f1.y*x0.w + f2.x*x1.x + f2.y*x1.y + f3.x*x1.z + f3.y*x1.w;
        }
    }
#pragma unroll
    for (int c = 0; c < NC; ++c) {
#pragma unroll
        for (int off = L/2; off > 0; off >>= 1) {
            sum[c] += __shfl_xor_sync(0xffffffff, sum[c], off, WARP_SIZE);
        }
    }
    if (valid && lane == 0) {
#pragma unroll
        for (int c = 0; c < NC; ++c) dst[(int64_t) c*nr + row] = sum[c];
    }
}

// RAON_F16_MV_NCOLS=0: single-token only
static bool raon_f16_mv_ncols_on() {
    static const bool on = raon_env_on("RAON_F16_MV_NCOLS");
    return on;
}

raon_fb_pending g_raon_fb_pending;

bool ggml_cuda_raon_fb_fold_enabled() {
    static const bool on = raon_env_on("RAON_FB_FOLD");
    return on;
}

void ggml_cuda_raon_fb_launch(const raon_fb_pending & fb, cudaStream_t stream) {
    raon_mv_params p = {};
    p.w[0] = fb.w; p.dst[0] = fb.dst; p.s1[0] = fb.s1; p.row0[0] = 0; p.row0[1] = fb.nrows; p.njob = 1;
    const int64_t grid = ((int64_t) fb.nrows*16 + 255)/256;
    k_raon_f16_mv<16, 1><<<grid, 256, 0, stream>>>(p, fb.x, 128, fb.nrows);
}

static bool raon_mv_ok(const ggml_tensor * mm) {
    const ggml_tensor * w = mm->src[0];
    const ggml_tensor * x = mm->src[1];
    return mm->op == GGML_OP_MUL_MAT && w->type == GGML_TYPE_F16 && x->type == GGML_TYPE_F32 && mm->type == GGML_TYPE_F32 &&
           w->ne[2] == 1 && w->ne[3] == 1 && w->nb[0] == sizeof(half) && w->nb[1] % 16 == 0 && raon_aligned16(w->data) &&
           w->ne[0] % 8 == 0 && w->ne[0] == x->ne[0] && x->nb[0] == sizeof(float) && raon_aligned16(x->data) &&
           x->ne[2] == 1 && x->ne[3] == 1 && x->ne[1] >= 1 && x->ne[1] <= 8 && (x->ne[1] == 1 || (x->nb[1] % 16 == 0 && raon_f16_mv_ncols_on())) &&
           ggml_is_contiguous(mm) && mm->ne[0] == w->ne[1] && mm->ne[1] == x->ne[1] && ggml_nrows(mm) == x->ne[1] && w->ne[1] <= (1 << 20);
}

int ggml_cuda_raon_f16_mv(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int i) {
    static const bool enabled = raon_env_on("RAON_F16_MV");
    if (!enabled || !ctx.fusion) return -1;

    const ggml_tensor * mm0 = cgraph->nodes[i];
    if (!raon_mv_ok(mm0)) return -1;
    const ggml_tensor * x = mm0->src[1];
    const int64_t ne0 = x->ne[0];

    raon_mv_params p = {};
    int last = i;
    int nrows = 0;
    p.njob = 0;
    for (int j = i; j < cgraph->n_nodes && p.njob < RAON_MV_MAX_JOBS; ) {
        const ggml_tensor * mm = cgraph->nodes[j];
        if (j != i && (!raon_mv_ok(mm) || mm->src[1] != x)) break;
        p.w[p.njob]    = (const half *) mm->src[0]->data;
        p.dst[p.njob]  = (float *) mm->data;
        p.s1[p.njob]   = mm->src[0]->nb[1]/sizeof(half);
        p.row0[p.njob] = nrows;
        nrows += (int) mm->ne[0];
        ++p.njob;
        last = j;
        j = raon_next_node(cgraph, j);
    }
    p.row0[p.njob] = nrows;
    // a lone matvec with long rows stays on mul_mat_vec_f
    if (p.njob == 1 && ne0 > 512) return -1;

    const int n8 = (int) (ne0/8);
    const int L  = n8 >= 32 ? 32 : n8 >= 16 ? 16 : n8 >= 8 ? 8 : n8 >= 4 ? 4 : 0;
    if (L == 0) return -1;
    const int nc = (int) x->ne[1];
    if (nc > 1) {
        p.x_s1 = (int64_t) (x->nb[1]/sizeof(float));
        const int nthreads = 256;
        const int64_t grid = ((int64_t) nrows*L + nthreads - 1)/nthreads;
#define RAON_MV_L(LL) switch (nc) { \
            case 2: k_raon_f16_mv_nc<LL, 2><<<grid, nthreads, 0, ctx.stream()>>>(p, (const float *) x->data, (int) ne0, nrows); break; \
            case 3: k_raon_f16_mv_nc<LL, 3><<<grid, nthreads, 0, ctx.stream()>>>(p, (const float *) x->data, (int) ne0, nrows); break; \
            case 4: k_raon_f16_mv_nc<LL, 4><<<grid, nthreads, 0, ctx.stream()>>>(p, (const float *) x->data, (int) ne0, nrows); break; \
            case 5: k_raon_f16_mv_nc<LL, 5><<<grid, nthreads, 0, ctx.stream()>>>(p, (const float *) x->data, (int) ne0, nrows); break; \
            case 6: k_raon_f16_mv_nc<LL, 6><<<grid, nthreads, 0, ctx.stream()>>>(p, (const float *) x->data, (int) ne0, nrows); break; \
            case 7: k_raon_f16_mv_nc<LL, 7><<<grid, nthreads, 0, ctx.stream()>>>(p, (const float *) x->data, (int) ne0, nrows); break; \
            default: k_raon_f16_mv_nc<LL, 8><<<grid, nthreads, 0, ctx.stream()>>>(p, (const float *) x->data, (int) ne0, nrows); break; }
        switch (L) {
            case 32: RAON_MV_L(32) break;
            case 16: RAON_MV_L(16) break;
            case  8: RAON_MV_L( 8) break;
            case  4: RAON_MV_L( 4) break;
        }
#undef RAON_MV_L
        ++g_raon_fused_f16_mv;
        return last;
    }

    // RAON_FB_FOLD: a lone [128 -> H*128] matvec feeding a one-token RDA_DECODE is folded into the RDA prep kernel
    if (p.njob == 1 && ne0 == 128 && ggml_cuda_raon_fb_fold_enabled()) {
        const int jr = raon_next_node(cgraph, last);
        const ggml_tensor * rda = jr < cgraph->n_nodes ? cgraph->nodes[jr] : nullptr;
        if (rda && rda->op == GGML_OP_RDA_DECODE && rda->src[8] && rda->src[8]->data == (void *) p.dst[0] &&
            rda->op_params[GGML_RDA_P_NSEQS] == 1 && rda->src[0]->ne[1] == 1 && rda->src[12]->ne[0] == 128 &&
            rda->src[8]->type == GGML_TYPE_F32 && ggml_nelements(rda->src[8]) == nrows &&
            nrows == rda->src[10]->ne[0]*128 && g_raon_fb_pending.dst == nullptr) {
            // the RDA op must be the only reader of the matvec output
            bool other = false;
            for (int j = jr + 1; j < cgraph->n_nodes && !other; ++j) {
                for (int k = 0; k < GGML_MAX_SRC; ++k) {
                    const ggml_tensor * sj = cgraph->nodes[j]->src[k];
                    if (sj && (sj == mm0 || sj->view_src == mm0)) { other = true; break; }
                }
            }
            if (other) goto no_fold;
            g_raon_fb_pending.w = p.w[0]; g_raon_fb_pending.x = (const float *) x->data; g_raon_fb_pending.dst = p.dst[0];
            g_raon_fb_pending.s1 = p.s1[0]; g_raon_fb_pending.nrows = nrows;
            ++g_raon_fused_f16_mv;
            return last;
        }
    }
no_fold:
    const int nthreads = 256;
    const int64_t grid = ((int64_t) nrows*L + nthreads - 1)/nthreads;
    static const int unr = [] { const char * e = getenv("RAON_F16_MV_U"); return e ? atoi(e) : 10; }();
    const bool big = unr > 1 && n8 > L;    // rows longer than one pass: several loads in flight per lane
#define RAON_MV_LAUNCH(LL) big ? k_raon_f16_mv<LL, 10><<<grid, nthreads, 0, ctx.stream()>>>(p, (const float *) x->data, (int) ne0, nrows) \
                               : k_raon_f16_mv<LL, 1><<<grid, nthreads, 0, ctx.stream()>>>(p, (const float *) x->data, (int) ne0, nrows)
    switch (L) {
        case 32: RAON_MV_LAUNCH(32); break;
        case 16: RAON_MV_LAUNCH(16); break;
        case  8: RAON_MV_LAUNCH( 8); break;
        case  4: RAON_MV_LAUNCH( 4); break;
    }
#undef RAON_MV_LAUNCH
    ++g_raon_fused_f16_mv;
    return last;
}

// MLA small ops (one token), RAON_MLA_FUSE:
//   kv_a latent: RMS_NORM -> MUL(w) -> CONCAT(extra, latent) -> CPY to the q8_0 K cache row (CPY node's graph slot)
//   q_a / any:   RMS_NORM -> MUL(1-row weight)
//   gate:        sigmoid -> MUL(kqv, sigmoid)
// The unfused expression order is kept.

#include "cpy-utils.cuh"
#include "graph.cuh"

int64_t g_raon_fused_mla = 0;

template <int NT>
static __device__ __forceinline__ float raon_block_sum(float v, float * s_sum) {
    v = warp_reduce_sum(v);
    if (NT > WARP_SIZE) {
        if (threadIdx.x % WARP_SIZE == 0) s_sum[threadIdx.x / WARP_SIZE] = v;
        __syncthreads();
        v = threadIdx.x % WARP_SIZE < NT/WARP_SIZE ? s_sum[threadIdx.x % WARP_SIZE] : 0.0f;
        v = warp_reduce_sum(v);
    }
    return v;
}

// rms_norm(x)*w -> y (and optionally the q8_0 cache row [extra | y] plus the f32 concat)
template <int NT>
static __global__ void __launch_bounds__(NT) k_raon_rms_mul(const float * __restrict__ x, const float * __restrict__ w,
        float * __restrict__ y, const int n, const float eps,
        const float * __restrict__ extra, const int nx, float * __restrict__ cat, char * cache, char ** cache_ind, const int cache_idx) {
    __shared__ float s_sum[WARP_SIZE];
    __shared__ float s_row[4096];
    float ss = 0.0f;
    for (int i = threadIdx.x; i < n; i += NT) { const float v = x[i]; ss += v*v; }
    ss = raon_block_sum<NT>(ss, s_sum);
    const float scale = rsqrtf(ss/n + eps);
    for (int i = threadIdx.x; i < n; i += NT) {
        const float v = (scale*x[i])*w[i];
        y[i] = v;
        if (cat) { s_row[nx + i] = v; cat[nx + i] = v; }
    }
    if (!cat) return;
    for (int i = threadIdx.x; i < nx; i += NT) { const float v = extra[i]; s_row[i] = v; cat[i] = v; }
    __syncthreads();
    char * dst = cache_ind ? cache_ind[cache_idx] : cache;
    for (int b = threadIdx.x; b < (nx + n)/QK8_0; b += NT) {
        quantize_f32_q8_0_block(s_row + b*QK8_0, (block_q8_0 *) dst + b);
    }
}

static bool raon_row_f32(const ggml_tensor * t) {
    return t->type == GGML_TYPE_F32 && ggml_nrows(t) == 1 && t->nb[0] == sizeof(float);
}

static bool raon_mla_enabled() {
    static const bool on = raon_env_on("RAON_MLA_FUSE");
    return on;
}

int ggml_cuda_raon_rms_norm_mul(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int i) {
    if (!raon_mla_enabled() || !ctx.fusion) return -1;
    const ggml_tensor * rms = cgraph->nodes[i];
    const ggml_tensor * x = rms->src[0];
    if (!raon_row_f32(x) || !raon_row_f32(rms) || x->ne[0] > 4096) return -1;
    const int n = (int) x->ne[0];
    const int jm = raon_next_node(cgraph, i);
    if (jm >= cgraph->n_nodes) return -1;
    const ggml_tensor * mul = cgraph->nodes[jm];
    if (mul->op != GGML_OP_MUL || mul->src[0] != rms || !raon_row_f32(mul) || mul->ne[0] != n) return -1;
    const ggml_tensor * w = mul->src[1];
    if (!raon_row_f32(w) || w->ne[0] != n) return -1;
    float eps;
    memcpy(&eps, rms->op_params, sizeof(float));

    // kv_a tail: CONCAT(extra, mul) dim 0 -> CPY to a q8_0 row
    int last = jm;
    const float * extra = nullptr; float * cat = nullptr; char * cache = nullptr; char ** cache_ind = nullptr;
    int nx = 0, cache_idx = -1;
    const int jc = raon_next_node(cgraph, jm);
    if (jc < cgraph->n_nodes) {
        const ggml_tensor * cc = cgraph->nodes[jc];
        const int jp = raon_next_node(cgraph, jc);
        if (cc->op == GGML_OP_CONCAT && ((const int32_t *) cc->op_params)[0] == 0 && cc->src[1] == mul && raon_row_f32(cc) &&
            raon_row_f32(cc->src[0]) && ggml_is_contiguous(cc) && cc->ne[0] == cc->src[0]->ne[0] + n && cc->ne[0] <= 4096 &&
            jp < cgraph->n_nodes) {
            const ggml_tensor * cp = cgraph->nodes[jp];
            if (cp->op == GGML_OP_CPY && cp->src[0] == cc && cp->src[1]->type == GGML_TYPE_Q8_0 && cp->src[1]->ne[0] == cc->ne[0] &&
                ggml_nrows(cp->src[1]) == 1 && cc->ne[0] % QK8_0 == 0 && cc->src[0]->ne[0] % QK8_0 == 0) {
                extra = (const float *) cc->src[0]->data;
                nx    = (int) cc->src[0]->ne[0];
                cat   = (float *) cc->data;
                cache = (char *) cp->src[1]->data;
                if (ctx.cur_graph && ctx.cur_graph->use_cpy_indirection) {
                    cache_ind = ctx.cur_graph->dest_ptrs_d;
                    cache_idx = ctx.cur_graph->graph_cpynode_index++;
                }
                last = jp;
            }
        }
    }
    k_raon_rms_mul<512><<<1, 512, 0, ctx.stream()>>>((const float *) x->data, (const float *) w->data, (float *) mul->data, n, eps,
            extra, nx, cat, cache, cache_ind, cache_idx);
    ++g_raon_fused_mla;
    return last;
}

static __global__ void k_raon_sigmoid_mul(const float * __restrict__ g, float * __restrict__ sg, const float * __restrict__ a,
        float * __restrict__ dst, const int n) {
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float s = 1.0f / (1.0f + expf(-g[i]));
    sg[i] = s;
    dst[i] = a[i]*s;
}

int ggml_cuda_raon_sigmoid_mul(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int i) {
    if (!raon_mla_enabled() || !ctx.fusion) return -1;
    const ggml_tensor * u = cgraph->nodes[i];
    if (ggml_get_unary_op(u) != GGML_UNARY_OP_SIGMOID || u->type != GGML_TYPE_F32 || !ggml_is_contiguous(u) ||
        !ggml_is_contiguous(u->src[0]) || u->src[0]->type != GGML_TYPE_F32) return -1;
    const int jm = raon_next_node(cgraph, i);
    if (jm >= cgraph->n_nodes) return -1;
    const ggml_tensor * mul = cgraph->nodes[jm];
    if (mul->op != GGML_OP_MUL || mul->src[1] != u) return -1;
    const ggml_tensor * a = mul->src[0];
    if (a->type != GGML_TYPE_F32 || mul->type != GGML_TYPE_F32 || !ggml_is_contiguous(a) || !ggml_is_contiguous(mul) ||
        !ggml_are_same_shape(a, u) || !ggml_are_same_shape(mul, u)) return -1;
    const int n = (int) ggml_nelements(u);
    k_raon_sigmoid_mul<<<(n + 255)/256, 256, 0, ctx.stream()>>>((const float *) u->src[0]->data, (float *) u->data,
            (const float *) a->data, (float *) mul->data, n);
    ++g_raon_fused_mla;
    return jm;
}
