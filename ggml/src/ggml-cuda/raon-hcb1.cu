// mHC boundary, single-token path. Same node pattern as ggml_cuda_raon_hc_boundary; anything else returns -1.

#include "common.cuh"
#include "raon-smallfuse.cuh"
#include "raon-hcb1.cuh"
#include "raon-hcb1-kern.cuh"

static int raon_hcb1_next_node(const ggml_cgraph * cgraph, int i) {
    int j = i + 1;
    while (j < cgraph->n_nodes && ggml_is_noop(cgraph->nodes[j])) ++j;
    return j;
}

static bool raon_hcb1_a16(const void * p) { return ((uintptr_t) p) % 16 == 0; }

static int raon_hcb1_env(const char * name, int def) {
    const char * e = getenv(name);
    return e ? atoi(e) : def;
}

int ggml_cuda_raon_hcb1(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int i) {
    static const int variant = raon_hcb1_env("RAON_HCB1", 1) ? raon_hcb1_env("RAON_HCB1_VARIANT", 14) : 0;
    static const bool hcb_on = raon_hcb1_env("RAON_HC_BOUNDARY", 1) != 0;
    if (variant <= 0 || !hcb_on || !ctx.fusion) return -1;

    constexpr int S = 4;
    constexpr int M = 2*S + (S - 1)*(S - 1);

    const ggml_tensor * hp   = cgraph->nodes[i];
    const ggml_tensor * x    = hp->src[0];
    const ggml_tensor * post = hp->src[1];
    const ggml_tensor * res  = hp->src[2];
    const ggml_tensor * comb = hp->src[3];
    if (hp->type != GGML_TYPE_F32 || x->type != GGML_TYPE_F32 || post->type != GGML_TYPE_F32 ||
        res->type != GGML_TYPE_F32 || comb->type != GGML_TYPE_F32) return -1;
    const int64_t E = x->ne[0];
    if (ggml_nrows(x) != 1 || x->nb[0] != sizeof(float) || post->ne[0] != S || ggml_nrows(post) != 1 || post->nb[0] != sizeof(float)) return -1;
    if (res->ne[0] != E || res->ne[1] != S || res->ne[2] != 1 || res->ne[3] != 1 || res->nb[0] != sizeof(float) ||
        res->nb[1] % (4*sizeof(float)) != 0) return -1;
    if (comb->ne[0] != S || comb->ne[1] != S || comb->ne[2] != 1 || comb->ne[3] != 1 || comb->nb[0] != sizeof(float) ||
        comb->nb[1] != S*sizeof(float)) return -1;
    if (!ggml_is_contiguous(hp) || hp->ne[0] != E || hp->ne[1] != S || hp->ne[2] != 1 || hp->ne[3] != 1) return -1;

    const int jm = raon_hcb1_next_node(cgraph, i);
    if (jm >= cgraph->n_nodes) return -1;
    const ggml_tensor * mm = cgraph->nodes[jm];
    if (mm->op != GGML_OP_MUL_MAT) return -1;
    const ggml_tensor * fn = mm->src[0];
    const ggml_tensor * y1 = mm->src[1];
    if (y1->view_src != hp && y1 != hp) return -1;
    if (y1->data != hp->data || ggml_nelements(y1) != E*S || y1->ne[0] != E*S) return -1;
    if (fn->type != GGML_TYPE_F16 || fn->ne[0] != E*S || fn->ne[1] != M || fn->ne[2] != 1 || fn->ne[3] != 1 ||
        fn->nb[0] != sizeof(half) || fn->nb[1] % 16 != 0) return -1;
    if (mm->type != GGML_TYPE_F32 || !ggml_is_contiguous(mm) || mm->ne[0] != M || ggml_nrows(mm) != 1) return -1;

    const int jp = raon_hcb1_next_node(cgraph, jm);
    if (jp >= cgraph->n_nodes) return -1;
    const ggml_tensor * pre = cgraph->nodes[jp];
    if (pre->op != GGML_OP_RAON_HC_PRE || pre->src[0] != hp || pre->src[1] != mm) return -1;
    const ggml_tensor * scale = pre->src[2];
    const ggml_tensor * base  = pre->src[3];
    if (pre->type != GGML_TYPE_F32 || !ggml_is_contiguous(pre) || ggml_nelements(pre) != E + S + S*S) return -1;
    if (scale->type != GGML_TYPE_F32 || base->type != GGML_TYPE_F32 || !ggml_is_contiguous(scale) || !ggml_is_contiguous(base)) return -1;

    const int jn = raon_hcb1_next_node(cgraph, jp);
    if (jn >= cgraph->n_nodes) return -1;
    const ggml_tensor * nrm = cgraph->nodes[jn];
    if (nrm->op != GGML_OP_FUSED_RMS_NORM || nrm->op_params[1] > 1) return -1;
    const ggml_tensor * zin = nrm->src[0];
    const ggml_tensor * nw  = nrm->src[1];
    if (zin->data != pre->data || zin->ne[0] != E || ggml_nrows(zin) != 1 || zin->type != GGML_TYPE_F32) return -1;
    if (!nw || nw->type != GGML_TYPE_F32 || nw->ne[0] != E || ggml_nrows(nw) != 1 || !ggml_is_contiguous(nw)) return -1;
    if (nrm->type != GGML_TYPE_F32 || !ggml_is_contiguous(nrm) || nrm->ne[0] != E || ggml_nrows(nrm) != 1) return -1;

    if (!raon_hcb1_a16(x->data) || !raon_hcb1_a16(res->data) || !raon_hcb1_a16(hp->data) || !raon_hcb1_a16(fn->data) ||
        !raon_hcb1_a16(pre->data) || !raon_hcb1_a16(nw->data) || !raon_hcb1_a16(nrm->data) || E % 8 != 0) return -1;

    raon_hcb1_args a = {};
    a.x = (const float *) x->data; a.post = (const float *) post->data; a.res = (const float *) res->data;
    a.comb = (const float *) comb->data; a.res_s1 = (int64_t) (res->nb[1]/sizeof(float)); a.hc_out = (float *) hp->data;
    a.fn = (const half *) fn->data; a.fn_s1 = (int64_t) (fn->nb[1]/sizeof(half));
    a.scale = (const float *) scale->data; a.base = (const float *) base->data;
    memcpy(&a.rms_eps,  &pre->op_params[0], sizeof(float));
    memcpy(&a.mag,      &pre->op_params[1], sizeof(float));
    memcpy(&a.norm_eps, &nrm->op_params[0], sizeof(float));
    a.mixes = (float *) mm->data; a.all = (float *) pre->data; a.nw = (const float *) nw->data; a.normed = (float *) nrm->data;
    a.E = (int) E;
    a.fn_next = nullptr;
    // RAON_HCB1_PREFETCH=1: prefetch the next boundary's hc_fn into L2
    static const bool prefetch = raon_hcb1_env("RAON_HCB1_PREFETCH", 0) != 0;
    if (prefetch) {
        for (int k = jn + 1; k < cgraph->n_nodes; ++k) {
            const ggml_tensor * t = cgraph->nodes[k];
            if (t->op != GGML_OP_HC_POST) continue;
            const int km = raon_hcb1_next_node(cgraph, k);
            if (km < cgraph->n_nodes && cgraph->nodes[km]->op == GGML_OP_MUL_MAT) {
                const ggml_tensor * f2 = cgraph->nodes[km]->src[0];
                if (f2->type == fn->type && f2->ne[0] == fn->ne[0] && f2->ne[1] == fn->ne[1] && f2->nb[1] == fn->nb[1] &&
                    ggml_nbytes(f2) == ggml_nbytes(fn) && f2->data != fn->data) {
                    a.fn_next = (const half *) f2->data;
                }
            }
            break;
        }
    }
    if (!raon_hcb1_launch(a, variant, ctx.stream(), nullptr)) return -1;
    ++g_raon_fused_hc_boundary;
    return jn;
}
