#include <algorithm>
#include "../llama-build-context.h"
#include "../llama-context.h"
#include "../llama-delta-net.h"
#include "../llama-model.h"

// Raon-1: RDA linear attention (KDA with a separate write gate) + absorbed MLA, blocks wrapped in mHC streams
// mixed by a doubly stochastic Helmert chart. Dense lead FFN, then sigmoid-routed MoE + shared expert, SiTU FFNs.
// Reference: KRAFTON's llama.cpp fork, src/models/raon1.cpp.

// constant [ne0, ne1] tensor built from fills so it stays on the backend that runs the graph
static ggml_tensor * raon1_const_2d(ggml_context * ctx0, const std::vector<float> & v, int64_t ne0, int64_t ne1) {
    ggml_tensor * out = nullptr;
    for (int64_t i1 = 0; i1 < ne1; ++i1) {
        ggml_tensor * row = nullptr;
        for (int64_t i0 = 0; i0 < ne0; ++i0) {
            ggml_tensor * c = ggml_fill(ctx0, ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, 1), v[i1*ne0 + i0]);
            row = row ? ggml_concat(ctx0, row, c, 0) : c;
        }
        row = ggml_reshape_2d(ctx0, row, ne0, 1);
        out = out ? ggml_concat(ctx0, out, row, 1) : row;
    }
    return out;
}

struct raon1_hc_consts {
    ggml_tensor * helmert;    // [hc-1, hc]
    ggml_tensor * identity;   // [hc, hc]
    ggml_tensor * chart_base; // [hc, hc]: 0.5*I + 0.5/hc
};

// RAON_MHC_FUSED: 1 (default) fused HC_PRE with rms_norm folded in, 2 fused without the fold, 0 composed graph
static int raon1_mhc_mode() {
    static const int mode = [] {
        const char * e = getenv("RAON_MHC_FUSED");
        const int m = e ? atoi(e) : 1;
        return m >= 0 && m <= 2 ? m : 1;
    }();
    return mode;
}

// fused mHC pre: same outputs as raon1_build_hc_pre
static ggml_tensor * raon1_build_hc_pre_fused(llm_build_context & llm,
        ggml_tensor * x, ggml_tensor * fn, ggml_tensor * scale, ggml_tensor * base,
        ggml_tensor ** post_out, ggml_tensor ** comb_out, int il, bool fold_rms) {
    auto & ctx0    = llm.ctx0;
    auto & hparams = llm.hparams;
    auto & cb      = llm.cb;

    const int64_t n_embd = hparams.n_embd;
    const int64_t hc = hparams.dsv4_hc_mult;
    const int64_t nt = x->ne[2];

    if (!ggml_is_contiguous(x)) {
        x = ggml_cont(ctx0, x);
    }
    // (fn @ x) / rms(x) == fn @ rms_norm(x)
    ggml_tensor * flat  = ggml_reshape_2d(ctx0, x, n_embd*hc, nt);
    ggml_tensor * mixes = ggml_mul_mat(ctx0, fn, fold_rms ? flat : ggml_rms_norm(ctx0, flat, hparams.dsv4_hc_eps));
    if (fold_rms) {
        // fn is F16 and the un-normalized dot products can overflow an F16 result
        ggml_mul_mat_set_prec(mixes, GGML_PREC_F32);
    }
    cb(mixes, "hc_mixes", il);

    ggml_tensor * all = ggml_raon_hc_pre(ctx0, x, mixes, scale, base,
            fold_rms ? hparams.dsv4_hc_eps : -1.0f, hparams.hc_magnitude);
    cb(all, "hc_pre_fused", il);

    const size_t fs = sizeof(float);
    *post_out = ggml_view_2d(ctx0, all, hc, nt, hc*fs, n_embd*nt*fs);
    *comb_out = ggml_view_3d(ctx0, all, hc, hc, nt, hc*fs, hc*hc*fs, (n_embd + hc)*nt*fs);
    return ggml_view_2d(ctx0, all, n_embd, nt, n_embd*fs, 0);
}

// RAON_MK_MOE=0 disables the fused single-token MoE block
static bool raon1_mk_moe_enabled() {
    static const bool on = [] {
        const char * e = getenv("RAON_MK_MOE");
        return !(e && atoi(e) == 0);
    }();
    return on;
}

// RAON_MK_MOE_LAYERS=a-b (or a single index): fuse only those layers
static bool raon1_mk_moe_layer_selected(int il) {
    static const std::pair<int, int> range = [] {
        const char * e = getenv("RAON_MK_MOE_LAYERS");
        int a = 0, b = 1 << 30;
        if (e && *e) {
            if (sscanf(e, "%d-%d", &a, &b) == 1) b = a;
        }
        return std::make_pair(a, b);
    }();
    return il >= range.first && il <= range.second;
}

// only when the CUDA kernel can run it and the composed path would add nothing (moe cache, LoRA, Hadamard)
static bool raon1_can_use_mk_moe(const llm_build_context & llm, const llama_layer & layer, int64_t n_tokens,
        int64_t n_expert_used, int il) {
    if (!raon1_mk_moe_enabled() || n_tokens != 1 || !raon1_mk_moe_layer_selected(il)) return false;
    const auto & lctx = llm.lctx;
    const auto & hp   = llm.hparams;
    if (!lctx.cparams.fused_moe_up_gate || lctx.moe_cache || !lctx.lora_adapters.empty()) return false;
    if ((llm_expert_gating_func_type) hp.expert_gating_func != LLM_EXPERT_GATING_FUNC_SIGMOID) return false;
    // llm_build_context uses n_expert_used = n_expert for the warmup graph: that one stays composed
    if (n_expert_used != 6) return false;
    const ggml_tensor * w[9] = { layer.ffn_gate_inp, layer.ffn_exp_probs_b,
        layer.ffn_up_exps, layer.ffn_gate_exps, layer.ffn_down_exps,
        layer.ffn_up_shexp, layer.ffn_gate_shexp, layer.ffn_down_shexp, nullptr };
    for (int i = 0; i < 8; ++i) {
        if (!w[i] || w[i]->extra || !w[i]->buffer || ggml_backend_buffer_is_host(w[i]->buffer)) return false;
        if (lctx.model.hadamard_rotation(w[i]) != nullptr) return false;
    }
    return layer.ffn_gate_inp->type   == GGML_TYPE_F32  && layer.ffn_exp_probs_b->type == GGML_TYPE_F32 &&
           layer.ffn_up_exps->type    == GGML_TYPE_IQ3_KS && layer.ffn_gate_exps->type == GGML_TYPE_IQ3_KS &&
           layer.ffn_down_exps->type  == GGML_TYPE_IQ4_KS_R16 &&
           layer.ffn_up_shexp->type   == GGML_TYPE_Q6_K && layer.ffn_gate_shexp->type == GGML_TYPE_Q6_K &&
           layer.ffn_down_shexp->type == GGML_TYPE_Q8_0 &&
           layer.ffn_gate_inp->ne[0] == 2560 && layer.ffn_gate_inp->ne[1] == 128 &&
           layer.ffn_up_exps->ne[1] == 1088 && layer.ffn_up_shexp->ne[1] == 1088;
}

// returns the pre-weighted stream sum [n_embd, nt]; post [hc, nt], comb [hc, hc, nt] with comb[dst + src*hc]
static ggml_tensor * raon1_build_hc_pre(llm_build_context & llm, const raon1_hc_consts & k,
        ggml_tensor * x, ggml_tensor * fn, ggml_tensor * scale, ggml_tensor * base,
        ggml_tensor ** post_out, ggml_tensor ** comb_out, int il) {
    auto & ctx0    = llm.ctx0;
    auto & hparams = llm.hparams;
    auto & cb      = llm.cb;

    const int64_t n_embd = hparams.n_embd;
    const int64_t hc = hparams.dsv4_hc_mult;
    const int64_t nm = hc - 1;
    const int64_t nt = x->ne[2];

    if (!ggml_is_contiguous(x)) {
        x = ggml_cont(ctx0, x);
    }
    // (fn @ x) / rms(x) == fn @ rms_norm(x)
    ggml_tensor * flat  = ggml_reshape_2d(ctx0, x, n_embd*hc, nt);
    ggml_tensor * mixes = ggml_mul_mat(ctx0, fn, ggml_rms_norm(ctx0, flat, hparams.dsv4_hc_eps));
    cb(mixes, "hc_mixes", il);

    auto affine = [&](int64_t offset, int64_t width, int64_t si) {
        ggml_tensor * z = ggml_view_2d(ctx0, mixes, width, nt, mixes->nb[1], offset*mixes->nb[0]);
        ggml_tensor * a = ggml_view_1d(ctx0, scale, 1, si*scale->nb[0]);
        ggml_tensor * b = ggml_view_1d(ctx0, base, width, offset*base->nb[0]);
        return ggml_add(ctx0, ggml_mul(ctx0, z, a), b);
    };
    ggml_tensor * pre = ggml_sigmoid(ctx0, affine(0, hc, 0));
    *post_out = ggml_scale(ctx0, ggml_sigmoid(ctx0, affine(hc, hc, 1)), hparams.hc_magnitude);

    // e = H z H^T, rho = 1 + max(0, max(-e / chart_base)), phi = chart_base + e / rho,
    // comb = I + sigmoid(eta) * (phi - I)
    ggml_tensor * z = ggml_cont_3d(ctx0, affine(2*hc, nm*nm, 2), nm, nm, nt);
    ggml_tensor * e = ggml_mul_mat(ctx0, k.helmert, z);
    e = ggml_mul_mat(ctx0, k.helmert, ggml_cont(ctx0, ggml_transpose(ctx0, e)));
    ggml_tensor * ratio = ggml_neg(ctx0, ggml_div(ctx0, e, k.chart_base));
    ratio = ggml_reshape_2d(ctx0, ratio, hc*hc, nt);
    // pool_2d has a CUDA kernel, pool_1d does not
    ggml_tensor * rho = ggml_pool_2d(ctx0, ratio, GGML_OP_POOL_MAX, hc*hc, 1, hc*hc, 1, 0, 0);
    rho = ggml_scale_bias(ctx0, ggml_clamp(ctx0, rho, 0.0f, INFINITY), 1.0f, 1.0f);
    rho = ggml_reshape_3d(ctx0, rho, 1, 1, nt);
    ggml_tensor * phi = ggml_add(ctx0, ggml_div(ctx0, e, rho), k.chart_base);
    ggml_tensor * gamma = ggml_sigmoid(ctx0, ggml_view_1d(ctx0, scale, 1, 3*scale->nb[0]));
    *comb_out = ggml_add(ctx0, ggml_mul(ctx0, ggml_sub(ctx0, phi, k.identity), gamma), k.identity);
    cb(pre, "hc_pre", il);
    cb(*post_out, "hc_post", il);
    cb(*comb_out, "hc_comb", il);

    return llm.build_mhc_weighted_sum(x, pre, n_embd, hc);
}

// RAON_MLA_FUSE=0: always ggml_cont the MLA views (also disables the CUDA MLA small-op fusions)
static bool raon1_mla_fuse() {
    static const bool on = [] { const char * e = getenv("RAON_MLA_FUSE"); return !e || atoi(e) != 0; }();
    return on;
}

static ggml_tensor * raon1_cont(ggml_context * ctx, ggml_tensor * t) {
    return raon1_mla_fuse() && ggml_is_contiguous(t) ? t : ggml_cont(ctx, t);
}

// absorbed MLA. Cache row is [extra (n_rot), latent]; the extra dims are dotted without rotation
static ggml_tensor * raon1_build_mla(llm_build_context & llm, ggml_cgraph * gf, int il,
        ggml_tensor * cur, ggml_tensor * KQ_mask) {
    auto & lctx    = llm.lctx;
    auto & ctx0    = llm.ctx0;
    auto & hparams = llm.hparams;
    auto & kv_self = lctx.kv_self;
    auto & cb      = llm.cb;
    auto & layer   = llm.model.layers[il];

    const int64_t n_head   = llm.n_head;
    const int64_t n_tokens = llm.n_tokens;
    const int64_t n_kv     = llm.n_kv;
    const int64_t nr   = hparams.n_rot;
    const int64_t nk   = hparams.n_embd_head_k_full;
    const int64_t nn   = nk - nr;
    const int64_t nv   = hparams.n_embd_head_v_full;
    const int64_t rank = hparams.n_lora_kv;
    const float   eps  = hparams.f_norm_rms_eps_latent;
    const float kq_scale = 1.0f/sqrtf(float(nk));

    ggml_tensor * q = ggml_mul_mat(ctx0, layer.wq_a, cur);
    ggml_tensor * kv = ggml_mul_mat(ctx0, layer.wkv_a_mqa, cur);
    cb(kv, "kv_a_mqa", il);
    ggml_tensor * gate = ggml_mul_mat(ctx0, layer.wqkv_gate, cur);
    // keep the three projections of cur adjacent so CUDA can quantize cur once (RAON_MLA_PROJ_FIRST=0 disables)
    static const bool proj_first = [] { const char * e = getenv("RAON_MLA_PROJ_FIRST"); return !e || atoi(e) != 0; }();
    if (proj_first) {
        ggml_build_forward_expand(gf, q);
        ggml_build_forward_expand(gf, kv);
        ggml_build_forward_expand(gf, gate);
    }
    q = ggml_mul(ctx0, ggml_rms_norm(ctx0, q, eps), layer.attn_q_a_norm);
    q = ggml_mul_mat(ctx0, layer.wq_b, q);
    cb(q, "q", il);
    ggml_tensor * latent = raon1_cont(ctx0, ggml_view_2d(ctx0, kv, rank, n_tokens, kv->nb[1], 0));
    latent = ggml_mul(ctx0, ggml_rms_norm(ctx0, latent, eps), layer.attn_kv_a_norm);
    cb(latent, "kv_compressed", il);
    ggml_tensor * extra = ggml_view_2d(ctx0, kv, nr, n_tokens, kv->nb[1], rank*ggml_element_size(kv));

    ggml_tensor * kvr = ggml_concat(ctx0, extra, latent, 0);
    cb(kvr, "kvr", il);

    const auto row_size = ggml_row_size(kv_self.k_l[il]->type, rank + nr);
    ggml_tensor * k_cache_view = ggml_view_2d(ctx0, kv_self.k_l[il], rank + nr, n_tokens, row_size, row_size*llm.kv_head);
    lctx.cache_copies[2*il + 0].cpy  = ggml_cpy(ctx0, kvr, k_cache_view);
    lctx.cache_copies[2*il + 0].step = row_size;
    ggml_build_forward_expand(gf, lctx.cache_copies[2*il + 0].cpy);

    ggml_tensor * kv_cache = ggml_view_2d(ctx0, kv_self.k_l[il], rank + nr, n_kv, row_size, 0);
    cb(kv_cache, "kv_cache", il);

    q = ggml_reshape_3d(ctx0, q, nk, n_head, n_tokens);
    ggml_tensor * q_nope  = ggml_view_3d(ctx0, q, nn, n_head, n_tokens, q->nb[1], q->nb[2], 0);
    ggml_tensor * q_extra = ggml_view_3d(ctx0, q, nr, n_head, n_tokens, q->nb[1], q->nb[2], nn*ggml_element_size(q));
    q_nope = ggml_permute(ctx0, q_nope, 0, 2, 1, 3);                 // [nn, n_tokens, n_head]
    ggml_tensor * q_abs = ggml_mul_mat(ctx0, layer.wk_b, q_nope);    // [rank, n_tokens, n_head]
    cb(q_abs, "q_nope_absorbed", il);
    q = ggml_concat(ctx0, ggml_permute(ctx0, q_extra, 0, 2, 1, 3), q_abs, 0); // [nr + rank, n_tokens, n_head]
    cb(q, "q_full", il);

    ggml_tensor * kqv_cmpr = nullptr;
    if (lctx.cparams.flash_attn) {
        ggml_tensor * kv_cache_lora = ggml_view_2d(ctx0, kv_self.k_l[il], rank, n_kv, row_size,
                ggml_row_size(kv_self.k_l[il]->type, nr));
        kqv_cmpr = ggml_flash_attn_ext(ctx0, q, kv_cache, kv_cache_lora, KQ_mask, kq_scale, hparams.f_max_alibi_bias, 0.f);
        ggml_flash_attn_ext_set_prec(kqv_cmpr, GGML_PREC_F32);
        cb(kqv_cmpr, "kqv_compressed", il);
        kqv_cmpr = ggml_permute(ctx0, kqv_cmpr, 0, 2, 1, 3);         // [rank, n_tokens, n_head]
    } else {
        ggml_tensor * kv_cache_trans = nullptr;
        if (lctx.cparams.mla_attn == 1 && kv_self.v_l.size() > (size_t) il && kv_self.v_l[il]) {
            ggml_tensor * v_view = ggml_view_2d(ctx0, kv_self.v_l[il], n_tokens, rank,
                    ggml_row_size(kv_self.v_l[il]->type, kv_self.size), ggml_row_size(kv_self.v_l[il]->type, llm.kv_head));
            ggml_tensor * v_cpy = ggml_cpy(ctx0, ggml_transpose(ctx0, latent), v_view);
            // registered so update_cache_copies() can re-point it and the graph can be reused;
            // the transposed cache advances one element per cell. RAON_NOFA_V_REUSE=0 disables
            static const bool reg_v = [] { const char * e = getenv("RAON_NOFA_V_REUSE"); return !e || atoi(e) != 0; }();
            if (reg_v && !ggml_is_quantized(kv_self.v_l[il]->type)) {
                lctx.cache_copies[2*il + 1].cpy  = v_cpy;
                lctx.cache_copies[2*il + 1].step = ggml_row_size(kv_self.v_l[il]->type, 1);
            }
            ggml_build_forward_expand(gf, v_cpy);
            kv_cache_trans = ggml_view_2d(ctx0, kv_self.v_l[il], n_kv, rank,
                    ggml_row_size(kv_self.v_l[il]->type, kv_self.size), 0);
        } else {
            ggml_tensor * kv_cache_lora = ggml_view_2d(ctx0, kv_self.k_l[il], rank, n_kv, row_size,
                    ggml_row_size(kv_self.k_l[il]->type, nr));
            kv_cache_trans = ggml_cont(ctx0, ggml_transpose(ctx0, kv_cache_lora));
        }
        cb(kv_cache_trans, "kv_cache_trans", il);

        ggml_tensor * kq = ggml_mul_mat(ctx0, kv_cache, q);            // [n_kv, n_tokens, n_head]
        ggml_mul_mat_set_prec(kq, GGML_PREC_F32);
        kq = ggml_soft_max_ext(ctx0, kq, KQ_mask, kq_scale, hparams.f_max_alibi_bias);
        cb(kq, "kq_soft_max", il);
        kqv_cmpr = ggml_mul_mat(ctx0, kv_cache_trans, kq);             // [rank, n_tokens, n_head]
        cb(kqv_cmpr, "kqv_compressed", il);
    }

    ggml_tensor * kqv = ggml_mul_mat(ctx0, layer.wv_b, kqv_cmpr);      // [nv, n_tokens, n_head]
    kqv = raon1_cont(ctx0, ggml_permute(ctx0, kqv, 0, 2, 1, 3));       // [nv, n_head, n_tokens]
    kqv = ggml_reshape_2d(ctx0, kqv, nv*n_head, n_tokens);
    cb(kqv, "kqv", il);

    kqv = ggml_mul(ctx0, kqv, ggml_sigmoid(ctx0, gate));
    cb(kqv, "mla_gated", il);

    return llm_build_context::llm_build_lora_mm(lctx, ctx0, layer.wo, kqv);
}

ggml_cgraph * llm_build_context::build_raon1() {
    ggml_cgraph * gf = new_graph_custom();

    const int64_t hc = hparams.dsv4_hc_mult;
    const int64_t nm = hc - 1;

    const int mhc_mode = raon1_mhc_mode();

    // only the composed path needs the chart constants
    raon1_hc_consts hk = {};
    if (mhc_mode == 0) {
        std::vector<float> h(nm*hc), id(hc*hc), cb_(hc*hc);
        for (int64_t i = 0; i < hc; ++i) {
            for (int64_t j = 0; j < nm; ++j) {
                h[i*nm + j] = (i <= j ? 1.0f : i == j + 1 ? -(float) (j + 1) : 0.0f)/sqrtf((j + 1)*(j + 2));
            }
            for (int64_t j = 0; j < hc; ++j) {
                id[i*hc + j]  = i == j ? 1.0f : 0.0f;
                cb_[i*hc + j] = 0.5f*id[i*hc + j] + 0.5f/hc;
            }
        }
        hk.helmert    = raon1_const_2d(ctx0, h,   nm, hc);
        hk.identity   = raon1_const_2d(ctx0, id,  hc, hc);
        hk.chart_base = raon1_const_2d(ctx0, cb_, hc, hc);
    }

    delta_net delta(lctx, batch);

    auto inpL = llm_build_inp_embd(ctx0, lctx, hparams, batch, model.tok_embd, cb);
    auto KQ_mask = build_inp_KQ_mask();
    auto inp_out_ids = build_inp_out_ids();

    // only the unfused KDA core reads inp_s_seq_qnext; unused, it would never get a buffer
    // and llama_set_inputs would fail filling it
    bool need_s_seq_qnext = false;
    for (int il = 0; il < n_layer && !need_s_seq_qnext; ++il) {
        need_s_seq_qnext = hparams.is_recurrent(il) && !delta.rda_fused_eligible(il);
    }
    if (need_s_seq_qnext) {
        lctx.inp_s_seq_qnext = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, 1, n_tokens);
        cb(lctx.inp_s_seq_qnext, "inp_s_seq_qnext", -1);
        ggml_set_input(lctx.inp_s_seq_qnext);
    }

    inpL = ggml_reshape_3d(ctx0, inpL, n_embd, 1, n_tokens);
    inpL = ggml_repeat_4d(ctx0, inpL, n_embd, hc, n_tokens, 1);
    cb(inpL, "hc_init", -1);

    for (int il = 0; il < n_layer; ++il) {
        const auto & layer = model.layers[il];

        ggml_tensor * post, * comb;
        auto residual = inpL;
        auto cur = mhc_mode == 0
            ? raon1_build_hc_pre(*this, hk, inpL, layer.hc_attn_fn, layer.hc_attn_scale, layer.hc_attn_base,
                &post, &comb, il)
            : raon1_build_hc_pre_fused(*this, inpL, layer.hc_attn_fn, layer.hc_attn_scale, layer.hc_attn_base,
                &post, &comb, il, mhc_mode == 1);
        cb(cur, "hc_attn_pre", il);

        if (hparams.is_recurrent(il)) {
            // the KDA core applies attn_norm itself
            cur = delta.build_layer_attn_kda(ctx0, gf, cur, nullptr, il, cb, /*add_residual = */ false);
        } else {
            cur = llm_build_norm(ctx0, cur, hparams, layer.attn_norm, nullptr, LLM_NORM_RMS, cb, il);
            cb(cur, "attn_norm", il);
            cur = raon1_build_mla(*this, gf, il, cur, KQ_mask);
        }
        cb(cur, "attn_out", il);

        inpL = build_mhc_post(cur, post, residual, comb, n_embd, hc, true);
        cb(inpL, "hc_attn_out", il);

        residual = inpL;
        cur = mhc_mode == 0
            ? raon1_build_hc_pre(*this, hk, inpL, layer.hc_ffn_fn, layer.hc_ffn_scale, layer.hc_ffn_base,
                &post, &comb, il)
            : raon1_build_hc_pre_fused(*this, inpL, layer.hc_ffn_fn, layer.hc_ffn_scale, layer.hc_ffn_base,
                &post, &comb, il, mhc_mode == 1);
        cur = llm_build_norm(ctx0, cur, hparams, layer.ffn_norm, nullptr, LLM_NORM_RMS, cb, il);
        cb(cur, "ffn_norm", il);

        if ((uint32_t) il < hparams.n_layer_dense_lead) {
            cur = llm_build_ffn(ctx0, lctx, nullptr, cur,
                    layer.ffn_up,   nullptr, nullptr,
                    layer.ffn_gate, nullptr, nullptr,
                    layer.ffn_down, nullptr, nullptr,
                    nullptr,
                    LLM_FFN_SITU, LLM_FFN_PAR, cb, il);
        } else if (raon1_can_use_mk_moe(*this, layer, n_tokens, n_expert_used, il) && ggml_is_contiguous(cur)) {
            // same result as the branch below (ffn_moe_out + ffn_shexp), one op
            cur = ggml_raon_moe_block(ctx0, cur, layer.ffn_gate_inp, layer.ffn_exp_probs_b,
                    layer.ffn_up_exps, layer.ffn_gate_exps, layer.ffn_down_exps,
                    layer.ffn_up_shexp, layer.ffn_gate_shexp, layer.ffn_down_shexp,
                    n_expert_used, hparams.expert_weights_norm, hparams.expert_weights_scale,
                    hparams.situ_beta, hparams.situ_linear_beta);
            cb(cur, "ffn_moe_block", il);
        } else {
            auto moe_out = llm_build_moe_ffn(ctx0, lctx, cur,
                    layer.ffn_gate_inp, nullptr,
                    layer.ffn_up_exps, nullptr,
                    layer.ffn_gate_exps, nullptr,
                    layer.ffn_down_exps, nullptr,
                    layer.ffn_exp_probs_b,
                    layer.ffn_gate_inp->ne[1], n_expert_used,
                    LLM_FFN_SITU, hparams.expert_weights_norm,
                    true, hparams.expert_weights_scale,
                    (llm_expert_gating_func_type) hparams.expert_gating_func,
                    cb, il, gf, false, nullptr, nullptr, nullptr, nullptr, nullptr);
            cb(moe_out, "ffn_moe_out", il);
            if (layer.ffn_gate_shexp) {
                auto shexp = llm_build_ffn(ctx0, lctx, nullptr, cur,
                        layer.ffn_up_shexp,   nullptr, nullptr,
                        layer.ffn_gate_shexp, nullptr, nullptr,
                        layer.ffn_down_shexp, nullptr, nullptr,
                        nullptr,
                        LLM_FFN_SITU, LLM_FFN_PAR, cb, il);
                cb(shexp, "ffn_shexp", il);
                moe_out = ggml_add(ctx0, moe_out, shexp);
            }
            cur = moe_out;
        }
        cb(cur, "ffn_out", il);

        inpL = build_mhc_post(cur, post, residual, comb, n_embd, hc, true);
        inpL = lctx.cvec.apply_to(ctx0, inpL, il);
        if (lctx.dflash.capture) {
            // DFlash tap: all hc streams, [n_embd*hc, n_tokens]. target_layers id k is the input of layer k.
            // Other layers get a name the callback ignores, so they are not copied.
            const auto & ids = lctx.dflash.capture->layer_ids;
            if (std::find(ids.begin(), ids.end(), il + 1) != ids.end()) {
                static const bool tap_view = [] { const char * e = getenv("RAON_DFLASH_TAP_VIEW"); return e && atoi(e) != 0; }();
                auto tap = tap_view && ggml_is_contiguous(inpL) ? ggml_reshape_2d(ctx0, inpL, n_embd*hc, n_tokens)
                                                                : ggml_cont_2d(ctx0, inpL, n_embd*hc, n_tokens);
                cb(tap, "l_out", il + 1);
                ggml_build_forward_expand(gf, tap);
            } else {
                cb(inpL, "l_hc_out", il);
            }
        } else {
            cb(inpL, "l_out", il);
        }
    }

    // average the mHC streams before the output head
    {
        auto flat = ggml_reshape_2d(ctx0, inpL, n_embd*hc, n_tokens);
        if (inp_out_ids && n_outputs != n_tokens) {
            flat = ggml_get_rows(ctx0, flat, inp_out_ids);
        }
        const int64_t n_out = flat->ne[1];
        ggml_tensor * summed = nullptr;
        for (int64_t s = 0; s < hc; ++s) {
            auto stream = ggml_view_2d(ctx0, flat, n_embd, n_out, flat->nb[1], s*n_embd*ggml_element_size(flat));
            summed = summed ? ggml_add(ctx0, summed, stream) : stream;
        }
        inpL = ggml_scale(ctx0, summed, 1.0f/hc);
        cb(inpL, "hc_collapse", -1);
    }

    auto cur = build_output(lctx, ctx0, inpL, model.output, model.output_norm, cb);
    cb(cur, "result_output", -1);
    ggml_build_forward_expand(gf, cur);
    // GPU top-m logits candidates (llama-gsample.h)
    if (llama_gsample_env_enabled() && !lctx.cparams.embeddings && !lctx.cparams.mtp && !lctx.dflash.capture &&
            cur->ne[1] >= 1 && cur->ne[1] <= LLAMA_GSAMPLE_MAX_ROWS && cur->type == GGML_TYPE_F32) {
        auto cand = ggml_logits_topm(ctx0, cur, llama_gsample_env_m());
        cb(cand, "gsample_cand", -1);
        ggml_build_forward_expand(gf, cand);
    }
    return gf;
}
