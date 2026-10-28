#include "llama-kda.h"
#include "llama-hparams.h"
#include "llama-cparams.h"
#include "llama-model.h"
#include "llama-context.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <cstdlib>

static std::pair<ggml_tensor *, ggml_tensor *> build_kda_qkvz(llama_context & lctx, ggml_context * ctx0,
        ggml_tensor * wq, ggml_tensor * wk, ggml_tensor * wv, ggml_tensor * ssm_g_a, ggml_tensor * ssm_g_b,
        ggml_tensor * input, int il, const llm_build_cb & cb, ggml_cgraph * gf) {
    auto q = llm_build_context::llm_build_lora_mm(lctx, ctx0, wq, input);
    auto k = llm_build_context::llm_build_lora_mm(lctx, ctx0, wk, input);
    auto v = llm_build_context::llm_build_lora_mm(lctx, ctx0, wv, input);
    auto z = llm_build_context::llm_build_lora_mm(lctx, ctx0, ssm_g_a, input);
    if (ssm_g_b) {
        // GLM-5.3-Flash / kimi-k3 low-rank KDA: g = g_b(g_a(x)), two chained projections
        z = llm_build_context::llm_build_lora_mm(lctx, ctx0, ssm_g_b, z);
    }
    cb(q, "q", il);
    cb(k, "k", il);
    cb(v, "v", il);
    cb(z, "z", il);

    auto qkv = ggml_concat(ctx0, q, k, 0);
    qkv = ggml_concat(ctx0, qkv, v, 0);
    cb(qkv, "qkv_mixed", il);
    ggml_build_forward_expand(gf, qkv);
    ggml_build_forward_expand(gf, z);
    return { qkv, z };
}

static std::pair<ggml_tensor *, ggml_tensor *> build_kda_beta_gate(llama_context & lctx, ggml_context * ctx0,
        ggml_tensor * ssm_beta, ggml_tensor * ssm_f_a, ggml_tensor * ssm_f_b, ggml_tensor * ssm_dt_b, ggml_tensor * ssm_a,
        ggml_tensor * input, int64_t head_dim, int64_t n_head, float lower_bound,
        int il, const llm_build_cb & cb, ggml_cgraph * gf) {
    const int64_t n_tok = input->ne[1];

    auto beta = llm_build_context::llm_build_lora_mm(lctx, ctx0, ssm_beta, input);
    beta = ggml_reshape_4d(ctx0, beta, n_head, 1, n_tok, 1);
    cb(beta, "beta", il);

    auto raw = llm_build_context::llm_build_lora_mm(lctx, ctx0, ssm_f_a, input);
    if (ssm_f_b) {
        // GLM-5.3-Flash / kimi-k3 low-rank KDA: f = f_b(f_a(x)), two chained projections
        raw = llm_build_context::llm_build_lora_mm(lctx, ctx0, ssm_f_b, raw);
    }
    raw = ggml_reshape_4d(ctx0, raw, head_dim, n_head, n_tok, 1);
    cb(raw, "decay_raw", il);

    auto dt = ggml_reshape_4d(ctx0, ssm_dt_b, head_dim, n_head, 1, 1);
    auto a  = ggml_reshape_4d(ctx0, ssm_a, 1, n_head, 1, 1);
    auto log_decay = ggml_add(ctx0, raw, dt);
    log_decay = ggml_mul(ctx0, log_decay, a);
    if (ssm_f_b) {
        // GLM-5.3-Flash formula (upstream build_kda_layer): ssm_a holds -exp(A_log), so the
        // sigmoid is taken on -(ssm_a*(f + dt)) == exp(A_log)*(f + dt)
        log_decay = ggml_sigmoid(ctx0, ggml_scale(ctx0, log_decay, -1.0f));
    } else {
        log_decay = ggml_sigmoid(ctx0, log_decay);
    }
    log_decay = ggml_scale(ctx0, log_decay, lower_bound);
    cb(log_decay, "log_decay", il);

    ggml_build_forward_expand(gf, beta);
    ggml_build_forward_expand(gf, log_decay);
    return { beta, log_decay };
}

static ggml_tensor * build_kda_conv(ggml_context * ctx0,
        ggml_tensor * ssm_conv1d_q, ggml_tensor * ssm_conv1d_k, ggml_tensor * ssm_conv1d_v) {
    const int64_t d_conv = ssm_conv1d_q->ne[0];
    auto q = ggml_reshape_2d(ctx0, ssm_conv1d_q, d_conv, ggml_nrows(ssm_conv1d_q));
    auto k = ggml_reshape_2d(ctx0, ssm_conv1d_k, d_conv, ggml_nrows(ssm_conv1d_k));
    auto v = ggml_reshape_2d(ctx0, ssm_conv1d_v, d_conv, ggml_nrows(ssm_conv1d_v));
    auto qkv = ggml_concat(ctx0, q, k, 1);
    return ggml_concat(ctx0, qkv, v, 1);
}

static ggml_tensor * build_kda_gated_output(llama_context & lctx, ggml_context * ctx0, ggml_tensor * ssm_norm, ggml_tensor * ssm_out, ggml_tensor * output, ggml_tensor * z,
        int64_t head_v_dim, int64_t num_v_heads, int64_t n_tok, int il, const llm_build_cb & cb) {

    ggml_tensor * attn_out_2d = ggml_reshape_2d(ctx0, output, head_v_dim, num_v_heads * n_tok);
    ggml_tensor * z_2d        = ggml_reshape_2d(ctx0, z,      head_v_dim, num_v_heads * n_tok);

    ggml_tensor * attn_out_norm = llm_build_context::llm_build_norm(ctx0, attn_out_2d, lctx.model.hparams, ssm_norm, nullptr, LLM_NORM_RMS, cb, il);
    cb(attn_out_norm, "attn_rms_norm", il);
    attn_out_norm = ggml_mul(ctx0, ggml_sigmoid(ctx0, z_2d), attn_out_norm);
    cb(attn_out_norm, "attn_out_norm", il);

    ggml_tensor * final_output = ggml_reshape_2d(ctx0, attn_out_norm, head_v_dim*num_v_heads, n_tok);
    cb(final_output, "final_output", il);

    ggml_tensor * out = llm_build_context::llm_build_lora_mm(lctx, ctx0, ssm_out, final_output);
    cb(out, "linear_attn_out", il);

    return ggml_reshape_2d(ctx0, out, lctx.model.hparams.n_embd, n_tok);
}

ggml_tensor * delta_net::build_layer_attn_kda_core(ggml_context * ctx0, ggml_cgraph * gf,
        ggml_tensor * delta_input, ggml_tensor * inp_s_seq_qnext, ggml_tensor * inp_out_ids,
        uint32_t state_seq_id_local, bool reset_state_local, int il, const llm_build_cb & cb,
        bool add_residual) const {
    const int64_t n_tok = delta_input->ne[1];
    const int64_t head_dim = lctx.model.hparams.ssm_d_state;

    auto & model   = lctx.model;
    auto & hparams = model.hparams;
    auto & kv_self = lctx.kv_self;
    auto & layer   = model.layers[il];

    if (model.split_mode == LLAMA_SPLIT_MODE_GRAPH && kv_self.s_l[il]->extra) {
        GGML_ASSERT(!layer.ssm_rda && "Raon-1 RDA layers do not support split mode graph");
        auto split_s_l = (ggml_split_tensor_t *) kv_self.s_l[il]->extra;
        GGML_ASSERT(split_s_l && split_s_l->n_device > 1);

        std::vector<ggml_tensor *> results(split_s_l->n_device, nullptr);
        bool input_added = false;
        for (int id = 0; id < split_s_l->n_device; ++id) {
            if (!split_s_l->splits[id]) {
                continue;
            }

            auto split = [id](ggml_tensor * tensor) {
                auto data = (ggml_split_tensor_t *) tensor->extra;
                GGML_ASSERT(data && data->splits[id]);
                return data->splits[id];
            };

            const int il_cb = 1000 * il + id;
            auto input = llm_build_context::get_input_tensor_sm_graph(ctx0, delta_input, id);
            auto cur = llm_build_context::llm_build_norm(ctx0, input, hparams,
                    split(layer.attn_norm), nullptr, LLM_NORM_RMS, cb, il_cb);

            auto ssm_out = split(layer.ssm_out);
            const int64_t n_head = ssm_out->ne[0] / head_dim;
            auto split_opt = [&split](ggml_tensor * t) { return t ? split(t) : nullptr; };
            auto [qkv_mixed, z] = build_kda_qkvz(lctx, ctx0,
                    split(layer.wq), split(layer.wk), split(layer.wv), split(layer.ssm_g_a), split_opt(layer.ssm_g_b),
                    cur, il_cb, cb, gf);
            auto [beta, log_decay] = build_kda_beta_gate(lctx, ctx0,
                    split(layer.ssm_beta), split(layer.ssm_f_a), split_opt(layer.ssm_f_b), split(layer.ssm_dt_b), split(layer.ssm_a),
                    cur, head_dim, n_head, hparams.kda_gate_lower_bound, il_cb, cb, gf);
            auto conv = build_kda_conv(ctx0,
                    split(layer.ssm_conv1d_q), split(layer.ssm_conv1d_k), split(layer.ssm_conv1d_v));

            ggml_tensor * per_step_ckpt = nullptr;
            if (save_per_step_states && il < (int) kv_self.ckpt.per_step_ssm.size()) {
                per_step_ckpt = kv_self.ckpt.per_step_ssm[il][id];
            }
            auto per_step_conv = save_per_step_states && il < (int) kv_self.ckpt.per_step_conv.size() &&
                                 id < (int) kv_self.ckpt.per_step_conv[il].size()
                               ? kv_self.ckpt.per_step_conv[il][id] : nullptr;

            const uint32_t qnext_state_slots = split_s_l->splits[id]->ne[1];
            auto output = build_qkv(ctx0, split_s_l->splits[id], conv, qkv_mixed,
                    inp_s_seq_qnext, beta, log_decay,
                    head_dim, n_head, head_dim, n_head, hparams.ssm_d_conv,
                    state_seq_id_local, qnext_state_slots, reset_state_local,
                    hparams.f_norm_rms_eps, 1, il_cb, cb, gf, per_step_ckpt, per_step_conv);

            auto gated_output = build_kda_gated_output(lctx, ctx0, split(layer.ssm_norm), ssm_out, output, z,
                    head_dim, n_head, n_tok, il_cb, cb);
            if (inp_out_ids) {
                gated_output = ggml_get_rows(ctx0, gated_output, inp_out_ids);
            }
            if (!input_added) {
                if (inp_out_ids) {
                    input = ggml_get_rows(ctx0, input, inp_out_ids);
                }
                if (add_residual) {
                    gated_output = ggml_add(ctx0, gated_output, input);
                }
                input_added = true;
            }
            if (gated_output->ne[1] > 32 && lctx.cparams.reduce_type != GGML_TYPE_F32) {
                gated_output = ggml_cast(ctx0, gated_output, lctx.cparams.reduce_type);
            }
            ggml_build_forward_expand(gf, gated_output);
            results[id] = gated_output;
        }

        auto output = ggml_reduce(ctx0, results.data(), split_s_l->n_device, GGML_OP_ADD);
        ggml_build_forward_expand(gf, output);
        return output;
    }

    const uint32_t qnext_state_slots = llm_build_context::llama_kv_qnext_state_slots(kv_self);
    int idx = model.default_layer_device[il];
    auto input = delta_input;
    if (input->op == GGML_OP_REDUCE) {
        const int idx_s_l = ggml_backend_sched_get_backend_idx(lctx.sched, kv_self.s_l[il]->buffer);
        if (idx_s_l >= 0) {
            idx = idx_s_l;
        }
        if (input->src[idx]) {
            input->view_src = input->src[idx];
        }
    }

    auto norm = layer.attn_norm->extra
              ? ((ggml_split_tensor_t *) layer.attn_norm->extra)->splits[idx]
              : layer.attn_norm;
    auto cur = llm_build_context::llm_build_norm(ctx0, input, hparams, norm, nullptr, LLM_NORM_RMS, cb, il);

    const int64_t n_head = hparams.ssm_dt_rank;
    auto [qkv_mixed, z] = build_kda_qkvz(lctx, ctx0,
            layer.wq, layer.wk, layer.wv, layer.ssm_g_a, layer.ssm_g_b, cur, il, cb, gf);
    auto [beta, log_decay] = build_kda_beta_gate(lctx, ctx0,
            layer.ssm_beta, layer.ssm_f_a, layer.ssm_f_b, layer.ssm_dt_b, layer.ssm_a,
            cur, head_dim, n_head, hparams.kda_gate_lower_bound, il, cb, gf);
    // Raon-1 loads the q/k/v conv weights as views of one merged tensor: avoids the concat per graph
    auto conv = layer.ssm_conv1d && layer.ssm_conv1d_q->view_src == layer.ssm_conv1d
              ? layer.ssm_conv1d
              : build_kda_conv(ctx0, layer.ssm_conv1d_q, layer.ssm_conv1d_k, layer.ssm_conv1d_v);

    // Raon-1 RDA: erase gate sigmoid(beta), write gate sigmoid(beta + rda(x)); the op applies the erase gate
    // and V is scaled by the write/erase ratio
    ggml_tensor * v_scale = nullptr;
    float eps_l2 = hparams.f_norm_rms_eps;
    if (layer.ssm_rda) {
        auto delta = llm_build_context::llm_build_lora_mm(lctx, ctx0, layer.ssm_rda, cur);
        delta = ggml_reshape_4d(ctx0, delta, n_head, 1, n_tok, 1);
        auto erase = ggml_softplus(ctx0, ggml_neg(ctx0, beta));
        auto write = ggml_softplus(ctx0, ggml_neg(ctx0, ggml_add(ctx0, beta, delta)));
        v_scale = ggml_exp(ctx0, ggml_sub(ctx0, erase, write));
        v_scale = ggml_reshape_4d(ctx0, v_scale, 1, n_head, n_tok, 1);
        cb(v_scale, "rda_v_scale", il);
        // same as HF F.normalize(eps=1e-6)
        eps_l2 = 1e-6f;
    }

    ggml_tensor * per_step_ckpt = nullptr;
    if (save_per_step_states && il < (int) kv_self.ckpt.per_step_ssm.size() && !kv_self.ckpt.per_step_ssm[il].empty()) {
        per_step_ckpt = kv_self.ckpt.per_step_ssm[il].front();
    }
    auto per_step_conv = save_per_step_states && il < (int) kv_self.ckpt.per_step_conv.size() &&
                         !kv_self.ckpt.per_step_conv[il].empty()
                       ? kv_self.ckpt.per_step_conv[il].front() : nullptr;

    auto output = build_qkv(ctx0, kv_self.s_l[il], conv, qkv_mixed,
            inp_s_seq_qnext, beta, log_decay,
            head_dim, n_head, head_dim, n_head, hparams.ssm_d_conv,
            state_seq_id_local, qnext_state_slots, reset_state_local,
            eps_l2, 1, il, cb, gf, per_step_ckpt, per_step_conv, v_scale, replay_layer(il));
    auto gated_output = build_kda_gated_output(lctx, ctx0, layer.ssm_norm, layer.ssm_out, output, z,
            head_dim, n_head, n_tok, il, cb);

    if (inp_out_ids) {
        gated_output = ggml_get_rows(ctx0, gated_output, inp_out_ids);
        input = ggml_get_rows(ctx0, input, inp_out_ids);
    }
    output = add_residual ? ggml_add(ctx0, gated_output, input) : gated_output;
    cb(output, "ssm_output", il);
    return output;
}

// RAON_RDA_FUSED: 1 (default) fused GGML_OP_RDA_DECODE for small batches with the state in device memory,
// 2 also with the state in host memory, 0 unfused graph
static int raon_rda_fused_mode() {
    static const int mode = [] {
        const char * e = getenv("RAON_RDA_FUSED");
        const int m = e ? atoi(e) : 1;
        return m >= 0 && m <= 2 ? m : 1;
    }();
    return mode;
}

static constexpr int RAON_RDA_FUSED_MAX_TOKENS = 4; // <= GGML_RDA_MAX_SEQS and GGML_RDA_MAX_TOKENS
static constexpr int GGML_RDA_MAX_TOKENS_LLAMA = 8; // = GGML_RDA_MAX_TOKENS
static constexpr int GGML_RDA_P_FLAGS_LLAMA    = 13; // = GGML_RDA_P_FLAGS
static constexpr int GGML_RDA_F_NO_STATE_WRITE_LLAMA = 1;

bool delta_net::rda_fused_eligible(int il) const {
    const int mode = raon_rda_fused_mode();
    if (mode == 0) {
        return false;
    }
    const auto & model   = lctx.model;
    const auto & hparams = model.hparams;
    const auto & layer   = model.layers[il];
    const ggml_tensor * s_l = lctx.kv_self.s_l[il];

    if (!layer.ssm_rda || !layer.ssm_f_b || layer.attn_norm->extra) {
        return false;
    }
    if (save_per_step_states) {
        // spec verify: only replay-rollback windows; per-step state checkpoints need the unfused delta_net.
        // RAON_RDA_FUSED_VERIFY=0 keeps verify windows unfused.
        static const bool fused_verify = [] {
            const char * e = getenv("RAON_RDA_FUSED_VERIFY");
            return e == nullptr || atoi(e) != 0;
        }();
        if (!fused_verify || !replay_window || !all_same_seq || batch.n_tokens > GGML_RDA_MAX_TOKENS_LLAMA ||
                replay_layer(il) == nullptr || replay_layer(il)->rda == nullptr) {
            return false;
        }
    } else {
        // RAON_RDA_FUSED_MAX: max tokens of one sequence to fuse (default 4, up to 8)
        static const int fused_max = [] {
            const char * e = getenv("RAON_RDA_FUSED_MAX");
            const int m = e ? atoi(e) : RAON_RDA_FUSED_MAX_TOKENS;
            return m < 1 ? 1 : m > GGML_RDA_MAX_TOKENS_LLAMA ? GGML_RDA_MAX_TOKENS_LLAMA : m;
        }();
        if (batch.n_tokens > (all_same_seq ? fused_max : RAON_RDA_FUSED_MAX_TOKENS)) {
            return false;
        }
    }
    if (!all_same_seq && !has_unique_seq_ids) {
        return false;
    }
    if (model.split_mode == LLAMA_SPLIT_MODE_GRAPH || !s_l || s_l->extra || s_l->type != GGML_TYPE_F32 || !s_l->buffer) {
        return false;
    }

    const int64_t S  = hparams.ssm_d_state;
    const int64_t H  = hparams.ssm_dt_rank;
    const int64_t HS = H*S;
    if (hparams.ssm_d_conv != 4 || hparams.ssm_n_group != H || hparams.ssm_d_inner != HS) {
        return false;
    }
    for (const ggml_tensor * c : { layer.ssm_conv1d_q, layer.ssm_conv1d_k, layer.ssm_conv1d_v }) {
        if (c->type != GGML_TYPE_F32 || !ggml_is_contiguous(c) || c->ne[0] != 4 || ggml_nelements(c) != 4*HS || !c->buffer) {
            return false;
        }
    }
    for (const ggml_tensor * t : { layer.ssm_dt_b, layer.ssm_a, layer.ssm_norm }) {
        if (t->type != GGML_TYPE_F32 || !ggml_is_contiguous(t)) {
            return false;
        }
    }
    if (ggml_nelements(layer.ssm_dt_b) != HS || ggml_nelements(layer.ssm_a) != H || layer.ssm_norm->ne[0] != S) {
        return false;
    }
    if (s_l->ne[0] < 3*3*HS + HS*S) {
        return false;
    }

    // the op updates s_l in place and is scheduled with the conv weights, so both must share a buffer type
    const bool state_on_host = ggml_backend_buffer_is_host(s_l->buffer);
    if (mode == 1 && state_on_host) {
        return false;
    }
    if (ggml_backend_buffer_get_type(layer.ssm_conv1d_q->buffer) != ggml_backend_buffer_get_type(s_l->buffer)) {
        return false;
    }
    return true;
}

ggml_tensor * delta_net::build_layer_attn_rda_fused(ggml_context * ctx0, ggml_cgraph * gf,
        ggml_tensor * input, ggml_tensor * inp_out_ids, int il, const llm_build_cb & cb,
        bool add_residual) const {
    const auto & model   = lctx.model;
    const auto & hparams = model.hparams;
    const auto & layer   = model.layers[il];
    const int64_t n_tok  = input->ne[1];

    auto cur = llm_build_context::llm_build_norm(ctx0, input, hparams, layer.attn_norm, nullptr, LLM_NORM_RMS, cb, il);

    // projections back to back so CUDA quantizes cur once
    auto q = llm_build_context::llm_build_lora_mm(lctx, ctx0, layer.wq, cur);
    auto k = llm_build_context::llm_build_lora_mm(lctx, ctx0, layer.wk, cur);
    auto v = llm_build_context::llm_build_lora_mm(lctx, ctx0, layer.wv, cur);
    auto z = llm_build_context::llm_build_lora_mm(lctx, ctx0, layer.ssm_g_a, cur);
    cb(q, "q", il);
    cb(k, "k", il);
    cb(v, "v", il);
    ggml_build_forward_expand(gf, q);
    ggml_build_forward_expand(gf, k);
    ggml_build_forward_expand(gf, v);
    ggml_build_forward_expand(gf, z);
    if (layer.ssm_g_b) {
        z = llm_build_context::llm_build_lora_mm(lctx, ctx0, layer.ssm_g_b, z);
    }
    cb(z, "z", il);

    auto beta = llm_build_context::llm_build_lora_mm(lctx, ctx0, layer.ssm_beta, cur);
    auto rda  = llm_build_context::llm_build_lora_mm(lctx, ctx0, layer.ssm_rda,  cur);
    auto f    = llm_build_context::llm_build_lora_mm(lctx, ctx0, layer.ssm_f_a,  cur);
    f = llm_build_context::llm_build_lora_mm(lctx, ctx0, layer.ssm_f_b, f);
    cb(beta, "beta", il);
    cb(rda,  "rda_delta", il);
    cb(f,    "decay_raw", il);

    // one group per sequence
    int32_t slots[RAON_RDA_FUSED_MAX_TOKENS];
    bool    reset[RAON_RDA_FUSED_MAX_TOKENS];
    const int n_seqs = all_same_seq ? 1 : (int) n_tok;
    for (int g = 0; g < n_seqs; ++g) {
        slots[g] = (int32_t) token_seq_ids[g];
        reset[g] = batch.pos != nullptr && batch.pos[g] == 0;
    }

    // l2 eps 1e-6 as in HF F.normalize
    auto out = ggml_rda_decode(ctx0, q, k, v,
            layer.ssm_conv1d_q, layer.ssm_conv1d_k, layer.ssm_conv1d_v,
            beta, rda, f, layer.ssm_dt_b, layer.ssm_a, z, layer.ssm_norm, lctx.kv_self.s_l[il],
            n_seqs, slots, reset, hparams.kda_gate_lower_bound, 1e-6f, hparams.f_norm_rms_eps);
    cb(out, "rda_fused_out", il);

    if (llama_replay_layer * rl = replay_window ? replay_layer(il) : nullptr) {
        // replay-rollback verify window: leave the slot untouched, save the op inputs for the commit
        GGML_ASSERT(n_seqs == 1 && !reset[0]);
        out->op_params[GGML_RDA_P_FLAGS_LLAMA] |= GGML_RDA_F_NO_STATE_WRITE_LLAMA;
        llama_replay_build_save_fused(ctx0, gf, *rl, out, q, k, v, f, beta, rda);
    }

    out = llm_build_context::llm_build_lora_mm(lctx, ctx0, layer.ssm_out, out);
    cb(out, "linear_attn_out", il);

    if (inp_out_ids) {
        out   = ggml_get_rows(ctx0, out, inp_out_ids);
        input = ggml_get_rows(ctx0, input, inp_out_ids);
    }
    out = add_residual ? ggml_add(ctx0, out, input) : out;
    cb(out, "ssm_output", il);
    return out;
}

ggml_tensor * delta_net::build_layer_attn_kda(ggml_context * ctx0, ggml_cgraph * gf,
        ggml_tensor * cur, ggml_tensor * inp_out_ids, int il, const llm_build_cb & cb,
        bool add_residual) const {
    auto & layer = lctx.model.layers[il];
    GGML_ASSERT(lctx.model.hparams.is_recurrent(il));
    GGML_ASSERT(layer.wq && layer.wk && layer.wv);
    GGML_ASSERT(layer.ssm_conv1d_q && layer.ssm_conv1d_k && layer.ssm_conv1d_v);
    GGML_ASSERT(layer.ssm_f_a && layer.ssm_g_a && layer.ssm_beta);
    GGML_ASSERT(layer.ssm_a && layer.ssm_dt_b && layer.ssm_norm && layer.ssm_out);

    if (rda_fused_eligible(il)) {
        return build_layer_attn_rda_fused(ctx0, gf, cur, inp_out_ids, il, cb, add_residual);
    }

    // only created when some layer is not fused
    GGML_ASSERT(lctx.inp_s_seq_qnext != nullptr);

    if (all_same_seq) {
        const bool reset_state = batch.pos != nullptr && batch.pos[0] == 0;
        return build_layer_attn_kda_core(ctx0, gf, cur, lctx.inp_s_seq_qnext, inp_out_ids,
                token_seq_ids.front(), reset_state, il, cb, add_residual);
    }

    GGML_ASSERT(has_unique_seq_ids && "bailingmoe3 mixed-sequence batches require unique sequence IDs per token");

    ggml_tensor * out = nullptr;
    for (int64_t i = 0; i < batch.n_tokens; ++i) {
        auto cur_i = ggml_view_2d(ctx0, cur, cur->ne[0], 1, cur->nb[1], (size_t) i * cur->nb[1]);
        auto inp_s_seq_qnext_i = ggml_view_2d(ctx0, lctx.inp_s_seq_qnext, 1, 1,
                lctx.inp_s_seq_qnext->nb[1], (size_t) i * lctx.inp_s_seq_qnext->nb[1]);
        const bool reset_state = batch.pos != nullptr && batch.pos[i] == 0;
        auto out_i = build_layer_attn_kda_core(ctx0, gf, cur_i, inp_s_seq_qnext_i, inp_out_ids,
                (uint32_t) token_seq_ids[i], reset_state, il, cb, add_residual);
        out = out == nullptr ? out_i : ggml_concat(ctx0, out, out_i, 1);
    }
    return out;
}
