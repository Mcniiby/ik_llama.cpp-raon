#pragma once

// Replay-based recurrent-state rollback for speculative verify (Strata-style).
// Instead of saving the SSM state after every verify token, the verify leaves the state slot untouched and
// saves the op inputs of the window; the commit re-runs the recurrence over the accepted rows into the slot.
// Only ggml is used here so the CPU test can include it without libllama.

#include "ggml.h"

#include <cstdint>
#include <cstring>

struct llama_replay_layer {
    // persistent buffers, allocated for max_tokens rows on the state's backend
    ggml_tensor * q    = nullptr; // [S_k * H_k * max_tokens]
    ggml_tensor * k    = nullptr; // [S_k * H_k * max_tokens]
    ggml_tensor * v    = nullptr; // [S_v * H_v * max_tokens]
    ggml_tensor * g    = nullptr; // [S_v * H_v * max_tokens], per-channel (KDA) gate
    ggml_tensor * beta = nullptr; // [H_v * max_tokens]
    ggml_tensor * rda  = nullptr; // [H_v * max_tokens], fused RDA windows only (write-gate delta)

    // fused RDA window: q/k/v hold the raw pre-conv projections, g the decay input f, beta/rda the raw gates
    bool fused = false;
    int32_t fused_op_params[16] = {};
    ggml_tensor * f_conv_q = nullptr, * f_conv_k = nullptr, * f_conv_v = nullptr;
    ggml_tensor * f_dt_b = nullptr, * f_a = nullptr, * f_norm = nullptr;

    int64_t S_k = 0, H_k = 0, S_v = 0, H_v = 0;
    int64_t g_ne1 = 0;      // S_v (per-channel gate)
    int32_t repeat_type = 0;
    int32_t n_window = 0;   // N of the last saved window

    bool valid() const { return q && k && v && g && beta; }
};

// Graph side of a verify: copy the ggml_delta_net() inputs into the window buffers (contiguous, same shapes).
// q, k must be contiguous; v, g, beta may have any strides.
static inline void llama_replay_build_save(ggml_context * ctx, ggml_cgraph * gf, llama_replay_layer & rl,
        ggml_tensor * q, ggml_tensor * k, ggml_tensor * v, ggml_tensor * g, ggml_tensor * beta, int32_t repeat_type) {
    GGML_ASSERT(rl.valid());
    GGML_ASSERT(q->ne[3] == 1 && "replay rollback supports one sequence per verify");
    GGML_ASSERT(g->ne[1] == v->ne[0] && "replay rollback supports the per-channel (KDA) gate only");
    const int64_t T = q->ne[1];

    rl.S_k = q->ne[0];
    rl.H_k = q->ne[2];
    rl.S_v = v->ne[0];
    rl.H_v = v->ne[2];
    rl.g_ne1 = g->ne[1];
    rl.repeat_type = repeat_type;
    rl.n_window = (int32_t) T;
    rl.fused = false;

    GGML_ASSERT(ggml_nelements(q)    <= ggml_nelements(rl.q));
    GGML_ASSERT(ggml_nelements(k)    <= ggml_nelements(rl.k));
    GGML_ASSERT(ggml_nelements(v)    <= ggml_nelements(rl.v));
    GGML_ASSERT(ggml_nelements(g)    <= ggml_nelements(rl.g));
    GGML_ASSERT(ggml_nelements(beta) <= ggml_nelements(rl.beta));

    auto dst_like = [ctx](ggml_tensor * buf, ggml_tensor * src) {
        return ggml_view_4d(ctx, buf, src->ne[0], src->ne[1], src->ne[2], src->ne[3],
                ggml_row_size(GGML_TYPE_F32, src->ne[0]),
                ggml_row_size(GGML_TYPE_F32, src->ne[0] * src->ne[1]),
                ggml_row_size(GGML_TYPE_F32, src->ne[0] * src->ne[1] * src->ne[2]), 0);
    };
    ggml_build_forward_expand(gf, ggml_cpy(ctx, q,    dst_like(rl.q,    q)));
    ggml_build_forward_expand(gf, ggml_cpy(ctx, k,    dst_like(rl.k,    k)));
    ggml_build_forward_expand(gf, ggml_cpy(ctx, v,    dst_like(rl.v,    v)));
    ggml_build_forward_expand(gf, ggml_cpy(ctx, g,    dst_like(rl.g,    g)));
    ggml_build_forward_expand(gf, ggml_cpy(ctx, beta, dst_like(rl.beta, beta)));
}

// Commit side: re-run the recurrence over the first n_rows saved rows from `state` and write it back.
// The trailing copy is fused by ggml_delta_net_find_state_cpy() so the op updates the slot in place.
static inline ggml_tensor * llama_replay_build_commit(ggml_context * ctx, ggml_cgraph * gf, const llama_replay_layer & rl,
        int32_t n_rows, ggml_tensor * state) {
    GGML_ASSERT(rl.valid() && rl.n_window > 0 && n_rows > 0 && n_rows <= rl.n_window);
    const int64_t T = rl.n_window, A = n_rows;
    const size_t f = sizeof(float);

    // the op wants q/k contiguous, so gather the A-row prefix of each head
    auto qk_rows = [&](ggml_tensor * buf, int64_t S, int64_t H) {
        ggml_tensor * view = ggml_view_4d(ctx, buf, S, A, H, 1, S * f, S * T * f, S * T * H * f, 0);
        return A == T ? view : ggml_cont(ctx, view);
    };
    ggml_tensor * q = qk_rows(rl.q, rl.S_k, rl.H_k);
    ggml_tensor * k = qk_rows(rl.k, rl.S_k, rl.H_k);
    ggml_tensor * v = ggml_view_4d(ctx, rl.v, rl.S_v, A, rl.H_v, 1,
            rl.S_v * f, rl.S_v * T * f, rl.S_v * T * rl.H_v * f, 0);
    ggml_tensor * g = ggml_view_4d(ctx, rl.g, A, rl.g_ne1, rl.H_v, 1,
            T * f, T * rl.g_ne1 * f, T * rl.g_ne1 * rl.H_v * f, 0);
    ggml_tensor * beta = ggml_view_4d(ctx, rl.beta, 1, A, rl.H_v, 1,
            f, T * f, T * rl.H_v * f, 0);

    GGML_ASSERT(state->ne[0] == rl.S_v && state->ne[1] == rl.S_v * rl.H_v && ggml_is_contiguous(state));

    ggml_tensor * dn = ggml_delta_net(ctx, q, k, v, g, beta, state, nullptr);
    dn->op_params[0] = rl.repeat_type;

    const int64_t output_size = rl.S_v * rl.H_v * A;
    const int64_t state_size  = rl.S_v * rl.S_v * rl.H_v;
    ggml_tensor * tail = ggml_view_1d(ctx, dn, state_size, output_size * f);
    ggml_tensor * cpy = ggml_cpy(ctx, tail, state);
    ggml_build_forward_expand(gf, cpy);
    return cpy;
}


// Graph side of a fused RDA verify window: save the per-token op inputs and the op parameters for the commit.
static inline void llama_replay_build_save_fused(ggml_context * ctx, ggml_cgraph * gf, llama_replay_layer & rl,
        ggml_tensor * op, ggml_tensor * q, ggml_tensor * k, ggml_tensor * v, ggml_tensor * f,
        ggml_tensor * beta, ggml_tensor * rda) {
    GGML_ASSERT(rl.valid() && rl.rda != nullptr);
    const int64_t T  = q->ne[1];
    const int64_t HS = q->ne[0];
    const int64_t H  = beta->ne[0];
    GGML_ASSERT(HS*T <= ggml_nelements(rl.q) && HS*T <= ggml_nelements(rl.k) && HS*T <= ggml_nelements(rl.v));
    GGML_ASSERT(HS*T <= ggml_nelements(rl.g) && H*T <= ggml_nelements(rl.beta) && H*T <= ggml_nelements(rl.rda));
    rl.fused    = true;
    rl.n_window = (int32_t) T;
    rl.H_v      = H;
    rl.S_v      = HS / H;
    memcpy(rl.fused_op_params, op->op_params, sizeof(rl.fused_op_params));
    rl.f_conv_q = op->src[3];  rl.f_conv_k = op->src[4];  rl.f_conv_v = op->src[5];
    rl.f_dt_b   = op->src[9];  rl.f_a      = op->src[10]; rl.f_norm   = op->src[12];
    auto save = [&](ggml_tensor * buf, ggml_tensor * src) {
        ggml_tensor * dst = ggml_view_2d(ctx, buf, src->ne[0], T, ggml_row_size(GGML_TYPE_F32, src->ne[0]), 0);
        ggml_build_forward_expand(gf, ggml_cpy(ctx, src, dst));
    };
    save(rl.q, q); save(rl.k, k); save(rl.v, v); save(rl.g, f); save(rl.beta, beta); save(rl.rda, rda);
}

// Commit side of a fused window: GGML_OP_RDA_DECODE over the first n_rows rows, updating the slot in place.
static inline ggml_tensor * llama_replay_build_commit_fused(ggml_context * ctx, ggml_cgraph * gf,
        const llama_replay_layer & rl, int32_t n_rows, ggml_tensor * state, int32_t slot) {
    GGML_ASSERT(rl.valid() && rl.fused && n_rows > 0 && n_rows <= rl.n_window);
    const int64_t HS = rl.S_v * rl.H_v, H = rl.H_v, A = n_rows;
    auto rows = [&](ggml_tensor * buf, int64_t w) {
        return ggml_view_2d(ctx, buf, w, A, ggml_row_size(GGML_TYPE_F32, w), 0);
    };
    ggml_tensor * f = rows(rl.g, HS);
    const int32_t slots[1] = { slot };
    const bool    reset[1] = { false };
    float lb, eps_l2, eps_rms;
    memcpy(&lb,      rl.fused_op_params + 2, sizeof(float)); // GGML_RDA_P_LB
    memcpy(&eps_l2,  rl.fused_op_params + 3, sizeof(float)); // GGML_RDA_P_EPS_L2
    memcpy(&eps_rms, rl.fused_op_params + 4, sizeof(float)); // GGML_RDA_P_EPS_RMS
    // z only feeds the (discarded) gated output; f has the right shape
    ggml_tensor * out = ggml_rda_decode(ctx, rows(rl.q, HS), rows(rl.k, HS), rows(rl.v, HS),
            rl.f_conv_q, rl.f_conv_k, rl.f_conv_v, rows(rl.beta, H), rows(rl.rda, H), f,
            rl.f_dt_b, rl.f_a, f, rl.f_norm, state, 1, slots, reset, lb, eps_l2, eps_rms);
    ggml_build_forward_expand(gf, out);
    return out;
}
