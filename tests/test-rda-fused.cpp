// GGML_OP_RDA_DECODE vs the unfused KDA/RDA graph it replaces, at Raon-1 shapes, over decode steps of 1-4
// tokens (one sequence and one token per sequence, with slot resets). Compares outputs and the state buffer.
// usage: test-rda-fused [n_threads]     (CPU only)

#include "ggml.h"
#include "../src/llama-delta-net.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

static constexpr int64_t S         = 128;
static constexpr int64_t H         = 40;
static constexpr int64_t HS        = H*S;
static constexpr int64_t D_CONV    = 4;
static constexpr int64_t CONV_DIM  = 3*HS;
static constexpr int64_t STATE_DIM = (D_CONV - 1)*CONV_DIM + HS*S;
static constexpr int64_t N_SLOTS   = 5;
static constexpr float   LB        = -5.0f;
static constexpr float   EPS_L2    = 1e-6f;
static constexpr float   EPS_RMS   = 1e-6f;

static std::mt19937 rng(1234);

static void fill_uniform(float * x, size_t n, float lo, float hi) {
    std::uniform_real_distribution<float> d(lo, hi);
    for (size_t i = 0; i < n; ++i) x[i] = d(rng);
}

static ggml_tensor * new_f32(ggml_context * ctx, int64_t ne0, int64_t ne1, float lo, float hi) {
    ggml_tensor * t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ne0, ne1);
    fill_uniform((float *) t->data, ggml_nelements(t), lo, hi);
    return t;
}

struct rda_weights {
    ggml_tensor * cq, * ck, * cv, * dt, * a, * norm;
};

struct rda_inputs {
    ggml_tensor * q, * k, * v, * f, * z, * beta, * rda;
};

static const llm_build_cb cb = [](ggml_tensor * t, const char * name, int il) {
    ggml_format_name(t, "%s-%d", name, il);
};

// the unfused graph of build_layer_attn_kda_core between the projections and ssm_out
static ggml_tensor * build_unfused_core(ggml_context * ctx, ggml_cgraph * gf, ggml_tensor * state,
        const rda_weights & w, const rda_inputs & in, int slot, bool reset) {
    const int64_t n_tok = in.q->ne[1];

    auto qkv = ggml_concat(ctx, ggml_concat(ctx, in.q, in.k, 0), in.v, 0);

    // build_kda_beta_gate
    auto beta = ggml_reshape_4d(ctx, in.beta, H, 1, n_tok, 1);
    auto raw  = ggml_reshape_4d(ctx, in.f, S, H, n_tok, 1);
    auto log_decay = ggml_add(ctx, raw, ggml_reshape_4d(ctx, w.dt, S, H, 1, 1));
    log_decay = ggml_mul(ctx, log_decay, ggml_reshape_4d(ctx, w.a, 1, H, 1, 1));
    log_decay = ggml_sigmoid(ctx, ggml_scale(ctx, log_decay, -1.0f));
    log_decay = ggml_scale(ctx, log_decay, LB);

    // build_kda_conv
    auto conv = ggml_concat(ctx, ggml_reshape_2d(ctx, w.cq, D_CONV, HS), ggml_reshape_2d(ctx, w.ck, D_CONV, HS), 1);
    conv = ggml_concat(ctx, conv, ggml_reshape_2d(ctx, w.cv, D_CONV, HS), 1);

    // RDA V scale
    auto delta = ggml_reshape_4d(ctx, in.rda, H, 1, n_tok, 1);
    auto erase = ggml_softplus(ctx, ggml_neg(ctx, beta));
    auto write = ggml_softplus(ctx, ggml_neg(ctx, ggml_add(ctx, beta, delta)));
    auto v_scale = ggml_reshape_4d(ctx, ggml_exp(ctx, ggml_sub(ctx, erase, write)), 1, H, n_tok, 1);

    auto ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 1, n_tok);
    memset(ids->data, 0, ggml_nbytes(ids));

    auto output = delta_net::build_qkv(ctx, state, conv, qkv, ids, beta, log_decay,
            S, H, S, H, D_CONV, slot, N_SLOTS, reset, EPS_L2, 1, 0, cb, gf, nullptr, nullptr, v_scale);

    // build_kda_gated_output (up to ssm_out)
    auto o  = ggml_reshape_2d(ctx, output, S, H*n_tok);
    auto z2 = ggml_reshape_2d(ctx, in.z, S, H*n_tok);
    o = ggml_fused_rms_norm(ctx, o, w.norm, EPS_RMS);
    o = ggml_mul(ctx, ggml_sigmoid(ctx, z2), o);
    return ggml_reshape_2d(ctx, o, HS, n_tok);
}

static ggml_tensor * col(ggml_context * ctx, ggml_tensor * t, int64_t i) {
    return ggml_view_2d(ctx, t, t->ne[0], 1, t->nb[1], i*t->nb[1]);
}

struct step_cfg {
    bool one_seq;              // all tokens one sequence, else one token per sequence
    std::vector<int> slots;    // one per group
    std::vector<int> resets;   // groups whose slot starts from zero (pos == 0)
    int n_tok;
};

int main(int argc, char ** argv) {
    const int n_threads = argc > 1 ? atoi(argv[1]) : 8;

    ggml_init_params wp = { (size_t) 256*1024*1024, nullptr, false };
    ggml_context * ctx_w = ggml_init(wp);

    rda_weights w;
    w.cq   = new_f32(ctx_w, D_CONV, HS, -0.5f, 0.5f);
    w.ck   = new_f32(ctx_w, D_CONV, HS, -0.5f, 0.5f);
    w.cv   = new_f32(ctx_w, D_CONV, HS, -0.5f, 0.5f);
    w.dt   = new_f32(ctx_w, HS, 1, -1.0f, 1.0f);
    w.a    = new_f32(ctx_w, H, 1, -1.0f, 1.0f);
    for (int64_t h = 0; h < H; ++h) {
        float * a = (float *) w.a->data;
        a[h] = -expf(a[h]); // ssm_a = -exp(A_log)
    }
    w.norm = new_f32(ctx_w, S, 1, 0.5f, 1.5f);
    // the conv weights as the model loads them: [d_conv, 1, H*S]
    w.cq = ggml_reshape_3d(ctx_w, w.cq, D_CONV, 1, HS);
    w.ck = ggml_reshape_3d(ctx_w, w.ck, D_CONV, 1, HS);
    w.cv = ggml_reshape_3d(ctx_w, w.cv, D_CONV, 1, HS);

    // two copies of a padded multi-slot state buffer (row = conv state | delta state | padding)
    const int64_t row = STATE_DIM + 16;
    ggml_tensor * state_ref = ggml_new_tensor_2d(ctx_w, GGML_TYPE_F32, row, N_SLOTS);
    ggml_tensor * state_fus = ggml_new_tensor_2d(ctx_w, GGML_TYPE_F32, row, N_SLOTS);
    for (int64_t s = 0; s < N_SLOTS; ++s) {
        float * r = (float *) ((char *) state_ref->data + s*state_ref->nb[1]);
        fill_uniform(r, (D_CONV - 1)*CONV_DIM, -2.0f, 2.0f);
        fill_uniform(r + (D_CONV - 1)*CONV_DIM, HS*S, -0.5f, 0.5f);
        fill_uniform(r + STATE_DIM, row - STATE_DIM, 7.0f, 7.0f);
    }
    memcpy(state_fus->data, state_ref->data, ggml_nbytes(state_ref));

    std::vector<step_cfg> sched;
    for (int round = 0; round < 3; ++round) {
        sched.push_back({ true,  { 2 }, round == 0 ? std::vector<int>{ 0 } : std::vector<int>{}, 1 });
        sched.push_back({ true,  { 2 }, {}, 1 });
        sched.push_back({ true,  { 2 }, {}, 2 });
        sched.push_back({ true,  { 1 }, {}, 3 });
        sched.push_back({ true,  { 2 }, {}, 4 });
        sched.push_back({ false, { 0, 3 }, {}, 2 });
        sched.push_back({ false, { 1, 2, 4 }, round == 1 ? std::vector<int>{ 1 } : std::vector<int>{}, 3 });
        sched.push_back({ false, { 4, 0, 2, 3 }, {}, 4 });
        sched.push_back({ true,  { 3 }, {}, 1 });
        sched.push_back({ true,  { 4 }, round == 2 ? std::vector<int>{ 0 } : std::vector<int>{}, 4 });
    }

    double worst_out = 0.0, worst_state = 0.0;
    bool ok = true;

    for (size_t si = 0; si < sched.size(); ++si) {
        const step_cfg & c = sched[si];
        const int n = c.n_tok;

        ggml_init_params gp = { (size_t) 512*1024*1024, nullptr, false };
        ggml_context * ctx = ggml_init(gp);

        rda_inputs in;
        in.q    = new_f32(ctx, HS, n, -1.0f, 1.0f);
        in.k    = new_f32(ctx, HS, n, -1.0f, 1.0f);
        in.v    = new_f32(ctx, HS, n, -1.0f, 1.0f);
        in.f    = new_f32(ctx, HS, n, -3.0f, 2.0f);
        in.z    = new_f32(ctx, HS, n, -2.0f, 2.0f);
        in.beta = new_f32(ctx, H, n, -3.0f, 3.0f);
        in.rda  = new_f32(ctx, H, n, -2.0f, 2.0f);

        const int n_groups = c.one_seq ? 1 : n;
        std::vector<int32_t> slots(c.slots.begin(), c.slots.end());
        bool reset[8] = {};
        for (int g : c.resets) reset[g] = true;

        // unfused reference
        ggml_cgraph * gf_ref = ggml_new_graph_custom(ctx, 8192, false);
        ggml_tensor * out_ref = nullptr;
        if (c.one_seq) {
            out_ref = build_unfused_core(ctx, gf_ref, state_ref, w, in, slots[0], reset[0]);
        } else {
            for (int i = 0; i < n; ++i) {
                rda_inputs in_i = { col(ctx, in.q, i), col(ctx, in.k, i), col(ctx, in.v, i), col(ctx, in.f, i),
                                    col(ctx, in.z, i), col(ctx, in.beta, i), col(ctx, in.rda, i) };
                auto out_i = build_unfused_core(ctx, gf_ref, state_ref, w, in_i, slots[i], reset[i]);
                out_ref = out_ref ? ggml_concat(ctx, out_ref, out_i, 1) : out_i;
            }
        }
        ggml_build_forward_expand(gf_ref, out_ref);
        if (si < 2) {
            int n_real = 0;
            std::string ops;
            for (int i = 0; i < ggml_graph_n_nodes(gf_ref); ++i) {
                ggml_tensor * t = ggml_graph_node(gf_ref, i);
                if (ggml_is_noop(t)) continue;
                ++n_real;
                ops += std::string(t->op == GGML_OP_UNARY ? ggml_unary_op_name(ggml_get_unary_op(t)) : ggml_op_name(t->op)) + " ";
            }
            printf("unfused core, n_tokens=%d%s: %d nodes, %d non-view: %s\n", n, reset[0] ? " (reset)" : "",
                    ggml_graph_n_nodes(gf_ref), n_real, ops.c_str());
        }
        ggml_graph_compute_with_ctx(ctx, gf_ref, n_threads);

        // fused
        ggml_cgraph * gf_fus = ggml_new_graph(ctx);
        ggml_tensor * out_fus = ggml_rda_decode(ctx, in.q, in.k, in.v, w.cq, w.ck, w.cv, in.beta, in.rda, in.f,
                w.dt, w.a, in.z, w.norm, state_fus, n_groups, slots.data(), reset, LB, EPS_L2, EPS_RMS);
        ggml_build_forward_expand(gf_fus, out_fus);
        ggml_graph_compute_with_ctx(ctx, gf_fus, n_threads);

        GGML_ASSERT(ggml_nelements(out_ref) == ggml_nelements(out_fus));
        const float * a = (const float *) out_ref->data;
        const float * b = (const float *) out_fus->data;
        double max_ref = 0.0, max_err = 0.0;
        for (int64_t i = 0; i < ggml_nelements(out_ref); ++i) {
            max_ref = std::max(max_ref, (double) fabsf(a[i]));
            max_err = std::max(max_err, (double) fabsf(a[i] - b[i]));
            if (!std::isfinite(b[i])) max_err = INFINITY;
        }
        double smax = 0.0, serr = 0.0, cerr = 0.0;
        for (int64_t s = 0; s < N_SLOTS; ++s) {
            const float * ra = (const float *) ((const char *) state_ref->data + s*state_ref->nb[1]);
            const float * rb = (const float *) ((const char *) state_fus->data + s*state_fus->nb[1]);
            for (int64_t i = 0; i < row; ++i) {
                const double e = fabs((double) ra[i] - rb[i]);
                if (i < (D_CONV - 1)*CONV_DIM) {
                    cerr = std::max(cerr, e);
                } else {
                    smax = std::max(smax, (double) fabsf(ra[i]));
                    serr = std::max(serr, e);
                }
                if (!std::isfinite(rb[i])) serr = INFINITY;
            }
        }
        const double rel_out = max_err/std::max(max_ref, 1e-30);
        const double rel_st  = serr/std::max(smax, 1e-30);
        worst_out   = std::max(worst_out, rel_out);
        worst_state = std::max(worst_state, rel_st);
        const bool step_ok = rel_out < 1e-4 && rel_st < 1e-4 && cerr == 0.0;
        ok = ok && step_ok;

        std::string sl;
        for (int s : c.slots) sl += std::to_string(s) + " ";
        printf("step %2zu: %s n=%d slots[ %s] reset=%zu | out max %.3e err %.3e (rel %.2e) | delta state max %.3e err %.3e (rel %.2e) | conv state err %.1e %s\n",
                si, c.one_seq ? "1 seq " : "n seqs", n, sl.c_str(), c.resets.size(),
                max_ref, max_err, rel_out, smax, serr, rel_st, cerr, step_ok ? "ok" : "FAIL");

        ggml_free(ctx);
    }

    printf("worst relative error: output %.3e, delta state %.3e -> %s\n", worst_out, worst_state, ok ? "PASS" : "FAIL");
    ggml_free(ctx_w);
    return ok ? 0 : 1;
}
