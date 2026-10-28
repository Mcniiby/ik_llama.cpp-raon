// GGML_OP_RAON_MOE_BLOCK vs the composed MoE + shared expert graph, CPU, synthetic Raon-1 shapes and types.
//   test-raon-moe-block [n_seeds (3)] [n_threads (8)] [n_unique experts (16)]
#include "ggml.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

static const int64_t H = 2560, F = 1088, E = 128, K = 6;

static void fill_f32(ggml_tensor * t, float range, std::mt19937 & rng) {
    std::uniform_real_distribution<float> d(-range, range);
    float * p = (float *) t->data;
    for (int64_t i = 0; i < ggml_nelements(t); ++i) p[i] = d(rng);
}

static void fill_q(ggml_tensor * t, int n_unique, float range, std::mt19937 & rng) {
    const int64_t n_mat = t->ne[2], n_rows = t->ne[1], n_per = t->ne[0];
    const int64_t nu = std::min<int64_t>(n_unique, n_mat);
    std::uniform_real_distribution<float> d(-range, range);
    std::vector<float> f(n_rows*n_per);
    for (int64_t u = 0; u < nu; ++u) {
        for (auto & v : f) v = d(rng);
        char * dst = (char *) t->data + u*t->nb[2];
        ggml_quantize_chunk(t->type, f.data(), dst, 0, n_rows, n_per, nullptr, nullptr);
        for (int64_t m = u + nu; m < n_mat; m += nu) memcpy((char *) t->data + m*t->nb[2], dst, t->nb[2]);
    }
}

static void set_situ(ggml_tensor * t, float beta, float lb) {
    memcpy(t->op_params + 1, &beta, sizeof(float));
    memcpy(t->op_params + 2, &lb,   sizeof(float));
}

int main(int argc, char ** argv) {
    const int n_seeds   = argc > 1 ? atoi(argv[1]) : 3;
    const int n_threads = argc > 2 ? atoi(argv[2]) : 8;
    const int n_unique  = argc > 3 ? atoi(argv[3]) : 16;
    const float beta = 4.0f, lb = 25.0f;

    ggml_init_params ip = { /*mem_size*/ (size_t) 1100 << 20, /*mem_buffer*/ nullptr, /*no_alloc*/ false };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * x  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, H, 1);
    ggml_tensor * rw = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, H, E);
    ggml_tensor * rb = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, E);
    ggml_tensor * ue = ggml_new_tensor_3d(ctx, GGML_TYPE_IQ3_KS, H, F, E);
    ggml_tensor * ge = ggml_new_tensor_3d(ctx, GGML_TYPE_IQ3_KS, H, F, E);
    ggml_tensor * de = ggml_new_tensor_3d(ctx, GGML_TYPE_IQ4_KS_R16, F, H, E);
    ggml_tensor * us = ggml_new_tensor_2d(ctx, GGML_TYPE_Q6_K, H, F);
    ggml_tensor * gs = ggml_new_tensor_2d(ctx, GGML_TYPE_Q6_K, H, F);
    ggml_tensor * ds = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, F, H);

    int n_fail = 0;
    for (int seed = 0; seed < n_seeds; ++seed) {
        const bool  norm_w  = seed != 1;
        const float w_scale = seed == 1 ? 2.5f : 1.0f;
        std::mt19937 rng(1000 + seed);
        fill_f32(x, 1.5f, rng); fill_f32(rw, 0.05f, rng); fill_f32(rb, 0.2f, rng);
        fill_q(ue, n_unique, 0.05f, rng); fill_q(ge, n_unique, 0.05f, rng); fill_q(de, n_unique, 0.05f, rng);
        fill_q(us, 1, 0.05f, rng); fill_q(gs, 1, 0.05f, rng); fill_q(ds, 1, 0.05f, rng);

        // composed graph
        ggml_tensor * logits = ggml_mul_mat(ctx, rw, x);
        ggml_tensor * probs  = ggml_sigmoid(ctx, logits);
        ggml_tensor * selp   = ggml_add(ctx, probs, rb);
        ggml_tensor * ids    = ggml_top_k(ctx, selp, K);
        ggml_tensor * w      = ggml_get_rows(ctx, ggml_reshape_3d(ctx, probs, 1, E, 1), ids);
        if (norm_w) {
            w = ggml_reshape_2d(ctx, w, K, 1);
            w = ggml_div(ctx, w, ggml_sum_rows(ctx, w));
            w = ggml_reshape_3d(ctx, w, 1, K, 1);
        }
        if (std::abs(w_scale - 1) > 1e-5f) w = ggml_scale(ctx, w, w_scale);
        ggml_tensor * par = ggml_moe_up_gate(ctx, ue, ge, ggml_reshape_3d(ctx, x, H, 1, 1), ids, GGML_UNARY_OP_SITU);
        set_situ(par, beta, lb);
        ggml_tensor * experts = ggml_mul_mat_id(ctx, de, par, ids);
        ggml_tensor * moe = ggml_mul_multi_add(ctx, experts, w);
        ggml_tensor * sh = ggml_fused_up_gate(ctx, us, gs, x, GGML_UNARY_OP_SITU);
        set_situ(sh, beta, lb);
        sh = ggml_mul_mat(ctx, ds, sh);
        ggml_tensor * y_ref = ggml_add(ctx, moe, sh);

        ggml_tensor * y_op = ggml_raon_moe_block(ctx, x, rw, rb, ue, ge, de, us, gs, ds, K, norm_w, w_scale, beta, lb);

        ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, y_ref);
        ggml_build_forward_expand(gf, y_op);
        ggml_graph_compute_with_ctx(ctx, gf, n_threads);

        // ids printed are the composed graph's top-k
        const int32_t * id = (const int32_t *) ids->data;
        const float * a = (const float *) y_ref->data;
        const float * b = (const float *) y_op->data;
        double num = 0, den = 0, maxd = 0;
        for (int64_t i = 0; i < H; ++i) {
            const double d = (double) a[i] - b[i];
            num += d*d; den += (double) a[i]*a[i]; maxd = std::max(maxd, std::fabs(d));
        }
        const double nmse = num/den;
        const bool ok = nmse < 1e-6 && std::isfinite(nmse);
        n_fail += !ok;
        printf("seed %d norm_w %d w_scale %.1f  ids %d %d %d %d %d %d  y rms %.4f  NMSE %.3e  max|d| %.3e  %s\n",
               seed, norm_w, w_scale, id[0], id[1], id[2], id[3], id[4], id[5], std::sqrt(den/H), nmse, maxd,
               ok ? "PASS" : "FAIL");
        fflush(stdout);
    }
    ggml_free(ctx);
    printf("%s\n", n_fail ? "RESULT: FAIL" : "RESULT: PASS");
    return n_fail ? 1 : 0;
}
