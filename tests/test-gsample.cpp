// CPU check for the GPU logits-candidate sampling path (LLAMA_GSAMPLE): the full sampling path vs
// common_sampler_topk_exact and vs GGML_OP_LOGITS_TOPM + common_sampler_cand_topk, on synthetic rows.
// Accepted candidates must be bit-identical to the full path and sample the same token.

#include "ggml.h"
#include "llama.h"
#include "../common/sampling-topk.h"

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <random>
#include <unordered_map>
#include <vector>

static const int N_VOCAB = 259840;
static const int N_ROW   = 206168;

struct scenario {
    const char * name;
    float sigma;       // logit spread
    float quant;       // > 0: round logits to multiples of quant (ties)
    int   n_peak;      // tokens pushed far above the rest
    float nan_rate;    // probability that a row gets a NaN
    int   win_top;     // window tokens drawn from the top-win_top of the row (repetition)
    float rep, freq, pres;
    bool  eos_ban;     // logit_bias[eos] = -inf (ignore_eos)
    bool  pos_bias;    // a positive bias entry (must fall back)
    int   k;           // top-k (1 = greedy)
    bool  untrimmed;
};

static void topm_ref(const float * x, int n, int m, std::vector<std::pair<float, int>> & out) {
    out.clear();
    for (int i = 0; i < n; ++i) if (x[i] > -INFINITY) out.push_back({ x[i], i });
    std::sort(out.begin(), out.end(), [](const auto & a, const auto & b) {
        return a.first > b.first || (a.first == b.first && a.second < b.second); });
    if ((int) out.size() > m) out.resize(m);
}

static llama_token draw(std::vector<llama_token_data> cand, std::mt19937 & rng) {
    llama_token_data_array a = { cand.data(), cand.size(), 0, true };
    llama_sample_top_p(nullptr, &a, 0.95f, 1);
    llama_sample_temp(nullptr, &a, 1.0f);
    if (a.size < 2) return a.data[0].id;
    std::vector<float> probs(a.size);
    float max = a.data[0].logit;
    for (size_t j = 0; j < a.size; ++j) { probs[j] = a.data[j].logit; max = std::max(max, probs[j]); }
    float sump = 0;
    for (size_t j = 0; j < a.size; ++j) { sump += expf(probs[j] - max); probs[j] = sump; }
    probs.back() += sump;
    auto r = rng();
    auto p = sump * r / rng.max();
    auto it = std::upper_bound(probs.begin(), probs.end(), p);
    return a.data[it - probs.begin()].id;
}

static bool same(const llama_token_data * a, const llama_token_data * b, int k) {
    for (int i = 0; i < k; ++i) {
        if (a[i].id != b[i].id || memcmp(&a[i].logit, &b[i].logit, sizeof(float)) != 0 || a[i].p != b[i].p) return false;
    }
    return true;
}

int main(int argc, char ** argv) {
    const int trials = argc > 1 ? atoi(argv[1]) : 200;
    const int M      = argc > 2 ? atoi(argv[2]) : 64;
    std::mt19937 gen(1234);

    // trimmed head: a random sorted subset of the vocab
    std::vector<uint32_t> out_ids(N_VOCAB);
    for (int i = 0; i < N_VOCAB; ++i) out_ids[i] = i;
    std::shuffle(out_ids.begin(), out_ids.end(), gen);
    out_ids.resize(N_ROW);
    std::sort(out_ids.begin(), out_ids.end());
    const llama_token eos = (llama_token) out_ids[17];
    const llama_token nl  = (llama_token) out_ids[200];

    const scenario scen[] = {
        //  name                sigma quant peak nan   win  rep   freq pres  eos   pos    k   untr
        { "prod",               2.5f, 0.0f,  8, 0.0f,  80, 1.0f, 0.0f, 1.5f, true,  false, 20, false },
        { "prod-repetitive",    2.5f, 0.0f,  3, 0.0f,  25, 1.0f, 0.0f, 1.5f, true,  false, 20, false },
        { "prod-flat",          0.4f, 0.0f,  0, 0.0f,  60, 1.0f, 0.0f, 1.5f, false, false, 20, false },
        { "rep+freq",           2.5f, 0.0f,  5, 0.0f,  40, 1.1f, 0.3f, 0.5f, true,  false, 40, false },
        { "ties",               2.0f, 0.5f,  0, 0.0f,  50, 1.0f, 0.0f, 1.5f, false, false, 20, false },
        { "nan",                2.5f, 0.0f,  5, 0.5f,  50, 1.0f, 0.0f, 1.5f, false, false, 20, false },
        { "pos-bias",           2.5f, 0.0f,  5, 0.0f,  50, 1.0f, 0.0f, 1.5f, false, true,  20, false },
        { "rep<1",              2.5f, 0.0f,  5, 0.0f,  50, 0.9f, 0.0f, 0.0f, false, false, 20, false },
        { "greedy",             2.5f, 0.0f,  5, 0.0f,  50, 1.0f, 0.0f, 1.5f, true,  false,  1, false },
        { "greedy-ties",        2.0f, 1.0f,  0, 0.0f,  50, 1.0f, 0.0f, 1.5f, false, false,  1, false },
        { "untrimmed",          2.5f, 0.0f,  5, 0.0f,  50, 1.0f, 0.0f, 1.5f, true,  false, 20, true  },
    };

    int bad = 0;
    std::vector<float> row, full;
    std::vector<llama_token_data> work, cpu_out, gpu_out, ref_full;
    std::vector<std::pair<float, int>> bf;

    const size_t mem = (size_t) N_VOCAB*sizeof(float)*2 + 16*1024*1024;
    std::vector<uint8_t> arena(mem);

    printf("trials %d per scenario, M = %d\n", trials, M);
    printf("%-18s %8s %8s %8s %8s %8s\n", "scenario", "cpu_acc", "gpu_acc", "gpu_fb", "mismatch", "op_err");
    for (const auto & sc : scen) {
        const int n_row = sc.untrimmed ? N_VOCAB : N_ROW;
        int n_cpu = 0, n_gpu = 0, n_fb = 0, n_mis = 0, n_operr = 0;
        for (int t = 0; t < trials; ++t) {
            std::normal_distribution<float> nd(0.0f, sc.sigma);
            row.resize(n_row);
            for (int i = 0; i < n_row; ++i) {
                float v = nd(gen);
                if (sc.quant > 0) v = std::round(v/sc.quant)*sc.quant;
                row[i] = v;
            }
            for (int p = 0; p < sc.n_peak; ++p) row[gen() % n_row] += 4.0f + 2.0f*p;
            if (sc.nan_rate > 0 && std::uniform_real_distribution<float>(0, 1)(gen) < sc.nan_rate) {
                const float specials[3] = { NAN, INFINITY, -INFINITY };
                for (int q = 0; q <= t % 3; ++q) row[gen() % n_row] = specials[(t + q) % 3];
            }
            auto vid = [&](int r) { return sc.untrimmed ? (llama_token) r : (llama_token) out_ids[r]; };

            // window: half from the row's top-win_top, half random (some outside the head)
            std::vector<std::pair<float, int>> top;
            topm_ref(row.data(), n_row, std::max(sc.win_top, 1), top);
            std::vector<llama_token> window;
            for (int i = 0; i < 64; ++i) {
                if (i % 2 == 0 && !top.empty()) window.push_back(vid(top[gen() % top.size()].second));
                else window.push_back((llama_token) (gen() % N_VOCAB));
            }
            if (t % 3 == 0) window.push_back(nl);
            const bool penalize_nl = (t % 2) == 0;

            std::map<llama_token, float> logit_bias;
            if (sc.eos_ban) logit_bias[eos] = -INFINITY;
            if (t % 4 == 1 && !top.empty()) logit_bias[vid(top[0].second)] = -0.75f;
            if (sc.pos_bias) logit_bias[(llama_token) (gen() % N_VOCAB)] = 30.0f;

            // ---- ref: the full path
            full.resize(N_VOCAB);
            if (sc.untrimmed) {
                full = row;
            } else {
                float mx = row[0];
                for (int i = 0; i < n_row; ++i) mx = row[i] > mx ? row[i] : mx;
                std::fill(full.begin(), full.end(), mx - 60.0f);
                for (int i = 0; i < n_row; ++i) full[out_ids[i]] = row[i];
            }
            for (const auto & b : logit_bias) full[b.first] += b.second;
            ref_full.resize(N_VOCAB);
            for (int i = 0; i < N_VOCAB; ++i) ref_full[i] = { i, full[i], 0.0f };
            llama_token_data_array cur_p = { ref_full.data(), ref_full.size(), 0, false };
            const float nl_logit = full[nl];
            llama_sample_repetition_penalties(nullptr, &cur_p, window.data(), window.size(), sc.rep, sc.freq, sc.pres);
            if (!penalize_nl) {
                for (size_t i = 0; i < cur_p.size; ++i) if (cur_p.data[i].id == nl) { cur_p.data[i].logit = nl_logit; break; }
            }
            const int k = sc.k;
            bool ref_ok = true;
            for (size_t i = 0; i < cur_p.size; ++i) if (cur_p.data[i].logit != cur_p.data[i].logit) ref_ok = false;
            llama_token ref_greedy = -1;
            if (k == 1) {
                auto it = std::max_element(cur_p.data, cur_p.data + cur_p.size,
                        [](const llama_token_data & a, const llama_token_data & b) { return a.logit < b.logit; });
                ref_greedy = it->id;
            }
            llama_sample_top_k(nullptr, &cur_p, k, 1);

            // penalty counts as sampling.cpp builds them
            std::unordered_map<llama_token, int> tc;
            for (auto w : window) tc[w]++;
            std::vector<std::pair<llama_token, int>> counts;
            std::vector<std::pair<llama_token, float>> pen;
            const bool pen_on = !(sc.rep == 1.0f && sc.freq == 0.0f && sc.pres == 0.0f);
            if (pen_on) {
                for (const auto & [tok, c] : tc) {
                    if (tok < 0 || tok >= N_VOCAB || (!penalize_nl && tok == nl)) continue;
                    counts.emplace_back(tok, c);
                    float l = full[tok];
                    if (l <= 0) l *= sc.rep; else l /= sc.rep;
                    l -= float(c) * sc.freq + float(c > 0) * sc.pres;
                    pen.emplace_back(tok, l);
                }
            }

            // ---- cpu: fast path
            cpu_out.resize(k);
            const bool cpu_acc = ref_ok && common_sampler_topk_exact(full.data(), N_VOCAB, k, pen.data(), (int) pen.size(), work, cpu_out.data());
            if (cpu_acc) {
                n_cpu++;
                if (!same(cpu_out.data(), cur_p.data, k)) { n_mis++; printf("  cpu mismatch %s trial %d\n", sc.name, t); }
            }

            // ---- gpu: the op + candidate top-k
            ggml_init_params ip = { mem, arena.data(), false };
            ggml_context * gctx = ggml_init(ip);
            ggml_tensor * x = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, n_row, 1);
            memcpy(x->data, row.data(), n_row*sizeof(float));
            ggml_tensor * y = ggml_logits_topm(gctx, x, M);
            ggml_cgraph * gf = ggml_new_graph(gctx);
            ggml_build_forward_expand(gf, y);
            ggml_graph_compute_with_ctx(gctx, gf, 4);
            const float * h = (const float *) y->data;
            {
                int nbad = 0;
                for (int i = 0; i < n_row; ++i) nbad += !std::isfinite(row[i]);
                topm_ref(row.data(), n_row, M, bf);
                bool ok = (int) h[2*M] == nbad && (int) h[2*M + 1] == (int) bf.size();
                for (size_t q = 0; q < bf.size() && ok; ++q) ok = memcmp(&h[q], &bf[q].first, 4) == 0 && (int) h[M + q] == bf[q].second;
                if (!ok) { n_operr++; printf("  op mismatch %s trial %d\n", sc.name, t); }
            }
            bool gpu_acc = false;
            const int nv = (int) h[2*M + 1];
            if (h[2*M] == 0.0f && nv > 0) {
                std::vector<llama_token> ids(nv);
                for (int q = 0; q < nv; ++q) ids[q] = vid((int) h[M + q]);
                float thr = nv < n_row ? h[nv - 1] : -INFINITY;
                if (!sc.untrimmed) { const float fill = h[0] - 60.0f; thr = fill > thr ? fill : thr; }
                std::vector<std::pair<llama_token, float>> bias(logit_bias.begin(), logit_bias.end());
                gpu_out.resize(k);
                gpu_acc = nv >= k && common_sampler_cand_topk(ids.data(), h, nv, thr, bias.data(), (int) bias.size(),
                        counts.data(), (int) counts.size(), sc.rep, sc.freq, sc.pres, k, work, gpu_out.data());
            }
            ggml_free(gctx);
            bool row_finite = true;
            for (int i = 0; i < n_row; ++i) row_finite &= std::isfinite(row[i]);
            if (gpu_acc && !row_finite) { n_mis++; printf("  gpu accepted a non-finite row %s trial %d\n", sc.name, t); }
            if (gpu_acc) {
                n_gpu++;
                bool ok = ref_ok && same(gpu_out.data(), cur_p.data, k);
                if (ok && k == 1) ok = gpu_out[0].id == ref_greedy;
                if (ok && k > 1) {
                    std::mt19937 r1(t + 1), r2(t + 1);
                    std::vector<llama_token_data> a(cur_p.data, cur_p.data + k), b(gpu_out.begin(), gpu_out.end());
                    ok = draw(a, r1) == draw(b, r2);
                }
                if (!ok) { n_mis++; printf("  gpu mismatch %s trial %d\n", sc.name, t); }
            } else {
                n_fb++;
            }
        }
        printf("%-18s %8d %8d %8d %8d %8d\n", sc.name, n_cpu, n_gpu, n_fb, n_mis, n_operr);
        bad += n_mis + n_operr;
        // expectations: these must always fall back
        if ((!strcmp(sc.name, "pos-bias") || !strcmp(sc.name, "rep<1")) && n_gpu != 0) { printf("  expected no gpu accepts\n"); bad++; }
    }
    printf(bad ? "FAIL (%d)\n" : "OK\n", bad);
    return bad ? 1 : 0;
}
