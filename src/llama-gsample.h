#pragma once
// GPU logits candidates: decode copies only the GGML_OP_LOGITS_TOPM rows; the full logits rows are copied
// lazily the first time something asks for them.
// LLAMA_GSAMPLE=0        no candidate op, logits copied eagerly
// LLAMA_GSAMPLE_M=<n>    candidates per row (default 64, clamped to [8, 256])
// LLAMA_GSAMPLE_STATS=1  print counters when the context is freed

#include "ggml.h"
#include "ggml-backend.h"

#include <cstdint>
#include <cstdlib>
#include <vector>

#define LLAMA_GSAMPLE_MAX_ROWS 16

static inline bool llama_gsample_env_enabled() {
    static const bool enabled = [] {
        const char * e = getenv("LLAMA_GSAMPLE");
        return !(e && e[0] == '0');
    }();
    return enabled;
}

static inline int llama_gsample_env_m() {
    static const int m = [] {
        const char * e = getenv("LLAMA_GSAMPLE_M");
        int v = e ? atoi(e) : 64;
        return v < 8 ? 8 : v > 256 ? 256 : v;
    }();
    return m;
}

struct llama_gsample {
    // lazy full-logits state for the outputs of the last decode
    bool                  pending  = false;    // logits rows [0, src.size()) not copied yet
    ggml_backend_t        backend  = nullptr;
    ggml_tensor         * res      = nullptr;  // result_output of the last graph
    std::vector<int64_t>  src;                 // res row of each output row
    int64_t               n_row    = 0;        // res->ne[0]

    // candidates of the last decode, one entry per output row
    int                   m        = 0;
    std::vector<float>    host;                // [n_out][2m + 2], raw op output
    std::vector<int32_t>  ids;                 // [n_out][m] vocab ids
    std::vector<float>    thr;                 // [n_out]
    std::vector<int32_t>  n_cand;              // [n_out]; 0 = unusable
    std::vector<uint8_t>  usable;              // [n_out]; cleared when consumed or materialized
    bool                  decoded  = false;    // host -> ids/thr done

    // biases the sampler applied to its candidates, applied to the row on materialization
    struct bias { int32_t row; int32_t id; float delta; };
    std::vector<bias>     pending_bias;

    int64_t n_decode_lazy = 0, n_materialize = 0, n_cand_used = 0, n_cand_refused = 0;
};
