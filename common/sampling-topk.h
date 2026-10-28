#pragma once
// Exact top-k for the common sampler: one threshold scan over the row plus sparse penalty updates, instead of
// building and partial-sorting the full candidate list. Returns false (caller uses the original path) unless
// the top-k is unique with a unique order, i.e. identical to what std::partial_sort would give.

#include "llama.h"

#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>

#if defined(__SSE2__) || defined(_M_X64)
#include <emmintrin.h>
#define COMMON_TOPK_SSE2 1
#endif

// pen: (id, value) for every id whose candidate logit differs from logits[id]
static inline bool common_sampler_topk_exact(
        const float * logits, int n_vocab, int k,
        const std::pair<llama_token, float> * pen, int n_pen,
        std::vector<llama_token_data> & work, llama_token_data * out) {
    if (k <= 0 || k > n_vocab) {
        return false;
    }
    const int keep = k + n_pen;
    const int cap  = std::max(4*keep, 512);
    work.clear();
    work.reserve(cap + n_pen + 1);

    auto desc = [](const llama_token_data & a, const llama_token_data & b) { return a.logit > b.logit; };

    float thr = -INFINITY;
    bool  has_nan = false;
    int i = 0;
    auto take = [&](int j) -> bool {
        const float v = logits[j];
        has_nan |= (v != v);
        if (v > thr) {
            work.push_back({ (llama_token) j, v, 0.0f });
            if ((int) work.size() >= cap) {
                std::nth_element(work.begin(), work.begin() + (keep - 1), work.end(), desc);
                thr = work[keep - 1].logit;
                // dropping later ties at thr is fine: acceptance requires the k-th value > thr
                work.erase(std::remove_if(work.begin(), work.end(),
                            [thr](const llama_token_data & c) { return c.logit < thr; }), work.end());
                if ((int) work.size() >= cap) {
                    return false;
                }
            }
        }
        return true;
    };
#ifdef COMMON_TOPK_SSE2
    for (; i + 16 <= n_vocab; i += 16) {
        const __m128 t  = _mm_set1_ps(thr);
        const __m128 a0 = _mm_loadu_ps(logits + i +  0), a1 = _mm_loadu_ps(logits + i +  4);
        const __m128 a2 = _mm_loadu_ps(logits + i +  8), a3 = _mm_loadu_ps(logits + i + 12);
        // !(a <= thr) is true for a > thr and for NaN
        const __m128 m = _mm_or_ps(_mm_or_ps(_mm_cmpnle_ps(a0, t), _mm_cmpnle_ps(a1, t)),
                                   _mm_or_ps(_mm_cmpnle_ps(a2, t), _mm_cmpnle_ps(a3, t)));
        if (_mm_movemask_ps(m)) {
            for (int j = i; j < i + 16; ++j) {
                if (!take(j)) return false;
            }
        }
    }
#endif
    for (; i < n_vocab; ++i) {
        if (!take(i)) return false;
    }
    if (has_nan) {
        return false;
    }

    if (n_pen > 0) {
        // replace penalized ids with their final value
        work.erase(std::remove_if(work.begin(), work.end(), [pen, n_pen](const llama_token_data & c) {
                    for (int j = 0; j < n_pen; ++j) if (pen[j].first == c.id) return true;
                    return false; }), work.end());
        for (int j = 0; j < n_pen; ++j) {
            if (pen[j].second != pen[j].second) {
                return false;
            }
            work.push_back({ pen[j].first, pen[j].second, 0.0f });
        }
    }

    if ((int) work.size() < k) {
        return false;
    }
    std::partial_sort(work.begin(), work.begin() + k, work.end(), desc);
    const float kth = work[k - 1].logit;
    if (!(kth > thr)) {
        return false;                           // an id outside the scan could tie the k-th value
    }
    for (int i = 1; i < k; ++i) {
        if (!(work[i - 1].logit > work[i].logit)) {
            return false;                       // tie inside the top-k
        }
    }
    for (size_t i = k; i < work.size(); ++i) {
        if (!(work[i].logit < kth)) {
            return false;                       // tie at the boundary
        }
    }
    std::copy(work.begin(), work.begin() + k, out);
    return true;
}

// Exact top-k of the final row from the GPU top-n candidates (llama_get_logits_cand_ith). work holds the
// candidates with their final logits; since final <= raw for every id, ids outside the list have final <= thr.
static inline bool common_sampler_topk_from_cand(std::vector<llama_token_data> & work, int k, float thr,
        llama_token_data * out) {
    if (k <= 0 || (int) work.size() < k) {
        return false;
    }
    for (const auto & c : work) {
        if (c.logit != c.logit) {
            return false;
        }
    }
    auto desc = [](const llama_token_data & a, const llama_token_data & b) { return a.logit > b.logit; };
    std::partial_sort(work.begin(), work.begin() + k, work.end(), desc);
    const float kth = work[k - 1].logit;
    if (!(kth > thr)) {
        return false;
    }
    for (int i = 1; i < k; ++i) {
        if (!(work[i - 1].logit > work[i].logit)) {
            return false;
        }
    }
    for (size_t i = k; i < work.size(); ++i) {
        if (!(work[i].logit < kth)) {
            return false;
        }
    }
    std::copy(work.begin(), work.begin() + k, out);
    return true;
}

// Applies bias and penalties with the same float ops as llama_sampling_prepare_impl, then
// common_sampler_topk_from_cand. Fails on any bias > 0 or a penalty that can raise a logit.
static inline bool common_sampler_cand_topk(
        const llama_token * ids, const float * logits, int n, float thr,
        const std::pair<llama_token, float> * bias, int n_bias,
        const std::pair<llama_token, int> * counts, int n_counts,
        float penalty_repeat, float penalty_freq, float penalty_present,
        int k, std::vector<llama_token_data> & work, llama_token_data * out) {
    for (int b = 0; b < n_bias; ++b) {
        if (!(bias[b].second <= 0.0f)) {
            return false;
        }
    }
    if (n_counts > 0 && !(penalty_repeat >= 1.0f && penalty_freq >= 0.0f && penalty_present >= 0.0f)) {
        return false;
    }
    work.resize(n);
    for (int i = 0; i < n; ++i) {
        float l = logits[i];
        for (int b = 0; b < n_bias; ++b) {
            if (bias[b].first == ids[i]) {
                l += bias[b].second;
                break;
            }
        }
        for (int c = 0; c < n_counts; ++c) {
            if (counts[c].first == ids[i]) {
                const int count = counts[c].second;
                if (l <= 0) {
                    l *= penalty_repeat;
                } else {
                    l /= penalty_repeat;
                }
                l -= float(count) * penalty_freq + float(count > 0) * penalty_present;
                break;
            }
        }
        work[i] = { ids[i], l, 0.0f };
    }
    return common_sampler_topk_from_cand(work, k, thr, out);
}
