// RDA small-batch decode step (GGML_OP_RDA_DECODE) in two kernels; the math is in ../ggml-rda.h.
// rda_prep_f32 stages the per-channel conv, norm and gate values and zeroes the per-head counters; rda_rec_f32 runs
// the recurrence with the state in registers, and the last CTA of a head (threadfence + counter) applies the norm.
// Fusion scope after FlashInfer's fused_kda_decode (Apache-2.0); recurrence lane split after llama.cpp's
// gated_delta_net_cuda (MIT, ggml-cuda/gated_delta_net.cu); conv / gate / norm staging after Strata's k_conv_seq
// and k_gdn_seq (MIT, src/qwen35/q36_engine.cu).
// CUDA-graph safe: scratch from the stream-ordered pool, counters zeroed by the prep kernel of the same op.

#include "rda-decode.cuh"
#include "raon-leftover.cuh"
#include "../ggml-rda.h"

#include <cstdint>
#include <cstring>

struct rda_decode_params {
    const float * q;
    const float * k;
    const float * v;
    const float * cq;
    const float * ck;
    const float * cv;
    const float * beta;
    const float * rda;
    const float * f;
    const float * dt;
    const float * a;
    const float * z;
    const float * nw;
    float       * state;
    float       * dst;
    int64_t q_nb1, k_nb1, v_nb1, beta_nb1, rda_nb1, f_nb1, z_nb1, dst_nb1, state_nb1; // in floats
    int     H;
    int     T;
    int     reset_mask;
    int     no_write;   // leave the state slot untouched
    int     slots[GGML_RDA_MAX_SEQS];
    float   lower_bound;
    float   eps_l2;
    float   eps_rms;
    // scratch per token (n = g*T + t): [n][H*S] for q, k, v, decay, o; [n][H] for beta, attn
    float        * sq;
    float        * sk;
    float        * sv;
    float        * sd;
    float        * so;
    float        * sb;
    float        * sa;
    unsigned int * cnt; // [n_seqs][H]
    // RAON_FB_FOLD (T = 1, one sequence): f = fb_w @ fb_x computed in the prep kernel
    const half  * fb_w;
    const float * fb_x;
    int64_t       fb_s1;
};

static constexpr int RDA_ROWS_PER_CTA = 16; // 2 warps x 8 rows

template <int S, bool FB = false>
__launch_bounds__(4*S, 1)
static __global__ void rda_prep_f32(const rda_decode_params p) {
    constexpr int NT = 4*S;
    constexpr int NW = NT/WARP_SIZE;

    const int h    = blockIdx.x;
    const int g    = blockIdx.y;
    const int tid  = threadIdx.x;
    const int warp = tid / WARP_SIZE;
    const int lane = tid % WARP_SIZE;
    const int T    = p.T;
    const int64_t tok0 = (int64_t) g*T;
    const int64_t HS   = (int64_t) p.H*S;
    const bool rst = (p.reset_mask >> g) & 1;
    const int  r   = tid % S;

    float * row = p.state + (int64_t) p.slots[g]*p.state_nb1;

    __shared__ float sQ[GGML_RDA_MAX_TOKENS*S];
    __shared__ float sK[GGML_RDA_MAX_TOKENS*S];
    __shared__ float sVs[GGML_RDA_MAX_TOKENS];

    __shared__ float sF[FB ? S : 1];
    if constexpr (FB) {
        // f for this head's S channels (T = 1): same lane split and shuffle order as k_raon_f16_mv<16, 1>
        static_assert(S == 128, "f_b fold expects 128-wide f_a");
        if (tid >= 3*S) {
            const int t3 = tid - 3*S;
            const int l  = t3 % 16;
            const int c0 = t3 / 16;                  // 0..7, channel group
            const float4 x0 = ((const float4 *) p.fb_x)[2*l + 0];
            const float4 x1 = ((const float4 *) p.fb_x)[2*l + 1];
            constexpr int NI = S/8;                  // 16 channels per 16-lane group
            uint4 wv[NI];
#pragma unroll
            for (int i = 0; i < NI; ++i) {
                const int64_t row = (int64_t) h*S + c0 + 8*i;
                wv[i] = ((const uint4 *) (p.fb_w + row*p.fb_s1))[l];
            }
#pragma unroll
            for (int i = 0; i < NI; ++i) {
                const half2 * hh = (const half2 *) &wv[i];
                const float2 f0 = __half22float2(hh[0]);
                const float2 f1 = __half22float2(hh[1]);
                const float2 f2 = __half22float2(hh[2]);
                const float2 f3 = __half22float2(hh[3]);
                float sum = 0.0f;
                sum += f0.x*x0.x + f0.y*x0.y + f1.x*x0.z + f1.y*x0.w + f2.x*x1.x + f2.y*x1.y + f3.x*x1.z + f3.y*x1.w;
#pragma unroll
                for (int off = 8; off > 0; off >>= 1) {
                    sum += __shfl_xor_sync(0xffffffff, sum, off, WARP_SIZE);
                }
                if (l == 0) sF[c0 + 8*i] = sum;
            }
        }
    }
    if (tid == 0) {
        p.cnt[g*p.H + h] = 0;
    }
    if (tid < T) {
        const float b = p.beta[(tok0 + tid)*p.beta_nb1 + h];
        const float d = p.rda [(tok0 + tid)*p.rda_nb1  + h];
        p.sb[(tok0 + tid)*p.H + h] = ggml_rda_sigmoid(b);
        sVs[tid] = ggml_rda_v_scale(b, d);
    }
    __syncthreads();

    if (tid < 3*S) {
        const int part = tid / S;
        const int64_t c_in = (int64_t) h*S + r;
        const float * x     = part == 0 ? p.q  : part == 1 ? p.k  : p.v;
        const int64_t x_nb1 = part == 0 ? p.q_nb1 : part == 1 ? p.k_nb1 : p.v_nb1;
        const float * wc    = (part == 0 ? p.cq : part == 1 ? p.ck : p.cv) + c_in*GGML_RDA_D_CONV;
        float * cs          = row + (part*HS + c_in)*(GGML_RDA_D_CONV - 1);

        const float wc0 = wc[0], wc1 = wc[1], wc2 = wc[2], wc3 = wc[3];
        float w0 = rst ? 0.0f : cs[0];
        float w1 = rst ? 0.0f : cs[1];
        float w2 = rst ? 0.0f : cs[2];
        for (int t = 0; t < T; ++t) {
            const float xt = x[(tok0 + t)*x_nb1 + c_in];
            float y = 0.0f;
            y += w0*wc0;
            y += w1*wc1;
            y += w2*wc2;
            y += xt*wc3;
            y = ggml_rda_silu(y);
            if (part == 0) {
                sQ[t*S + r] = y;
            } else if (part == 1) {
                sK[t*S + r] = y;
            } else {
                p.sv[(tok0 + t)*HS + c_in] = y*sVs[t];
            }
            w0 = w1; w1 = w2; w2 = xt;
        }
        if (!p.no_write) {
            cs[0] = w0; cs[1] = w1; cs[2] = w2;
        }
    } else {
        const int64_t c = (int64_t) h*S + r;
        const float dt = p.dt[c];
        const float a  = p.a[h];
        for (int t = 0; t < T; ++t) {
            const float fv = FB ? sF[r] : p.f[(tok0 + t)*p.f_nb1 + c];
            p.sd[(tok0 + t)*HS + c] = ggml_rda_decay(fv, dt, a, p.lower_bound);
        }
    }
    __syncthreads();

    // L2 norms, one warp per q or k vector; q also takes the 1/sqrt(S) scale
    const float scale = rsqrtf((float) S);
    for (int vi = warp; vi < 2*T; vi += NW) {
        float * x = ((vi & 1) ? sK : sQ) + (vi >> 1)*S;
        float ss = 0.0f;
        for (int j = lane; j < S; j += WARP_SIZE) {
            ss += x[j]*x[j];
        }
        ss = warp_reduce_sum(ss);
        const float inv = 1.0f/fmaxf(sqrtf(ss), p.eps_l2);
        float * out = ((vi & 1) ? p.sk : p.sq) + (tok0 + (vi >> 1))*HS + (int64_t) h*S;
        for (int j = lane; j < S; j += WARP_SIZE) {
            const float y = (vi & 1) ? x[j]*inv : x[j]*inv*scale;
            x[j]   = y;
            out[j] = y;
        }
    }
    __syncthreads();
    for (int t = warp; t < T; t += NW) {
        float s = 0.0f;
        for (int j = lane; j < S; j += WARP_SIZE) {
            s += sK[t*S + j]*sQ[t*S + j];
        }
        s = warp_reduce_sum(s);
        if (lane == 0) {
            p.sa[(tok0 + t)*p.H + h] = s;
        }
    }
}

// lane = 8*kg + rl: state row (v index) rr = cta_row0 + 8*warp + rl, columns (k index) {kg*SPL + i}.
// For a fixed i the warp reads 4 contiguous 32-byte row segments.
template <int S>
__launch_bounds__(64)
static __global__ void rda_rec_f32(const rda_decode_params p) {
    constexpr int SPL       = S/4;                  // state values per lane
    constexpr int CTA_PER_H = S/RDA_ROWS_PER_CTA;

    const int h    = blockIdx.x / CTA_PER_H;
    const int sub  = blockIdx.x % CTA_PER_H;
    const int g    = blockIdx.y;
    const int warp = threadIdx.x / WARP_SIZE;
    const int lane = threadIdx.x % WARP_SIZE;
    const int rl   = lane % 8;
    const int kg   = lane / 8;
    const int rr   = sub*RDA_ROWS_PER_CTA + warp*8 + rl;
    const int T    = p.T;
    const int64_t tok0 = (int64_t) g*T;
    const int64_t HS   = (int64_t) p.H*S;
    const bool rst = (p.reset_mask >> g) & 1;

    float * ssm = p.state + (int64_t) p.slots[g]*p.state_nb1 + (GGML_RDA_D_CONV - 1)*3*HS + (int64_t) h*S*S;
    const int c0 = kg*SPL;

    float st[SPL];
#pragma unroll
    for (int i = 0; i < SPL; ++i) {
        st[i] = rst ? 0.0f : ssm[(int64_t) (c0 + i)*S + rr];
    }

    for (int t = 0; t < T; ++t) {
        const int64_t n  = tok0 + t;
        const float * qt = p.sq + n*HS + (int64_t) h*S + c0;
        const float * kt = p.sk + n*HS + (int64_t) h*S + c0;
        const float * dt = p.sd + n*HS + (int64_t) h*S + c0;
        float p1 = 0.0f, p2 = 0.0f;
#pragma unroll
        for (int i = 0; i < SPL; ++i) {
            const float s = st[i]*dt[i];
            st[i] = s;
            p1 += s*kt[i];
            p2 += s*qt[i];
        }
        p1 += __shfl_xor_sync(0xffffffff, p1, 8);
        p2 += __shfl_xor_sync(0xffffffff, p2, 8);
        p1 += __shfl_xor_sync(0xffffffff, p1, 16);
        p2 += __shfl_xor_sync(0xffffffff, p2, 16);

        const float v_new = p.sb[n*p.H + h]*(p.sv[n*HS + (int64_t) h*S + rr] - p1);
        if (kg == 0) {
            p.so[n*HS + (int64_t) h*S + rr] = p2 + v_new*p.sa[n*p.H + h];
        }
#pragma unroll
        for (int i = 0; i < SPL; ++i) {
            st[i] = ggml_rda_clamp_state(st[i] + v_new*kt[i]);
        }
    }

    if (!p.no_write) {
#pragma unroll
        for (int i = 0; i < SPL; ++i) {
            ssm[(int64_t) (c0 + i)*S + rr] = st[i];
        }
    }

    // the last CTA of the head applies the gated RMS norm to all S rows
    __shared__ bool is_last;
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) {
        is_last = atomicAdd(p.cnt + g*p.H + h, 1u) == CTA_PER_H - 1;
    }
    __syncthreads();
    if (!is_last) {
        return;
    }
    __threadfence();
    for (int t = warp; t < T; t += 2) {
        const int64_t n = tok0 + t;
        const float * o = p.so + n*HS + (int64_t) h*S;
        float ss = 0.0f;
        for (int j = lane; j < S; j += WARP_SIZE) {
            const float x = __ldcg(o + j);
            ss += x*x;
        }
        ss = warp_reduce_sum(ss);
        const float rs = 1.0f/sqrtf(ss/S + p.eps_rms);
        const float * zt = p.z + n*p.z_nb1 + (int64_t) h*S;
        float * out = p.dst + n*p.dst_nb1 + (int64_t) h*S;
        for (int j = lane; j < S; j += WARP_SIZE) {
            out[j] = (__ldcg(o + j)*rs*p.nw[j])*ggml_rda_sigmoid(zt[j]);
        }
    }
}

bool ggml_cuda_rda_decode_supported(const ggml_tensor * dst) {
    const int64_t S = dst->src[12]->ne[0];
    const int n_seqs = dst->op_params[GGML_RDA_P_NSEQS];
    if (S != 64 && S != 128) {
        return false;
    }
    if (n_seqs < 1 || n_seqs > GGML_RDA_MAX_SEQS || dst->src[0]->ne[1] % n_seqs != 0) {
        return false;
    }
    return dst->src[0]->ne[1]/n_seqs <= GGML_RDA_MAX_TOKENS;
}

void ggml_cuda_op_rda_decode(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(ggml_cuda_rda_decode_supported(dst));

    const int64_t S      = dst->src[12]->ne[0];
    const int     H      = (int) dst->src[10]->ne[0];
    const int     n_seqs = dst->op_params[GGML_RDA_P_NSEQS];
    const int64_t n_t    = dst->src[0]->ne[1];
    const int     T      = (int) (n_t/n_seqs);
    const int64_t HS     = (int64_t) H*S;

    rda_decode_params p;
    p.q     = (const float *) dst->src[0]->data;
    p.k     = (const float *) dst->src[1]->data;
    p.v     = (const float *) dst->src[2]->data;
    p.cq    = (const float *) dst->src[3]->data;
    p.ck    = (const float *) dst->src[4]->data;
    p.cv    = (const float *) dst->src[5]->data;
    p.beta  = (const float *) dst->src[6]->data;
    p.rda   = (const float *) dst->src[7]->data;
    p.f     = (const float *) dst->src[8]->data;
    p.dt    = (const float *) dst->src[9]->data;
    p.a     = (const float *) dst->src[10]->data;
    p.z     = (const float *) dst->src[11]->data;
    p.nw    = (const float *) dst->src[12]->data;
    p.state = (float *) dst->src[13]->data;
    p.dst   = (float *) dst->data;
    p.q_nb1     = dst->src[0]->nb[1]/sizeof(float);
    p.k_nb1     = dst->src[1]->nb[1]/sizeof(float);
    p.v_nb1     = dst->src[2]->nb[1]/sizeof(float);
    p.beta_nb1  = dst->src[6]->nb[1]/sizeof(float);
    p.rda_nb1   = dst->src[7]->nb[1]/sizeof(float);
    p.f_nb1     = dst->src[8]->nb[1]/sizeof(float);
    p.z_nb1     = dst->src[11]->nb[1]/sizeof(float);
    p.dst_nb1   = dst->nb[1]/sizeof(float);
    p.state_nb1 = dst->src[13]->nb[1]/sizeof(float);
    p.H          = H;
    p.T          = T;
    p.reset_mask = dst->op_params[GGML_RDA_P_RESET];
    p.no_write   = (dst->op_params[GGML_RDA_P_FLAGS] & GGML_RDA_F_NO_STATE_WRITE) != 0;
    for (int g = 0; g < GGML_RDA_MAX_SEQS; ++g) {
        p.slots[g] = g < n_seqs ? dst->op_params[GGML_RDA_P_SLOTS + g] : 0;
    }
    memcpy(&p.lower_bound, dst->op_params + GGML_RDA_P_LB,      sizeof(float));
    memcpy(&p.eps_l2,      dst->op_params + GGML_RDA_P_EPS_L2,  sizeof(float));
    memcpy(&p.eps_rms,     dst->op_params + GGML_RDA_P_EPS_RMS, sizeof(float));

    // scratch: 5 [n_t][HS] + 2 [n_t][H] floats + n_seqs*H counters
    const size_t n_scratch = 5*n_t*HS + 2*n_t*H + (size_t) n_seqs*H;
    ggml_cuda_pool_alloc<float> scratch(ctx.pool(), n_scratch);
    float * s = scratch.get();
    p.sq  = s;              s += n_t*HS;
    p.sk  = s;              s += n_t*HS;
    p.sv  = s;              s += n_t*HS;
    p.sd  = s;              s += n_t*HS;
    p.so  = s;              s += n_t*HS;
    p.sb  = s;              s += n_t*H;
    p.sa  = s;              s += n_t*H;
    p.cnt = (unsigned int *) s;

    cudaStream_t stream = ctx.stream();
    p.fb_w = nullptr; p.fb_x = nullptr; p.fb_s1 = 0;
    bool fb = false;
    if (g_raon_fb_pending.dst != nullptr) {
        const raon_fb_pending pend = g_raon_fb_pending;
        g_raon_fb_pending = raon_fb_pending{};
        if (pend.dst == p.f && S == 128 && T == 1 && n_seqs == 1 && pend.nrows == HS) {
            p.fb_w = pend.w; p.fb_x = pend.x; p.fb_s1 = pend.s1;
            fb = true;
        } else {
            ggml_cuda_raon_fb_launch(pend, stream);   // not fused: run the deferred matvec first
        }
    }
    if (S == 128 && fb) {
        rda_prep_f32<128, true><<<dim3(H, n_seqs), 4*128, 0, stream>>>(p);
        rda_rec_f32<128><<<dim3(H*(128/RDA_ROWS_PER_CTA), n_seqs), 64, 0, stream>>>(p);
    } else if (S == 128) {
        rda_prep_f32<128><<<dim3(H, n_seqs), 4*128, 0, stream>>>(p);
        rda_rec_f32<128><<<dim3(H*(128/RDA_ROWS_PER_CTA), n_seqs), 64, 0, stream>>>(p);
    } else {
        rda_prep_f32<64><<<dim3(H, n_seqs), 4*64, 0, stream>>>(p);
        rda_rec_f32<64><<<dim3(H*(64/RDA_ROWS_PER_CTA), n_seqs), 64, 0, stream>>>(p);
    }
    CUDA_CHECK(cudaGetLastError());
}
