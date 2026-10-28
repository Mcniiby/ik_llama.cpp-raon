#include "common.cuh"
#include "sinkhorn.cuh"

// Sinkhorn normalization of T independent S x S matrices (S <= 8, so a matrix is
// at most 64 floats). One thread per token: the whole matrix lives in a thread-local
// array and the 6-node-per-iteration graph chain collapses into this single kernel.
// Semantics match the reference: softmax over columns, column normalization,
// then (iters - 1) rounds of row + column normalization (ends on columns).
// Input is the flat [S*S, T] row-major tensor (column index fastest); output is
// [S, S, T] with ne0 = row, i.e. transposed on write.

template <int S>
static __global__ void k_sinkhorn(const float * __restrict__ x, float * __restrict__ dst,
                                  const int64_t T, const int iters, const float eps,
                                  const int transposed, const int64_t nb1) {
    const int64_t t = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    if (t >= T) {
        return;
    }

    const float * xt = (const float *)((const char *) x + t*nb1);
    float m[S*S];

    #pragma unroll
    for (int r = 0; r < S; ++r) {
        float mx = xt[r*S];
        for (int c = 1; c < S; ++c) mx = fmaxf(mx, xt[r*S + c]);
        float sum = 0.0f;
        for (int c = 0; c < S; ++c) { m[r*S + c] = expf(xt[r*S + c] - mx); sum += m[r*S + c]; }
        for (int c = 0; c < S; ++c) m[r*S + c] = m[r*S + c]/sum + eps;
    }
    #pragma unroll
    for (int c = 0; c < S; ++c) {
        float sum = eps;
        for (int r = 0; r < S; ++r) sum += m[r*S + c];
        for (int r = 0; r < S; ++r) m[r*S + c] /= sum;
    }
    for (int i = 0; i < iters - 1; ++i) {
        #pragma unroll
        for (int r = 0; r < S; ++r) {
            float sum = eps;
            for (int c = 0; c < S; ++c) sum += m[r*S + c];
            for (int c = 0; c < S; ++c) m[r*S + c] /= sum;
        }
        #pragma unroll
        for (int c = 0; c < S; ++c) {
            float sum = eps;
            for (int r = 0; r < S; ++r) sum += m[r*S + c];
            for (int r = 0; r < S; ++r) m[r*S + c] /= sum;
        }
    }

    float * yt = dst + t*S*S;
    if (transposed) {
        #pragma unroll
        for (int c = 0; c < S; ++c) {
            for (int r = 0; r < S; ++r) yt[c*S + r] = m[r*S + c];
        }
    } else {
        #pragma unroll
        for (int k = 0; k < S*S; ++k) yt[k] = m[k];
    }
}

template <int S>
static __global__ void k_hc_pre(const float * __restrict__ x, const float * __restrict__ scale,
        const float * __restrict__ bias, float * __restrict__ dst,
        const int64_t T, const int iters, const float eps, const int64_t nb1) {
    const int64_t t = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    if (t >= T) {
        return;
    }

    float m[S*S];

    const float * x_pre  = (const float *)((const char *) x + t*nb1);
    const float * x_post = x_pre + S;
    const float * x_comb = x_pre + 2*S;

    float * y_pre  = dst + S*t;
    float * y_post = dst + S*(T + t);
    float * y_comb = dst + 2*S*T + S*S*t;

    #pragma unroll
    for (int i = 0; i < S; ++i) {
        float val = x_pre[i] * scale[0] + bias[i];
        y_pre[i]  = 1.f / (1.f + expf(-val)) + eps;
    }

    #pragma unroll
    for (int i = 0; i < S; ++i) {
        float val = x_post[i] * scale[1] + bias[S + i];
        y_post[i]  = 2.f / (1.f + expf(-val));
    }

    #pragma unroll
    for (int i = 0; i < S*S; ++i) {
        m[i] = x_comb[i] * scale[2] + bias[2*S + i];
    }

    #pragma unroll
    for (int r = 0; r < S; ++r) {
        float mx = m[r*S];
        for (int c = 1; c < S; ++c) mx = fmaxf(mx, m[r*S + c]);
        float sum = 0.0f;
        for (int c = 0; c < S; ++c) { m[r*S + c] = expf(m[r*S + c] - mx); sum += m[r*S + c]; }
        for (int c = 0; c < S; ++c) m[r*S + c] = m[r*S + c]/sum + eps;
    }
    #pragma unroll
    for (int c = 0; c < S; ++c) {
        float sum = eps;
        for (int r = 0; r < S; ++r) sum += m[r*S + c];
        for (int r = 0; r < S; ++r) m[r*S + c] /= sum;
    }
    for (int i = 0; i < iters - 1; ++i) {
        #pragma unroll
        for (int r = 0; r < S; ++r) {
            float sum = eps;
            for (int c = 0; c < S; ++c) sum += m[r*S + c];
            for (int c = 0; c < S; ++c) m[r*S + c] /= sum;
        }
        #pragma unroll
        for (int c = 0; c < S; ++c) {
            float sum = eps;
            for (int r = 0; r < S; ++r) sum += m[r*S + c];
            for (int r = 0; r < S; ++r) m[r*S + c] /= sum;
        }
    }

    #pragma unroll
    for (int i = 0; i < S*S; ++i) y_comb[i] = m[i];
}

// x2/xdst (optional, fused ADD): x is x + x2, stored to xdst; x2 and xdst use x's strides.
// No __restrict__ on x/x2/xdst: the ADD output may be in-place over an operand.
template <int S>
static __global__ void k_hc_post(int ne0, const float * x, const float * __restrict__ post,
        const float * __restrict__ res, const float * __restrict__ comb, float * __restrict__ dst,
        const int nelem, size_t nbx1, size_t nbp1, size_t nbc2, size_t nbr1, size_t nbr2,
        size_t nb1, size_t nb2, const float * x2 = nullptr, float * xdst = nullptr) {

    int ii = blockIdx.x*blockDim.x + threadIdx.x;
    if (ii >= nelem) {
        return;
    }
    int i0 = ii % ne0;
    int i1 = ii / ne0;

    float r[S];

    const float * x_r    = (const float *)((const char *)x + i1*nbx1);
    const float * post_r = (const float *)((const char *)post + i1*nbp1);
    const float * comb_r = (const float *)((const char *)comb + i1*nbc2);

    #pragma unroll
    for (int j = 0; j < S; ++j) {
        const float * res_r  = (const float *)((const char *)res + i1*nbr2 + j*nbr1);
        r[j] = res_r[i0];
    }

    float xv = x_r[i0];
    if (x2) {
        xv += ((const float *)((const char *)x2 + i1*nbx1))[i0];
        ((float *)((char *)xdst + i1*nbx1))[i0] = xv;
    }

    #pragma unroll
    for (int i = 0; i < S; ++i) {
        float sum = xv * post_r[i];
        #pragma unroll
        for (int j = 0; j < S; ++j) {
            sum += comb_r[j*S + i] * r[j];
        }
        float * dst_r = (float *)((char *)dst + i1*nb2 + i*nb1);
        dst_r[i0] = sum;
    }

}

void ggml_cuda_op_sinkhorn(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];

    const int S     = dst->op_params[0];
    const int iters = dst->op_params[1];
    float eps;
    memcpy(&eps, &dst->op_params[2], sizeof(float));
    const int transposed = dst->op_params[3];
    const int64_t T = src0->ne[1];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(S >= 1 && S <= 8);
    GGML_ASSERT(src0->ne[0] == (int64_t) S * S);
    GGML_ASSERT(ggml_is_contiguous(dst));

    if (T == 0) {
        return;
    }

    const int block = 256;
    const int64_t grid = (T + block - 1)/block;
    cudaStream_t stream = ctx.stream();

    const float * x = (const float *) src0->data;
    float * y = (float *) dst->data;

    switch (S) {
        case 1: k_sinkhorn<1><<<grid, block, 0, stream>>>(x, y, T, iters, eps, transposed, src0->nb[1]); break;
        case 2: k_sinkhorn<2><<<grid, block, 0, stream>>>(x, y, T, iters, eps, transposed, src0->nb[1]); break;
        case 3: k_sinkhorn<3><<<grid, block, 0, stream>>>(x, y, T, iters, eps, transposed, src0->nb[1]); break;
        case 4: k_sinkhorn<4><<<grid, block, 0, stream>>>(x, y, T, iters, eps, transposed, src0->nb[1]); break;
        case 5: k_sinkhorn<5><<<grid, block, 0, stream>>>(x, y, T, iters, eps, transposed, src0->nb[1]); break;
        case 6: k_sinkhorn<6><<<grid, block, 0, stream>>>(x, y, T, iters, eps, transposed, src0->nb[1]); break;
        case 7: k_sinkhorn<7><<<grid, block, 0, stream>>>(x, y, T, iters, eps, transposed, src0->nb[1]); break;
        case 8: k_sinkhorn<8><<<grid, block, 0, stream>>>(x, y, T, iters, eps, transposed, src0->nb[1]); break;
        default: GGML_ABORT("sinkhorn: unsupported S");
    }
}

void ggml_cuda_op_hc_pre(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    const ggml_tensor * src2 = dst->src[2];

    const int S     = dst->op_params[0];
    const int iters = dst->op_params[1];
    float eps;
    memcpy(&eps, &dst->op_params[2], sizeof(float));
    const int64_t T = src0->ne[1];

    GGML_ASSERT(src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 && src2->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(S >= 1 && S <= 8);
    GGML_ASSERT(src0->ne[0] == (int64_t) S * S + 2 * S);
    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(src1->ne[0] >= 3 && ggml_nrows(src1) == 1);
    GGML_ASSERT(src2->ne[0] >= S*(S+2) && ggml_nrows(src2) == 1);

    if (T == 0) {
        return;
    }

    const int block = 256;
    const int64_t grid = (T + block - 1)/block;
    cudaStream_t stream = ctx.stream();

    const float * x = (const float *) src0->data;
    const float * s = (const float *) src1->data;
    const float * b = (const float *) src2->data;
    float * y = (float *) dst->data;

    switch (S) {
        case 1: k_hc_pre<1><<<grid, block, 0, stream>>>(x, s, b, y, T, iters, eps, src0->nb[1]); break;
        case 2: k_hc_pre<2><<<grid, block, 0, stream>>>(x, s, b, y, T, iters, eps, src0->nb[1]); break;
        case 3: k_hc_pre<3><<<grid, block, 0, stream>>>(x, s, b, y, T, iters, eps, src0->nb[1]); break;
        case 4: k_hc_pre<4><<<grid, block, 0, stream>>>(x, s, b, y, T, iters, eps, src0->nb[1]); break;
        case 5: k_hc_pre<5><<<grid, block, 0, stream>>>(x, s, b, y, T, iters, eps, src0->nb[1]); break;
        case 6: k_hc_pre<6><<<grid, block, 0, stream>>>(x, s, b, y, T, iters, eps, src0->nb[1]); break;
        case 7: k_hc_pre<7><<<grid, block, 0, stream>>>(x, s, b, y, T, iters, eps, src0->nb[1]); break;
        case 8: k_hc_pre<8><<<grid, block, 0, stream>>>(x, s, b, y, T, iters, eps, src0->nb[1]); break;
        default: GGML_ABORT("hc_pre: unsupported S");
    }
}

// add (optional): the ADD node producing dst->src[0]; its two operands are summed inside the kernel
static void ggml_cuda_op_hc_post_impl(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * add) {
    const ggml_tensor * x    = add ? add->src[0] : dst->src[0];
    const ggml_tensor * post = dst->src[1];
    const ggml_tensor * res  = dst->src[2];
    const ggml_tensor * comb = dst->src[3];

    GGML_ASSERT(x->type == GGML_TYPE_F32 && post->type == GGML_TYPE_F32 && res->type == GGML_TYPE_F32 && comb->type == GGML_TYPE_F32);
    GGML_ASSERT(x->ne[0] == res->ne[0]);
    GGML_ASSERT(x->ne[1] == res->ne[2] && x->ne[1] == post->ne[1] && x->ne[1] == comb->ne[2]);
    GGML_ASSERT(x->ne[2] == 1 && x->ne[3] == 1 && post->ne[2] == 1 && post->ne[3] == 1 && res->ne[3] == 1 && comb->ne[3] == 1);
    GGML_ASSERT(post->ne[0] == res->ne[1] && post->ne[0] == comb->ne[0] && post->ne[0] == comb->ne[1]);

    int S = post->ne[0];

    constexpr int kBlockSize = 256;
    int nelem = x->ne[0]*x->ne[1];
    int nblock = (nelem + kBlockSize - 1)/kBlockSize;
    auto x_d = (const float *)x->data;
    auto post_d = (const float *)post->data;
    auto res_d = (const float *)res->data;
    auto comb_d = (const float *)comb->data;
    auto dst_d = (float *)dst->data;
    const float * x2_d = nullptr;
    float * xdst_d = nullptr;
    if (add) {
        GGML_ASSERT(add == dst->src[0] && add->type == GGML_TYPE_F32 && add->src[1]->type == GGML_TYPE_F32);
        GGML_ASSERT(ggml_are_same_shape(add->src[1], x) && ggml_are_same_shape(add, x));
        GGML_ASSERT(add->src[1]->nb[1] == x->nb[1] && add->nb[1] == x->nb[1]);
        x2_d   = (const float *)add->src[1]->data;
        xdst_d = (float *)add->data;
    }

    switch (S) {
        case 1: k_hc_post<1><<<nblock, kBlockSize, 0, ctx.stream()>>>(x->ne[0], x_d, post_d, res_d, comb_d, dst_d,
                        nelem, x->nb[1], post->nb[1], comb->nb[2], res->nb[1], res->nb[2], dst->nb[1], dst->nb[2], x2_d, xdst_d); break;
        case 2: k_hc_post<2><<<nblock, kBlockSize, 0, ctx.stream()>>>(x->ne[0], x_d, post_d, res_d, comb_d, dst_d,
                        nelem, x->nb[1], post->nb[1], comb->nb[2], res->nb[1], res->nb[2], dst->nb[1], dst->nb[2], x2_d, xdst_d); break;
        case 3: k_hc_post<3><<<nblock, kBlockSize, 0, ctx.stream()>>>(x->ne[0], x_d, post_d, res_d, comb_d, dst_d,
                        nelem, x->nb[1], post->nb[1], comb->nb[2], res->nb[1], res->nb[2], dst->nb[1], dst->nb[2], x2_d, xdst_d); break;
        case 4: k_hc_post<4><<<nblock, kBlockSize, 0, ctx.stream()>>>(x->ne[0], x_d, post_d, res_d, comb_d, dst_d,
                        nelem, x->nb[1], post->nb[1], comb->nb[2], res->nb[1], res->nb[2], dst->nb[1], dst->nb[2], x2_d, xdst_d); break;
        case 5: k_hc_post<5><<<nblock, kBlockSize, 0, ctx.stream()>>>(x->ne[0], x_d, post_d, res_d, comb_d, dst_d,
                        nelem, x->nb[1], post->nb[1], comb->nb[2], res->nb[1], res->nb[2], dst->nb[1], dst->nb[2], x2_d, xdst_d); break;
        case 6: k_hc_post<6><<<nblock, kBlockSize, 0, ctx.stream()>>>(x->ne[0], x_d, post_d, res_d, comb_d, dst_d,
                        nelem, x->nb[1], post->nb[1], comb->nb[2], res->nb[1], res->nb[2], dst->nb[1], dst->nb[2], x2_d, xdst_d); break;
        case 7: k_hc_post<7><<<nblock, kBlockSize, 0, ctx.stream()>>>(x->ne[0], x_d, post_d, res_d, comb_d, dst_d,
                        nelem, x->nb[1], post->nb[1], comb->nb[2], res->nb[1], res->nb[2], dst->nb[1], dst->nb[2], x2_d, xdst_d); break;
        case 8: k_hc_post<8><<<nblock, kBlockSize, 0, ctx.stream()>>>(x->ne[0], x_d, post_d, res_d, comb_d, dst_d,
                        nelem, x->nb[1], post->nb[1], comb->nb[2], res->nb[1], res->nb[2], dst->nb[1], dst->nb[2], x2_d, xdst_d); break;
        default: GGML_ABORT("hc_post: unsupported S");
    }
}

void ggml_cuda_op_hc_post(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_hc_post_impl(ctx, dst, nullptr);
}

void ggml_cuda_op_add_hc_post(ggml_backend_cuda_context & ctx, ggml_tensor * add, ggml_tensor * hc_post) {
    ggml_cuda_op_hc_post_impl(ctx, hc_post, add);
}

// Fused mHC pre-mix (GGML_OP_RAON_HC_PRE), one block per token: sum of squares over all S streams (the folded
// rms_norm), the Helmert chart (ggml-raon-hc.h) on thread 0, then y = sum_s pre_s * x_s.

#include "../ggml-raon-hc.h"

template <int S, int block_size>
static __global__ void k_raon_hc_pre(const float * __restrict__ x, const float * __restrict__ mixes,
        const float * __restrict__ scale, const float * __restrict__ base,
        float * __restrict__ y, float * __restrict__ post, float * __restrict__ comb,
        const int64_t ne0, const int64_t nbx2, const int64_t nbm1, const float rms_eps, const float mag) {
    const int64_t t   = blockIdx.x;
    const int     tid = threadIdx.x;

    const float * xt = (const float *) ((const char *) x + t*nbx2);
    const float * mt = (const float *) ((const char *) mixes + t*nbm1);

    __shared__ float s_sum[block_size/WARP_SIZE];
    __shared__ float s_pre[S];

    float r = 1.0f;
    if (rms_eps >= 0.0f) {
        float tmp = 0.0f;
        for (int64_t i = tid; i < ne0*S; i += block_size) {
            const float xi = xt[i];
            tmp += xi*xi;
        }
        tmp = warp_reduce_sum(tmp);
        if (block_size > WARP_SIZE) {
            const int warp_id = tid/WARP_SIZE;
            const int lane_id = tid%WARP_SIZE;
            if (lane_id == 0) {
                s_sum[warp_id] = tmp;
            }
            __syncthreads();
            tmp = lane_id < block_size/WARP_SIZE ? s_sum[lane_id] : 0.0f;
            tmp = warp_reduce_sum(tmp);
        }
        const float mean = tmp/(ne0*S);
        r = rsqrtf(mean + rms_eps);
    }

    if (tid == 0) {
        float pre[S], pst[S], cmb[S*S];
        raon_hc_coeffs(S, mt, r, scale, base, mag, pre, pst, cmb);
#pragma unroll
        for (int i = 0; i < S; ++i) {
            s_pre[i]       = pre[i];
            post[t*S + i]  = pst[i];
        }
#pragma unroll
        for (int i = 0; i < S*S; ++i) {
            comb[t*S*S + i] = cmb[i];
        }
    }
    __syncthreads();

    float w[S];
#pragma unroll
    for (int s = 0; s < S; ++s) {
        w[s] = s_pre[s];
    }
    float * yt = y + t*ne0;
    for (int64_t i = tid; i < ne0; i += block_size) {
        float sum = 0.0f;
#pragma unroll
        for (int s = 0; s < S; ++s) {
            sum += xt[i + s*ne0]*w[s];
        }
        yt[i] = sum;
    }
}

void ggml_cuda_op_raon_hc_pre(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * x     = dst->src[0];
    const ggml_tensor * mixes = dst->src[1];
    const ggml_tensor * scale = dst->src[2];
    const ggml_tensor * base  = dst->src[3];

    GGML_ASSERT(x->type == GGML_TYPE_F32 && mixes->type == GGML_TYPE_F32 &&
                scale->type == GGML_TYPE_F32 && base->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(x) && ggml_is_contiguous(dst));
    GGML_ASSERT(mixes->nb[0] == sizeof(float));

    float rms_eps, mag;
    memcpy(&rms_eps, &dst->op_params[0], sizeof(float));
    memcpy(&mag,     &dst->op_params[1], sizeof(float));

    const int     S   = (int) x->ne[1];
    const int64_t ne0 = x->ne[0];
    const int64_t T   = x->ne[2];
    if (T == 0) {
        return;
    }

    const float * x_d = (const float *) x->data;
    const float * m_d = (const float *) mixes->data;
    const float * s_d = (const float *) scale->data;
    const float * b_d = (const float *) base->data;
    float * y_d    = (float *) dst->data;
    float * post_d = y_d + ne0*T;
    float * comb_d = post_d + S*T;

    constexpr int block = 512;
    const dim3 grid((unsigned) T, 1, 1);
    cudaStream_t stream = ctx.stream();

#define RAON_HC_LAUNCH(SS) k_raon_hc_pre<SS, block><<<grid, block, 0, stream>>>(x_d, m_d, s_d, b_d, y_d, post_d, comb_d, \
            ne0, x->nb[2], mixes->nb[1], rms_eps, mag)
    switch (S) {
        case 2: RAON_HC_LAUNCH(2); break;
        case 3: RAON_HC_LAUNCH(3); break;
        case 4: RAON_HC_LAUNCH(4); break;
        case 5: RAON_HC_LAUNCH(5); break;
        case 6: RAON_HC_LAUNCH(6); break;
        case 7: RAON_HC_LAUNCH(7); break;
        case 8: RAON_HC_LAUNCH(8); break;
        default: GGML_ABORT("raon_hc_pre: unsupported S");
    }
#undef RAON_HC_LAUNCH
}
