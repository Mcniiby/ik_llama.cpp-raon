// F32/F16/BF16 matrix x F32 vector(s) for small batch sizes.
// Ported from mainline llama.cpp ggml-cuda/mmvf.cu (plain MUL_MAT only: no MUL_MAT_ID, GLU/bias fusion or PDL).
// F16 weights always accumulate in F32 (mainline uses half2 at GGML_PREC_DEFAULT) to avoid F16 overflow.

#include "mmvf.cuh"

#include <cstdlib>
#include <cstring>

template <typename T>
static __device__ __forceinline__ float2 mmvf_load2(const T * __restrict__ x, const int i);

template <>
__device__ __forceinline__ float2 mmvf_load2<float>(const float * __restrict__ x, const int i) {
    return ((const float2 *) x)[i];
}

template <>
__device__ __forceinline__ float2 mmvf_load2<half>(const half * __restrict__ x, const int i) {
    return __half22float2(((const half2 *) x)[i]);
}

template <>
__device__ __forceinline__ float2 mmvf_load2<nv_bfloat16>(const nv_bfloat16 * __restrict__ x, const int i) {
    const nv_bfloat162 v = ((const nv_bfloat162 *) x)[i];
    return make_float2(__bfloat162float(v.x), __bfloat162float(v.y));
}

template <typename T, int ncols_dst, int block_size>
static __global__ void mul_mat_vec_f(
        const T * __restrict__ x, const float * __restrict__ y, float * __restrict__ dst,
        const int ncols2, const int64_t stride_row, const int64_t stride_col_y2, const int64_t stride_col_dst,
        const int channel_ratio, const int64_t stride_channel_x, const int64_t stride_channel_y, const int64_t stride_channel_dst,
        const int sample_ratio,  const int64_t stride_sample_x,  const int64_t stride_sample_y,  const int64_t stride_sample_dst) {
    const int row         = blockIdx.x;
    const int channel_dst = blockIdx.y;
    const int sample_dst  = blockIdx.z;
    const int channel_x   = channel_dst / channel_ratio;
    const int sample_x    = sample_dst  / sample_ratio;
    const int tid         = threadIdx.x;

    x   += sample_x  *stride_sample_x   + channel_x  *stride_channel_x   + row*stride_row;
    y   += sample_dst*stride_sample_y   + channel_dst*stride_channel_y;
    dst += sample_dst*stride_sample_dst + channel_dst*stride_channel_dst;

    const float2 * y2 = (const float2 *) y;

    __shared__ float buf_iw[WARP_SIZE];
    if (block_size > WARP_SIZE) {
        if (tid < WARP_SIZE) {
            buf_iw[tid] = 0.0f;
        }
        __syncthreads();
    }

    float sumf[ncols_dst] = {0.0f};

    for (int col2 = tid; col2 < ncols2; col2 += block_size) {
        const float2 tmpx = mmvf_load2<T>(x, col2);
#pragma unroll
        for (int j = 0; j < ncols_dst; ++j) {
            const float2 tmpy = y2[j*stride_col_y2 + col2];
            sumf[j] += tmpx.x*tmpy.x;
            sumf[j] += tmpx.y*tmpy.y;
        }
    }

#pragma unroll
    for (int j = 0; j < ncols_dst; ++j) {
        sumf[j] = warp_reduce_sum(sumf[j]);

        if (block_size > WARP_SIZE) {
            buf_iw[tid/WARP_SIZE] = sumf[j];
            __syncthreads();
            if (tid < WARP_SIZE) {
                sumf[j] = buf_iw[tid];
                sumf[j] = warp_reduce_sum(sumf[j]);
            }
            __syncthreads();
        }
    }

    if (tid >= ncols_dst) {
        return;
    }

    dst[tid*stride_col_dst + row] = sumf[tid];
}

struct mmvf_args {
    int     ncols2;
    int64_t nrows;
    int64_t stride_row, stride_col_y2, stride_col_dst;
    int     channel_ratio;
    int64_t nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst;
    int     sample_ratio;
    int64_t nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst;
};

template <typename T, int ncols_dst, int block_size>
static void mul_mat_vec_f_launch(const T * x, const float * y, float * dst, const mmvf_args & a, cudaStream_t stream) {
    const dim3 block_nums(a.nrows, a.nchannels_dst, a.nsamples_dst);
    const dim3 block_dims(block_size, 1, 1);
    mul_mat_vec_f<T, ncols_dst, block_size><<<block_nums, block_dims, 0, stream>>>(
        x, y, dst, a.ncols2, a.stride_row, a.stride_col_y2, a.stride_col_dst,
        a.channel_ratio, a.stride_channel_x, a.stride_channel_y, a.stride_channel_dst,
        a.sample_ratio,  a.stride_sample_x,  a.stride_sample_y,  a.stride_sample_dst);
}

template <typename T, int ncols_dst>
static void mul_mat_vec_f_switch_block_size(const T * x, const float * y, float * dst, const mmvf_args & a,
        const int block_size, cudaStream_t stream) {
    switch (block_size) {
        case  32: mul_mat_vec_f_launch<T, ncols_dst,  32>(x, y, dst, a, stream); break;
        case  64: mul_mat_vec_f_launch<T, ncols_dst,  64>(x, y, dst, a, stream); break;
        case  96: mul_mat_vec_f_launch<T, ncols_dst,  96>(x, y, dst, a, stream); break;
        case 128: mul_mat_vec_f_launch<T, ncols_dst, 128>(x, y, dst, a, stream); break;
        case 160: mul_mat_vec_f_launch<T, ncols_dst, 160>(x, y, dst, a, stream); break;
        case 192: mul_mat_vec_f_launch<T, ncols_dst, 192>(x, y, dst, a, stream); break;
        case 224: mul_mat_vec_f_launch<T, ncols_dst, 224>(x, y, dst, a, stream); break;
        case 256: mul_mat_vec_f_launch<T, ncols_dst, 256>(x, y, dst, a, stream); break;
        default: GGML_ABORT("fatal error");
    }
}

template <typename T>
static void mul_mat_vec_f_cuda(const T * x, const float * y, float * dst, const mmvf_args & a, const int64_t ncols_dst,
        const int cc, cudaStream_t stream) {
    // Same block size choice as mainline: fewest loop iterations, ties go to the smaller block.
    const int64_t ncols = 2*int64_t(a.ncols2);
    int     block_size_best = WARP_SIZE;
    int64_t niter_best      = (ncols + 2*WARP_SIZE - 1) / (2*WARP_SIZE);
    int     max_block_size  = 256;
    if (GGML_CUDA_CC_IS_AMD(cc) && cc < CC_RDNA1) {
        max_block_size = 128;
    }
    for (int block_size = 2*WARP_SIZE; block_size <= max_block_size; block_size += WARP_SIZE) {
        const int64_t niter = (ncols + 2*block_size - 1) / (2*block_size);
        if (niter < niter_best) {
            niter_best      = niter;
            block_size_best = block_size;
        }
    }

    switch (ncols_dst) {
        case 1: mul_mat_vec_f_switch_block_size<T, 1>(x, y, dst, a, block_size_best, stream); break;
        case 2: mul_mat_vec_f_switch_block_size<T, 2>(x, y, dst, a, block_size_best, stream); break;
        case 3: mul_mat_vec_f_switch_block_size<T, 3>(x, y, dst, a, block_size_best, stream); break;
        case 4: mul_mat_vec_f_switch_block_size<T, 4>(x, y, dst, a, block_size_best, stream); break;
        case 5: mul_mat_vec_f_switch_block_size<T, 5>(x, y, dst, a, block_size_best, stream); break;
        case 6: mul_mat_vec_f_switch_block_size<T, 6>(x, y, dst, a, block_size_best, stream); break;
        case 7: mul_mat_vec_f_switch_block_size<T, 7>(x, y, dst, a, block_size_best, stream); break;
        case 8: mul_mat_vec_f_switch_block_size<T, 8>(x, y, dst, a, block_size_best, stream); break;
        default: GGML_ABORT("fatal error");
    }
}

void ggml_cuda_mul_mat_vec_f(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);

    GGML_TENSOR_BINARY_OP_LOCALS;

    const size_t ts_src0 = ggml_type_size(src0->type);
    const size_t ts_src1 = ggml_type_size(src1->type);
    const size_t ts_dst  = ggml_type_size(dst->type);

    GGML_ASSERT(ne00 % 2 == 0 && ne10 == ne00);
    GGML_ASSERT(nb00 == ts_src0 && nb10 == ts_src1 && nb0 == ts_dst);
    GGML_ASSERT(ne11 >= 1 && ne11 <= MMVF_MAX_BATCH_SIZE && ne1 == ne11);
    GGML_ASSERT(ne12 == ne2 && ne13 == ne3);
    GGML_ASSERT(ne2 % ne02 == 0 && ne3 % ne03 == 0);

    mmvf_args a;
    a.ncols2             = int(ne00/2);
    a.nrows              = ne01;
    a.stride_row         = nb01/ts_src0;
    a.stride_col_y2      = (nb11/ts_src1)/2;
    a.stride_col_dst     = nb1/ts_dst;
    a.channel_ratio      = int(ne2/ne02);
    a.nchannels_dst      = ne2;
    a.stride_channel_x   = nb02/ts_src0;
    a.stride_channel_y   = nb12/ts_src1;
    a.stride_channel_dst = nb2/ts_dst;
    a.sample_ratio       = int(ne3/ne03);
    a.nsamples_dst       = ne3;
    a.stride_sample_x    = nb03/ts_src0;
    a.stride_sample_y    = nb13/ts_src1;
    a.stride_sample_dst  = nb3/ts_dst;

    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    const float * src1_d = (const float *) src1->data;
    float       * dst_d  = (float       *) dst->data;
    cudaStream_t stream  = ctx.stream();

    switch (src0->type) {
        case GGML_TYPE_F32:
            mul_mat_vec_f_cuda((const float       *) src0->data, src1_d, dst_d, a, ne11, cc, stream);
            break;
        case GGML_TYPE_F16:
            mul_mat_vec_f_cuda((const half        *) src0->data, src1_d, dst_d, a, ne11, cc, stream);
            break;
        case GGML_TYPE_BF16:
            mul_mat_vec_f_cuda((const nv_bfloat16 *) src0->data, src1_d, dst_d, a, ne11, cc, stream);
            break;
        default:
            GGML_ABORT("unsupported type: %s", ggml_type_name(src0->type));
    }
    CUDA_CHECK(cudaGetLastError());
}

static bool ggml_cuda_mmvf_disabled() {
    static const bool disabled = [] {
        const char * e = getenv("GGML_CUDA_NO_MMVF");
        return e != nullptr && *e != '\0' && strcmp(e, "0") != 0;
    }();
    return disabled;
}

bool ggml_cuda_should_use_mmvf(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst, const int cc) {
    if (ggml_cuda_mmvf_disabled()) {
        return false;
    }
    const ggml_type type = src0->type;
    if ((type != GGML_TYPE_F32 && type != GGML_TYPE_F16 && type != GGML_TYPE_BF16) ||
            src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return false;
    }

    const int64_t ne11 = src1->ne[1];
    if (ne11 < 1 || ne11 > MMVF_MAX_BATCH_SIZE || dst->ne[1] != ne11) {
        return false;
    }

    // src0: contiguous rows, half2/float2/nv_bfloat162 aligned row/channel/sample starts
    const size_t ts = ggml_type_size(type);
    if (src0->ne[0] % 2 != 0 || src0->nb[0] != ts || ((uintptr_t) src0->data) % (2*ts) != 0) {
        return false;
    }
    for (int i = 1; i < GGML_MAX_DIMS; ++i) {
        if (src0->nb[i] % (2*ts) != 0) {
            return false;
        }
    }

    // src1: read as float2, so same requirements
    if (src1->ne[0] != src0->ne[0] || src1->nb[0] != sizeof(float) || ((uintptr_t) src1->data) % (2*sizeof(float)) != 0) {
        return false;
    }
    for (int i = 1; i < GGML_MAX_DIMS; ++i) {
        if (src1->nb[i] % (2*sizeof(float)) != 0) {
            return false;
        }
    }
    if (dst->nb[0] != sizeof(float)) {
        return false;
    }

    // broadcasting of src0 over dims 2/3 only; grid limits
    if (src1->ne[2] != dst->ne[2] || src1->ne[3] != dst->ne[3] ||
            dst->ne[2] % src0->ne[2] != 0 || dst->ne[3] % src0->ne[3] != 0 ||
            dst->ne[2] > 65535 || dst->ne[3] > 65535 || src0->ne[1] > INT32_MAX || src0->ne[0] > INT32_MAX) {
        return false;
    }

    // Batch-size heuristic from mainline's ggml_cuda_should_use_mmvf. ik has no MMF, so the F16/BF16 alternative
    // is cuBLAS, which MMVF beats up to MMVF_MAX_BATCH_SIZE.
    switch (type) {
        case GGML_TYPE_F32:
            if (GGML_CUDA_CC_IS_NVIDIA(cc)) {
                if (cc >= CC_AMPERE) {
                    return ne11 <= 3;
                }
                if (cc >= CC_TURING) {
                    return ne11 <= 4;
                }
                return ne11 <= 3;
            }
            return ne11 <= MMVF_MAX_BATCH_SIZE;
        case GGML_TYPE_F16:
        case GGML_TYPE_BF16:
            if (GGML_CUDA_CC_IS_NVIDIA(cc)) {
                const bool src0_small = src0->ne[1] <= 512 || src0->ne[2]*src0->ne[3] == 1;
                if (src0_small) {
                    return ne11 <= MMVF_MAX_BATCH_SIZE;
                }
                // large batched src0 (e.g. KQ/KQV without FA): mainline leaves these to MMF/cuBLAS on tensor-core GPUs
                return cc < CC_VOLTA && ne11 <= MMVF_MAX_BATCH_SIZE;
            }
            return ne11 <= MMVF_MAX_BATCH_SIZE;
        default:
            return false;
    }
}
