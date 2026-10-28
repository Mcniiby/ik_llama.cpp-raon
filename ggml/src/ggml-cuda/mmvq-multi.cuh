#pragma once
// Decode quantized matvec: several weight matrices sharing one q8_1 src1 in one launch, plus tuned thread layouts.
//
//   GGML_CUDA_MMVQ_MULTI=0  turns the multi-matrix launch off (one mmvq launch per MUL_MAT)
//   GGML_CUDA_MMVQ_TUNE=0   keeps the stock thread layout (4 warps, 1 row per block) for every shape
//   GGML_CUDA_MMVQ_CFG=<n>  forces layout n for every shape (tuning only, see mmvq_multi_cfg_rows)
// All read once.

#include "common.cuh"

#define MMVQ_MULTI_MAX 8
#define MMVQ_MULTI_MAX_COLS 8

struct mmvq_multi_set {
    const void * vx [MMVQ_MULTI_MAX];
    float      * dst[MMVQ_MULTI_MAX];
    int          nrows  [MMVQ_MULTI_MAX];
    int          blk_end[MMVQ_MULTI_MAX];   // cumulative CUDA block counts, filled by ggml_cuda_mmvq_multi
    int          n;
    int          ncols;          // src1 columns (1..MMVQ_MULTI_MAX_COLS; 0 = 1): y column j at j*y_col_blocks q8_1 blocks,
    int          y_col_blocks;   // dst column j at j*nrows floats of each matrix
    // channel strides in bytes (blockIdx.y), only with n == 1
    uint64_t     nb02, nb12, nb2;
};

bool ggml_cuda_mmvq_multi_enabled();      // GGML_CUDA_MMVQ_MULTI
bool ggml_cuda_mmvq_tune_enabled();       // GGML_CUDA_MMVQ_TUNE or GGML_CUDA_MMVQ_CFG
bool ggml_cuda_mmvq_multi_type_supported(ggml_type type);

// vy: src1 quantized to q8_1 (one column per channel), ncols_x = ne00. s.n matrices of the same type and ne00,
// contiguous rows. ne2 channels (blockIdx.y) only for s.n == 1.
void ggml_cuda_mmvq_multi(ggml_type type, mmvq_multi_set & s, const void * vy, int ncols_x, int ne2, cudaStream_t stream);
