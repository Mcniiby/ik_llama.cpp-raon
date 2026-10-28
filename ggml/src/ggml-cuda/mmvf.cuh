#pragma once

#include "common.cuh"

#define MMVF_MAX_BATCH_SIZE 8 // Max. batch size for which to use MMVF kernels.

// F32/F16/BF16 weights x F32 activations, ne11 <= MMVF_MAX_BATCH_SIZE (port of mainline llama.cpp mmvf, plain MUL_MAT only).
// Always accumulates in F32, regardless of dst precision.
void ggml_cuda_mul_mat_vec_f(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);

// Type/shape/stride/alignment checks plus the batch-size heuristic. Honours GGML_CUDA_NO_MMVF=1 (always false).
bool ggml_cuda_should_use_mmvf(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst, int cc);
