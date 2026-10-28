#pragma once

#include "common.cuh"

// fused RDA decode step (GGML_OP_RDA_DECODE)
bool ggml_cuda_rda_decode_supported(const ggml_tensor * dst);
void ggml_cuda_op_rda_decode(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
