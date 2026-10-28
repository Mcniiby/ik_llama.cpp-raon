#pragma once
#include "common.cuh"

// GGML_OP_RAON_MOE_BLOCK (Raon-1 mini single-token MoE layer, persistent megakernel)
bool ggml_cuda_raon_moe_block_supported(const ggml_tensor * op);
void ggml_cuda_op_raon_moe_block(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
