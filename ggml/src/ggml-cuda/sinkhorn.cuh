#include "common.cuh"

void ggml_cuda_op_sinkhorn(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_hc_pre(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_hc_post(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// ADD (same-shape contiguous F32) fused into the HC_POST that consumes it; the ADD output is still written
void ggml_cuda_op_add_hc_post(ggml_backend_cuda_context & ctx, ggml_tensor * add, ggml_tensor * hc_post);

void ggml_cuda_op_raon_hc_pre(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
