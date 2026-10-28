#include "common.cuh"

// GGML_OP_LOGITS_TOPM: exact top-M of each logits row (see ggml_logits_topm in ggml.h)
void ggml_cuda_op_logits_topm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// launch geometry used by ggml_cuda_op_logits_topm; false if the row/M combination is not supported
bool ggml_cuda_logits_topm_geometry(int64_t n, int m, int * n_warps, int * n_blocks);
