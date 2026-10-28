//
// Copyright (C) 2024 Iwan Kawrakow
// MIT license
// SPDX-License-Identifier: MIT
//

#include "iqk_mmvq.cuh"
#include "iqk_mmvq_templates.cuh"
#include "devquants.cuh"

void iqk_mul_mat_vec_q(ggml_type type, const mmvq_args & args, cudaStream_t stream) {
    switch (type) {
#if GGML_CUDA_DEVQ_ON(IQ1_BN)
        case GGML_TYPE_IQ1_BN:
            mul_mat_vec_iq1_bn_q8_1_cuda(args, stream);
            break;
#endif
#if GGML_CUDA_DEVQ_ON(IQ2_BN)
        case GGML_TYPE_IQ2_BN:
            mul_mat_vec_iq2_bn_q8_1_cuda(args, stream);
            break;
#endif
#if GGML_CUDA_DEVQ_ON(PQ2_0)
        case GGML_TYPE_PQ2_0:
            mul_mat_vec_pq2_0_q8_1_cuda(args, stream);
            break;
#endif
#if GGML_CUDA_DEVQ_ON(IQ2_K)
        case GGML_TYPE_IQ2_K:
            mul_mat_vec_iq2_k_q8_1_cuda(args, stream);
            break;
#endif
#if GGML_CUDA_DEVQ_ON(IQ3_K)
        case GGML_TYPE_IQ3_K:
            mul_mat_vec_iq3_k_q8_1_cuda(args, stream);
            break;
#endif
#if GGML_CUDA_DEVQ_ON(IQ2_KL)
        case GGML_TYPE_IQ2_KL:
            mul_mat_vec_iq2_kl_q8_1_cuda(args, stream);
            break;
#endif
#if GGML_CUDA_DEVQ_ON(IQ3_KS)
        case GGML_TYPE_IQ3_KS:
            mul_mat_vec_iq3_ks_q8_1_cuda(args, stream);
            break;
#endif
#if GGML_CUDA_DEVQ_ON(IQ4_K)
        case GGML_TYPE_IQ4_K:
            mul_mat_vec_iq4_k_q8_1_cuda(args, stream);
            break;
#endif
#if GGML_CUDA_DEVQ_ON(IQ4_KS)
        case GGML_TYPE_IQ4_KS:
            mul_mat_vec_iq4_ks_q8_1_cuda(args, stream);
            break;
#endif
#if GGML_CUDA_DEVQ_ON(IQ4_KSS)
        case GGML_TYPE_IQ4_KSS:
            mul_mat_vec_iq4_kss_q8_1_cuda(args, stream);
            break;
#endif
#if GGML_CUDA_DEVQ_ON(IQ1_KT)
        case GGML_TYPE_IQ1_KT:
            mul_mat_vec_iq1_kt_q8_1_cuda(args, stream);
            break;
#endif
#if GGML_CUDA_DEVQ_ON(IQ2_KT)
        case GGML_TYPE_IQ2_KT:
            mul_mat_vec_iq2_kt_q8_1_cuda(args, stream);
            break;
#endif
#if GGML_CUDA_DEVQ_ON(IQ3_KT)
        case GGML_TYPE_IQ3_KT:
            mul_mat_vec_iq3_kt_q8_1_cuda(args, stream);
            break;
#endif
#if GGML_CUDA_DEVQ_ON(IQ4_KT)
        case GGML_TYPE_IQ4_KT:
            mul_mat_vec_iq4_kt_q8_1_cuda(args, stream);
            break;
#endif
#if GGML_CUDA_DEVQ_ON(IQ2_KS)
        case GGML_TYPE_IQ2_KS:
            mul_mat_vec_iq2_ks_q8_1_cuda(args, stream);
            break;
#endif
#if GGML_CUDA_DEVQ_ON(IQ5_K)
        case GGML_TYPE_IQ5_K:
            mul_mat_vec_iq5_k_q8_1_cuda(args, stream);
            break;
#endif
#if GGML_CUDA_DEVQ_ON(IQ5_KS)
        case GGML_TYPE_IQ5_KS:
            mul_mat_vec_iq5_ks_q8_1_cuda(args, stream);
            break;
#endif
#if GGML_CUDA_DEVQ_ON(IQ6_K)
        case GGML_TYPE_IQ6_K:
            mul_mat_vec_iq6_k_q8_1_cuda(args, stream);
            break;
#endif
#if GGML_CUDA_DEVQ_ON(IQ2_K_R4)
        case GGML_TYPE_IQ2_K_R4:
            mul_mat_vec_iq2_k_r4_q8_1_cuda(args, stream);
            break;
#endif
#if GGML_CUDA_DEVQ_ON(IQ3_K_R4)
        case GGML_TYPE_IQ3_K_R4:
            mul_mat_vec_iq3_k_r4_q8_1_cuda(args, stream);
            break;
#endif
#if GGML_CUDA_DEVQ_ON(IQ4_K_R4)
        case GGML_TYPE_IQ4_K_R4:
            mul_mat_vec_iq4_k_r4_q8_1_cuda(args, stream);
            break;
#endif
#if GGML_CUDA_DEVQ_ON(IQ4_KS_R4)
        case GGML_TYPE_IQ4_KS_R4:
            mul_mat_vec_iq4_ks_r4_q8_1_cuda(args, stream);
            break;
#endif
#if GGML_CUDA_DEVQ_ON(IQ5_K_R4)
        case GGML_TYPE_IQ5_K_R4:
            mul_mat_vec_iq5_k_r4_q8_1_cuda(args, stream);
            break;
#endif
#if GGML_CUDA_DEVQ_ON(IQ5_KS_R4)
        case GGML_TYPE_IQ5_KS_R4:
            mul_mat_vec_iq5_ks_r4_q8_1_cuda(args, stream);
            break;
#endif
#if GGML_CUDA_DEVQ_ON(IQ1_S_R4)
        case GGML_TYPE_IQ1_S_R4:
            mul_mat_vec_iq1_s_r4_q8_1_cuda(args, stream);
            break;
#endif
#if GGML_CUDA_DEVQ_ON(IQ1_M_R4)
        case GGML_TYPE_IQ1_M_R4:
            mul_mat_vec_iq1_m_r4_q8_1_cuda(args, stream);
            break;
#endif
#if GGML_CUDA_DEVQ_ON(IQ3_KS_R16)
        case GGML_TYPE_IQ3_KS_R16:
            mul_mat_vec_iq3ks_r16_q8_1_cuda(args, stream);
            break;
#endif
#if GGML_CUDA_DEVQ_ON(IQ4_KS_R16)
        case GGML_TYPE_IQ4_KS_R16:
            mul_mat_vec_iq4_ks_r16_q8_1_cuda(args, stream);
            break;
#endif
#if GGML_CUDA_DEVQ_ON(IQ5_KS_R16)
        case GGML_TYPE_IQ5_KS_R16:
            mul_mat_vec_iq5_ks_r16_q8_1_cuda(args, stream);
            break;
#endif
        default:
            ggml_cuda_devq_check_type(type);
            GGML_ABORT("fatal error");
            break;
    }
}
