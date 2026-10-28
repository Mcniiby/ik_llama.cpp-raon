# ik_llama.cpp for Raon-1 mini (fast single-GPU decode)

A fork of [ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp) that runs KRAFTON's
[Raon-1 mini preview](https://huggingface.co/KRAFTON/Raon-1-mini-preview) (28.8B total / 4.3B active MoE, hybrid
delta-attention + MLA, 4-stream mHC residual) on a single 16 GB GPU, with CUDA decode work that makes it about 2.3x
faster at single-stream decode than a plain port.

| RTX 5070 Ti (16 GB), IQ3_KS GGUF, q8_0 KV, FA on | tok/s |
|---|---|
| Plain port (unfused SiTU / mHC, no CUDA graphs) | ~97 |
| **This fork, tg128** | **~220-230** |
| This fork, decode at 16K context | ~190 |
| This fork, prompt processing pp4096 (`-ub 2048` / `-ub 512`) | ~4,900 / ~3,500 |

Decode is launch-bound enough that absolute numbers drift ~3% between sessions with background CPU load; compare
within one run. At batch 1 a token reads ~2.6 GB of weights, so the card's practical ceiling is ~290 tok/s.

Upstream ik_llama.cpp has its own Raon support since #2759. This fork is a separate, earlier port with the speed work
below; it isn't a drop-in replacement for upstream.

## Supported model and hardware

- **Model:** Raon-1 mini preview (`general.architecture = raon1`). The matching GGUF is
  `raon-1-mini-preview-IQ3_KS.gguf` ([McNiiby/Raon-1-mini-preview-IQ3_KS-GGUF](https://huggingface.co/McNiiby/Raon-1-mini-preview-IQ3_KS-GGUF)); KRAFTON's Q8_0/Q4_K_M files load too. IQ3_KS / IQ4_KS_R16 are
  ik-only quant types, and the trimmed output head (`raon1.output_token_ids`) needs this fork.
- **Tested:** RTX 5070 Ti (sm_120), CUDA 12.8, Linux.
- **RTX 30/40 (sm_86/89):** compile and fall back to the `staged` megakernel (the default `pipe` variant needs sm_90+).
  Not tested on real RTX 30/40 cards; reports welcome. `GGML_CUDA_MMVQ_TUNE` layouts are tuned for sm_120.
- **CPU:** the port and every new op have CPU implementations (the CPU is the reference for the GPU tests).

## Build

```
cmake -B build -G Ninja -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120 -DCMAKE_BUILD_TYPE=Release \
  -DLLAMA_CURL=OFF -DGGML_IQK_FA_ALL_QUANTS=OFF -DGGML_CUDA_FA_ALL_QUANTS=OFF -DGGML_CUDA_DEV_QUANTS=ON
cmake --build build -j --target llama-server llama-bench llama-perplexity llama-cli test-backend-ops
```

Use `-DCMAKE_CUDA_ARCHITECTURES=86` or `89` for RTX 30/40. `GGML_CUDA_DEV_QUANTS=ON` only compiles CUDA kernels for
F32 F16 BF16 Q8_0 Q6_K Q5_K Q4_K Q5_0 IQ4_NL IQ4_XS IQ3_S IQ3_KS IQ4_KS IQ4_KS_R16 weights and F16/Q8_0 KV (any other
type aborts with a message), which cuts build time a lot. Leave it OFF for a general-purpose build.

## Run

Chat server (128K context fits a 16 GB card with ~1.2 GB to spare; KRAFTON's sampling settings):

```
build/bin/llama-server -m raon-1-mini-preview-IQ3_KS.gguf --alias raon \
  -ngl 99 -fa on --ctx-size 131072 --parallel 1 -ub 256 -ctk q8_0 -ctv q8_0 \
  --jinja --reasoning-format deepseek --reasoning on \
  --chat-template-kwargs '{"truncate_history_thinking": false}' \
  --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0 --presence-penalty 1.5
```

- Raon-1 is meant to run with thinking on (see KRAFTON's card).
- `--spec-type ngram-simple` (lossless n-gram speculation) adds ~6% in chat and 12-22% when the reply copies from
  the prompt (summaries, code edits).
- `--parallel N` works. Larger `-ub` speeds up prefill but costs VRAM.

Benchmark:

```
build/bin/llama-bench -m raon-1-mini-preview-IQ3_KS.gguf -ngl 99 -fa 1 -ctk q8_0 -ctv q8_0 -ub 256 -p 0 -n 128 -r 3
```

## What's in it

| Change | Speedup when added |
|---|---|
| SiTU in ik's fused MoE path + fused mHC pre-mix (CUDA graphs back on; ~1,500 -> ~150 launches per token) | +31% |
| Small F16/F32 matvec kernel (ported from mainline) instead of cuBLAS | +12% |
| Fused recurrent-layer (RDA) decode op | +9% |
| Single-token MoE megakernel (router + 6 routed + shared expert per layer, one persistent launch) | +7% |
| Merged dense matvecs sharing an input, per-shape tuning, small fusions | +13% |
| Pipelined megakernel (TMA ring, next expert loads while the current one computes) | +5% |
| One-kernel mHC sublayer boundary | +3.1% |
| Small-kernel fixes (layer-0 FFN, recurrent-layer matvecs) | +1.2% |
| Flash-attention decode split over all SMs past 4K tokens | +2.3-3% at 16K |
| Barrier-free KDA scan for prompt processing | +35% on a 4K prompt |
| GPU top-M sampling candidates, exact fast top-k sampler | ~15% in real chat |

Each row was measured with the rows above it already on, in one binary with the change's switch on vs off. Every
change was also checked for quality the same way: same-binary KLD 0.003-0.005 (the rounding-order floor is 0.0048,
the same build at `-ub 4` vs `-ub 8`), with identical greedy text or a divergence only at measured near-tie tokens.
Port correctness vs KRAFTON's reference implementation: identical tokenizer ids and greedy tokens, KLD 0.010
(reference FA on vs off: 0.0087).

Also included: grouped 2-8 token MoE matvec and fused RDA for speculative-decoding verify batches (verifying 2/4/8
tokens costs 1.3/1.7/2.2x one token), DFlash2 speculative decoding (works, but at this size it is slower than plain
decode on a 16 GB card), multi-slot fixes, mainline's MMQ stream-k (#22298) and CUDA-graph re-arm (#19754).

## Runtime switches

Every speed patch has an environment switch, so A/B runs use one binary. All are read once at startup, default on;
`=0` restores the original path unless noted.

| Variable | Meaning |
|---|---|
| `RAON_MHC_FUSED` | Fused mHC pre-mix op; 2 = fused but rms_norm kept separate |
| `RAON_RDA_FUSED` | Fused RDA decode op for <= 4 tokens |
| `RAON_KDA_FAST` | Barrier-free KDA recurrence kernel for prefill |
| `RAON_MK_MOE` | Single-token MoE block megakernel |
| `RAON_MK_MOE_VARIANT` | `pipe` (default, sm_90+), `staged` (sm_80+), `direct` |
| `RAON_MK_SAFE_ABORT` | Megakernel abort path that can't hang the GPU on a bad launch |
| `RAON_HC_BOUNDARY`, `RAON_HCB1` | One-kernel mHC sublayer boundary (general / single-token version) |
| `RAON_HC_COLLAPSE` | Final stream collapse + output norm in one kernel |
| `RAON_F16_MV` | Batched small F16 matvecs (`RAON_F16_MV_U` = unroll, default 10) |
| `RAON_FB_FOLD`, `RAON_IQ3KS_MV` | Small-kernel fixes for the recurrent layers and layer-0 FFN |
| `RAON_MLA_FUSE`, `RAON_MLA_PROJ_FIRST` | MLA decode fusions; projections emitted together |
| `RAON_FA_NSM_FULL` | Flash-attention decode split over all SMs at long context |
| `RAON_MOEG_TPB` | Grouped-expert kernel threads per block (default 4) |
| `RAON_MMVQ_NC_TUNE` | Tuned layouts for multi-column matvecs |
| `RAON_CONV_MERGE` | Load the RDA conv q/k/v weights as one tensor |
| `GGML_CUDA_NO_MMVF` | `=1` turns off the small-batch F16/F32 GEMV and goes back to cuBLAS |
| `GGML_CUDA_MMQ_V2` | MMQ stream-k scheduling from mainline |
| `GGML_CUDA_MMVQ_MULTI`, `GGML_CUDA_MMVQ_TUNE` | One launch for matvecs sharing an input; per-shape layouts |
| `GGML_CUDA_MOE_GROUPED` | Grouped 2-8 token MoE matvec |
| `GGML_CUDA_ROUTER_NORM_FUSE`, `GGML_CUDA_ADD_HC_POST_FUSE` | Router norm into top-k; MoE combine into HC_POST |
| `GGML_CUDA_GRAPH_REARM` | Re-arm CUDA graphs after property changes instead of disabling them |
| `LLAMA_SAMPLER_FAST_TOPK`, `LLAMA_GSAMPLE` | Exact fast top-k sampler; GPU top-M candidates (`LLAMA_GSAMPLE_M`, default 64) |

Experimental megakernel variants, off by default because they measured no gain on the 5070 Ti, are kept for anyone
who wants to build on them: `RAON_MK_SHQ` (shared expert split into smaller tasks), `RAON_MK_DNSPLIT` (down projection
split into two task groups), `RAON_MK_RTL2` (router weights pinned in L2).

Debug and test-only switches (`RAON_NAN_CHECK`, `RAON_DUMP_NODES`, `RAON_MK_MOE_DEBUG`, `RAON_MK_MOE_LAYERS`,
`RAON_HCB_DEBUG`, `GGML_CUDA_GRAPH_STATS`, `GGML_CUDA_POOL_LOG`, `GGML_CUDA_DEVQ_SKIP`, `RAON_MMQ_TESTS`,
`RAON_MMVQ_TESTS`, ...) are off by default. Op tests: `test-backend-ops -o RAON_HC_PRE | RDA_DECODE | DELTA_NET |
RAON_MOE_BLOCK | RAON_SMALLFUSE | LOGITS_TOPM | MUL_MAT_ID | MOE_FUSED_UP_GATE`; CPU checks: `test-rda-fused`,
`test-raon-moe-block`, `test-gsample`.

## The GGUF

`raon-1-mini-preview-IQ3_KS.gguf` (12.9 GiB): routed gate/up IQ3_KS, routed down IQ4_KS_R16, shared expert and
attention Q8_0/Q6_K, recurrent-layer parameters F16, token embedding Q4_K, Q6_K output head trimmed from 259,840 to
206,168 rows (Korean/Chinese/Japanese-only tokens dropped; tokenizer and embeddings stay full, dropped tokens get
-inf). KLD vs KRAFTON's Q4_K_M: 0.093, same top token 87%. The importance matrix and vocab trim were calibrated on
public text plus private chat data that isn't released. `scripts/raon/` has the trim and imatrix-conversion scripts.

## Known issues

- Presence penalty in ik applies to the last 64 tokens including the prompt; KRAFTON's card applies it to all
  generated tokens. That may matter for long thinking traces.
- Speculative verify batches (2-8 tokens) don't get the single-token megakernel, so DFlash2 doesn't pay off yet.
- Multi-slot greedy text isn't bit-repeatable (batch composition depends on timing); single-stream is deterministic.
- Split mode `graph` is not supported for RDA layers; `-muge` (merged up/gate) is not supported for Raon.
- New ops change `GGML_OP_COUNT` and the RPC protocol version, so RPC peers must run this fork.

## Credits

- [ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp) by Iwan Kawrakow and contributors (MIT): the base and the
  KDA/DELTA_NET, MLA, mHC and IQK quant kernels this builds on.
- [llama.cpp](https://github.com/ggml-org/llama.cpp) / ggml (MIT): mmvf, MMQ stream-k (#22298) and CUDA-graph re-arm
  (#19754) are ported from mainline.
- ik_llama.cpp PR #2733: the DFlash2 routing for Raon was adapted from it.
- [KRAFTON](https://huggingface.co/KRAFTON) for Raon-1 mini (Apache-2.0), its DFlash2 drafter, and the reference
  llama.cpp implementation used to validate this port.
- Strata (MIT): fused-kernel and grouped-expert designs (k_moe_out_small, native_expert_grouped / EXPDEC, split
  top-k, k_conv_seq / k_gdn_seq). DeepSeek's Mega-mHC layout (arXiv 2609.19969) informed the mHC fusion.

## How this was built

Written with Claude Code (Anthropic's Claude) under my direction. Each change was validated before merging: CUDA vs
CPU op tests, a same-binary KLD against the switch off, greedy-text parity, and interleaved llama-bench rounds. Bugs
and judgement calls are mine to own; please open an issue.

License: MIT (same as ik_llama.cpp).
