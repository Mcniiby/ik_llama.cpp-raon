# ik_llama.cpp for Raon-1 mini

A fork of [ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp) that runs KRAFTON's
[Raon-1 mini preview](https://huggingface.co/KRAFTON/Raon-1-mini-preview) (28.8B total / 4.3B active MoE) on a single
16 GB GPU, at about 2.3x the decode speed of a plain port.

| RTX 5070 Ti, 16 GB | tok/s |
|---|---|
| Plain port | ~97 |
| **This fork, decode** | **~220-230** |
| This fork, decode at 16K context | ~190 |

The matching GGUF (3-bit experts, 13.8 GB, fits 128K context in 16 GB):
[McNiiby/Raon-1-mini-preview-IQ3_KS-GGUF](https://huggingface.co/McNiiby/Raon-1-mini-preview-IQ3_KS-GGUF).
It only runs on this fork.

## Quick start

Linux, CUDA 12.8. Use `-DCMAKE_CUDA_ARCHITECTURES=86` for RTX 30, `89` for RTX 40, `120` for RTX 50.

```
cmake -B build -G Ninja -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120 -DCMAKE_BUILD_TYPE=Release \
  -DLLAMA_CURL=OFF -DGGML_IQK_FA_ALL_QUANTS=OFF -DGGML_CUDA_FA_ALL_QUANTS=OFF -DGGML_CUDA_DEV_QUANTS=ON
cmake --build build -j --target llama-server llama-bench

build/bin/llama-server -m raon-1-mini-preview-IQ3_KS.gguf --alias raon \
  -ngl 99 -fa on --ctx-size 131072 -ub 256 -ctk q8_0 -ctv q8_0 \
  --jinja --reasoning-format deepseek --reasoning on \
  --chat-template-kwargs '{"truncate_history_thinking": false}' \
  --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0 --presence-penalty 1.5
```

Add `--spec-type ngram-simple` for lossless n-gram speculation (about +6% in chat, more when the reply copies the
prompt).

## Good to know

- **GPUs:** tested on an RTX 5070 Ti. RTX 30/40 build and fall back to a ~5% slower MoE kernel, but haven't been
  tested on real cards; reports welcome. The GGUF needs 16 GB of VRAM.
- **English-focused GGUF:** Korean/Chinese/Japanese-only tokens were trimmed from the output layer. Input in any
  language works; CJK output is slower and sometimes worded differently.
- Upstream ik_llama.cpp has its own Raon support since #2759. This is a separate port with the speed work below.

## More

- [docs/raon/README.md](docs/raon/README.md): every change and its speedup, runtime switches for A/B testing,
  how the numbers were measured, known issues and credits.
- [docs/README-ik_llama.cpp.md](docs/README-ik_llama.cpp.md): upstream's README.

Written with Claude Code under my direction; every change was checked against the original path for quality (KLD,
greedy text) and speed before merging. MIT, same as ik_llama.cpp.
