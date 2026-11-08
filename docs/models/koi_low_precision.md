# Koi weight-only MXFP8 and NVFP4

Koi accepts packed MXFP8 checkpoints and mixed BF16/MXFP8/NVFP4 checkpoints.
The adapters use vLLM's existing **Marlin W8A16 and W4A16 kernels**. Activations
remain BF16. Loading repacks weights for Marlin; serving does not retain full
BF16 copies of quantized matrices or reconstruct them before each GEMM.

## Supported configurations

| Checkpoint | Body | LM head | Embedding |
| --- | --- | --- | --- |
| BF16 (unchanged) | BF16 | BF16 | BF16 |
| MXFP8 body | MXFP8 matrices | BF16 | BF16 |
| MXFP8 body and head | MXFP8 matrices | MXFP8 | BF16 |
| Mixed FFNs/head | NVFP4 FFNs, BF16 elsewhere | MXFP8 | BF16 |
| Mixed body/FFNs/head | NVFP4 FFNs, MXFP8 other matrices | MXFP8 | BF16 |
| Per-matrix allocation | BF16, MXFP8 or NVFP4 per matrix | From allocation | BF16 |

Vectors and convolution weights remain unquantized. Per-matrix allocations
split fused GDP and QKV projections so each original matrix keeps its selected
format and independent NVFP4 tensor scale. Loop count and recurrent cache mode
retain their normal Koi meanings.

The packed paths currently require an NVIDIA GPU supported by Marlin,
`--dtype bfloat16`, and tensor/pipeline parallel sizes of **1**. These paths do
not enable activation quantization or provide CPU, ROCm or Metal support.

## Run a packed checkpoint

```bash
vllm serve /path/to/packed-koi \
  --dtype bfloat16 \
  --tensor-parallel-size 1 \
  --pipeline-parallel-size 1 \
  --hf-overrides '{"n_recurrent_loops":4,"recurrent_kv_cache_mode":"per_loop"}'
```

The checkpoint includes tokenizer assets, its Koi `config.json`, and packed
safetensors. The loader selects the adapter from the configuration markers
below; no `--quantization` override is needed. Adding a marker to a BF16
checkpoint does **not** quantize it. The packed values and scales must already
exist. This change adds serving support, not a PTQ/QAD exporter.

## Checkpoint contract

MXFP8 checkpoints set this field in `config.json`:

```json
{"koi_mxfp8":{"version":1,"block_size":32,"activations":"bfloat16","lm_head":true}}
```

Set `lm_head` to `false` to keep the head BF16. Each quantized linear stores
`weight` as E4M3 `[N,K]` and `weight_scale` as E8M0 bytes `[N,K/32]`.

Mixed checkpoints set `koi_mixed` instead:

```json
{"koi_mixed":{"version":1,"recipe":"body-mxfp8-ffn-nvfp4","activations":"bfloat16","nvfp4_scale_policy":"dynamic_max","execution":"per_gemm_bf16_dequant"}}
```

The other version-1 recipe is `ffn-nvfp4-head-mxfp8`. The `execution` string is
retained for compatibility with existing exported checkpoints; it describes
the old export schema, **not** the current execution backend. Both recipes use
native Marlin. NVFP4 tensors store packed E2M1 `weight` bytes `[N,K/2]`, E4M3
`weight_scale` `[N,K/16]`, and scalar FP32 `weight_scale_2`. Low nibbles precede
high nibbles. The effective scale is the block scale times the tensor scale.

Per-matrix checkpoints use the existing version-2 export schema:

```json
{"koi_mixed":{"version":2,"recipe":"searched-3gb","allocation":"quantization_allocation.json","activations":"bfloat16","execution":"requires_per_matrix_serving_qualification"}}
```

The allocation file contains a `formats` object mapping **every original HF
parameter name** to `BF16`, `MXFP8` or `NVFP4`. Only matrices may be quantized;
`model.embed_tokens.weight` must be BF16. The recipe name is a compatibility
identifier, not an enforced model size. Keep the allocation alongside the
checkpoint. GDP source projections use `z_proj`, `v_proj`, `k_proj`, `q_proj`,
`b_proj`, `a_proj`; attention uses `q_proj`, `k_proj`, `v_proj`. The loader checks
coverage, shapes, dtypes and duplicate source tensors, and rejects incomplete
checkpoints instead of leaving parameters uninitialized.
