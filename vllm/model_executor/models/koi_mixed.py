# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Packed NVFP4 FFNs/MXFP8 head through built-in Marlin W4A16/W8A16."""

import torch

from .koi_mxfp8 import install_packed_weights as install_mxfp8
from .koi_packed import KoiNvFp4Method as KoiNvFp4Method
from .koi_packed import install_packed_linear


def install_packed_weights(model, vllm_config):
    recipe = model.config.koi_mixed
    if recipe.get("version") == 2:
        from .koi_quantization import install_packed_weights as install_searched

        return install_searched(model, vllm_config)
    full_body = recipe.get("recipe") == "body-mxfp8-ffn-nvfp4"
    expected = dict(
        version=1,
        recipe="body-mxfp8-ffn-nvfp4" if full_body else "ffn-nvfp4-head-mxfp8",
        activations="bfloat16",
        nvfp4_scale_policy="dynamic_max",
        execution="per_gemm_bf16_dequant",
    )
    if recipe != expected:
        raise ValueError("Unqualified mixed checkpoint recipe")
    pc = vllm_config.parallel_config
    if pc.tensor_parallel_size != 1 or pc.pipeline_parallel_size != 1:
        raise ValueError("Mixed Koi requires TP=PP=1")
    if vllm_config.model_config.dtype != torch.bfloat16:
        raise ValueError("Mixed Koi requires BF16 activations")
    if full_body:
        model.config.koi_mxfp8 = dict(
            version=1, block_size=32, activations="bfloat16", lm_head=True
        )
        install_mxfp8(model, vllm_config)
        del model.config.koi_mxfp8
    count = 0
    for i, layer in enumerate(model.model.layers):
        if model.config.layer_types[i] != "mlp":
            continue
        for role in ("up_proj", "down_proj"):
            module = getattr(layer.mixer, role)
            install_packed_linear(module, "NVFP4")
            count += 1
    assert count == 2 * model.config.layer_types.count("mlp")
    if full_body:
        return
    install_packed_linear(model.lm_head, "MXFP8")


def load_packed_weights(model, weights):
    if model.config.koi_mixed.get("version") == 2:
        from .koi_quantization import load_packed_weights as load_searched

        return load_searched(model, weights)
    params = dict(model.named_parameters())
    seen, loaded = set(), set()
    for name, value in weights:
        if name in seen:
            raise ValueError(f"Duplicate tensor {name}")
        seen.add(name)
        target, offset, fused = name, 0, False
        if name.startswith("model.layers.") and ".mixer." in name:
            stem, leaf = name.rsplit(".mixer.", 1)
            role, _, suffix = leaf.partition(".")
            idx = int(stem.split(".")[-1])
            mixer = model.model.layers[idx].mixer
            if model.config.layer_types[idx] == "linear_attention" and role in (
                "z_proj",
                "v_proj",
                "k_proj",
                "q_proj",
                "b_proj",
                "a_proj",
            ):
                m = mixer.num_householder
                sizes = [
                    mixer.value_dim,
                    m * mixer.value_dim,
                    m * mixer.key_dim,
                    mixer.key_dim,
                    m * mixer.num_v_heads,
                    mixer.num_v_heads,
                ]
                j = ["z_proj", "v_proj", "k_proj", "q_proj", "b_proj", "a_proj"].index(
                    role
                )
                target, offset, fused = (
                    f"{stem}.mixer.in_proj.{suffix}",
                    sum(sizes[:j]),
                    True,
                )
                assert value.shape[0] == sizes[j]
            elif model.config.layer_types[idx] == "full_attention" and role in (
                "q_proj",
                "k_proj",
                "v_proj",
            ):
                sizes = [mixer.q_size, mixer.kv_size, mixer.kv_size]
                j = ["q_proj", "k_proj", "v_proj"].index(role)
                target, offset, fused = (
                    f"{stem}.mixer.qkv_proj.{suffix}",
                    sum(sizes[:j]),
                    True,
                )
                assert value.shape[0] == sizes[j]
        p = params[target]
        dest = p.data[offset : offset + value.shape[0]] if fused else p.data
        if dest.shape != value.shape:
            raise ValueError(f"Shape mismatch {name}: {dest.shape} != {value.shape}")
        if p.dtype != value.dtype and not name.endswith(".A_log"):
            raise ValueError(f"Dtype mismatch {name}: {p.dtype} != {value.dtype}")
        dest.copy_(value)
        loaded.add(target)
    full_body = model.config.koi_mixed["recipe"] == "body-mxfp8-ffn-nvfp4"
    extra_per_gdp, extra_per_attention = (10, 4) if full_body else (5, 2)
    expected = len(params) + sum(
        extra_per_gdp
        if kind == "linear_attention"
        else extra_per_attention
        if kind == "full_attention"
        else 0
        for kind in model.config.layer_types
    )
    if set(params) != loaded or len(seen) != expected:
        raise ValueError(
            f"Incomplete checkpoint: {set(params) - loaded}, source count {len(seen)}"
        )
    # The model-level post-load hook builds the loop tables after quant finalize.
    return loaded
