# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-source-matrix Koi allocation through native Marlin W4A16/W8A16.

Fused GDP/QKV projections are split so each original matrix retains its own
format and NVFP4 global scale. No persistent BF16 reconstruction is created.
"""

import json
from pathlib import Path

import torch
from torch import nn

from vllm.model_executor.layers.linear import LinearBase
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead

from .koi_mxfp8 import KoiMxFp8Method, KoiMxFp8TorchLinear


class SplitProjection(nn.Module):
    def __init__(self, roles, sizes, k, device):
        super().__init__()
        self.parts = nn.ModuleDict(
            {
                role: nn.Linear(k, n, bias=False, device=device, dtype=torch.bfloat16)
                for role, n in zip(roles, sizes)
            }
        )

    def forward(self, x):
        return torch.cat([part(x) for part in self.parts.values()], dim=-1), None


def mapped_name(model, name):
    if name.startswith("model.layers.") and ".mixer." in name:
        stem, leaf = name.rsplit(".mixer.", 1)
        role, _, suffix = leaf.partition(".")
        kind = model.config.layer_types[int(stem.split(".")[-1])]
        if kind == "linear_attention" and role in (
            "z_proj",
            "v_proj",
            "k_proj",
            "q_proj",
            "b_proj",
            "a_proj",
        ):
            return f"{stem}.mixer.in_proj.parts.{role}.{suffix}"
        if kind == "full_attention" and role in ("q_proj", "k_proj", "v_proj"):
            return f"{stem}.mixer.qkv_proj.parts.{role}.{suffix}"
    return name


def install_packed_weights(model, vllm_config):
    from .koi_mixed import KoiNvFp4Method

    marker = model.config.koi_mixed
    if marker != dict(
        version=2,
        recipe="searched-3gb",
        allocation="quantization_allocation.json",
        activations="bfloat16",
        execution="requires_per_matrix_serving_qualification",
    ):
        raise ValueError("Unsupported searched Koi schema")
    pc = vllm_config.parallel_config
    if pc.tensor_parallel_size != 1 or pc.pipeline_parallel_size != 1:
        raise ValueError("Searched Koi requires TP=PP=1")
    if vllm_config.model_config.dtype != torch.bfloat16:
        raise ValueError("Searched Koi requires BF16 activations")
    allocation = json.loads(
        (Path(vllm_config.model_config.model) / marker["allocation"]).read_text()
    )
    formats = allocation["formats"]
    if set(formats.values()) - {"NVFP4", "MXFP8", "BF16"}:
        raise ValueError("Unsupported per-matrix precision")
    if formats.get("model.embed_tokens.weight") != "BF16":
        raise ValueError("Koi embeddings must remain BF16")
    model._searched_formats = formats
    for i, layer in enumerate(model.model.layers):
        mixer = layer.mixer
        kind = model.config.layer_types[i]
        if kind == "linear_attention":
            original = mixer.in_proj
            m = mixer.num_householder
            roles = ["z_proj", "v_proj", "k_proj", "q_proj", "b_proj", "a_proj"]
            sizes = [
                mixer.value_dim,
                m * mixer.value_dim,
                m * mixer.key_dim,
                mixer.key_dim,
                m * mixer.num_v_heads,
                mixer.num_v_heads,
            ]
            mixer.in_proj = SplitProjection(
                roles, sizes, original.weight.shape[1], original.weight.device
            )
        elif kind == "full_attention":
            original = mixer.qkv_proj
            mixer.qkv_proj = SplitProjection(
                ["q_proj", "k_proj", "v_proj"],
                [mixer.q_size, mixer.kv_size, mixer.kv_size],
                original.weight.shape[1],
                original.weight.device,
            )
    modules = dict(model.named_modules())
    params = dict(model.named_parameters())
    expected = {mapped_name(model, name) for name in formats}
    if expected != set(params):
        raise ValueError(
            f"Parameter mapping mismatch: missing {expected - set(params)}, "
            f"extra {set(params) - expected}"
        )
    for name, fmt in formats.items():
        if fmt == "BF16":
            continue
        target = mapped_name(model, name)
        if not name.endswith(".weight") or params[target].ndim != 2:
            raise ValueError(f"Only matrix weights may be quantized: {name}")
        module = modules[target.rsplit(".", 1)[0]]
        w = module.weight
        if not isinstance(module, (nn.Linear, LinearBase, ParallelLMHead)):
            raise TypeError(f"Unsupported matrix module {name}: {type(module)}")
        n, k = w.shape
        w.requires_grad_(False)
        if fmt == "MXFP8":
            w.data = torch.empty(n, k, dtype=torch.float8_e4m3fn, device=w.device)
            scale = torch.empty(n, k // 32, dtype=torch.uint8, device=w.device)
            module.quant_method = KoiMxFp8Method((n, k))
        else:
            w.data = torch.empty(n, k // 2, dtype=torch.uint8, device=w.device)
            scale = torch.empty(n, k // 16, dtype=torch.float8_e4m3fn, device=w.device)
            module.register_parameter(
                "weight_scale_2",
                nn.Parameter(
                    torch.empty((), device=w.device, dtype=torch.float32),
                    requires_grad=False,
                ),
            )
            module.quant_method = KoiNvFp4Method()
        module.register_parameter(
            "weight_scale", nn.Parameter(scale, requires_grad=False)
        )
        if isinstance(module, nn.Linear):
            module.__class__ = KoiMxFp8TorchLinear


def load_packed_weights(model, weights):
    params = dict(model.named_parameters())
    seen = set()
    loaded = set()
    for name, value in weights:
        if name in seen:
            raise ValueError(f"Duplicate source tensor {name}")
        seen.add(name)
        target = mapped_name(model, name)
        p = params[target]
        if value.shape != p.shape or (
            value.dtype != p.dtype and not name.endswith(".A_log")
        ):
            raise ValueError(f"Packed shape/dtype mismatch: {name}")
        p.data.copy_(value)
        loaded.add(target)
    expected = sum(
        {"BF16": 1, "MXFP8": 2, "NVFP4": 3}[fmt]
        for fmt in model._searched_formats.values()
    )
    if loaded != set(params) or len(seen) != expected:
        raise ValueError(
            f"Incomplete searched checkpoint: missing {set(params) - loaded}, "
            f"count {len(seen)}"
        )
    return loaded
