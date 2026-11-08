# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-source-matrix Koi allocation through native Marlin W4A16/W8A16.

Fused GDP/QKV projections are split so each original matrix retains its own
format and NVFP4 global scale. No persistent BF16 reconstruction is created.
"""

import torch
from torch import nn

from vllm.model_executor.layers.linear import LinearBase
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.transformers_utils.repo_utils import get_hf_file_to_dict

from .koi_packed import PACKED_FORMATS, install_packed_linear


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


def resolve_formats(allocation, parameter_names, matrix_names):
    default = allocation.get("default", "BF16")
    overrides = allocation.get("formats", {})
    supported = {"BF16", *PACKED_FORMATS}
    if not isinstance(overrides, dict):
        raise ValueError("Koi formats must map parameter names to precisions")
    if default not in supported or any(
        fmt not in supported for fmt in overrides.values()
    ):
        raise ValueError("Unsupported per-matrix precision")
    unknown = set(overrides) - parameter_names
    if unknown:
        raise ValueError(f"Unknown Koi parameter names: {sorted(unknown)}")
    formats = {}
    for name in parameter_names:
        fmt = overrides.get(name, default if name in matrix_names else "BF16")
        if fmt != "BF16" and name not in matrix_names:
            raise ValueError(f"Only linear matrix weights may be quantized: {name}")
        formats[name] = fmt
    return formats


def install_packed_weights(model, vllm_config):
    marker = getattr(model.config, "koi_quantization", None)
    legacy = marker is None
    if legacy:
        marker = model.config.koi_mixed
    if not isinstance(marker, dict):
        raise ValueError("Koi per-matrix quantization metadata must be an object")
    if marker.get("version") != (2 if legacy else 1):
        raise ValueError("Unsupported Koi per-matrix checkpoint version")
    if marker.get("activations", "bfloat16") != "bfloat16":
        raise ValueError("Koi packed weights require BF16 activations")
    pc = vllm_config.parallel_config
    if pc.tensor_parallel_size != 1 or pc.pipeline_parallel_size != 1:
        raise ValueError("Koi packed weights require TP=PP=1")
    if vllm_config.model_config.dtype != torch.bfloat16:
        raise ValueError("Koi packed weights require BF16 activations")
    if model.config.tie_word_embeddings:
        raise ValueError("Per-matrix Koi checkpoints require untied embeddings")
    allocation = marker
    if "allocation" in marker:
        if "formats" in marker or "default" in marker:
            raise ValueError("Specify inline formats or an allocation file, not both")
        allocation = get_hf_file_to_dict(
            marker["allocation"],
            vllm_config.model_config.model,
            revision=vllm_config.model_config.revision,
        )
        if allocation is None:
            raise ValueError("Koi quantization allocation file was not found")
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
    # Resolve formats in original HF names, before adding packed scale tensors.
    source_params = {}
    for name, param in params.items():
        source = name.replace(".in_proj.parts.", ".").replace(".qkv_proj.parts.", ".")
        source_params[source] = param
    matrix_names = {
        name
        for name, param in source_params.items()
        if name.endswith(".weight")
        and param.ndim == 2
        and isinstance(
            modules[mapped_name(model, name).rsplit(".", 1)[0]],
            (nn.Linear, LinearBase, ParallelLMHead),
        )
    }
    formats = resolve_formats(allocation, set(source_params), matrix_names)
    model._koi_formats = formats
    for name in source_params:
        fmt = formats[name]
        if fmt == "BF16":
            continue
        target = mapped_name(model, name)
        if not name.endswith(".weight") or params[target].ndim != 2:
            raise ValueError(f"Only matrix weights may be quantized: {name}")
        module = modules[target.rsplit(".", 1)[0]]
        if not isinstance(module, (nn.Linear, LinearBase, ParallelLMHead)):
            raise TypeError(f"Unsupported matrix module {name}: {type(module)}")
        install_packed_linear(module, fmt)


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
        1 if fmt == "BF16" else 2 + PACKED_FORMATS[fmt].tensor_scale
        for fmt in model._koi_formats.values()
    )
    if loaded != set(params) or len(seen) != expected:
        raise ValueError(
            f"Incomplete per-matrix checkpoint: missing {set(params) - loaded}, "
            f"count {len(seen)}"
        )
    return loaded
