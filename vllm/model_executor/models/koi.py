# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Inference-only Koi: recurrent gated delta product and attention layers."""

import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch import nn

from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    LinearBase,
    LinearMethodBase,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.mamba.gdn.gated_delta_product import (
    GatedDeltaProductAttention,
)
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateCopyFuncCalculator,
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.configs.koi import KoiConfig
from vllm.transformers_utils.repo_utils import get_hf_file_to_dict

from .interfaces import HasInnerState, IsHybrid
from .nemotron_h import NemotronHMLP
from .utils import extract_layer_index, maybe_prefix

if TYPE_CHECKING:
    from vllm.model_executor.kernels.linear.mxfp8.marlin import MarlinMxfp8LinearKernel
    from vllm.model_executor.kernels.linear.nvfp4.marlin import MarlinNvFp4LinearKernel


class KoiPackedMethod(LinearMethodBase):
    kernel: "MarlinMxfp8LinearKernel | MarlinNvFp4LinearKernel"

    def __init__(self, shape):
        self.shape = shape

    def create_weights(self, *args, **kwargs):
        raise RuntimeError("Koi packed weights are installed before checkpoint load")

    def process_weights_after_loading(self, layer):
        layer.output_size_per_partition, layer.input_size_per_partition = self.shape
        layer.params_dtype = torch.bfloat16
        self.kernel.process_weights_after_loading(layer)
        assert layer.weight.dtype == torch.int32
        assert self.kernel.input_quant_key() is None

    def apply(self, layer, x, bias=None):
        if x.dtype != torch.bfloat16:
            raise ValueError("Koi packed kernels require BF16 activations")
        return self.kernel.apply_weights(layer, x, bias)


class KoiMxFp8Method(KoiPackedMethod):
    def __init__(self, shape):
        super().__init__(shape)
        from vllm.model_executor.kernels.linear.mxfp8 import Mxfp8LinearLayerConfig
        from vllm.model_executor.kernels.linear.mxfp8.marlin import (
            MarlinMxfp8LinearKernel,
        )

        self.kernel = MarlinMxfp8LinearKernel(Mxfp8LinearLayerConfig(shape))


class KoiNvFp4Method(KoiPackedMethod):
    def __init__(self, shape=None):
        super().__init__(shape)
        from vllm.model_executor.kernels.linear.nvfp4 import NvFp4LinearLayerConfig
        from vllm.model_executor.kernels.linear.nvfp4.marlin import (
            MarlinNvFp4LinearKernel,
        )

        self.kernel = MarlinNvFp4LinearKernel(NvFp4LinearLayerConfig())

    def process_weights_after_loading(self, layer):
        assert layer.weight.dtype == torch.uint8
        self.shape = (layer.weight.shape[0], layer.weight.shape[1] * 2)
        layer.register_parameter(
            "weight_global_scale", layer._parameters.pop("weight_scale_2")
        )
        super().process_weights_after_loading(layer)


class KoiPackedLinear(nn.Linear):
    def forward(self, x):
        return self.quant_method.apply(self, x, self.bias)


@dataclass(frozen=True)
class PackedFormat:
    block_size: int
    elements_per_byte: int
    weight_dtype: torch.dtype
    scale_dtype: torch.dtype
    method: type[KoiPackedMethod]
    tensor_scale: bool = False


PACKED_FORMATS = {
    "MXFP8": PackedFormat(32, 1, torch.float8_e4m3fn, torch.uint8, KoiMxFp8Method),
    "NVFP4": PackedFormat(
        16, 2, torch.uint8, torch.float8_e4m3fn, KoiNvFp4Method, True
    ),
}


def install_packed_linear(module, fmt):
    """Allocate source-format storage while retaining vLLM parameter metadata."""
    spec = PACKED_FORMATS[fmt]
    w = module.weight
    n, k = w.shape
    if k % spec.block_size:
        raise ValueError(f"{fmt} block size does not divide input size {k}")
    w.requires_grad_(False)
    w.data = torch.empty(
        n, k // spec.elements_per_byte, dtype=spec.weight_dtype, device=w.device
    )
    module.register_parameter(
        "weight_scale",
        nn.Parameter(
            torch.empty(
                n, k // spec.block_size, dtype=spec.scale_dtype, device=w.device
            ),
            requires_grad=False,
        ),
    )
    if spec.tensor_scale:
        module.register_parameter(
            "weight_scale_2",
            nn.Parameter(
                torch.empty((), dtype=torch.float32, device=w.device),
                requires_grad=False,
            ),
        )
    module.quant_method = spec.method((n, k))
    if isinstance(module, nn.Linear):
        module.__class__ = KoiPackedLinear


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


def _install_per_matrix_weights(model, vllm_config):
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


def _load_per_matrix_weights(model, weights):
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


def _install_mxfp8_weights(model, vllm_config):
    recipe = model.config.koi_mxfp8
    if recipe.get("version") != 1 or recipe.get("block_size") != 32:
        raise ValueError("Unsupported Koi MXFP8 checkpoint schema")
    if recipe.get("activations") != "bfloat16":
        raise ValueError("Only W8A16 Koi checkpoints are supported")
    pc = vllm_config.parallel_config
    if pc.tensor_parallel_size != 1 or pc.pipeline_parallel_size != 1:
        raise ValueError("Koi MXFP8 currently requires TP=PP=1")
    if vllm_config.model_config.dtype != torch.bfloat16:
        raise ValueError("Koi MXFP8 requires --dtype bfloat16")
    for name, module in model.named_modules():
        w = getattr(module, "weight", None)
        if w is None or w.ndim != 2 or name == "model.embed_tokens":
            continue
        if name == "lm_head" and not recipe["lm_head"]:
            continue
        if not isinstance(module, (nn.Linear, LinearBase, ParallelLMHead)):
            raise TypeError(f"Unsupported matrix module: {name}: {type(module)}")
        install_packed_linear(module, "MXFP8")


def _load_mxfp8_weights(model, weights):
    params = dict(model.named_parameters())
    loaded = set()
    seen = set()
    for name, value in weights:
        if name in seen:
            raise ValueError(f"Duplicate checkpoint tensor: {name}")
        seen.add(name)
        target, offset = name, 0
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
                roles = ["z_proj", "v_proj", "k_proj", "q_proj", "b_proj", "a_proj"]
                i = roles.index(role)
                if value.shape[0] != sizes[i]:
                    raise ValueError(f"Incorrect fused GDP shape: {name}")
                target = f"{stem}.mixer.in_proj.{suffix}"
                offset = sum(sizes[:i])
            elif model.config.layer_types[idx] == "full_attention" and role in (
                "q_proj",
                "k_proj",
                "v_proj",
            ):
                sizes = [mixer.q_size, mixer.kv_size, mixer.kv_size]
                i = ["q_proj", "k_proj", "v_proj"].index(role)
                if value.shape[0] != sizes[i]:
                    raise ValueError(f"Incorrect QKV shape: {name}")
                target = f"{stem}.mixer.qkv_proj.{suffix}"
                offset = sum(sizes[:i])
        param = params[target]
        if param.dtype in (torch.float8_e4m3fn, torch.uint8):
            if param.dtype != value.dtype:
                raise ValueError(f"Packed dtype mismatch: {name}")
            dest = param.data[offset : offset + value.shape[0]]
            if dest.shape != value.shape:
                raise ValueError(f"Packed shape mismatch: {name}")
            dest.copy_(value)
        else:
            getattr(param, "weight_loader", default_weight_loader)(param, value)
        loaded.add(target)
    missing = set(params) - loaded
    if missing:
        raise ValueError(f"Missing checkpoint tensors: {sorted(missing)}")
    # Each fused GDP replaces six weight/scale pairs, QKV replaces three.
    expected_tensors = len(params) + sum(
        10 if kind == "linear_attention" else 4 if kind == "full_attention" else 0
        for kind in model.config.layer_types
    )
    if len(seen) != expected_tensors:
        raise ValueError(f"Incomplete source projection set: {len(seen)} tensors")
    # Koi's model-level post-load hook builds conditioning tables after Marlin
    # has repacked all quantized linears, including the conditioner MLP.
    return loaded


def _install_mixed_weights(model, vllm_config):
    recipe = model.config.koi_mixed
    if recipe.get("version") == 2:
        return _install_per_matrix_weights(model, vllm_config)
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
        _install_mxfp8_weights(model, vllm_config)
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


def _load_mixed_weights(model, weights):
    if model.config.koi_mixed.get("version") == 2:
        return _load_per_matrix_weights(model, weights)
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


class KoiAttention(nn.Module):
    """Grouped-query attention without positional embeddings."""

    def __init__(self, config, vllm_config, prefix, cache_prefixes):
        super().__init__()
        tp_size = get_tensor_model_parallel_world_size()
        self.num_heads = config.num_attention_heads // tp_size
        self.num_kv_heads = max(1, config.num_key_value_heads // tp_size)
        self.head_dim = config.head_dim
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.qkv_proj = QKVParallelLinear(
            config.hidden_size,
            self.head_dim,
            config.num_attention_heads,
            config.num_key_value_heads,
            bias=False,
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            config.num_attention_heads * self.head_dim,
            config.hidden_size,
            bias=False,
            prefix=f"{prefix}.o_proj",
        )
        self.per_loop_kv = config.recurrent_kv_cache_mode == "per_loop"
        self.attn = nn.ModuleList(
            [
                Attention(
                    self.num_heads,
                    self.head_dim,
                    scale=self.head_dim**-0.5,
                    num_kv_heads=self.num_kv_heads,
                    cache_config=vllm_config.cache_config,
                    prefix=f"{p}.attn",
                )
                for p in (cache_prefixes if self.per_loop_kv else cache_prefixes[:1])
            ]
        )

    def forward(self, hidden_states, loop_idx=0):
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        # Each loop overwrites this step's KV; earlier steps retain last-loop KV.
        output, _ = self.o_proj(self.attn[loop_idx if self.per_loop_kv else 0](q, k, v))
        return output


def _sinusoidal_embedding(
    value: float, dim: int, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(
        -math.log(100.0) * torch.arange(half, device=device, dtype=torch.float32) / half
    )
    args = value * freqs * 100.0
    return torch.cat([torch.cos(args), torch.sin(args)]).to(dtype)


class KoiLoopConditioner(nn.Module):
    """``(progress, budget) -> c``: sinusoidal embeddings, Linear, SiLU, Linear."""

    def __init__(self, cond_dim: int, dtype: torch.dtype) -> None:
        super().__init__()
        self.cond_dim = cond_dim
        self.dtype = dtype
        self.mlp = nn.Sequential(
            nn.Linear(2 * cond_dim, cond_dim, dtype=dtype),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim, dtype=dtype),
        )

    def forward(self, progress: float, budget: float) -> torch.Tensor:
        device = self.mlp[0].weight.device
        return self.mlp(
            torch.cat(
                [
                    _sinusoidal_embedding(progress, self.cond_dim, device, self.dtype),
                    _sinusoidal_embedding(budget, self.cond_dim, device, self.dtype),
                ]
            )
        )


class KoiLoopConditioning(nn.Module):
    """Adaptive normalization and a gated low-rank adapter."""

    def __init__(self, config: KoiConfig, prefix: str):
        super().__init__()
        hidden, cond, rank = (
            config.hidden_size,
            config.loop_conditioning_hidden_size,
            config.loop_adapter_rank,
        )
        self.hidden_size = hidden
        self.adapter_rank = rank
        self.gamma_head = nn.Linear(cond, hidden)
        self.beta_head = nn.Linear(cond, hidden)
        self.gate_head = nn.Linear(cond, hidden)
        self.adapter_gate = nn.Linear(cond, rank) if rank else None
        self.up_proj = (
            ReplicatedLinear(rank, hidden, bias=False, prefix=f"{prefix}.up_proj")
            if rank
            else None
        )
        self.down_proj = (
            ReplicatedLinear(hidden, rank, bias=False, prefix=f"{prefix}.down_proj")
            if rank
            else None
        )
        self.register_buffer(
            "table",
            torch.zeros(config.n_recurrent_loops + 1, 3 * hidden + rank),
            persistent=False,
        )

    def update(self, conds):
        rows = []
        for c in conds:
            parts = [1 + self.gamma_head(c), self.beta_head(c), 1 + self.gate_head(c)]
            if self.adapter_gate is not None:
                parts.append(self.adapter_gate(c))
            rows.append(torch.cat(parts))
        self.table.copy_(torch.stack(rows))

    def forward(self, hidden_states, step):
        scale, shift, gate, adapter_gate = self.table[step].split(
            [self.hidden_size] * 3 + [self.adapter_rank]
        )
        hidden_states = scale * hidden_states + shift
        if self.down_proj is not None and self.up_proj is not None:
            adapter, _ = self.down_proj(hidden_states)
            adapter, _ = self.up_proj(adapter_gate * adapter)
            hidden_states = hidden_states + adapter
        return hidden_states, gate


class KoiLoopSeam(nn.Module):
    """Conditioned recurrent entry and exit."""

    def __init__(self, config: KoiConfig):
        super().__init__()
        hidden, cond = config.hidden_size, config.loop_conditioning_hidden_size
        self.hidden_size = hidden
        self.entry_norm = RMSNorm(hidden, eps=config.rms_norm_eps)
        self.entry_gamma = nn.Linear(cond, hidden)
        self.entry_beta = nn.Linear(cond, hidden)
        self.exit_gate = nn.Linear(cond, hidden)
        self.input_mix = nn.Linear(cond, hidden) if config.loop_input_mixing else None
        self.register_buffer(
            "table",
            torch.zeros(
                config.n_recurrent_loops,
                hidden * (4 if self.input_mix is not None else 3),
            ),
            persistent=False,
        )

    def update(self, conds):
        rows = []
        for step, c in enumerate(conds):
            parts = [
                1 + self.entry_gamma(c),
                self.entry_beta(c),
                (1.0 if step == 0 else 0.0) + self.exit_gate(c),
            ]
            if self.input_mix is not None:
                parts.append(self.input_mix(c))
            rows.append(torch.cat(parts))
        self.table.copy_(torch.stack(rows))

    def entry(self, loop_input, input_h, step):
        parts = self.table[step].split(self.hidden_size)
        hidden_states = parts[0] * self.entry_norm(loop_input) + parts[1]
        if self.input_mix is not None:
            hidden_states = hidden_states + parts[3] * input_h
        return hidden_states

    def exit(self, state, new_state, step):
        gate = self.table[step, 2 * self.hidden_size : 3 * self.hidden_size]
        return state + gate * (new_state - state)


class KoiStateUpdate(nn.Module):
    """``state + sigmoid(W x + b) * softplus(base_alpha_raw) * (new - state)``, with
    ``x`` the loop input."""

    def __init__(self, config: KoiConfig, prefix: str) -> None:
        super().__init__()
        self.proj = ReplicatedLinear(
            config.hidden_size, config.hidden_size, bias=True, prefix=f"{prefix}.proj"
        )
        self.base_alpha_raw = nn.Parameter(torch.empty(config.hidden_size))

    def forward(
        self, state: torch.Tensor, new_state: torch.Tensor, loop_input: torch.Tensor
    ) -> torch.Tensor:
        logits, _ = self.proj(loop_input)
        beta = torch.sigmoid(logits) * F.softplus(self.base_alpha_raw)
        return state + beta * (new_state - state)


class KoiDecoderLayer(nn.Module):
    def __init__(self, config, vllm_config, prefix, layer_idx):
        super().__init__()
        self.layer_type = config.layer_types[layer_idx]
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        n_in, n_rec = config.n_input_layers, config.n_recurrent_layers
        loops = config.n_recurrent_loops if n_in <= layer_idx < n_in + n_rec else 1
        # Virtual indices distinguish caches; projections keep physical indices.
        indices = [layer_idx + loop * config.num_hidden_layers for loop in range(loops)]
        cache_prefixes = [
            f"{prefix}.mixer".replace(f"layers.{layer_idx}.", f"layers.{i}.", 1)
            for i in indices
        ]
        if self.layer_type == "linear_attention":
            self.mixer = GatedDeltaProductAttention(
                config,
                vllm_config,
                prefix=f"{prefix}.mixer",
                cache_prefixes=cache_prefixes,
            )
        elif self.layer_type == "full_attention":
            self.mixer = KoiAttention(
                config, vllm_config, f"{prefix}.mixer", cache_prefixes
            )
        else:
            self.mixer = NemotronHMLP(
                config=None,
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                prefix=f"{prefix}.mixer",
            )
        self.loop_conditioning = (
            KoiLoopConditioning(config, f"{prefix}.loop_conditioning")
            if layer_idx >= n_in
            else None
        )

    def forward(self, hidden_states, loop_idx=0, cond_idx=0):
        residual = hidden_states
        hidden_states = self.norm(hidden_states)
        if self.loop_conditioning is not None:
            hidden_states, gate = self.loop_conditioning(hidden_states, cond_idx)
        if self.layer_type == "mlp":
            hidden_states = self.mixer(hidden_states)
        else:
            hidden_states = self.mixer(hidden_states, loop_idx)
        if self.loop_conditioning is not None:
            hidden_states = hidden_states * gate
        return residual + hidden_states


@support_torch_compile
class KoiModel(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.config = config
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size, config.hidden_size
        )
        self.layers = nn.ModuleList(
            [
                KoiDecoderLayer(config, vllm_config, f"{prefix}.layers.{i}", i)
                for i in range(config.num_hidden_layers)
            ]
        )
        self.norm_f = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.loop_conditioner = KoiLoopConditioner(
            config.loop_conditioning_hidden_size, vllm_config.model_config.dtype
        )
        self.loop_seam = KoiLoopSeam(config)
        self.state_update = KoiStateUpdate(config, f"{prefix}.state_update")

    @torch.no_grad()
    def build_loop_tables(self):
        config = self.config
        count = config.n_recurrent_loops
        budget = (
            min(count, config.loop_conditioning_max_loops)
            / config.loop_conditioning_max_loops
        )
        conds = [
            self.loop_conditioner((step + 1) / count, budget) for step in range(count)
        ]
        conds.append(self.loop_conditioner(1.0, budget))
        for layer in self.layers:
            if layer.loop_conditioning is not None:
                layer.loop_conditioning.update(conds)
        self.loop_seam.update(conds[:-1])

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h = inputs_embeds if inputs_embeds is not None else self.embed_tokens(input_ids)
        config = self.config
        n_in, n_rec = config.n_input_layers, config.n_recurrent_layers
        for layer in self.layers[:n_in]:
            h = layer(h)
        input_h = state = h
        for loop in range(config.n_recurrent_loops):
            loop_input = state + input_h
            h = self.loop_seam.entry(loop_input, input_h, loop) if loop else loop_input
            for layer in self.layers[n_in : n_in + n_rec]:
                h = layer(h, loop, loop)
            new_state = self.state_update(state, h, loop_input)
            state = self.loop_seam.exit(state, new_state, loop)
        h = state
        for layer in self.layers[n_in + n_rec :]:
            h = layer(h, cond_idx=config.n_recurrent_loops)
        return self.norm_f(h)


class KoiForCausalLM(nn.Module, HasInnerState, IsHybrid):
    packed_modules_mapping = {"qkv_proj": ["q_proj", "k_proj", "v_proj"]}

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = self.config = vllm_config.model_config.hf_config
        config.validate()
        if vllm_config.quant_config is not None:
            raise ValueError("Koi does not support quantized checkpoints")
        if vllm_config.speculative_config is not None:
            raise ValueError("Koi currently supports plain decoding only")
        if vllm_config.cache_config.mamba_cache_mode == "all":
            raise ValueError("Koi does not support mamba_cache_mode='all'")
        self.model = KoiModel(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        if config.tie_word_embeddings:
            self.lm_head = self.lm_head.tie_weights(self.model.embed_tokens)
        self.logits_processor = LogitsProcessor(config.vocab_size)
        if getattr(config, "koi_quantization", None):
            _install_per_matrix_weights(self, vllm_config)
        elif getattr(config, "koi_mixed", None):
            _install_mixed_weights(self, vllm_config)
        elif getattr(config, "koi_mxfp8", None):
            _install_mxfp8_weights(self, vllm_config)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.model(input_ids, positions, inputs_embeds)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)

    def process_weights_after_loading(self) -> None:
        # Also called by loaders that restore parameters without load_weights.
        self.model.build_loop_tables()

    @classmethod
    def get_mamba_state_dtype_from_config(cls, vllm_config):
        return MambaStateDtypeCalculator.gated_delta_net_state_dtype(
            vllm_config.model_config.dtype,
            vllm_config.cache_config.mamba_cache_dtype,
            vllm_config.cache_config.mamba_ssm_cache_dtype,
        )

    @classmethod
    def get_mamba_state_shape_from_config(cls, vllm_config):
        c = vllm_config.model_config.hf_config
        return MambaStateShapeCalculator.gated_delta_product_state_shape(
            vllm_config.parallel_config.tensor_parallel_size,
            c.linear_num_key_heads,
            c.linear_num_value_heads,
            c.linear_key_head_dim,
            c.linear_value_head_dim,
            c.linear_num_householder,
            c.linear_conv_kernel_dim,
        )

    @classmethod
    def get_mamba_state_copy_func(cls):
        return MambaStateCopyFuncCalculator.gated_delta_net_state_copy_func()

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        if getattr(self.config, "koi_quantization", None):
            return _load_per_matrix_weights(self, weights)
        if getattr(self.config, "koi_mixed", None):
            return _load_mixed_weights(self, weights)
        if getattr(self.config, "koi_mxfp8", None):
            return _load_mxfp8_weights(self, weights)
        params = dict(self.named_parameters())
        loaded = set()
        for name, weight in weights:
            if self.config.tie_word_embeddings and name.startswith("lm_head."):
                continue
            if name.startswith("model.layers."):
                layer_idx = extract_layer_index(name)
                stem, sep, leaf = name.rpartition(".mixer.")
                role = leaf.split(".")[0]
                layer_type = self.config.layer_types[layer_idx]
                if (
                    sep
                    and layer_type == "linear_attention"
                    and role
                    in ("z_proj", "v_proj", "k_proj", "q_proj", "b_proj", "a_proj")
                ):
                    self.model.layers[layer_idx].mixer.load_in_proj_weight(
                        role[0], weight
                    )
                    loaded.add(f"{stem}.mixer.in_proj.weight")
                    continue
                if (
                    sep
                    and layer_type == "full_attention"
                    and role in ("q_proj", "k_proj", "v_proj")
                ):
                    name = f"{stem}.mixer.qkv_proj.weight"
                    params[name].weight_loader(params[name], weight, role[0])
                    loaded.add(name)
                    continue
            param = params[name]
            getattr(param, "weight_loader", default_weight_loader)(param, weight)
            loaded.add(name)
        self.model.build_loop_tables()
        return loaded
