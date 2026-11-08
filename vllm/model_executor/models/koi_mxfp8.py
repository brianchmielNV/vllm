# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Koi TP=1 packed MXFP8 weights with built-in Marlin W8A16 GEMMs.

Uses the existing vLLM MXFP8 backend, with load-time dequantization disabled.
The custom koi_mxfp8 checkpoint marker avoids advertising support in unpatched
vLLM versions. Convolution, vectors, and embedding remain unquantized.
"""

import torch
from torch import nn

from vllm.model_executor.kernels.linear.mxfp8 import Mxfp8LinearLayerConfig
from vllm.model_executor.kernels.linear.mxfp8.marlin import (
    MarlinMxfp8LinearKernel,
)
from vllm.model_executor.layers.linear import LinearBase, LinearMethodBase
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.model_loader.weight_utils import default_weight_loader


class KoiMxFp8Method(LinearMethodBase):
    def __init__(self, shape):
        self.kernel = MarlinMxfp8LinearKernel(Mxfp8LinearLayerConfig(shape))
        self.shape = shape

    def create_weights(self, *args, **kwargs):
        raise RuntimeError("Koi MXFP8 weights are installed before checkpoint load")

    def process_weights_after_loading(self, layer):
        layer.output_size_per_partition, layer.input_size_per_partition = self.shape
        layer.params_dtype = torch.bfloat16
        self.kernel.process_weights_after_loading(layer)
        assert layer.weight.dtype == torch.int32
        assert self.kernel.input_quant_key() is None

    def apply(self, layer, x, bias=None):
        if x.dtype != torch.bfloat16:
            raise ValueError("Koi MXFP8 W8A16 requires BF16 activations")
        return self.kernel.apply_weights(layer, x, bias)


class KoiMxFp8TorchLinear(nn.Linear):
    def forward(self, x):
        return self.quant_method.apply(self, x, self.bias)


def install_packed_weights(model, vllm_config):
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
        n, k = w.shape
        if k % 32:
            raise ValueError(f"MXFP8 block size does not divide {name}")
        # Preserve parameter subclass/loader metadata; no BF16 matrix survives.
        w.requires_grad_(False)
        w.data = torch.empty((n, k), dtype=torch.float8_e4m3fn, device=w.device)
        module.register_parameter(
            "weight_scale",
            nn.Parameter(
                torch.empty((n, k // 32), dtype=torch.uint8, device=w.device),
                requires_grad=False,
            ),
        )
        module.quant_method = KoiMxFp8Method((n, k))
        if isinstance(module, nn.Linear):
            module.__class__ = KoiMxFp8TorchLinear


def load_packed_weights(model, weights):
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
