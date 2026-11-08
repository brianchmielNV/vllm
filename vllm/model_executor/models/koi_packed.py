# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared packed linear storage and native W8A16/W4A16 kernels for Koi."""

from dataclasses import dataclass

import torch
from torch import nn

from vllm.model_executor.kernels.linear.mxfp8 import Mxfp8LinearLayerConfig
from vllm.model_executor.kernels.linear.mxfp8.marlin import MarlinMxfp8LinearKernel
from vllm.model_executor.kernels.linear.nvfp4 import NvFp4LinearLayerConfig
from vllm.model_executor.kernels.linear.nvfp4.marlin import MarlinNvFp4LinearKernel
from vllm.model_executor.layers.linear import LinearMethodBase


class KoiPackedMethod(LinearMethodBase):
    kernel: MarlinMxfp8LinearKernel | MarlinNvFp4LinearKernel

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
        self.kernel = MarlinMxfp8LinearKernel(Mxfp8LinearLayerConfig(shape))


class KoiNvFp4Method(KoiPackedMethod):
    def __init__(self, shape=None):
        super().__init__(shape)
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
