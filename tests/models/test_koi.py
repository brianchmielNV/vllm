# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import pytest
import torch

from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config
from vllm.model_executor.models.koi import KoiLoopConditioning, KoiLoopSeam
from vllm.transformers_utils.configs.koi import KoiConfig


@pytest.mark.parametrize("seam", [False, True])
@torch.no_grad()
def test_loop_table_reload_preserves_captured_storage(seam):
    config = KoiConfig(
        hidden_size=8,
        loop_conditioning_hidden_size=4,
        loop_adapter_rank=0,
        n_recurrent_loops=2,
    )
    with set_current_vllm_config(VllmConfig(device_config=DeviceConfig(device="cpu"))):
        module = KoiLoopSeam(config) if seam else KoiLoopConditioning(config, "test")
    conds = [torch.randn(4) for _ in range(2 if seam else 3)]
    module.update(conds)
    captured_table = module.table.view_as(module.table)
    before = captured_table.clone()
    pointer = captured_table.data_ptr()
    head = module.entry_beta if seam else module.beta_head
    head.bias.add_(1)
    module.update(conds)
    assert module.table.data_ptr() == pointer
    torch.testing.assert_close(captured_table, module.table)
    torch.testing.assert_close(captured_table[:, 8:16], before[:, 8:16] + 1)
    assert "table" not in module.state_dict()


def _cache_layout(model):
    from vllm.model_executor.layers.attention import Attention
    from vllm.model_executor.layers.mamba.abstract import MambaBase

    return (
        sum(isinstance(m, Attention) for m in model.modules()),
        sum(isinstance(m, MambaBase) for m in model.modules()),
    )


@pytest.mark.usefixtures("enable_pickle")
@pytest.mark.parametrize("cache_mode", ["per_loop", "last_loop"])
def test_loop_cache_modes(vllm_runner, tmp_path, cache_mode):
    """Last-loop attention must retain independent GDP state through decoding."""
    from vllm import SamplingParams

    KoiConfig(
        architectures=["KoiForCausalLM"],
        vocab_size=64,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        layer_types=["linear_attention", "full_attention"],
        n_input_layers=0,
        n_recurrent_layers=2,
        n_output_layers=0,
        n_recurrent_loops=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=64,
        linear_num_value_heads=4,
        linear_num_key_heads=2,
        loop_conditioning_hidden_size=8,
        loop_adapter_rank=0,
    ).save_pretrained(tmp_path)
    with vllm_runner(
        str(tmp_path),
        load_format="dummy",
        skip_tokenizer_init=True,
        dtype="bfloat16",
        enforce_eager=True,
        max_model_len=64,
        max_num_seqs=2,
        enable_prefix_caching=False,
        hf_overrides={"recurrent_kv_cache_mode": cache_mode},
    ) as runner:
        outputs = runner.llm.generate(
            [{"prompt_token_ids": [1, 2, 3]}, {"prompt_token_ids": [4, 5]}],
            SamplingParams(temperature=0, max_tokens=4, ignore_eos=True, logprobs=1),
        )
        for output in outputs:
            completion = output.outputs[0]
            assert len(completion.token_ids) == 4
            assert all(
                torch.isfinite(torch.tensor(step[token].logprob))
                for token, step in zip(completion.token_ids, completion.logprobs)
            )
        attention_caches = 2 if cache_mode == "per_loop" else 1
        assert runner.apply_model(_cache_layout) == [(attention_caches, 2)]


@pytest.mark.parametrize("fmt", ["MXFP8", "NVFP4"])
@pytest.mark.parametrize("n,k", [(96, 256), (256, 256)])
@pytest.mark.parametrize("batch", [1, 32, 512])
@torch.inference_mode()
def test_packed_linear(fmt, n, k, batch):
    """Check decode/prefill, small GDP gates, bias and native packed storage."""
    from torch import nn
    from torch.nn import functional as F

    from vllm.model_executor.models.koi import install_packed_linear

    generator = torch.Generator(device="cuda").manual_seed(2341)
    layer = nn.Linear(k, n, device="cuda", dtype=torch.bfloat16)
    layer.requires_grad_(False)
    install_packed_linear(layer, fmt)
    bias = torch.randn(n, device="cuda", generator=generator).bfloat16() * 0.01
    layer.bias.copy_(bias)
    if fmt == "MXFP8":
        q = torch.randint(-8, 9, (n, k), device="cuda", generator=generator)
        layer.weight.data = q.to(torch.float8_e4m3fn)
        scales = torch.full((n, k // 32), 124, device="cuda", dtype=torch.uint8)
        reference_weight = q.bfloat16() * 0.125  # E8M0 exponent: 124 - 127.
    else:
        q = torch.randint(0, 256, (n, k // 2), device="cuda", generator=generator)
        layer.weight.data = q.to(torch.uint8)
        scales = torch.full((n, k // 16), 0.5, device="cuda", dtype=torch.float8_e4m3fn)
        layer.register_parameter(
            "weight_scale_2",
            nn.Parameter(torch.tensor(0.25, device="cuda"), requires_grad=False),
        )
        # Independent E2M1 decode, including the high/low nibble ordering.
        lut = torch.tensor(
            [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6],
            device="cuda",
            dtype=torch.bfloat16,
        )
        codes = torch.stack((q & 15, q >> 4), dim=-1).reshape(n, k)
        reference_weight = lut[codes] * 0.125
    layer.register_parameter("weight_scale", nn.Parameter(scales, requires_grad=False))
    method = layer.quant_method
    method.process_weights_after_loading(layer)
    assert layer.weight.dtype == torch.int32
    assert method.kernel.input_quant_key() is None
    x = torch.randn(batch, k, device="cuda", generator=generator).bfloat16() * 0.1
    actual = layer(x)
    expected = F.linear(x, reference_weight, bias)
    assert actual.dtype == torch.bfloat16
    assert torch.isfinite(actual).all()
    relative_rmse = (
        (actual.float() - expected.float()).square().mean()
        / expected.float().square().mean()
    ).sqrt()
    # BF16 Marlin and dense GEMM use different accumulation orders.
    assert relative_rmse < 0.02
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as p:
        method.apply(layer, x, layer.bias)
    assert any("marlin_gemm" in event.key for event in p.key_averages())
    with pytest.raises(ValueError, match="BF16 activations"):
        method.apply(layer, x.float(), layer.bias)


@pytest.mark.parametrize("default", ["BF16", "MXFP8", "NVFP4"])
def test_per_matrix_formats(default):
    from vllm.model_executor.models.koi import resolve_formats

    matrices = {"projection.weight", "lm_head.weight", "ffn.weight"}
    names = matrices | {"projection.bias", "model.embed_tokens.weight"}
    formats = resolve_formats(
        {
            "default": default,
            "formats": {"projection.weight": "NVFP4", "lm_head.weight": "MXFP8"},
        },
        names,
        matrices,
    )
    assert formats == {
        "projection.weight": "NVFP4",
        "lm_head.weight": "MXFP8",
        "ffn.weight": default,
        "projection.bias": "BF16",
        "model.embed_tokens.weight": "BF16",
    }


@pytest.mark.parametrize(
    "allocation,match",
    [
        ({"formats": {"missing.weight": "BF16"}}, "Unknown Koi parameter"),
        ({"formats": {"projection.bias": "NVFP4"}}, "Only linear matrix"),
        ({"formats": {"model.embed_tokens.weight": "MXFP8"}}, "Only linear matrix"),
        ({"default": "FP8"}, "Unsupported per-matrix"),
        ({"formats": {"projection.weight": "FP8"}}, "Unsupported per-matrix"),
    ],
)
def test_per_matrix_formats_reject_invalid(allocation, match):
    from vllm.model_executor.models.koi import resolve_formats

    with pytest.raises(ValueError, match=match):
        resolve_formats(
            allocation,
            {"projection.weight", "projection.bias", "model.embed_tokens.weight"},
            {"projection.weight"},
        )
