"""Quantized static matrix multiplication."""

# Based on https://github.com/triton-lang/triton/blob/main/python/triton/ops/matmul.py


import torch
import triton
import triton.language as tl
from triton.ops.matmul_perf_model import (
    estimate_matmul_time,
)

from qattn.nn.functional._matmul_configs import (
    int8_configs,
    early_config_prune,
)


def _estimate_matmul_time(*args, **kwargs):
    kwargs["SPLIT_K"] = 1
    return estimate_matmul_time(*args, **kwargs)


@triton.autotune(
    configs=int8_configs(),
    key=["M", "N", "K"],
    prune_configs_by={"early_config_prune": early_config_prune, "perf_model": _estimate_matmul_time, "top_k": 10},
)
@triton.heuristics(
    {
        "EVEN_K": lambda args: args["K"] % args["BLOCK_K"] == 0,
    }
)
@triton.jit
def _kernel(
    A,
    B,
    C,
    bias,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    a_scale_ptr,
    b_scale_ptr,
    out_scale_ptr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    EVEN_K: tl.constexpr,
    BIAS_ADD: tl.constexpr,
    A_PER_CHANNEL: tl.constexpr,
    B_PER_CHANNEL: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_z = tl.program_id(1)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)

    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // (group_size)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    rk = pid_z * BLOCK_K + tl.arange(0, BLOCK_K)

    A = A + (ram[:, None] * stride_am + rk[None, :] * stride_ak)
    B = B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        if EVEN_K:
            a = tl.load(A)
            b = tl.load(B)
        else:
            k_remaining = K - k * BLOCK_K
            _0 = tl.zeros((1, 1), dtype=tl.int8)
            a = tl.load(A, mask=rk[None, :] < k_remaining, other=_0)
            b = tl.load(B, mask=rk[:, None] < k_remaining, other=_0)
        acc += tl.dot(a, b, allow_tf32=True, out_dtype=tl.int32)
        A += BLOCK_K * stride_ak
        B += BLOCK_K * stride_bk
    if A_PER_CHANNEL:
        _0 = tl.zeros((1,), dtype=a_scale_ptr.dtype.element_ty)
        mask = ram < M
        a_scale = tl.load(a_scale_ptr + ram, mask=mask, other=_0)
    else:
        a_scale = tl.load(a_scale_ptr)
    if B_PER_CHANNEL:
        _0 = tl.zeros((1,), dtype=b_scale_ptr.dtype.element_ty)
        mask = rbn < N
        b_scale = tl.load(b_scale_ptr + rbn, mask=mask, other=_0)
    else:
        b_scale = tl.load(b_scale_ptr)
    if BIAS_ADD:
        bias = tl.load(bias + rn)
        if A_PER_CHANNEL and B_PER_CHANNEL:
            bias = tl.math.llrint(bias / (a_scale[:, None] * b_scale[None, :])).to(tl.int32)
            acc = acc + bias
        else:
            bias = tl.math.llrint(bias / (a_scale * b_scale)).to(tl.int32)
            acc = acc + bias[None, :]

    if A_PER_CHANNEL and B_PER_CHANNEL:
        mask = ram < M
        _0 = tl.zeros((1,), dtype=out_scale_ptr.dtype.element_ty)
        out_scale = tl.load(out_scale_ptr + ram, mask=mask, other=_0)
        acc = tl.math.llrint((acc.to(tl.float32) * a_scale[:, None] * b_scale[None, :] * out_scale[:, None])).to(
            tl.int8
        )
    else:
        out_scale = tl.load(out_scale_ptr)
        acc = tl.math.llrint((acc.to(tl.float32) * (a_scale * b_scale * out_scale))).to(tl.int8)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    C = C + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)
    mask = (rm < M)[:, None] & (rn < N)[None, :]
    tl.store(C, acc, mask=mask)


class _matmul(torch.autograd.Function):
    kernel = _kernel

    @staticmethod
    def forward(
        ctx,
        input,
        other,
        input_scale,
        other_scale,
        out_scale,
        bias=None,
        a_per_channel=False,
        b_per_channel=False,
    ) -> torch.Tensor:
        device = input.device
        if input.stride(0) > 1 and input.stride(1) > 1:
            input = input.contiguous()
        if other.stride(0) > 1 and other.stride(1) > 1:
            other = other.contiguous()
        assert input.shape[1] == other.shape[0], "incompatible dimensions"
        M, K = input.shape
        _, N = other.shape
        c = torch.empty((M, N), device=device, dtype=torch.int8)
        grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),)  # noqa: E731
        BIAS_ADD = 0 if bias is None else 1
        _kernel[grid](
            input,
            other,
            c,
            bias,
            M,
            N,
            K,
            input.stride(0),
            input.stride(1),
            other.stride(0),
            other.stride(1),
            c.stride(0),
            c.stride(1),
            a_scale_ptr=input_scale,
            b_scale_ptr=other_scale,
            out_scale_ptr=1.0 / out_scale,
            GROUP_M=8,
            BIAS_ADD=BIAS_ADD,
            A_PER_CHANNEL=a_per_channel,
            B_PER_CHANNEL=b_per_channel,
        )
        return c
