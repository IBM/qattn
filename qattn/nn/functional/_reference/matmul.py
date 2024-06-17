from functools import wraps

import torch

__all__ = [
    "_dynamic_matmul",
    "_static_matmul",
    "_static_matmul_fp32_out",
]


def _move_to_cpu(fn):
    @wraps(fn)
    def inner(*args, **kwargs):
        new_args = [arg.cpu() if isinstance(arg, torch.Tensor) else arg for arg in args]
        new_kwargs = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in kwargs.items()}
        return fn(*new_args, **new_kwargs)

    return inner


@_move_to_cpu
def _dynamic_matmul(x, w, x_scale, w_scale):
    x = torch.round(x / x_scale).to(torch.int8).to(torch.int32)
    w = w.to(torch.int32)
    c = x @ w
    c = c * (x_scale * w_scale)
    return c


@_move_to_cpu
def _static_matmul(x, w, x_scale, w_scale, out_scale):
    x = x.to(torch.int32)
    w = w.to(torch.int32)
    c = x @ w
    c = c * (x_scale * w_scale / out_scale)
    c = torch.round(c).to(torch.int8)
    return c


@_move_to_cpu
def _static_matmul_fp32_out(x, w, x_scale, w_scale):
    x = x.to(torch.int32)
    w = w.to(torch.int32)
    c = x @ w
    c = c * (x_scale * w_scale)
    return c
