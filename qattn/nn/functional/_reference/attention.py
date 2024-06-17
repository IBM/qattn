import math

import torch

from qattn.nn.functional import _reference


def attention(
    q,
    k,
    v,
    sm_scale=None,
    dtype=torch.float32,
):
    # reference implementation
    intermediate = {}
    if sm_scale is None:
        sm_scale = 1 / math.sqrt(k.size(-1))
    p = torch.matmul(q, k.transpose(2, 3)) * sm_scale
    intermediate["p"] = p.clone()
    p = torch.softmax(p.float(), dim=-1).to(dtype)
    intermediate["softmax"] = p.clone()
    out = torch.matmul(p, v)
    intermediate["out"] = out.clone()
    return out, intermediate


def static_attention(q, k, v, qkv_scale, out_scale, sm_scale=None):
    if sm_scale is None:
        sm_scale = 1 / math.sqrt(k.size(-1))
    q_scale = qkv_scale * sm_scale
    p = _reference._static_matmul_fp32_out(q, k.transpose(2, 3), q_scale, qkv_scale)
    p = torch.softmax(p.float().cuda(), dim=-1)
    out = p.half() @ (v * qkv_scale).half()
    out = torch.round(out / out_scale).to(torch.int8)
    return out
