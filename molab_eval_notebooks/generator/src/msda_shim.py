"""Pure-PyTorch stand-in for DeltaV's compiled `MultiScaleDeformableAttention` CUDA op.

DeltaV's INSTALL.md calls the compiled op optional ("a pure-PyTorch path is used
otherwise"), but ms_deform_attn_func.py imports it unconditionally. This module provides
the forward pass with the same maths as the CUDA kernel (bilinear sampling at
loc * size - 0.5, zero outside the map, attention-weighted sum), computed in float32
like the kernel, which only supports float/double. Inference only: no backward.
"""
import torch
import torch.nn.functional as F

__version__ = "pytorch-fallback"


def ms_deform_attn_forward(value, spatial_shapes, level_start_index, sampling_locations,
                           attention_weights, im2col_step=64):
    """value (N, S, M, D); sampling_locations (N, Lq, M, L, P, 2) as (x, y) in [0, 1];
    attention_weights (N, Lq, M, L, P). Returns (N, Lq, M * D)."""
    out_dtype = value.dtype
    value = value.float()
    sampling_locations = sampling_locations.float()
    attention_weights = attention_weights.float()
    n, _, m, d = value.shape
    _, lq, _, n_levels, n_points, _ = sampling_locations.shape
    shapes = [(int(h), int(w)) for h, w in spatial_shapes.tolist()]
    starts = [int(s) for s in level_start_index.tolist()]
    grids = 2 * sampling_locations - 1  # grid_sample(align_corners=False): pixel = loc * size - 0.5
    sampled = []
    for lid in range(n_levels):
        h, w = shapes[lid]
        v = value[:, starts[lid]:starts[lid] + h * w]                      # N, HW, M, D
        v = v.flatten(2).transpose(1, 2).reshape(n * m, d, h, w)          # N*M, D, H, W
        g = grids[:, :, :, lid].transpose(1, 2).flatten(0, 1)             # N*M, Lq, P, 2
        sampled.append(F.grid_sample(v, g, mode="bilinear", padding_mode="zeros", align_corners=False))
    weights = attention_weights.transpose(1, 2).reshape(n * m, 1, lq, n_levels * n_points)
    out = (torch.stack(sampled, dim=-2).flatten(-2) * weights).sum(-1).view(n, m * d, lq)
    return out.transpose(1, 2).contiguous().to(out_dtype)


def ms_deform_attn_backward(*args, **kwargs):
    raise NotImplementedError("PyTorch fallback for MultiScaleDeformableAttention is inference-only")
