"""Pure-PyTorch stand-in for `flash_attn.flash_attn_varlen_func`.

Only used when the real flash-attn package cannot be imported (prebuilt
flash-attn wheels rarely match the RTX PRO 6000 / sm_120 + newest torch combo
that molab ships). Semantics follow flash-attn >= 2.1:
  * q: (total_q, n_heads, d), k/v: (total_k, n_kv_heads, d)  (GQA supported)
  * causal masks are bottom-right aligned (query i of a length-Lq segment may
    attend to keys j <= i + (Lk - Lq)), which is what KV-cache decoding needs.
"""
import torch
import torch.nn.functional as F

__version__ = "2.7.0-sdpa-shim"


def flash_attn_varlen_func(
    q, k, v,
    cu_seqlens_q, cu_seqlens_k,
    max_seqlen_q=None, max_seqlen_k=None,
    dropout_p=0.0, softmax_scale=None, causal=False,
    window_size=(-1, -1), softcap=0.0, alibi_slopes=None,
    deterministic=False, return_attn_probs=False, block_table=None, **kwargs,
):
    n_heads, n_kv = q.shape[1], k.shape[1]
    if n_heads != n_kv:
        rep = n_heads // n_kv
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)
    cq = cu_seqlens_q.tolist()
    ck = cu_seqlens_k.tolist()
    out = torch.empty_like(q)
    for i in range(len(cq) - 1):
        qs, qe, ks, ke = cq[i], cq[i + 1], ck[i], ck[i + 1]
        lq, lk = qe - qs, ke - ks
        if lq == 0:
            continue
        qi = q[qs:qe].transpose(0, 1).unsqueeze(0)  # 1,H,Lq,D
        ki = k[ks:ke].transpose(0, 1).unsqueeze(0)
        vi = v[ks:ke].transpose(0, 1).unsqueeze(0)
        mask, is_causal = None, False
        if causal and lq > 1:
            if lq == lk:
                is_causal = True
            else:
                rows = torch.arange(lq, device=q.device).unsqueeze(1)
                cols = torch.arange(lk, device=q.device).unsqueeze(0)
                mask = cols <= rows + (lk - lq)
        oi = F.scaled_dot_product_attention(
            qi, ki, vi, attn_mask=mask, dropout_p=0.0,
            is_causal=is_causal, scale=softmax_scale,
        )
        out[qs:qe] = oi.squeeze(0).transpose(0, 1).to(out.dtype)
    return out


def flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False, **kwargs):
    """(B, L, H, D) layout wrapper, provided for completeness."""
    b, lq, lk = q.shape[0], q.shape[1], k.shape[1]
    cu_q = torch.arange(0, (b + 1) * lq, lq, device=q.device, dtype=torch.int32)
    cu_k = torch.arange(0, (b + 1) * lk, lk, device=q.device, dtype=torch.int32)
    o = flash_attn_varlen_func(
        q.reshape(b * lq, *q.shape[2:]), k.reshape(b * lk, *k.shape[2:]),
        v.reshape(b * lk, *v.shape[2:]), cu_q, cu_k, lq, lk,
        softmax_scale=softmax_scale, causal=causal,
    )
    return o.reshape(q.shape)
