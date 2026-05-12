"""3D Rotary Position Embedding for HTCS sparse patch tokens.

Stage 1 returns a variable-length sparse patch sequence with explicit
(t, h, w) coordinates. We need a position encoding that:

* Is permutation-invariant across the *order* of the kept patches.
* Encodes time-vs-space symmetrically so swapping frames or rows works.
* Plays well with bf16 (rotate-half RoPE survives low precision better than
  additive sinusoidal embeddings).

Implementation: split feature dim `d` into three equal even-sized chunks
for t/h/w axes; apply rotate-half RoPE per chunk using the corresponding
coordinate. If d is not divisible by 6, the trailing `d - 3*d_each`
dimensions are left unchanged.

Reference: HTCS impl doc §3.5.
"""

import torch


def apply_3d_rope(
    patches: torch.Tensor,
    coords: torch.Tensor,
    theta: float = 10000.0,
) -> torch.Tensor:
    """3D Sinusoidal Rotary Position Embedding (rotate_half convention).

    Args:
        patches: (B, N, d)  — Stage-1 sparse patch tokens (float32 or bf16).
        coords:  (B, N, 3)  — integer coordinates (t, h, w), dtype long.
        theta:   base for inverse frequencies (LLaMA / Qwen default 10000).

    Returns:
        (B, N, d) — same shape, with 3D positional information baked in.
    """
    B, N, d = patches.shape
    d_each = (d // 6) * 2                       # largest even ≤ d/3
    d_rope = d_each * 3
    if d_rope == 0:
        return patches

    device = patches.device
    dtype = patches.dtype
    half = d_each // 2

    # inv_freq[i] = 1 / theta^(i/half), shape (half,)
    inv_f = 1.0 / (theta ** (
        torch.arange(half, device=device, dtype=torch.float32) / half
    ))

    def rope_1d(x_slice: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        # x_slice: (B, N, d_each) float32;  pos: (B, N) long
        ang = pos.float().unsqueeze(-1) * inv_f                              # (B, N, half)
        cos_f = torch.cat([ang.cos(), ang.cos()], dim=-1)                    # (B, N, d_each)
        sin_f = torch.cat([ang.sin(), ang.sin()], dim=-1)
        x1, x2 = x_slice[..., :half], x_slice[..., half:]
        return x_slice * cos_f + torch.cat([-x2, x1], dim=-1) * sin_f

    out = patches.clone()
    for axis_i in range(3):
        sl = slice(axis_i * d_each, (axis_i + 1) * d_each)
        rotated = rope_1d(patches[..., sl].float(), coords[..., axis_i])
        out[..., sl] = rotated.to(dtype)

    return out
