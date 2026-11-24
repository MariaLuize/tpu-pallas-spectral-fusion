import functools
from typing import Tuple

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
import torch
import torch.nn.functional as F

# TorchTPU Pallas bridge
from torch.tpu import pallas as tpu_pallas


_TORCH_TO_JAX_DTYPE = {
    torch.float32: jnp.float32,
    torch.bfloat16: jnp.bfloat16,
}


def _fused_spectral_indices_2d_kernel(red_ref,green_ref,nir_ref,out_ndvi_ref,out_ndwi_ref, *,eps: float,):
    """Pallas kernel operating on 2D tiles inside TPU VMEM (SRAM)."""
    red = red_ref[...]
    green = green_ref[...]
    nir = nir_ref[...]

    orig_dtype = red.dtype

    # Compute in float32 inside the VPU to preserve precision for bfloat16 inputs
    red_f32 = red.astype(jnp.float32)
    green_f32 = green.astype(jnp.float32)
    nir_f32 = nir.astype(jnp.float32)

    ndvi = (nir_f32 - red_f32) / (nir_f32 + red_f32 + eps)
    ndwi = (green_f32 - nir_f32) / (green_f32 + nir_f32 + eps)

    out_ndvi_ref[...] = ndvi.astype(orig_dtype)
    out_ndwi_ref[...] = ndwi.astype(orig_dtype)


def _pad_to_block_multiple(tensor: torch.Tensor, block_h: int, block_w: int,) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """Pad a 2D tensor on the bottom/right so dimensions are multiples of block size."""
    h, w  = tensor.shape
    pad_h = (block_h - (h % block_h)) % block_h
    pad_w = (block_w - (w % block_w)) % block_w

    if pad_h == 0 and pad_w == 0:
        return tensor, (0, 0)

    # F.pad expects (pad_left, pad_right, pad_top, pad_bottom)
    padded = F.pad(tensor, (0, pad_w, 0, pad_h), mode="constant", value=0.0)
    return padded, (pad_h, pad_w)


def compute_indices_pallas(red: torch.Tensor,green: torch.Tensor,nir: torch.Tensor,block_shape: Tuple[int, int] = (128, 128),eps: float = 1e-6,) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute fused NDVI and NDWI on TPU using a 2D-tiled Pallas kernel.

    Args:
        red:   2D contiguous tensor of shape (H, W) on TPU (float32 or bfloat16).
        green: 2D contiguous tensor of shape (H, W) on TPU.
        nir:   2D contiguous tensor of shape (H, W) on TPU.
        block_shape: (block_h, block_w) tile dimensions for SRAM. Must align with
            TPU vector lanes (multiples of 8 and 128 recommended).
        eps: Small constant to prevent division by zero.

    Returns:
        Tuple of (ndvi, ndwi) tensors with the same shape and dtype as inputs.
    """
    if red.ndim != 2 or green.ndim != 2 or nir.ndim != 2:
        raise ValueError("All input bands must be 2D tensors of shape (H, W).")
    if not (red.shape == green.shape == nir.shape):
        raise ValueError("All input bands must have identical shapes.")
    if red.dtype not in _TORCH_TO_JAX_DTYPE:
        raise TypeError(
            f"Unsupported dtype {red.dtype}. Supported: {list(_TORCH_TO_JAX_DTYPE.keys())}"
        )

    orig_h, orig_w   = red.shape
    block_h, block_w = block_shape

    # Ensure contiguity to avoid expensive view relayouts on TPU
    red_c, _   = _pad_to_block_multiple(red.contiguous(), block_h, block_w)
    green_c, _ = _pad_to_block_multiple(green.contiguous(), block_h, block_w)
    nir_c, _   = _pad_to_block_multiple(nir.contiguous(), block_h, block_w)

    padded_h, padded_w = red_c.shape
    grid = (padded_h // block_h, padded_w // block_w)

    jax_dtype = _TORCH_TO_JAX_DTYPE[red.dtype]
    out_shape = [
        jax.ShapeDtypeStruct((padded_h, padded_w), jax_dtype),
        jax.ShapeDtypeStruct((padded_h, padded_w), jax_dtype),
    ]

    block_spec = pl.BlockSpec(
        block_shape=(block_h, block_w),
        index_map=lambda i, j: (i, j),
    )

    kernel_fn = functools.partial(_fused_spectral_indices_2d_kernel, eps=eps)

    fused_call = tpu_pallas.pallas_call(
        kernel_fn,
        out_shape=out_shape,
        in_specs=[block_spec, block_spec, block_spec],
        out_specs=[block_spec, block_spec],
        grid=grid,
    )

    ndvi_out, ndwi_out = fused_call(red_c, green_c, nir_c)

    # Slice back to original dimensions if padding was applied
    if padded_h != orig_h or padded_w != orig_w:
        ndvi_out = ndvi_out[:orig_h, :orig_w]
        ndwi_out = ndwi_out[:orig_h, :orig_w]

    return ndvi_out, ndwi_out