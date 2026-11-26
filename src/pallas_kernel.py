import functools
from typing import Tuple

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import torch
import torch_tpu  # Registers the native 'tpu' backend and torch.tpu.pallas


_SUPPORTED_DTYPES = (torch.float32, torch.bfloat16)


def _fused_spectral_indices_2d_kernel(
    red_ref,
    green_ref,
    nir_ref,
    out_ndvi_ref,
    out_ndwi_ref,
    *,
    eps: float,
):
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


@torch.tpu.pallas.jax_op("spectral::fused_ndvi_ndwi")
def _fused_spectral_indices_op(
    red: jax.Array,
    green: jax.Array,
    nir: jax.Array,
    block_h: int = 512,
    block_w: int = 512,
    eps: float = 1e-6,
) -> tuple[jax.Array, jax.Array]:
    """TorchTPU custom op wrapping the 2D-tiled Pallas spectral index kernel.

    Uses `pl.cdiv` so Mosaic TPU handles non-multiple spatial dimensions
    (such as Sentinel-2 10980x10980 tiles) directly on boundary DMAs without
    allocating extra padded HBM tensors.
    """
    h, w = red.shape
    grid = (pl.cdiv(h, block_h), pl.cdiv(w, block_w))

    out_shape = (
        jax.ShapeDtypeStruct((h, w), red.dtype),
        jax.ShapeDtypeStruct((h, w), red.dtype),
    )

    block_spec = pl.BlockSpec(
        block_shape=(block_h, block_w),
        index_map=lambda i, j: (i, j),
    )

    kernel_fn = functools.partial(_fused_spectral_indices_2d_kernel, eps=eps)
    compiler_params = (
        pltpu.CompilerParams(dimension_semantics=("parallel", "parallel"))
        if hasattr(pltpu, "CompilerParams")
        else None
    )

    ndvi_out, ndwi_out = pl.pallas_call(
        kernel_fn,
        out_shape=out_shape,
        in_specs=[block_spec, block_spec, block_spec],
        out_specs=(block_spec, block_spec),
        grid=grid,
        compiler_params=compiler_params,
    )(red, green, nir)

    return ndvi_out, ndwi_out


def compute_indices_pallas(
    red: torch.Tensor,
    green: torch.Tensor,
    nir: torch.Tensor,
    block_shape: Tuple[int, int] = (512, 512),
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
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
    if red.dtype not in _SUPPORTED_DTYPES:
        raise TypeError(
            f"Unsupported dtype {red.dtype}. Supported: {list(_SUPPORTED_DTYPES)}"
        )

    block_h, block_w = block_shape

    # Ensure contiguity to avoid expensive view relayouts on TPU
    red_c = red.contiguous()
    green_c = green.contiguous()
    nir_c = nir.contiguous()

    return _fused_spectral_indices_op(red_c, green_c, nir_c, block_h, block_w, eps)
