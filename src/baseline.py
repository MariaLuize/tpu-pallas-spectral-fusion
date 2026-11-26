from typing import Tuple
import torch

try:
    import torch_tpu  # noqa: F401
except ImportError:
    pass


def compute_indices_eager(red: torch.Tensor,green: torch.Tensor,nir: torch.Tensor,eps: float = 1e-6,) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute NDVI and NDWI using standard PyTorch ops on TPU.

    Upcasts to float32 during division for numerical stability when inputs
    are bfloat16, then casts back to the original dtype.

    Args:
        red: 2D tensor of shape (H, W) on TPU.
        green: 2D tensor of shape (H, W) on TPU.
        nir: 2D tensor of shape (H, W) on TPU.
        eps: Small constant to prevent division by zero.

    Returns:
        Tuple of (ndvi, ndwi) tensors with the same shape and dtype as inputs.
    """
    orig_dtype = red.dtype

    red_f32   = red.to(torch.float32)
    green_f32 = green.to(torch.float32)
    nir_f32   = nir.to(torch.float32)

    ndvi = (nir_f32 - red_f32) / (nir_f32 + red_f32 + eps)
    ndwi = (green_f32 - nir_f32) / (green_f32 + nir_f32 + eps)

    return ndvi.to(orig_dtype), ndwi.to(orig_dtype)


def get_compiled_indices_fn():
    """Return a torch.compile(backend='tpu') version of the baseline."""
    return torch.compile(compute_indices_eager, backend="tpu")