import pytest
import torch

from src.baseline import compute_indices_eager, get_compiled_indices_fn
from src.pallas_kernel import compute_indices_pallas


DEVICE = torch.device("tpu")


def _generate_synthetic_bands(
    shape: tuple[int, int],
    dtype: torch.dtype,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate synthetic reflectance bands in [0.0, 1.0] on TPU."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    red = torch.rand(shape, generator=gen, dtype=dtype).to(DEVICE)
    green = torch.rand(shape, generator=gen, dtype=dtype).to(DEVICE)
    nir = torch.rand(shape, generator=gen, dtype=dtype).to(DEVICE)
    return red, green, nir


@pytest.mark.parametrize(
    "shape,block_shape",
    [
        ((1024, 1024), (128, 128)),  # Exact multiple of block_shape
        ((1098, 1098), (128, 128)),  # Sentinel-2 style non-multiple (requires padding)
        ((500, 750), (128, 256)),    # Asymmetric shape and block_shape
    ],
)
@pytest.mark.parametrize(
    "dtype,rtol,atol",
    [
        (torch.float32, 1e-5, 1e-6),
        (torch.bfloat16, 1e-2, 1e-2),
    ],
)
def test_pallas_matches_eager(shape: tuple[int, int],block_shape: tuple[int, int],dtype: torch.dtype,rtol: float,atol: float,):
    """Verify Pallas kernel output matches eager baseline within dtype tolerances."""
    red, green, nir = _generate_synthetic_bands(shape, dtype)

    ref_ndvi, ref_ndwi = compute_indices_eager(red, green, nir)
    pal_ndvi, pal_ndwi = compute_indices_pallas(
        red, green, nir, block_shape=block_shape
    )

    assert pal_ndvi.shape == shape
    assert pal_ndwi.shape == shape
    assert pal_ndvi.dtype == dtype
    assert pal_ndwi.dtype == dtype

    torch.testing.assert_close(pal_ndvi.cpu(), ref_ndvi.cpu(), rtol=rtol, atol=atol)
    torch.testing.assert_close(pal_ndwi.cpu(), ref_ndwi.cpu(), rtol=rtol, atol=atol)


def test_pallas_matches_compiled():
    """Verify Pallas kernel matches torch.compile(backend='tpu') in bfloat16."""
    shape = (1098, 1098)
    red, green, nir = _generate_synthetic_bands(shape, torch.bfloat16)

    compiled_fn = get_compiled_indices_fn()
    comp_ndvi, comp_ndwi = compiled_fn(red, green, nir)
    pal_ndvi, pal_ndwi = compute_indices_pallas(
        red, green, nir, block_shape=(128, 128)
    )

    torch.testing.assert_close(pal_ndvi.cpu(), comp_ndvi.cpu(), rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(pal_ndwi.cpu(), comp_ndwi.cpu(), rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_zero_reflectance_stability(dtype: torch.dtype):
    """Ensure all-zero inputs (e.g., nodata/masked pixels) do not produce NaN or Inf."""
    shape = (300, 300)  # Forces padding + zero-division edge case
    zeros = torch.zeros(shape, dtype=dtype, device=DEVICE)

    ndvi, ndwi = compute_indices_pallas(zeros, zeros, zeros, block_shape=(128, 128))

    ndvi_cpu = ndvi.cpu()
    ndwi_cpu = ndwi.cpu()

    assert torch.isfinite(ndvi_cpu).all(), "NDVI produced NaN or Inf on zero input"
    assert torch.isfinite(ndwi_cpu).all(), "NDWI produced NaN or Inf on zero input"
    assert torch.all(ndvi_cpu == 0.0)
    assert torch.all(ndwi_cpu == 0.0)