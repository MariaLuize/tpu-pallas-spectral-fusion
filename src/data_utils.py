from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import torch


SENTINEL2_10M_SHAPE: Tuple[int, int] = (10980, 10980)
DEFAULT_SCALE_FACTOR: float = 10000.0
PB0400_RADIO_ADD_OFFSET: float = -1000.0

WindowLike = Union[Tuple[int, int, int, int], "rasterio.windows.Window"]


def _require_rasterio():
    try:
        import rasterio
        from rasterio.windows import Window
    except ImportError as exc:
        raise ImportError(
            "rasterio is required for Sentinel-2 GeoTIFF/JP2 I/O. "
            "Install it with: pip install rasterio"
        ) from exc
    return rasterio, Window


def generate_synthetic_bands(
    shape: Tuple[int, int] = SENTINEL2_10M_SHAPE,
    dtype: torch.dtype = torch.float32,
    device: Union[str, torch.device] = "tpu",
    seed: int = 42,
    realistic: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate synthetic Red (B04), Green (B03), and NIR (B08) reflectance bands.

    Values are generated in [0.0, 1.0] on CPU with a deterministic generator and
    transferred to the target device as contiguous 2D tensors.

    Args:
        shape: Spatial dimensions (H, W). Defaults to a full Sentinel-2 10m tile.
        dtype: Target tensor dtype (e.g., torch.float32 or torch.bfloat16).
        device: Target PyTorch device (e.g., 'tpu', 'cpu').
        seed: Random seed for reproducibility.
        realistic: If True, synthesize spatially coherent vegetation/water/soil
            patterns rather than i.i.d. uniform noise (useful for visual demos).

    Returns:
        Tuple of contiguous 2D tensors (red, green, nir) of shape (H, W).
    """
    h, w = shape
    gen = torch.Generator(device="cpu").manual_seed(seed)

    if not realistic:
        red = torch.rand((h, w), generator=gen, dtype=dtype).to(device).contiguous()
        green = torch.rand((h, w), generator=gen, dtype=dtype).to(device).contiguous()
        nir = torch.rand((h, w), generator=gen, dtype=dtype).to(device).contiguous()
        return red, green, nir

    y = torch.linspace(-1.0, 1.0, h, dtype=torch.float32).unsqueeze(1)
    x = torch.linspace(-1.0, 1.0, w, dtype=torch.float32).unsqueeze(0)

    # Smooth spatial masks for vegetation (high NIR, low Red) and water (high Green, low NIR)
    veg_mask = torch.sigmoid(4.0 * (torch.sin(3.0 * x) * torch.cos(3.0 * y) + 0.2))
    water_mask = torch.sigmoid(6.0 * (-(x + 0.35) ** 2 - (y - 0.2) ** 2 + 0.18))

    noise_r = 0.03 * torch.randn((h, w), generator=gen, dtype=torch.float32)
    noise_g = 0.03 * torch.randn((h, w), generator=gen, dtype=torch.float32)
    noise_n = 0.03 * torch.randn((h, w), generator=gen, dtype=torch.float32)

    red_f32 = (0.28 - 0.20 * veg_mask - 0.15 * water_mask + noise_r).clamp(0.0, 1.0)
    green_f32 = (0.24 - 0.10 * veg_mask + 0.18 * water_mask + noise_g).clamp(0.0, 1.0)
    nir_f32 = (0.26 + 0.48 * veg_mask - 0.23 * water_mask + noise_n).clamp(0.0, 1.0)

    red = red_f32.to(dtype=dtype, device=device).contiguous()
    green = green_f32.to(dtype=dtype, device=device).contiguous()
    nir = nir_f32.to(dtype=dtype, device=device).contiguous()
    return red, green, nir


def _read_single_band(
    path: Union[str, Path],*,scale_factor: Optional[float],offset: float,window: Optional[WindowLike],clip_range: Optional[Tuple[float, float]],
) -> Tuple[np.ndarray, dict]:
    """Read a single raster band, apply Sentinel-2 L2A scaling, and return (array, profile)."""
    rasterio, Window = _require_rasterio()

    win = None
    if window is not None:
        if isinstance(window, Window):
            win = window
        else:
            col_off, row_off, width, height = window
            win = Window(col_off=col_off, row_off=row_off, width=width, height=height)

    with rasterio.open(path) as src:
        raw = src.read(1, window=win)
        nodata = src.nodata
        profile = src.profile.copy()

        if win is not None:
            profile.update(
                height=raw.shape[0],
                width=raw.shape[1],
                transform=src.window_transform(win),
            )

    nodata_mask = (raw == nodata) if nodata is not None else None
    arr = raw.astype(np.float32)

    if scale_factor is not None and np.issubdtype(raw.dtype, np.integer):
        arr = (arr + float(offset)) / float(scale_factor)

    if clip_range is not None:
        arr = np.clip(arr, clip_range[0], clip_range[1])

    if nodata_mask is not None:
        arr[nodata_mask] = 0.0

    return np.ascontiguousarray(arr, dtype=np.float32), profile


def load_sentinel2_bands(
    red_path: Union[str, Path],
    green_path: Union[str, Path],
    nir_path: Union[str, Path],
    *,
    dtype: torch.dtype = torch.float32,
    device: Union[str, torch.device] = "tpu",
    scale_factor: Optional[float] = DEFAULT_SCALE_FACTOR,
    offset: float = 0.0,
    window: Optional[WindowLike] = None,
    clip_range: Optional[Tuple[float, float]] = (0.0, 1.0),
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    """Load Sentinel-2 Red (B04), Green (B03), and NIR (B08) GeoTIFF/JP2 bands.

    Handles Sentinel-2 L2A uint16 reflectance scaling:
        reflectance = (DN + offset) / scale_factor
    For Processing Baseline 04.00+ tiles (post-Jan 2022), pass
    `offset=-1000.0` (or `PB0400_RADIO_ADD_OFFSET`).

    Args:
        red_path: Path to Red band (B04) raster (.jp2 or .tif).
        green_path: Path to Green band (B03) raster (.jp2 or .tif).
        nir_path: Path to NIR band (B08) raster (.jp2 or .tif).
        dtype: Target PyTorch tensor dtype (e.g., torch.float32 or torch.bfloat16).
        device: Target PyTorch device (e.g., 'tpu', 'cpu').
        scale_factor: Reflectance quantification value (default 10000.0).
            Applied automatically when input rasters have integer dtypes; set to
            None to disable scaling.
        offset: Radiometric offset added before dividing by `scale_factor`
            (use -1000.0 for Sentinel-2 Processing Baseline 04.00+).
        window: Optional spatial crop as `(col_off, row_off, width, height)` or
            a `rasterio.windows.Window`.
        clip_range: Optional `(min_val, max_val)` clamp applied after scaling.

    Returns:
        Tuple of `(red, green, nir, profile)` where the first three elements are
        contiguous 2D tensors on `device` and `profile` is the rasterio metadata
        dict (updated for `window` if provided).
    """
    red_np, profile = _read_single_band(
        red_path,
        scale_factor=scale_factor,
        offset=offset,
        window=window,
        clip_range=clip_range,
    )
    green_np, _ = _read_single_band(
        green_path,
        scale_factor=scale_factor,
        offset=offset,
        window=window,
        clip_range=clip_range,
    )
    nir_np, _ = _read_single_band(
        nir_path,
        scale_factor=scale_factor,
        offset=offset,
        window=window,
        clip_range=clip_range,
    )

    if not (red_np.shape == green_np.shape == nir_np.shape):
        raise ValueError(
            "Loaded bands have mismatched shapes: "
            f"red={red_np.shape}, green={green_np.shape}, nir={nir_np.shape}."
        )

    red = torch.from_numpy(red_np).to(dtype=dtype, device=device).contiguous()
    green = torch.from_numpy(green_np).to(dtype=dtype, device=device).contiguous()
    nir = torch.from_numpy(nir_np).to(dtype=dtype, device=device).contiguous()

    return red, green, nir, profile


def save_index_geotiff(
    index_tensor: torch.Tensor,
    output_path: Union[str, Path],
    profile: dict,
    *,
    compress: str = "deflate",
    tiled: bool = True,
    blockxsize: int = 256,
    blockysize: int = 256,
    nodata: Optional[float] = None,
) -> Path:
    """Export a 2D spectral index tensor (NDVI or NDWI) as a compressed tiled GeoTIFF.

    Args:
        index_tensor: 2D tensor of shape (H, W) on TPU/CPU/GPU.
        output_path: Destination `.tif` file path.
        profile: Rasterio profile dict (e.g., returned by `load_sentinel2_bands`).
        compress: GeoTIFF compression codec (default: 'deflate').
        tiled: Whether to write internal GeoTIFF tiles (default: True).
        blockxsize: Internal tile width when `tiled=True` (must be multiple of 16).
        blockysize: Internal tile height when `tiled=True` (must be multiple of 16).
        nodata: Optional nodata value to record in the output metadata.

    Returns:
        Resolved `Path` to the written GeoTIFF file.
    """
    if index_tensor.ndim != 2:
        raise ValueError(
            f"Expected a 2D index tensor of shape (H, W), got shape {tuple(index_tensor.shape)}."
        )

    rasterio, _ = _require_rasterio()

    arr = index_tensor.detach().to(device="cpu", dtype=torch.float32).numpy()
    h, w = arr.shape

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    out_profile = profile.copy()
    out_profile.update(
        driver="GTiff",
        dtype="float32",
        count=1,
        height=h,
        width=w,
        compress=compress,
        tiled=tiled,
        nodata=nodata,
        BIGTIFF="IF_SAFER",
    )
    if tiled:
        out_profile.update(blockxsize=blockxsize, blockysize=blockysize)
    else:
        out_profile.pop("blockxsize", None)
        out_profile.pop("blockysize", None)

    # Remove driver-specific creation options that may conflict with GTiff (e.g., JP2)
    for key in ("photometric", "quality", "reversible", "interleave"):
        out_profile.pop(key, None)

    with rasterio.open(out_path, "w", **out_profile) as dst:
        dst.write(arr, 1)

    return out_path
