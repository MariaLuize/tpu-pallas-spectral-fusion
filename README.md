# Fused Spectral Index Computation with Custom Pallas Kernels on TorchTPU

Accelerating high-resolution multispectral remote sensing pipelines (Sentinel-2 10m tiles) on Cloud TPUs by fusing NDVI and NDWI computations into a single 2D-tiled Pallas kernel via `torch.tpu.pallas`.

## Why This Project Exists

Computing spectral indices on full Sentinel-2 tiles (`10980 x 10980` pixels) is heavily **memory-bandwidth bound**. In standard eager PyTorch execution, element-wise operations allocate intermediate tensors and trigger multiple HBM (High Bandwidth Memory) read/write round-trips for very little arithmetic.

This project fuses the computation of **NDVI** and **NDWI** into a single pass:
1. Loads 2D tiles of Red (B04), Green (B03), and NIR (B08) bands from HBM into TPU VMEM (SRAM).
2. Upcasts to `float32` inside the Vector Processing Unit (VPU) for numerical stability when inputs are `bfloat16`.
3. Computes both indices simultaneously and writes the results back to HBM in a single pass (theoretical minimum: 3 reads + 2 writes).

---

## Repository Structure

```text
tpu-pallas-spectral-fusion/
├── README.md
├── requirements.txt                # Pinned dependencies for Cloud TPU VM
├── src/
│   ├── baseline.py                 # PyTorch Eager & torch.compile(backend="tpu") baselines
│   ├── pallas_kernel.py            # Custom 2D-tiled Pallas kernel with auto-padding & BF16 support
│   └── data_utils.py               # Synthetic band generator & real Sentinel-2 GeoTIFF/JP2 I/O (rasterio)
├── tests/
│   └── test_numerical.py           # Pytest suite validating FP32, BF16, padding, and zero-division stability
├── benchmarks/
│   ├── bench_throughput.py         # Latency (ms) & effective HBM bandwidth (GB/s) benchmark + CSV export
│   ├── plot_results.py             # Generates HBM bandwidth, latency/speedup, and roofline charts from CSV
│   └── profile_xprof.py            # torch.profiler script generating XProf traces per execution mode
├── notebooks/
│   └── tutorial_pallas_tpu.ipynb   # Step-by-step tutorial notebook
└── results/                        # Generated CSV benchmarks and XProf traces (gitignored)
```

---

## Key Technical Decisions

* **2D Tiling & Alignment:** Default `block_shape=(512, 512)` (and `256x1024` for wide contiguous DMA bursts) aligns with TPU vector lanes (`8x128` sublanes) while fitting double-buffered inside TPU v5e's 16 MB VMEM. Non-multiple spatial dimensions (e.g., `10980 x 10980`) are handled natively on boundary DMAs via `pl.cdiv` with zero extra HBM padding copies.
* **Mixed Precision (`bfloat16`):** Inputs and outputs can stay in `bfloat16` to halve HBM traffic, while division is performed in `float32` inside SRAM (`eps=1e-6`) to prevent precision loss and `NaN`/`Inf` on nodata pixels.
* **Real Satellite Data Support:** `src/data_utils.py` handles Sentinel-2 L2A `uint16` reflectance scaling (`scale_factor=10000.0`), optional Processing Baseline 04.00+ radiometric `offset` (`-1000.0`), windowed reads, and compressed tiled GeoTIFF export.

---

## Quickstart (Cloud TPU VM)

### 1. Run Numerical Equivalence Tests

Validates the Pallas kernel against Eager and `torch.compile(backend="tpu")` across `float32`, `bfloat16`, padded shapes, and all-zero edge cases:

```bash
pytest -v tests/test_numerical.py
```

### 2. Run Throughput & Effective Bandwidth Benchmark

Sweeps image shapes, dtypes, and Pallas tile sizes, printing a Markdown table and saving a CSV:

```bash
python -m benchmarks.bench_throughput \
  --shapes 4096x4096 8192x8192 10980x10980 \
  --dtypes float32 bfloat16 \
  --block-shapes 128x128 256x256 512x512 \
  --output-csv results/throughput_v6e.csv
```

Effective HBM bandwidth is computed against the fused theoretical minimum (3 reads + 2 writes):
`Effective BW (GB/s) = (5 * H * W * sizeof(dtype) / 1e9) / (median_ms / 1e3)`

### 3. Capture XProf Traces

Generates separate TensorBoard/XProf runs for `eager`, `compiled`, and `pallas`:

```bash
python -m benchmarks.profile_xprof \
  --shape 10980x10980 \
  --dtype bfloat16 \
  --block-shape 256x256 \
  --logdir results/xprof_traces
```

Launch TensorBoard on the TPU VM and port-forward `6006` locally:

```bash
tensorboard --logdir results/xprof_traces --port 6006 --bind_all
```

---

## Benchmark Results (Cloud TPU v5e, `v5litepod-1`)

Measured on a single-chip Cloud TPU v5e (`v5litepod-1`, 819 GB/s theoretical peak HBM bandwidth) using `torch_tpu` (`torch 2.15.0`) and `jax 0.10.2`:

| Shape | Dtype | Method | Median (ms) | Std (ms) | Effective BW (GB/s) | Speedup vs Eager |
|---|---|---|---|---|---|---|
| 4096x4096 | float32 | Eager | 2.360 | 0.017 | 142.2 | 1.00x |
| 4096x4096 | float32 | torch.compile | 0.799 | 0.010 | 419.8 | 2.95x |
| 4096x4096 | float32 | Pallas (256x256) | 0.879 | 0.031 | 381.9 | 2.69x |
| 4096x4096 | float32 | Pallas (512x512) | 0.863 | 0.034 | 388.9 | 2.74x |
| 4096x4096 | float32 | Pallas (256x1024) | 0.868 | 0.046 | 386.7 | 2.72x |
| 8192x8192 | float32 | Eager | 8.950 | 0.031 | 150.0 | 1.00x |
| 8192x8192 | float32 | torch.compile | 2.313 | 0.008 | 580.3 | 3.87x |
| 8192x8192 | float32 | Pallas (256x256) | 2.422 | 0.041 | 554.3 | 3.70x |
| 8192x8192 | float32 | Pallas (512x512) | 2.347 | 0.055 | 571.8 | 3.81x |
| 8192x8192 | float32 | Pallas (256x1024) | 2.368 | 0.041 | 566.8 | 3.78x |
| 10980x10980 | float32 | Eager | 16.037 | 0.014 | 150.4 | 1.00x |
| 10980x10980 | float32 | torch.compile | 3.999 | 0.019 | 602.9 | 4.01x |
| 10980x10980 | float32 | Pallas (256x256) | 4.166 | 0.027 | 578.7 | 3.85x |
| 10980x10980 | float32 | Pallas (512x512) | 3.973 | 0.039 | 606.9 | 4.04x |
| 10980x10980 | float32 | Pallas (256x1024) | 3.931 | 0.029 | 613.4 | 4.08x |
| 4096x4096 | bfloat16 | Eager | 3.149 | 0.010 | 53.3 | 1.00x |
| 4096x4096 | bfloat16 | torch.compile | 0.553 | 0.007 | 303.3 | 5.69x |
| 4096x4096 | bfloat16 | Pallas (256x256) | 0.675 | 0.026 | 248.4 | 4.66x |
| 4096x4096 | bfloat16 | Pallas (512x512) | 0.620 | 0.026 | 270.6 | 5.08x |
| 4096x4096 | bfloat16 | Pallas (256x1024) | 0.627 | 0.025 | 267.6 | 5.02x |
| 8192x8192 | bfloat16 | Eager | 12.107 | 0.010 | 55.4 | 1.00x |
| 8192x8192 | bfloat16 | torch.compile | 1.291 | 0.011 | 519.7 | 9.37x |
| 8192x8192 | bfloat16 | Pallas (256x256) | 1.670 | 0.023 | 402.0 | 7.25x |
| 8192x8192 | bfloat16 | Pallas (512x512) | 1.370 | 0.032 | 489.9 | 8.84x |
| 8192x8192 | bfloat16 | Pallas (256x1024) | 1.370 | 0.023 | 490.0 | 8.84x |
| 10980x10980 | bfloat16 | Eager | 21.712 | 0.011 | 55.5 | 1.00x |
| 10980x10980 | bfloat16 | torch.compile | 2.102 | 0.008 | 573.5 | 10.33x |
| 10980x10980 | bfloat16 | Pallas (256x256) | 2.738 | 0.035 | 440.4 | 7.93x |
| 10980x10980 | bfloat16 | Pallas (512x512) | 2.239 | 0.050 | 538.5 | 9.70x |
| 10980x10980 | bfloat16 | Pallas (256x1024) | 2.175 | 0.062 | 554.3 | 9.98x |

### Key Takeaways
* **Beats `torch.compile` on full Sentinel-2 `float32` tiles (`10980x10980`):** Pallas `(256x1024)` achieves **3.931 ms** (**613.4 GB/s**, ~75% of v5e's 819 GB/s peak HBM, **4.08x** over Eager), edging out `torch.compile` (`3.999 ms`, `602.9 GB/s`).
* **~10x speedup in `bfloat16` (`10980x10980`):** Pallas `(256x1024)` drops latency from **21.712 ms** (Eager, `55.5 GB/s`) to **2.175 ms** (`554.3 GB/s`, **9.98x** faster).
* **Eliminating eager `F.pad` doubled `10980x10980` Pallas throughput:** Switching from eager `F.pad` + output slicing (`8.358 ms` fp32 / `4.329 ms` bf16) to native Mosaic TPU boundary tiling via `pl.cdiv` + `dimension_semantics=("parallel", "parallel")` cut latency by **>2x** (`3.931 ms` fp32 / `2.175 ms` bf16).
* **VMEM budget & tile aspect ratio:** `512x512` and `256x1024` (10 MB double-buffered in `float32`) fit inside TPU v5e's 16 MB VMEM and maximize DMA burst efficiency, while `512x1024` in `float32` (20 MB double-buffered) exceeds v5e VMEM.

---

## Roadmap & Next Steps (Context for Development)

* [x] Add `requirements.txt` and `.gitignore` (ignore `results/`, `__pycache__/`, `.pytest_cache/`).
* [x] Verify exact `torch.tpu.pallas` import path and `jax_op` / `pallas_call` signature against the sprint's TorchTPU build on a Cloud TPU v5e VM.
* [x] Run benchmarks on TPU VM and populate the results table in this README.
* [x] Add a plotting script (`benchmarks/plot_results.py`) to turn `results/*.csv` into roofline / bandwidth comparison charts for the blog post.
* [x] Build `notebooks/tutorial_pallas_tpu.ipynb` importing directly from `src/`.
* [x] Draft the companion technical blog post using the benchmark numbers and XProf Trace Viewer screenshots.
