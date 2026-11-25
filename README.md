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

* **2D Tiling & Alignment:** Default `block_shape=(128, 128)` aligns with TPU vector lanes (`8x128` sublanes). Non-multiple spatial dimensions (e.g., `10980 x 10980`) are automatically zero-padded before the kernel call and sliced back to `(H, W)` afterward.
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

## Roadmap & Next Steps (Context for Development)

* [x] Add `requirements.txt` and `.gitignore` (ignore `results/`, `__pycache__/`, `.pytest_cache/`).
* [ ] Verify exact `torch.tpu.pallas` import path and `pallas_call` signature against the sprint's TorchTPU build on a Cloud TPU v6e/v5e VM.
* [ ] Run benchmarks on TPU VM and populate the results table in this README.
* [x] Add a plotting script (`benchmarks/plot_results.py`) to turn `results/*.csv` into roofline / bandwidth comparison charts for the blog post.
* [x] Build `notebooks/tutorial_pallas_tpu.ipynb` importing directly from `src/`.
* [ ] Draft the companion technical blog post using the benchmark numbers and XProf Trace Viewer screenshots.
