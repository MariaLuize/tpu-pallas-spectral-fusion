"""Benchmark latency and effective HBM bandwidth (GB/s) on TorchTPU.

Compares:
1. PyTorch Eager mode on TPU
2. torch.compile(backend="tpu")
3. Custom 2D-tiled Pallas kernel (across configurable block shapes)
"""

import argparse
import csv
from pathlib import Path
import statistics
import time
from typing import Callable, List, Tuple

import torch
import torch_tpu  # noqa: F401

from src.baseline import compute_indices_eager, get_compiled_indices_fn
from src.pallas_kernel import compute_indices_pallas


DEVICE = torch.device("tpu")


def _sync_tpu() -> None:
    """Synchronize TPU device execution with host."""
    if hasattr(torch, "accelerator") and torch.accelerator.is_available():
        torch.accelerator.synchronize()
    elif hasattr(torch, "tpu") and hasattr(torch.tpu, "synchronize"):
        torch.tpu.synchronize()
    else:
        # Fallback for environments using torch_xla directly
        import torch_xla
        torch_xla.sync()


def _parse_shapes(raw_shapes: List[str]) -> List[Tuple[int, int]]:
    shapes = []
    for item in raw_shapes:
        h, w = item.lower().split("x")
        shapes.append((int(h), int(w)))
    return shapes


def _benchmark_op(
    fn: Callable,
    inputs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    warmup: int,
    iters: int,
) -> Tuple[float, float]:
    """Run warmup and timed iterations, returning (median_ms, std_ms)."""
    # Warmup (triggers JIT/XLA compilation and caches executables)
    for _ in range(warmup):
        _ = fn(*inputs)
    _sync_tpu()

    latencies_ms = []
    for _ in range(iters):
        t0 = time.perf_counter()
        _ = fn(*inputs)
        _sync_tpu()
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000.0)

    median_ms = statistics.median(latencies_ms)
    std_ms = statistics.stdev(latencies_ms) if len(latencies_ms) > 1 else 0.0
    return median_ms, std_ms


def _effective_bandwidth_gb_s(
    shape: Tuple[int, int],
    dtype: torch.dtype,
    latency_ms: float,
) -> float:
    """Compute effective HBM bandwidth assuming 3 reads + 2 writes."""
    h, w = shape
    elem_bytes = torch.tensor([], dtype=dtype).element_size()
    total_bytes = 5 * h * w * elem_bytes
    return (total_bytes / 1e9) / (latency_ms / 1e3)


def run_benchmarks(
    shapes: List[Tuple[int, int]],
    dtypes: List[torch.dtype],
    block_shapes: List[Tuple[int, int]],
    warmup: int,
    iters: int,
) -> List[dict]:
    """Execute the full benchmark matrix and return structured rows."""
    compiled_fn = get_compiled_indices_fn()
    results = []

    for dtype in dtypes:
        dtype_name = str(dtype).replace("torch.", "")

        for shape in shapes:
            h, w = shape
            red = torch.rand((h, w), dtype=dtype, device=DEVICE)
            green = torch.rand((h, w), dtype=dtype, device=DEVICE)
            nir = torch.rand((h, w), dtype=dtype, device=DEVICE)
            _sync_tpu()
            inputs = (red, green, nir)

            configs = [
                ("Eager", compute_indices_eager),
                ("torch.compile", compiled_fn),
            ]
            for bh, bw in block_shapes:
                configs.append(
                    (
                        f"Pallas ({bh}x{bw})",
                        lambda r, g, n, bs=(bh, bw): compute_indices_pallas(
                            r, g, n, block_shape=bs
                        ),
                    )
                )

            eager_ms = None
            for name, fn in configs:
                median_ms, std_ms = _benchmark_op(fn, inputs, warmup, iters)
                bw_gb_s = _effective_bandwidth_gb_s(shape, dtype, median_ms)

                if name == "Eager":
                    eager_ms = median_ms
                speedup = (eager_ms / median_ms) if eager_ms else 1.0

                results.append(
                    {
                        "shape": f"{h}x{w}",
                        "dtype": dtype_name,
                        "method": name,
                        "median_ms": round(median_ms, 3),
                        "std_ms": round(std_ms, 3),
                        "bandwidth_gb_s": round(bw_gb_s, 1),
                        "speedup_vs_eager": round(speedup, 2),
                    }
                )

    return results


def _print_markdown_table(rows: List[dict]) -> None:
    header = (
        "| Shape | Dtype | Method | Median (ms) | Std (ms) | "
        "Effective BW (GB/s) | Speedup vs Eager |"
    )
    sep = "|---|---|---|---|---|---|---|"
    print("\n" + header)
    print(sep)
    for r in rows:
        print(
            f"| {r['shape']} | {r['dtype']} | {r['method']} | "
            f"{r['median_ms']:.3f} | {r['std_ms']:.3f} | "
            f"{r['bandwidth_gb_s']:.1f} | {r['speedup_vs_eager']:.2f}x |"
        )
    print()


def _save_csv(rows: List[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Results saved to {path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark TorchTPU spectral index fusion."
    )
    parser.add_argument(
        "--shapes",
        nargs="+",
        default=["4096x4096", "8192x8192", "10980x10980"],
        help="2D image shapes (e.g., 4096x4096 10980x10980 for Sentinel-2 10m).",
    )
    parser.add_argument(
        "--dtypes",
        nargs="+",
        choices=["float32", "bfloat16"],
        default=["float32", "bfloat16"],
        help="Tensor dtypes to benchmark.",
    )
    parser.add_argument(
        "--block-shapes",
        nargs="+",
        default=["128x128", "256x256", "512x512"],
        help="Pallas 2D tile sizes to sweep.",
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Optional path to save benchmark results as CSV.",
    )
    args = parser.parse_args()

    dtype_map = {"float32": torch.float32, "bfloat16": torch.bfloat16}
    shapes = _parse_shapes(args.shapes)
    block_shapes = _parse_shapes(args.block_shapes)
    dtypes = [dtype_map[d] for d in args.dtypes]

    rows = run_benchmarks(
        shapes=shapes,
        dtypes=dtypes,
        block_shapes=block_shapes,
        warmup=args.warmup,
        iters=args.iters,
    )
    _print_markdown_table(rows)

    if args.output_csv:
        _save_csv(rows, args.output_csv)


if __name__ == "__main__":
    main()