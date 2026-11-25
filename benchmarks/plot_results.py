"""Generate blog-ready bandwidth, latency/speedup, and roofline charts from benchmark CSVs.

Expected CSV schema (produced by `benchmarks/bench_throughput.py`):
    shape, dtype, method, median_ms, std_ms, bandwidth_gb_s, speedup_vs_eager
"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")  # Headless backend for Cloud TPU VMs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Reference per-chip HBM bandwidth (GB/s) and VPU/compute ceilings (GFLOP/s)
TPU_PRESETS: Dict[str, Dict[str, float]] = {
    "v6e": {"peak_hbm_gb_s": 1600.0, "peak_gflops": 918000.0},
    "v5e": {"peak_hbm_gb_s": 819.0, "peak_gflops": 197000.0},
    "v5p": {"peak_hbm_gb_s": 2765.0, "peak_gflops": 459000.0},
    "v4": {"peak_hbm_gb_s": 1200.0, "peak_gflops": 275000.0},
}

# NDVI: (nir - red) / (nir + red + eps) -> 4 FLOPs
# NDWI: (green - nir) / (green + nir + eps) -> 4 FLOPs
FLOPS_PER_PIXEL: int = 8
DTYPE_BYTES: Dict[str, int] = {"float32": 4, "bfloat16": 2}


def _parse_shape_pixels(shape_str: str) -> int:
    h, w = shape_str.lower().split("x")
    return int(h) * int(w)


def _method_order(methods: List[str]) -> List[str]:
    """Keep Eager, torch.compile, then Pallas tile sizes in natural order."""
    priority = {"Eager": 0, "torch.compile": 1}
    return sorted(methods, key=lambda m: (priority.get(m, 2), m))


def _style_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.45)
    ax.set_axisbelow(True)


def plot_bandwidth_comparison(
    df: pd.DataFrame,
    output_path: Path,
    peak_hbm_gb_s: Optional[float] = None,
    tpu_label: Optional[str] = None,
) -> Path:
    """Plot Effective HBM Bandwidth (GB/s) grouped by shape and method, per dtype."""
    dtypes = list(df["dtype"].unique())
    shapes = sorted(df["shape"].unique(), key=_parse_shape_pixels)
    methods = _method_order(list(df["method"].unique()))

    fig, axes = plt.subplots(
        1, len(dtypes), figsize=(7.0 * len(dtypes), 4.8), squeeze=False
    )

    x = np.arange(len(shapes))
    bar_w = 0.78 / max(len(methods), 1)
    colors = plt.cm.viridis(np.linspace(0.15, 0.88, len(methods)))

    for idx, dtype in enumerate(dtypes):
        ax = axes[0, idx]
        sub = df[df["dtype"] == dtype]

        for m_idx, method in enumerate(methods):
            m_sub = sub[sub["method"] == method].set_index("shape")
            vals = [float(m_sub.loc[s, "bandwidth_gb_s"]) if s in m_sub.index else 0.0 for s in shapes]
            offset = (m_idx - (len(methods) - 1) / 2.0) * bar_w
            bars = ax.bar(
                x + offset,
                vals,
                width=bar_w * 0.92,
                label=method,
                color=colors[m_idx],
                edgecolor="white",
                linewidth=0.5,
            )
            for bar, val in zip(bars, vals):
                if val > 0:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        bar.get_height(),
                        f"{val:.0f}",
                        ha="center",
                        va="bottom",
                        fontsize=7.5,
                        rotation=45,
                    )

        if peak_hbm_gb_s is not None:
            label = f"Peak HBM ({tpu_label}: {peak_hbm_gb_s:.0f} GB/s)" if tpu_label else f"Peak HBM ({peak_hbm_gb_s:.0f} GB/s)"
            ax.axhline(
                peak_hbm_gb_s,
                color="#d62728",
                linestyle="--",
                linewidth=1.4,
                label=label,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(shapes)
        ax.set_title(f"Effective HBM Bandwidth ({dtype})", fontsize=12, fontweight="bold")
        ax.set_xlabel("Tile Shape (H x W)")
        ax.set_ylabel("Effective HBM Bandwidth (GB/s)")
        _style_axes(ax)
        ax.legend(frameon=False, fontsize=8.5, loc="upper left")

    fig.suptitle(
        "Fused Spectral Index (NDVI + NDWI) Effective HBM Bandwidth",
        fontsize=13.5,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_latency_and_speedup(
    df: pd.DataFrame,
    highlight_shape: str,
    output_path: Path,
) -> Path:
    """Plot median latency (± std) and speedup vs. Eager for a target tile shape."""
    sub = df[df["shape"] == highlight_shape]
    if sub.empty:
        raise ValueError(
            f"Shape '{highlight_shape}' not found in CSV. Available: {list(df['shape'].unique())}"
        )

    dtypes = list(sub["dtype"].unique())
    methods = _method_order(list(sub["method"].unique()))

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8))
    x = np.arange(len(methods))
    bar_w = 0.72 / max(len(dtypes), 1)
    palette = ["#4C72B0", "#55A868", "#C44E52", "#8172B3"]

    for d_idx, dtype in enumerate(dtypes):
        d_sub = sub[sub["dtype"] == dtype].set_index("method")
        offset = (d_idx - (len(dtypes) - 1) / 2.0) * bar_w

        med = [float(d_sub.loc[m, "median_ms"]) if m in d_sub.index else 0.0 for m in methods]
        std = [float(d_sub.loc[m, "std_ms"]) if m in d_sub.index else 0.0 for m in methods]
        spd = [float(d_sub.loc[m, "speedup_vs_eager"]) if m in d_sub.index else 0.0 for m in methods]

        bars_lat = axes[0].bar(
            x + offset,
            med,
            yerr=std,
            capsize=3,
            width=bar_w * 0.92,
            label=dtype,
            color=palette[d_idx % len(palette)],
        )
        for bar, val in zip(bars_lat, med):
            if val > 0:
                axes[0].text(
                    bar.get_x() + bar.get_width() / 2.0,
                    bar.get_height(),
                    f"{val:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

        bars_spd = axes[1].bar(
            x + offset,
            spd,
            width=bar_w * 0.92,
            label=dtype,
            color=palette[d_idx % len(palette)],
        )
        for bar, val in zip(bars_spd, spd):
            if val > 0:
                axes[1].text(
                    bar.get_x() + bar.get_width() / 2.0,
                    bar.get_height(),
                    f"{val:.2f}x",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

    for ax, ylabel, title in [
        (axes[0], "Median Latency (ms)", f"Latency on {highlight_shape} (Lower is Better)"),
        (axes[1], "Speedup vs. Eager (x)", f"Speedup on {highlight_shape} (Higher is Better)"),
    ]:
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=18, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=11.5, fontweight="bold")
        _style_axes(ax)
        ax.legend(frameon=False, fontsize=9)

    axes[1].axhline(1.0, color="#666666", linestyle=":", linewidth=1.1)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_roofline(
    df: pd.DataFrame,
    highlight_shape: str,
    output_path: Path,
    peak_hbm_gb_s: float,
    peak_gflops: float,
    tpu_label: Optional[str] = None,
) -> Path:
    """Plot a log-log roofline chart showing the memory-bound regime of NDVI+NDWI."""
    sub = df[df["shape"] == highlight_shape]
    if sub.empty:
        raise ValueError(
            f"Shape '{highlight_shape}' not found in CSV. Available: {list(df['shape'].unique())}"
        )

    pixels = _parse_shape_pixels(highlight_shape)
    total_flops = pixels * FLOPS_PER_PIXEL

    ai_min, ai_max = 0.1, 1000.0
    ai_grid = np.logspace(np.log10(ai_min), np.log10(ai_max), 400)
    roof = np.minimum(ai_grid * peak_hbm_gb_s, peak_gflops)

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    hw_title = f"TPU {tpu_label}" if tpu_label else "TPU"
    ax.loglog(
        ai_grid,
        roof,
        color="#222222",
        linewidth=2.0,
        label=f"{hw_title} Roofline ({peak_hbm_gb_s:.0f} GB/s HBM)",
    )

    methods = _method_order(list(sub["method"].unique()))
    markers = {"float32": "o", "bfloat16": "s"}
    colors = dict(zip(methods, plt.cm.viridis(np.linspace(0.15, 0.88, len(methods)))))

    for _, row in sub.iterrows():
        dtype = str(row["dtype"])
        method = str(row["method"])
        elem_bytes = DTYPE_BYTES.get(dtype, 4)
        # Fused theoretical minimum bytes: 3 reads + 2 writes per pixel
        ai = FLOPS_PER_PIXEL / (5.0 * elem_bytes)
        achieved_gflops = (total_flops / 1e9) / (float(row["median_ms"]) / 1e3)

        ax.scatter(
            ai,
            achieved_gflops,
            s=85,
            marker=markers.get(dtype, "o"),
            color=colors[method],
            edgecolors="black",
            linewidths=0.6,
            zorder=5,
            label=f"{method} ({dtype})",
        )

    ridge_ai = peak_gflops / peak_hbm_gb_s
    ax.axvline(ridge_ai, color="#999999", linestyle=":", linewidth=1.0)
    ax.text(
        ridge_ai * 1.08,
        peak_gflops * 0.25,
        f"Ridge Point\n({ridge_ai:.0f} FLOP/B)",
        fontsize=8.5,
        color="#555555",
    )

    ax.set_xlim(ai_min, ai_max)
    ax.set_xlabel("Arithmetic Intensity (FLOPs / Byte, Fused Minimum)", fontsize=10.5)
    ax.set_ylabel("Achieved Performance (GFLOP/s)", fontsize=10.5)
    ax.set_title(
        f"Roofline Analysis — Fused NDVI+NDWI on {highlight_shape}",
        fontsize=12,
        fontweight="bold",
    )
    ax.grid(True, which="both", linestyle="--", alpha=0.35)
    ax.legend(frameon=False, fontsize=8, loc="lower right", ncol=2)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot TorchTPU spectral fusion benchmark CSV results."
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        required=True,
        help="Path to CSV generated by benchmarks/bench_throughput.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/plots"),
        help="Directory to write generated PNG charts.",
    )
    parser.add_argument(
        "--highlight-shape",
        type=str,
        default=None,
        help="Tile shape for latency/speedup & roofline plots (defaults to largest shape in CSV).",
    )
    parser.add_argument(
        "--tpu-preset",
        choices=list(TPU_PRESETS.keys()),
        default="v6e",
        help="TPU hardware preset for HBM bandwidth & roofline ceilings.",
    )
    parser.add_argument(
        "--peak-hbm-gb-s",
        type=float,
        default=None,
        help="Override peak HBM bandwidth (GB/s) for reference lines.",
    )
    parser.add_argument(
        "--peak-gflops",
        type=float,
        default=None,
        help="Override peak compute ceiling (GFLOP/s) for the roofline plot.",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    if df.empty:
        raise ValueError(f"Input CSV is empty: {args.input_csv}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    highlight_shape = args.highlight_shape or max(
        df["shape"].unique(), key=_parse_shape_pixels
    )
    preset = TPU_PRESETS[args.tpu_preset]
    peak_hbm = args.peak_hbm_gb_s if args.peak_hbm_gb_s is not None else preset["peak_hbm_gb_s"]
    peak_gflops = args.peak_gflops if args.peak_gflops is not None else preset["peak_gflops"]

    bw_path = plot_bandwidth_comparison(
        df,
        args.output_dir / "bandwidth_comparison.png",
        peak_hbm_gb_s=peak_hbm,
        tpu_label=args.tpu_preset,
    )
    lat_path = plot_latency_and_speedup(
        df,
        highlight_shape=highlight_shape,
        output_path=args.output_dir / "latency_and_speedup.png",
    )
    roof_path = plot_roofline(
        df,
        highlight_shape=highlight_shape,
        output_path=args.output_dir / "roofline.png",
        peak_hbm_gb_s=peak_hbm,
        peak_gflops=peak_gflops,
        tpu_label=args.tpu_preset,
    )

    print("Saved plots:")
    for p in (bw_path, lat_path, roof_path):
        print(f"  - {p}")


if __name__ == "__main__":
    main()
