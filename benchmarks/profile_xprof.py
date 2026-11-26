import argparse
from pathlib import Path
from typing import Callable, List, Tuple

import torch
import torch_tpu  # noqa: F401
from torch.profiler import ProfilerActivity, profile, record_function, schedule

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
        import torch_xla
        torch_xla.sync()


def _get_profiler_activities() -> List[ProfilerActivity]:
    """Detect available profiler activities for the current TorchTPU build."""
    activities = [ProfilerActivity.CPU]
    for attr in ("TPU", "XPU", "PRIVATEUSE1"):
        if hasattr(ProfilerActivity, attr):
            activities.append(getattr(ProfilerActivity, attr))
            break
    return activities


def _parse_shape(raw: str) -> Tuple[int, int]:
    h, w = raw.lower().split("x")
    return int(h), int(w)


def profile_mode(
    name: str, fn: Callable, inputs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], logdir: Path, wait: int,
    warmup: int, active: int,
) -> None:
    """Profile a single execution mode and export XProf traces."""
    run_dir = logdir / name
    run_dir.mkdir(parents=True, exist_ok=True)

    # Pre-compile once before starting the profiler schedule so compilation
    # overhead does not pollute the steady-state device trace.
    _ = fn(*inputs)
    _sync_tpu()

    activities = _get_profiler_activities()
    total_steps = wait + warmup + active

    print(f"Profiling '{name}' ({total_steps} steps) -> {run_dir}")

    with profile(
        activities=activities,
        schedule=schedule(wait=wait, warmup=warmup, active=active, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(str(run_dir)),
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as prof:
        for step in range(total_steps):
            with record_function(f"{name}_step_{step}"):
                _ = fn(*inputs)
                _sync_tpu()
            prof.step()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate XProf traces for TorchTPU spectral index fusion."
    )
    parser.add_argument(
        "--shape",
        type=str,
        default="10980x10980",
        help="2D image shape (default: 10980x10980, full Sentinel-2 10m tile).",
    )
    parser.add_argument(
        "--dtype",
        choices=["float32", "bfloat16"],
        default="bfloat16",
        help="Input tensor dtype.",
    )
    parser.add_argument(
        "--block-shape",
        type=str,
        default="256x256",
        help="Pallas 2D tile shape.",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=["eager", "compiled", "pallas"],
        default=["eager", "compiled", "pallas"],
        help="Execution modes to profile.",
    )
    parser.add_argument(
        "--logdir",
        type=Path,
        default=Path("results/xprof_traces"),
        help="Output directory for XProf/TensorBoard traces.",
    )
    parser.add_argument("--wait", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--active", type=int, default=3)
    args = parser.parse_args()

    shape = _parse_shape(args.shape)
    block_shape = _parse_shape(args.block_shape)
    dtype = torch.float32 if args.dtype == "float32" else torch.bfloat16

    red = torch.rand(shape, dtype=dtype, device=DEVICE)
    green = torch.rand(shape, dtype=dtype, device=DEVICE)
    nir = torch.rand(shape, dtype=dtype, device=DEVICE)
    _sync_tpu()
    inputs = (red, green, nir)

    compiled_fn = get_compiled_indices_fn() if "compiled" in args.modes else None

    dispatch = {
        "eager": compute_indices_eager,
        "compiled": compiled_fn,
        "pallas": lambda r, g, n: compute_indices_pallas(
            r, g, n, block_shape=block_shape
        ),
    }

    for mode in args.modes:
        profile_mode(
            name=f"{mode}_{args.dtype}_{args.shape}",
            fn=dispatch[mode],
            inputs=inputs,
            logdir=args.logdir,
            wait=args.wait,
            warmup=args.warmup,
            active=args.active,
        )

    print(
        f"\nDone. Inspect traces with:\n"
        f"  tensorboard --logdir {args.logdir} --port 6006 --bind_all"
    )


if __name__ == "__main__":
    main()