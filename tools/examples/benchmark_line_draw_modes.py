#!/usr/bin/env python
"""Quick benchmark for line drawing modes on a synthetic neuron."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from neurotrack.data.image import Image


def _polyline_to_segments(points: np.ndarray) -> list[np.ndarray]:
    """Convert an ordered polyline of points into consecutive 3D segments."""
    segments: list[np.ndarray] = []
    for i in range(points.shape[0] - 1):
        segments.append(np.stack((points[i], points[i + 1]), axis=0))
    return segments


def build_dummy_neuron() -> list[torch.Tensor]:
    """Construct a simple branching neuron as a list of 2x3 segment tensors."""
    trunk = np.array(
        [
            [20, 64, 64],
            [28, 65, 64],
            [36, 65, 65],
            [44, 66, 65],
            [52, 66, 66],
            [60, 67, 66],
            [68, 67, 66],
            [76, 68, 67],
            [84, 68, 67],
            [92, 69, 68],
        ],
        dtype=np.float32,
    )

    b1 = np.array(
        [
            trunk[3],
            trunk[3] + [8, 10, 6],
            trunk[3] + [14, 16, 12],
            trunk[3] + [18, 20, 18],
        ],
        dtype=np.float32,
    )

    b2 = np.array(
        [
            trunk[5],
            trunk[5] + [6, -12, 8],
            trunk[5] + [12, -20, 14],
            trunk[5] + [18, -24, 20],
        ],
        dtype=np.float32,
    )

    b3 = np.array(
        [
            trunk[7],
            trunk[7] + [5, 9, -9],
            trunk[7] + [10, 15, -16],
            trunk[7] + [14, 19, -22],
        ],
        dtype=np.float32,
    )

    all_segments: list[np.ndarray] = []
    for polyline in (trunk, b1, b2, b3):
        all_segments.extend(_polyline_to_segments(polyline))

    return [torch.tensor(seg, dtype=torch.float32) for seg in all_segments]


def benchmark_mode(
    mode: str,
    segments: list[torch.Tensor],
    width: float,
    repeats: int,
    volume_shape: tuple[int, int, int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Draw all segments repeatedly and return final volume and per-run times."""
    mode_key = mode.lower()
    times = np.zeros(repeats, dtype=np.float64)
    final_volume = None

    for i in range(repeats):
        img = Image(torch.zeros(volume_shape, dtype=torch.float32))
        t0 = time.perf_counter()
        for seg in segments:
            img.draw_line_segment(seg, width=width, channel=0, mode=mode_key)
        times[i] = time.perf_counter() - t0
        final_volume = img.data[0].cpu().numpy()

    if final_volume is None:
        raise RuntimeError("Benchmark did not generate an output volume.")

    return final_volume, times


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark gaussian vs binary line drawing modes.")
    parser.add_argument("--repeats", type=int, default=5, help="Number of timed runs per mode.")
    parser.add_argument("--width", type=float, default=4.0, help="Segment drawing width.")
    parser.add_argument(
        "--size",
        type=int,
        default=128,
        help="Cubic volume size (one channel is used internally).",
    )
    parser.add_argument("--show", action="store_true", help="Display the plot window in addition to saving.")
    args = parser.parse_args()

    if args.repeats < 1:
        raise ValueError("--repeats must be >= 1")
    if args.size < 64:
        raise ValueError("--size must be >= 64")

    segments = build_dummy_neuron()
    volume_shape = (1, args.size, args.size, args.size)

    # Warm-up to reduce one-time overhead in timed runs.
    benchmark_mode("gaussian", segments, width=args.width, repeats=1, volume_shape=volume_shape)
    benchmark_mode("binary", segments, width=args.width, repeats=1, volume_shape=volume_shape)

    mode_labels = ["Gaussian", "binary"]
    results: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    print(f"Segments: {len(segments)} | width: {args.width} | volume: {volume_shape}")
    for mode_label in mode_labels:
        volume, times = benchmark_mode(
            mode_label,
            segments,
            width=args.width,
            repeats=args.repeats,
            volume_shape=volume_shape,
        )
        results[mode_label] = (volume, times)
        print(
            f"{mode_label:>8}: mean={times.mean() * 1e3:.2f} ms "
            f"std={times.std() * 1e3:.2f} ms min={times.min() * 1e3:.2f} ms"
        )

    gaussian_mean = results["Gaussian"][1].mean()
    binary_mean = results["binary"][1].mean()
    speedup = gaussian_mean / binary_mean if binary_mean > 0 else float("inf")
    print(f"Speedup (binary vs Gaussian): {speedup:.2f}x")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for ax, mode_label in zip(axes, mode_labels):
        volume, times = results[mode_label]
        mip = volume.max(axis=0)
        ax.imshow(mip, cmap="magma", origin="lower")
        ax.set_title(f"{mode_label} | mean {times.mean() * 1e3:.1f} ms")
        ax.axis("off")

    out_dir = REPO_ROOT / "outputs" / "media"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "line_draw_mode_comparison.png"
    fig.savefig(out_path, dpi=180)
    print(f"Saved comparison figure: {out_path}")

    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
