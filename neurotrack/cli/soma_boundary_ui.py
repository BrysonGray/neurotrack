#!/usr/bin/env python3
"""Interactive command-line UI for soma boundary selection and SWC filtering."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import tifffile as tif
from matplotlib import patches
from matplotlib.widgets import RectangleSelector

from neurotrack.data import loading, save
from neurotrack.data.seed_io import save_seeds_json


def _clip_box(box: Tuple[int, int, int, int, int, int], shape_zyx: Tuple[int, int, int]) -> Tuple[int, int, int, int, int, int]:
    z_max_img, y_max_img, x_max_img = shape_zyx
    x0, x1, y0, y1, z0, z1 = [int(v) for v in box]
    x0, x1 = sorted((max(0, x0), min(x_max_img, x1)))
    y0, y1 = sorted((max(0, y0), min(y_max_img, y1)))
    z0, z1 = sorted((max(0, z0), min(z_max_img, z1)))

    if x0 == x1 or y0 == y1 or z0 == z1:
        raise ValueError(f"Bounding box must have non-zero size in x, y, and z. Got {(x0, x1, y0, y1, z0, z1)}")

    return (x0, x1, y0, y1, z0, z1)


def _filter_swc(swc_array: np.ndarray, box: Tuple[int, int, int, int, int, int]) -> tuple[set[int], np.ndarray]:
    x0, x1, y0, y1, z0, z1 = box
    in_box = (
        (swc_array[:, 2] >= x0) & (swc_array[:, 2] < x1)
        & (swc_array[:, 3] >= y0) & (swc_array[:, 3] < y1)
        & (swc_array[:, 4] >= z0) & (swc_array[:, 4] < z1)
    )
    removed_ids = set(swc_array[in_box, 0].astype(int).tolist())
    swc_filtered = np.array([row.copy() for row in swc_array if int(row[0]) not in removed_ids])

    if swc_filtered.size == 0:
        return removed_ids, swc_filtered

    for row in swc_filtered:
        parent_id = int(row[6])
        if parent_id in removed_ids:
            row[6] = -1

    return removed_ids, swc_filtered


def _draw_box(axes: list, box: Tuple[int, int, int, int, int, int]) -> None:
    x0, x1, y0, y1, z0, z1 = box
    axes[0].plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], color="cyan", linewidth=1.5)
    axes[1].plot([x0, x1, x1, x0, x0], [z0, z0, z1, z1, z0], color="cyan", linewidth=1.5)
    axes[2].plot([y0, y1, y1, y0, y0], [z0, z0, z1, z1, z0], color="cyan", linewidth=1.5)


def _draw_swc_edges(axes: list, swc_points: np.ndarray, color: str, alpha: float) -> None:
    if swc_points.size == 0:
        return
    local_id_to_idx = {int(row[0]): idx for idx, row in enumerate(swc_points)}
    for row in swc_points:
        parent_id = int(row[6])
        if parent_id == -1 or parent_id not in local_id_to_idx:
            continue
        parent_row = swc_points[local_id_to_idx[parent_id]]
        x1, y1, z1 = row[2:5]
        x2, y2, z2 = parent_row[2:5]
        axes[0].plot([x1, x2], [y1, y2], color=color, alpha=alpha, linewidth=1.0)
        axes[1].plot([x1, x2], [z1, z2], color=color, alpha=alpha, linewidth=1.0)
        axes[2].plot([y1, y2], [z1, z2], color=color, alpha=alpha, linewidth=1.0)


def _draw_root_nodes(axes: list, swc_points: np.ndarray) -> None:
    if swc_points.size == 0:
        return
    root_points = swc_points[swc_points[:, 6] == -1]
    if root_points.size == 0:
        return
    axes[0].scatter(root_points[:, 2], root_points[:, 3], c="red", s=20, alpha=0.9)
    axes[1].scatter(root_points[:, 2], root_points[:, 4], c="red", s=20, alpha=0.9)
    axes[2].scatter(root_points[:, 3], root_points[:, 4], c="red", s=20, alpha=0.9)


def _seed_points_zyx_tuples(swc_points: np.ndarray) -> List[Tuple[float, float, float]]:
    if swc_points.size == 0:
        return []
    root_points = swc_points[swc_points[:, 6] == -1]
    if root_points.size == 0:
        return []
    seed_points_zyx = root_points[:, [4, 3, 2]]
    return [tuple(map(float, row)) for row in seed_points_zyx.tolist()]


def choose_soma_boundary_cli(
    swc_file: Path,
    img_file: Path,
    bounding_box: Tuple[int, int, int, int, int, int],
) -> tuple[Tuple[int, int, int, int, int, int], np.ndarray, List[Tuple[float, float, float]]]:
    """Interactive drag-and-approve UI for soma boundary selection."""
    img = tif.imread(str(img_file)).squeeze()
    swc_array = np.array(loading.swc(str(swc_file)), dtype=float)

    if swc_array.size == 0:
        raise ValueError(f"Empty SWC file: {swc_file}")

    if bounding_box == (0, 0, 0, 0, 0, 0):
        roots = swc_array[swc_array[:, 6] == -1]
        if len(roots) == 0:
            cx, cy, _ = np.mean(swc_array[:, 2:5], axis=0).astype(int)
        else:
            cx, cy, _ = roots[0][2:5].astype(int)
        r = 40
        bounding_box = (cx - r, cx + r, cy - r, cy + r, 0, img.shape[0])

    current_box = _clip_box(bounding_box, img.shape)
    state: dict = {
        "current_box": current_box,
        "swc_filtered": None,
        "removed_ids": set(),
        "seed_points": [],
        "approved_once": False,
        "done": False,
        "preview_fig": None,
        "preview_axes": None,
    }

    fig_sel, ax_sel = plt.subplots(1, 1, figsize=(8, 8))
    ax_sel.imshow(img.max(axis=0), cmap="gray", vmin=0, vmax=255)
    ax_sel.axis("off")
    ax_sel.set_title("Drag box on XY. Enter = preview filter. r = return result.")

    rect_patch = patches.Rectangle(
        (current_box[0], current_box[2]),
        current_box[1] - current_box[0],
        current_box[3] - current_box[2],
        fill=False,
        edgecolor="cyan",
        linewidth=2,
    )
    ax_sel.add_patch(rect_patch)

    def _update_rect(box: Tuple[int, int, int, int, int, int]) -> None:
        rect_patch.set_xy((box[0], box[2]))
        rect_patch.set_width(box[1] - box[0])
        rect_patch.set_height(box[3] - box[2])
        fig_sel.canvas.draw_idle()

    def _on_select(eclick, erelease) -> None:
        if eclick.xdata is None or eclick.ydata is None or erelease.xdata is None or erelease.ydata is None:
            return
        x0, x1 = sorted((int(round(eclick.xdata)), int(round(erelease.xdata))))
        y0, y1 = sorted((int(round(eclick.ydata)), int(round(erelease.ydata))))
        z0, z1 = state["current_box"][4], state["current_box"][5]
        try:
            state["current_box"] = _clip_box((x0, x1, y0, y1, z0, z1), img.shape)
            _update_rect(state["current_box"])
        except ValueError:
            return

    selector = RectangleSelector(
        ax_sel,
        _on_select,
        useblit=False,
        button=[1],
        minspanx=1,
        minspany=1,
        spancoords="pixels",
        interactive=True,
    )

    def _render_preview(box: Tuple[int, int, int, int, int, int], removed_ids: set[int], swc_filtered: np.ndarray) -> None:
        if state["preview_fig"] is None or not plt.fignum_exists(state["preview_fig"].number):
            preview_fig, preview_axes = plt.subplots(1, 3, figsize=(18, 6))
            state["preview_fig"] = preview_fig
            state["preview_axes"] = preview_axes
        else:
            preview_fig = state["preview_fig"]
            preview_axes = state["preview_axes"]
            for ax in preview_axes:
                ax.clear()

        preview_axes[0].imshow(img.max(axis=0), cmap="gray", vmin=0, vmax=255)
        preview_axes[0].set_title("XY (max Z)")
        preview_axes[1].imshow(img.max(axis=1), cmap="gray", vmin=0, vmax=255)
        preview_axes[1].set_title("XZ (max Y)")
        preview_axes[2].imshow(img.max(axis=2), cmap="gray", vmin=0, vmax=255)
        preview_axes[2].set_title("YZ (max X)")

        _draw_swc_edges(preview_axes, swc_array, color="red", alpha=0.25)
        _draw_swc_edges(preview_axes, swc_filtered, color="lime", alpha=0.7)
        _draw_root_nodes(preview_axes, swc_filtered)
        _draw_box(preview_axes, box)

        for ax in preview_axes:
            ax.axis("off")

        preview_fig.suptitle(
            f"{swc_file.name} | Box={box} | Removed={len(removed_ids)} nodes | Remaining={len(swc_filtered)}",
            y=1.02,
        )
        preview_fig.tight_layout()
        preview_fig.canvas.draw_idle()
        plt.show(block=False)

    def _approve_current_box() -> None:
        removed_ids, swc_filtered = _filter_swc(swc_array, state["current_box"])
        state["removed_ids"] = removed_ids
        state["swc_filtered"] = swc_filtered
        state["seed_points"] = _seed_points_zyx_tuples(swc_filtered)
        state["approved_once"] = True
        _render_preview(state["current_box"], removed_ids, swc_filtered)

    def _on_key(event) -> None:
        key = (event.key or "").lower()
        if key == "enter":
            _approve_current_box()
        elif key == "r":
            if not state["approved_once"]:
                _approve_current_box()
            state["done"] = True
            if state["preview_fig"] is not None and plt.fignum_exists(state["preview_fig"].number):
                plt.close(state["preview_fig"])
            plt.close(fig_sel)

    def _on_close(_event) -> None:
        state["done"] = True

    fig_sel.canvas.mpl_connect("key_press_event", _on_key)
    fig_sel.canvas.mpl_connect("close_event", _on_close)
    plt.show()

    selector.set_active(False)

    if state["swc_filtered"] is None:
        _approve_current_box()

    return state["current_box"], state["swc_filtered"], state["seed_points"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive soma boundary selection UI")
    parser.add_argument("--swc-file", required=True, type=Path, help="Path to input SWC file")
    parser.add_argument("--img-file", required=True, type=Path, help="Path to input TIFF file")
    parser.add_argument(
        "--bounding-box",
        nargs=6,
        type=int,
        metavar=("X_MIN", "X_MAX", "Y_MIN", "Y_MAX", "Z_MIN", "Z_MAX"),
        default=(0, 0, 0, 0, 0, 0),
        help="Optional initial box in x,y,z order",
    )
    parser.add_argument("--out-swc", type=Path, default=None, help="Optional output path for filtered SWC")
    parser.add_argument("--out-seeds", type=Path, default=None, help="Optional output path for seeds JSON")
    parser.add_argument(
        "--seed-key",
        type=str,
        default=None,
        help="Key to use in seeds JSON; defaults to image filename",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    current_box, swc_filtered, seed_points_zyx_tuples = choose_soma_boundary_cli(
        swc_file=args.swc_file,
        img_file=args.img_file,
        bounding_box=tuple(args.bounding_box),
    )

    print(f"Selected box: {current_box}")
    print(f"Seed points (z, y, x): {seed_points_zyx_tuples}")

    if args.out_swc is not None:
        save.write_swc(swc_filtered.tolist(), str(args.out_swc))
        print(f"Wrote filtered SWC: {args.out_swc}")

    if args.out_seeds is not None:
        seed_key = args.seed_key or args.img_file.name
        save_seeds_json(args.out_seeds, {seed_key: seed_points_zyx_tuples})
        print(f"Wrote seeds JSON: {args.out_seeds} (key={seed_key})")


if __name__ == "__main__":
    main()
