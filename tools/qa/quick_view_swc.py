#!/usr/bin/env python3

""" Lightweight script to quickly visualize an SWC file and its corresponding image patch. Useful for QA during patch extraction. """
# check if running in a notebook environment and if not, use the appropriate matplotlib backend
import sys
if not 'ipykernel' in sys.modules:
    import matplotlib
    matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
import numpy as np
import argparse
from pathlib import Path
from typing import List, Tuple
from neurotrack.data import loading
import tifffile as tf


TYPE_TO_COLOR = {
    0: "purple",
    1: "red",
    2: "blue",
    3: "green",
    4: "cyan",
    5: "magenta",
    6: "yellow",
    7: "pink",
    8: "orange",
}

TYPE_TO_LABEL = {
    0: "undefined",
    1: "soma",
    2: "axon",
    3: "basal dendrite",
    4: "apical dendrite",
    5: "custom",
    6: "unspecified neurite",
    7: "glia process",
    8: "unknown",
}

ROOT_NODE_COLOR = "lime"
ROOT_NODE_LABEL = "root node"


def _prep_swc_to_plot(swc_file) -> tuple[LineCollection, List[int]]:
    swc_list = loading.swc(swc_file)
    id_to_idx = {node[0]: idx for idx, node in enumerate(swc_list)}
    line_segments_ids: List[Tuple[int, int]] = []
    for node in swc_list:
        parent_id = node[6]
        node_id = node[0]
        if parent_id != -1:  # -1 indicates no parent
            line_segments_ids.append((node_id, parent_id))

    line_segments = []
    segment_types = []
    segment_radii = []
    for child_id, parent_id in line_segments_ids:
        child_idx = id_to_idx[child_id]
        parent_idx = id_to_idx[parent_id]
        child_node = swc_list[child_idx]
        parent_node = swc_list[parent_idx]
        line_segments.append([(parent_node[2], parent_node[3]), (child_node[2], child_node[3])]) # 2,3,4 -> x,y,z.
        segment_types.append(parent_node[1])
        segment_radii.append(parent_node[5])

    colors = [TYPE_TO_COLOR.get(segment_type, 'brown') for segment_type in segment_types]
    line_collection = LineCollection(
        line_segments,
        colors=colors,
        linewidths=segment_radii,
        zorder=3,
    )
    return line_collection, sorted(set(segment_types))


def _plot_root_nodes(ax, swc_file) -> bool:
    swc_list = loading.swc(swc_file)
    root_nodes = [node for node in swc_list if node[6] == -1]
    if not root_nodes:
        return False

    x_coords = [node[2] for node in root_nodes]
    y_coords = [node[3] for node in root_nodes]
    ax.scatter(
        x_coords,
        y_coords,
        color=ROOT_NODE_COLOR,
        s=50,
        edgecolors="black",
        linewidths=0.6,
        zorder=10,
    )
    return True


def _add_swc_legend(ax, segment_types: List[int], include_root_nodes: bool = False) -> None:
    legend_handles = [
        Line2D(
            [0],
            [0],
            color=TYPE_TO_COLOR.get(segment_type, 'brown'),
            lw=2,
            label=TYPE_TO_LABEL.get(segment_type, f'type {segment_type}'),
        )
        for segment_type in segment_types
    ]
    if include_root_nodes:
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                markersize=7,
                markerfacecolor=ROOT_NODE_COLOR,
                markeredgecolor="black",
                linestyle="None",
                label=ROOT_NODE_LABEL,
            )
        )
    if legend_handles:
        ax.legend(handles=legend_handles, loc='upper left', bbox_to_anchor=(1.02, 1.0), title='SWC types')


def _prep_image_to_plot(image) -> LineCollection:
    img = tf.imread(image)
    img = np.squeeze(img)  # Remove singleton dimensions
    if img.ndim == 3:
        img = np.max(img, axis=0)  # Max projection along the z-axis
    else:
        err = f"Warning: Expected image to have 3 dimensions not including singletons. Got {img.ndim} dimensions."
        raise ValueError(err)
    return img

def plot_image_with_swc(image, swc_file, figsize=(10,10)):
    swc_overlay, segment_types = _prep_swc_to_plot(swc_file)
    image_data = _prep_image_to_plot(image)
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(image_data, cmap='gray')
    ax.add_collection(swc_overlay)
    has_roots = _plot_root_nodes(ax, swc_file)
    _add_swc_legend(ax, segment_types, include_root_nodes=has_roots)
    ax.set_title(f"Image: {image} with SWC overlay")
    ax.axis('off')
    fig.tight_layout()
    plt.show()

def main() -> None:
    parser = argparse.ArgumentParser(description="Quickly visualize an SWC file and its corresponding image patch.")
    parser.add_argument("--swcdir", type=str, required=True, help="Path to the SWC file.")
    parser.add_argument("--imgdir", type=str, required=True, help="Directory containing the image patches.")
    args = parser.parse_args()
    img_to_swc_map = loading.map_tiff_to_swc(args.imgdir, args.swcdir)

    img_files = [f for f in Path(args.imgdir).glob("*.tif") if f.is_file()]


    i = 0
    while i < len(img_files):
        # get the swc file for the first image
        img_file = img_files[i]
        swc_file = img_to_swc_map.get(img_file)
        if swc_file is None:
            print(f"No SWC file found for image {img_file.name}. Skipping.")
            i += 1
            continue
        # prep the swc overlay
        swc_overlay, segment_types = _prep_swc_to_plot(swc_file)
        # prep the image
        image = _prep_image_to_plot(img_file)
        # plot the image and overlay the swc
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(image, cmap='gray')
        ax.add_collection(swc_overlay)
        has_roots = _plot_root_nodes(ax, swc_file)
        _add_swc_legend(ax, segment_types, include_root_nodes=has_roots)
        ax.set_title(f"Image: {img_file.name} with SWC overlay")
        ax.axis('off')
        fig.tight_layout()
        plt.show()
        # wait for user input to close the plot before moving on to the next image
        user_input = input("Press Enter to continue or 'q' to quit...")
        if user_input.lower() == 'q':
            break
        i += 1

if __name__ == "__main__":
    main()