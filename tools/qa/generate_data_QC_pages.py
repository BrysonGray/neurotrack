#%%
import pandas as pd
from pathlib import Path
import plotly.express as px
import matplotlib.pyplot as plt
import numpy as np
import tifffile as tf
import os
import shutil
import subprocess
import re
from difflib import SequenceMatcher

from neurotrack.data import loading as load
#%%

def _clean_dataset_name(dataset_key):
    # Drop prefixes like e_checked6_ / p_checked6_ while keeping source dataset text.
    return re.sub(r'^[a-z]_checked\d+_', '', str(dataset_key))

def _canonical_original_lookup_name(path_like):
    key = Path(path_like).name
    if '.v3dpbd' in key:
        key = key.split('.v3dpbd', 1)[0]
    elif '.v3draw' in key:
        key = key.split('.v3draw', 1)[0]

    return key


def _canonical_match_key(path_like):
    """Build a normalized key for matching SWCs to converted TIFF files."""
    name = _canonical_original_lookup_name(path_like)
    name = name.replace('.tif_uint8', '')
    name = name.replace('.tiff', '')
    name = name.replace('.tif', '')
    name = name.replace('.v3dpbd', '')
    name = name.replace('.v3draw', '')
    return name.lower()


def _build_original_metadata_map(original_files_by_dataset, lookup_df, leave_out_keys):
    """Return metadata keyed by original SWC path string for QC title rendering."""
    lookup_rows = []
    for row in lookup_df[['path', 'id']].itertuples(index=False):
        lookup_rows.append((str(row.path), int(row.id)))

    metadata = {}

    for dataset_key, orig_paths in original_files_by_dataset.items():
        dataset_clean = _clean_dataset_name(dataset_key)
        included = dataset_key not in leave_out_keys

        for orig_path in orig_paths:
            orig_path = Path(orig_path)
            lookup_name = _canonical_original_lookup_name(orig_path)

            matches = [(p, i) for (p, i) in lookup_rows if lookup_name == Path(p).stem]
            if len(matches) != 1:
                print(f'Warning: {len(matches)} matches for {orig_path.name} (lookup_name={lookup_name})')

            file_id = matches[0][1]

            metadata[str(orig_path)] = {
                'dataset_key': dataset_key,
                'dataset_clean': dataset_clean,
                'file_name': _canonical_match_key(orig_path),
                'lookup_name': lookup_name,
                'file_id': file_id,
                'included': included,
            }

    return metadata


def _to_original_swc_path(path_like):
    return Path(str(path_like).replace('_FIXED_PARENT_CONNECTIONS', ''))


def _to_fixed_swc_path(path_like):
    path_str = str(path_like)
    if '_FIXED_PARENT_CONNECTIONS' in path_str:
        return Path(path_str)
    if path_str.endswith('.swc'):
        return Path(path_str[:-4] + '_FIXED_PARENT_CONNECTIONS.swc')
    return Path(path_str + '_FIXED_PARENT_CONNECTIONS.swc')


def _match_tif_for_original_swc(dataset_key, original_swc_path, tif_files_by_dataset):
    """Map original SWC to converted TIFF path using canonical and fuzzy matching."""
    original_swc_path = Path(original_swc_path)
    tif_candidates = [Path(p) for p in tif_files_by_dataset.get(dataset_key, [])]
    if not tif_candidates:
        return None

    swc_key = _canonical_match_key(original_swc_path)
    tif_key_map = {}
    for tif_path in tif_candidates:
        key = _canonical_match_key(tif_path)
        tif_key_map.setdefault(key, []).append(tif_path)

    direct = tif_key_map.get(swc_key, [])
    if len(direct) == 1:
        return direct[0]
    if len(direct) > 1:
        parent_hint = original_swc_path.parent.name.lower()
        narrowed = [p for p in direct if parent_hint in str(p.parent).lower()]
        if len(narrowed) == 1:
            return narrowed[0]
        return sorted(direct, key=lambda p: len(str(p)))[0]

    # Fallback: choose best fuzzy key match if confidence is sufficient.
    best_path = None
    best_score = -1.0
    second_best = -1.0
    for tif_path in tif_candidates:
        score = SequenceMatcher(None, swc_key, _canonical_match_key(tif_path)).ratio()
        if score > best_score:
            second_best = best_score
            best_score = score
            best_path = tif_path
        elif score > second_best:
            second_best = score

    if best_score >= 0.80 and (best_score - second_best) >= 0.03:
        return best_path

    return None


def _build_qc_swc_img_lists(original_files_by_dataset, tif_files_by_dataset):
    """Build ordered original SWC paths and aligned image paths for QC generation."""
    all_swc_paths = []
    for dataset_key in sorted(original_files_by_dataset.keys()):
        all_swc_paths.extend(sorted(original_files_by_dataset[dataset_key], key=lambda p: str(p)))
    all_tif_paths = []
    for dataset_key in sorted(tif_files_by_dataset.keys()):
        all_tif_paths.extend(sorted(tif_files_by_dataset[dataset_key], key=lambda p: str(p)))

    def _normalize_original_swcs(path):
        key = path.stem
        if '.v3dpbd' in key:
            key = key.split('.v3dpbd', 1)[0]
        elif '.v3draw' in key:
            key = key.split('.v3draw', 1)[0]
        return key
    swc_paths = []
    img_paths = []
    for original_swc in all_swc_paths:
        original_swc = Path(original_swc)
        original_swc_key = _normalize_original_swcs(original_swc)
        for tif_path in all_tif_paths:
            tif_key = Path(tif_path).stem
            if original_swc_key == tif_key:
                swc_paths.append(original_swc)
                img_paths.append(Path(tif_path))
                break

    if len(swc_paths) < len(all_swc_paths):
        missing_count = len(all_swc_paths) - len(swc_paths)
        print(f'Warning: could not infer image path for {missing_count} SWC files.')
    
    return swc_paths, img_paths


def plot_swc_3d(swc_list, ax=None, title=None, zoom=1.0, pad_fraction=0.08):
    # get the orphan nodes (nodes whose parent is -1)
    all_coords = np.array([row[2:5] for row in swc_list], dtype=float)
    mins = all_coords.min(axis=0)
    maxs = all_coords.max(axis=0)
    span = np.maximum(maxs - mins, 1.0)
    pad = pad_fraction * span

    centered_coords = all_coords - all_coords.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered_coords, full_matrices=False)
    second_pc = vt[1].copy()
    third_pc = vt[2].copy()
    if second_pc[1] < 0:
        second_pc *= -1
    if third_pc[2] < 0:
        third_pc *= -1

    camera_direction = second_pc + 1.7 * third_pc
    camera_direction /= np.linalg.norm(camera_direction)
    camera_distance = 0.01

    orphan_coords = np.array([row[2:5] for row in swc_list if row[6] == -1])
    adj_dict = load.adjacency_dict(swc_list)
    end_coords = [row[2:5] for row in swc_list if len(adj_dict.get(row[0], [])) == 1 and row[6] != -1]  # get the end points (nodes with only one connection)

    swc_parsed = load.parse_swc_into_segments(swc_list)
    # Create a DataFrame for plotting
    data = []
    # Iterate through the sections list
    for i, segment in enumerate(swc_parsed):
        for row in segment:
            node_coord = row[2:5]
            data.append([i, node_coord[0], node_coord[1], node_coord[2]])
    df_sections = pd.DataFrame(data, columns=["section", "x", "y", "z"])

    # plot the sections and the unique orphan nodes as red points
    
    fig = px.line_3d(df_sections, x="x", y="y", z="z", color='section', labels=None)
    fig.update_traces(showlegend=False)

    # add orphan coords  as red points
    orphan_x = [coord[0] for coord in orphan_coords]
    orphan_y = [coord[1] for coord in orphan_coords]
    orphan_z = [coord[2] for coord in orphan_coords]
    fig.add_scatter3d(x=orphan_x, y=orphan_y, z=orphan_z, mode='markers', marker=dict(size=5, color='red'), name='No parent')

    # add end coords as green points
    end_x = [coord[0] for coord in end_coords]
    end_y = [coord[1] for coord in end_coords]
    end_z = [coord[2] for coord in end_coords]
    fig.add_scatter3d(x=end_x, y=end_y, z=end_z, mode='markers', marker=dict(size=3, color='green'), name='End points')

    if ax is None:
        center = (mins + maxs) / 2.0
        half_extent = np.maximum(((span * 0.5) + pad) / max(zoom, 1e-6), 1e-3)
        fig.update_layout(
            scene=dict(
                aspectmode='data',
            xaxis=dict(range=[center[0] - half_extent[0], center[0] + half_extent[0]]),
            yaxis=dict(range=[center[1] - half_extent[1], center[1] + half_extent[1]]),
            zaxis=dict(range=[center[2] - half_extent[2], center[2] + half_extent[2]]),
                camera=dict(
                    # Fixed viewpoint for reproducible exports, oriented by PCA.
                    eye=dict(
                        x=float(camera_direction[0] * camera_distance),
                        y=float(camera_direction[1] * camera_distance),
                        z=float(camera_direction[2] * camera_distance),
                    ),
                    up=dict(x=0.0, y=0.0, z=1.0),
                ),
            ),
            legend_title_text='',
            width=1000,
            height=600
        )
        if title:
            fig.update_layout(title=title)
        fig.show()
        return fig

    # Matplotlib rendering path (for embedding in QC report figure).
    for segment in swc_parsed:
        segment_coords = np.array([row[2:5] for row in segment], dtype=float)
        if len(segment_coords) == 0:
            continue
        ax.plot(
            segment_coords[:, 0],
            segment_coords[:, 1],
            segment_coords[:, 2],
            color='black',
            linewidth=0.8,
            alpha=0.8,
        )

    if len(orphan_coords) > 0:
        ax.scatter(orphan_coords[:, 0], orphan_coords[:, 1], orphan_coords[:, 2], c='red', s=14, alpha=0.9)
    if len(end_coords) > 0:
        end_arr = np.array(end_coords)
        ax.scatter(end_arr[:, 0], end_arr[:, 1], end_arr[:, 2], c='green', s=8, alpha=0.8)

    center = (mins + maxs) / 2.0
    half_extent = np.maximum(((span * 0.5) + pad) / max(zoom, 1e-6), 1e-3)
    ax.set_xlim(center[0] - half_extent[0], center[0] + half_extent[0])
    ax.set_ylim(center[1] - half_extent[1], center[1] + half_extent[1])
    ax.set_zlim(center[2] - half_extent[2], center[2] + half_extent[2])
    ax.set_box_aspect((half_extent[0] * 2, half_extent[1] * 2, half_extent[2] * 2))

    # Approximate plotly camera eye with matplotlib elev/azim.
    azim = np.degrees(np.arctan2(camera_direction[1], camera_direction[0]))
    elev = np.degrees(np.arctan2(camera_direction[2], np.linalg.norm(camera_direction[:2])))
    ax.view_init(elev=elev, azim=azim)

    ax.set_xlabel('x', labelpad=1)
    ax.set_ylabel('y', labelpad=1)
    ax.set_zlabel('z', labelpad=1)
    ax.tick_params(axis='both', which='major', labelsize=7, pad=0)
    if title:
        ax.set_title(title, fontsize=14, pad=0)

    return ax

#%%
def plot_qc_overview(
    original_swc_file,
    img_file,
    fixed_swc_file=None,
    crop_size=17,
    seed=None,
    figsize=(22, 18),
    dataset_name=None,
    original_file_name=None,
    file_id=None,
    included_in_final=None,
    swc_zoom=1.25,
):
    """
    Create a QC overview plot with whole image MIP on the left and 4 random subregions on the right.
    
    Parameters:
    -----------
    swc_file : Path or str
        Path to the SWC file
    img_file : Path or str
        Path to the TIFF image file
    crop_size : int, optional
        Half-size of the cropped region for subregions (default: 17, resulting in 35x35 pixel crops)
    seed : int, optional
        Random seed for reproducibility (default: None)
    figsize : tuple, optional
        Figure size (default: (20, 12))
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The generated figure
    """
    from matplotlib.gridspec import GridSpec
    from matplotlib.patches import Rectangle
    
    # Load SWC and image
    swc_list = load.swc(original_swc_file)
    img = tf.imread(img_file)
    img = np.squeeze(img)
    
    # Create edge list from SWC
    edges = []
    id_to_idx = {int(line[0]): idx for idx, line in enumerate(swc_list)}
    for line in swc_list:
        if line[6] == -1:  # Skip the root node
            continue
        parent_id = int(line[6])
        child_id = int(line[0])
        edges.append((parent_id, child_id))
    
    # Create figure with custom grid layout
    # Use nested GridSpec for fine control over spacing
    fig = plt.figure(figsize=figsize)
    
    # Outer layout: top row for 2D overview and zooms, bottom row for side-by-side 3D SWC.
    gs_outer = GridSpec(
        2,
        1,
        figure=fig,
        # Keep the bottom 3D row about the same visual height as the top panel.
        height_ratios=[1.0, 1.0],
        hspace=0.10,
    )

    # Top panel matches existing layout.
    gs_main = gs_outer[0].subgridspec(
        3,
        2,
        width_ratios=[0.4, 0.6],
        # Collapse top/bottom padding so gs_right height matches gs_whole.
        height_ratios=[0.1, 1.0, 0.0],
        wspace=0.0,
        hspace=0.0,
    )
    
    # Left panel: Whole image MIP along z-axis with SWC overlay
    ax_whole = fig.add_subplot(gs_main[:, 0])
    ax_whole.imshow(img.max(axis=0), cmap='gray')
    # Keep equal aspect while visually tightening the gap to the right cutout panel.
    # ax_whole.set_anchor('E')
    ax_whole.axis('off')
    
    # Right side layout with explicit spacer rows/column so vertical and horizontal
    # subregion gaps are tightly controlled and comparable.
    # Rows: [row1, spacer, row2, spacer, row3]
    # Cols: [triplet A (3 cols), spacer, triplet B (3 cols)]
    gs_right = gs_main[1, 1].subgridspec(
        5,
        7,
        width_ratios=[1, 1, 1, 0.08, 1, 1, 1],
        # Add a little more vertical spacing between cutout rows.
        height_ratios=[1, 0.01, 1, 0.01, 1],
        wspace=0.0,
        hspace=0.0,
    )
    
    # Plot edges on whole image
    for parent_id, child_id in edges:
        parent_idx = id_to_idx[parent_id]
        child_idx = id_to_idx[child_id]
        parent_coords = swc_list[parent_idx][2:5]  # x, y, z
        child_coords = swc_list[child_idx][2:5]  # x, y, z
        ax_whole.plot([parent_coords[0], child_coords[0]], 
                     [parent_coords[1], child_coords[1]], 
                     'r-', linewidth=0.3, alpha=0.7)
    
    # Right side: 2x3 grid of random subregions
    rng = np.random.default_rng(seed=seed)
    triplet_axes = {}
    subregion_bounds_xy = {}
    
    for subregion_idx in range(6):
        # Select random center point
        center = np.floor(np.array(swc_list[rng.integers(len(swc_list))])[2:5]).astype(int)  # x, y, z
        
        # Calculate crop boundaries
        x_min = max(center[0] - crop_size, 0)
        x_max = min(center[0] + crop_size, img.shape[2])
        y_min = max(center[1] - crop_size, 0)
        y_max = min(center[1] + crop_size, img.shape[1])
        z_min = max(center[2] - crop_size, 0)
        z_max = min(center[2] + crop_size, img.shape[0])
        subregion_bounds_xy[subregion_idx] = (x_min, y_min, x_max, y_max)
        
        # Crop image
        cropped_img = img[z_min:z_max, y_min:y_max, x_min:x_max]
        
        # Pad if needed
        pad_width = ((max(0, -(center[2] - crop_size)), max(0, center[2] + crop_size - img.shape[0])),
                     (max(0, -(center[1] - crop_size)), max(0, center[1] + crop_size - img.shape[1])), 
                     (max(0, -(center[0] - crop_size)), max(0, center[0] + crop_size - img.shape[2])))
        cropped_img = np.pad(cropped_img, pad_width, mode='constant', constant_values=0)
        pad_left_xyz = np.array([pad_width[2][0], pad_width[1][0], pad_width[0][0]])
        
        
        # Map subregion to explicit right-grid placement (2 columns x 3 rows)
        # 0..5 laid out row-major: [0,1], [2,3], [4,5]
        row_group = subregion_idx // 2
        row_idx = row_group * 2
        col_start = 0 if (subregion_idx % 2) == 0 else 4
        
        # Plot three orthogonal MIPs for this subregion
        titles = ['XY', 'XZ', 'YZ']
        axes_list = [0, 1, 2]
        coord_pairs = [(0, 1), (0, 2), (1, 2)]  # Which coords to use for each projection
        triplet_axes[subregion_idx] = []
        
        for mip_idx in range(3):
            c1, c2 = coord_pairs[mip_idx]
            
            # Get edges within cropped region
            cropped_edges = []
            for parent_id, child_id in edges:
                parent_idx = id_to_idx[parent_id]
                child_idx = id_to_idx[child_id]
                parent_coords = swc_list[parent_idx][2:5]  # x, y, z
                child_coords = swc_list[child_idx][2:5]  # x, y, z
                
                # Check if both nodes are within cropped region
                if (x_min <= parent_coords[0] < x_max and y_min <= parent_coords[1] < y_max and z_min <= parent_coords[2] < z_max) and \
                (x_min <= child_coords[0] < x_max and y_min <= child_coords[1] < y_max and z_min <= child_coords[2] < z_max):
                    cropped_edges.append((parent_id, child_id))

            ax = fig.add_subplot(gs_right[row_idx, col_start + mip_idx])
            ax.imshow(cropped_img.max(axis=axes_list[mip_idx]), cmap='gray', aspect='equal')
            ax.set_xlim(0, crop_size*2)
            ax.set_ylim(crop_size*2, 0)
            if row_group == 0:
                ax.set_title(titles[mip_idx], fontsize=16, pad=2)
            ax.axis('off')
            triplet_axes[subregion_idx].append(ax)
            
            # Plot edges
            for parent_id, child_id in cropped_edges:
                parent_idx = id_to_idx[parent_id]
                child_idx = id_to_idx[child_id]
                parent_coords = swc_list[parent_idx][2:5] - np.array([x_min, y_min, z_min]) + pad_left_xyz
                child_coords = swc_list[child_idx][2:5] - np.array([x_min, y_min, z_min]) + pad_left_xyz
                
                ax.plot([parent_coords[c1], child_coords[c1]], 
                       [parent_coords[c2], child_coords[c2]], 
                       'r-', linewidth=1.0, alpha=0.8)
    
    # Bottom row: original and fixed SWC 3D comparison.
    original_swc_file = _to_original_swc_path(original_swc_file)
    if fixed_swc_file is None:
        fixed_swc_file = _to_fixed_swc_path(original_swc_file)
    else:
        fixed_swc_file = Path(fixed_swc_file)

    gs_bottom = gs_outer[1].subgridspec(1, 2, wspace=0.0)
    ax_3d_orig = fig.add_subplot(gs_bottom[0, 0], projection='3d')
    ax_3d_fixed = fig.add_subplot(gs_bottom[0, 1], projection='3d')

    try:
        swc_orig_list = load.swc(original_swc_file)
        plot_swc_3d(swc_orig_list, ax=ax_3d_orig, title='Original SWC', zoom=swc_zoom)
    except Exception as exc:
        ax_3d_orig.text2D(0.03, 0.5, f'Original SWC load failed\n{exc}', transform=ax_3d_orig.transAxes)
        ax_3d_orig.set_axis_off()

    try:
        swc_fixed_list = load.swc(fixed_swc_file)
        plot_swc_3d(swc_fixed_list, ax=ax_3d_fixed, title='Fixed SWC', zoom=swc_zoom)
        from matplotlib.lines import Line2D
        # Show one legend per figure pair on the right side of the fixed SWC panel.
        legend_handles = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=7, label='No parent'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=6, label='End point'),
        ]
        ax_3d_fixed.legend(
            handles=legend_handles,
            loc='center left',
            bbox_to_anchor=(1.02, 0.5),
            frameon=False,
            borderaxespad=0.0,
            fontsize=9,
        )
    except Exception as exc:
        ax_3d_fixed.text2D(0.03, 0.5, f'Fixed SWC load failed\n{exc}', transform=ax_3d_fixed.transAxes)
        ax_3d_fixed.set_axis_off()

    # Leave headroom for header text while minimizing unused horizontal margin.
    fig.subplots_adjust(left=0.02, right=0.985, bottom=0.02, top=0.90)
    border_colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown']
    # border_pad = 0.0015
    border_pad = 0.0
    for subregion_idx in range(6):
        positions = [ax.get_position() for ax in triplet_axes[subregion_idx]]
        x0 = min(pos.x0 for pos in positions)
        y0 = min(pos.y0 for pos in positions)
        x1 = max(pos.x1 for pos in positions)
        y1 = max(pos.y1 for pos in positions)

        border = Rectangle(
            (x0 - border_pad, y0 - border_pad),
            (x1 - x0) + 2 * border_pad,
            (y1 - y0) + 2 * border_pad,
            transform=fig.transFigure,
            fill=False,
            edgecolor=border_colors[subregion_idx % len(border_colors)],
            linewidth=4.5,
            zorder=10,
        )
        fig.add_artist(border)

    for subregion_idx in range(6):
        x_min, y_min, x_max, y_max = subregion_bounds_xy[subregion_idx]
        crop_box = Rectangle(
            (x_min, y_min),
            x_max - x_min,
            y_max - y_min,
            transform=ax_whole.transData,
            fill=False,
            edgecolor=border_colors[subregion_idx % len(border_colors)],
            linewidth=2.0,
            zorder=11,
        )
        ax_whole.add_patch(crop_box)

    included_label = 'Unknown' if included_in_final is None else ('Yes' if included_in_final else 'No')
    id_label = 'N/A' if file_id is None else str(file_id)
    dataset_label = dataset_name if dataset_name is not None else 'Unknown'
    file_label = original_file_name if original_file_name is not None else Path(original_swc_file).name
    fig.suptitle(
        f'Dataset: {dataset_label} | Name: {file_label} | ID: {id_label} | Included: {included_label}',
        fontsize=19,
        y=0.965,
    )
    
    return fig

def _make_latex_qc_doc(tex_path, figure_paths, title='Neurotrack QC Report'):
    tex_path = Path(tex_path)
    tex_path.parent.mkdir(parents=True, exist_ok=True)

    rel_paths = [Path(os.path.relpath(p, tex_path.parent)).as_posix() for p in figure_paths]

    lines = [
        r'\documentclass[11pt]{article}',
        r'\usepackage[margin=0.5in]{geometry}',
        r'\usepackage{graphicx}',
        r'\usepackage{grffile}',
        r'\usepackage{float}',
        r'\begin{document}',
        rf'\section*{{{title}}}',
    ]

    for fig_pair in [rel_paths[i:i+2] for i in range(0, len(rel_paths), 2)]:
        # Show up to two figures per page; handle odd count without empty includegraphics.
        lines.extend([
            r'\begin{figure}[H]',
            r'\centering',
            rf'\includegraphics[width=\textwidth, keepaspectratio]{{{fig_pair[0]}}}',
            r'\end{figure}',
        ])
        if len(fig_pair) > 1:
            lines.extend([
                r'\begin{figure}[H]',
                r'\centering',
                rf'\includegraphics[width=\textwidth, keepaspectratio]{{{fig_pair[1]}}}',
                r'\end{figure}',
            ])
        lines.append(r'\clearpage')

    lines.append(r'\end{document}')
    tex_path.write_text('\n'.join(lines), encoding='utf-8')

#%%
def _compile_latex_to_pdf(tex_path):
    tex_path = Path(tex_path)
    pdflatex_exe = shutil.which('pdflatex')
    if pdflatex_exe is None:
        print('pdflatex not found. Skipping PDF compilation.')
        return None

    work_dir = tex_path.parent
    tex_name = tex_path.name
    pdf_path = tex_path.with_suffix('.pdf')

    try:
        for pass_idx in range(2):
            result = subprocess.run(
                [pdflatex_exe, '-interaction=nonstopmode', tex_name],
                cwd=work_dir,
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode != 0:
                print(f'pdflatex pass {pass_idx + 1} failed (exit {result.returncode}).')
                if result.stdout:
                    print(result.stdout[-2000:])
                if result.stderr:
                    print(result.stderr[-2000:])
                return None
    except Exception as exc:
        print(f'Error while compiling LaTeX: {exc}')
        return None

    if pdf_path.exists():
        print(f'PDF report: {pdf_path}')
        return pdf_path

    print('pdflatex completed but PDF not found.')
    return None


def generate_all_qc_pages(
    original_swc_paths,
    img_paths,
    swc_metadata,
    output_dir,
    crop_size=50,
    base_seed=42,
    figsize=(20, 12),
    dpi=200,
    compile_pdf=True,
):
    output_dir = Path(output_dir)
    figures_dir = output_dir / 'figures'
    figures_dir.mkdir(parents=True, exist_ok=True)

    saved_figure_paths = []
    n_pairs = min(len(original_swc_paths), len(img_paths))
    print(f'Generating QC figures for {n_pairs} image/SWC pairs...')

    for idx, (original_swc_path, img_path) in enumerate(zip(original_swc_paths, img_paths)):
        original_swc_path = Path(original_swc_path)
        img_path = Path(img_path)
        fixed_swc_path = _to_fixed_swc_path(original_swc_path)

        # Always use the original (non-fixed) name for metadata lookup and display.
        file_meta = swc_metadata.get(str(original_swc_path), {})

        seed = None if base_seed is None else base_seed + idx
        fig = plot_qc_overview(
            original_swc_file=original_swc_path,
            img_file=img_path,
            fixed_swc_file=fixed_swc_path,
            crop_size=crop_size,
            seed=seed,
            figsize=figsize,
            dataset_name=file_meta.get('dataset_clean'),
            original_file_name=file_meta.get('file_name', original_swc_path.name),
            file_id=file_meta.get('file_id'),
            included_in_final=file_meta.get('included'),
        )

        stem = f'{idx:04d}_{img_path.stem}'
        fig_path = figures_dir / f'{stem}_qc.png'
        fig.savefig(fig_path, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        saved_figure_paths.append(fig_path)

        if (idx + 1) % 10 == 0 or (idx + 1) == n_pairs:
            print(f'  saved {idx + 1}/{n_pairs}')

    tex_path = output_dir / 'qc_report.tex'
    _make_latex_qc_doc(
        tex_path=tex_path,
        figure_paths=saved_figure_paths,
        title='Neurotrack QC Report',
    )

    pdf_path = None
    if compile_pdf:
        pdf_path = _compile_latex_to_pdf(tex_path)

    print(f'Finished. Figures: {figures_dir}')
    print(f'LaTeX report: {tex_path}')
    return saved_figure_paths, tex_path, pdf_path

#%%
swc_root = Path("/home/brysongray/data/neurotrack_data/gold166/gold166_original")
tif_root = Path("/home/brysongray/data/neurotrack_data/gold166/gold166_converted")

original_swc_files = {}
fixed_swc_files = {}
for d in swc_root.iterdir():
    if not d.is_dir():
        continue
    fixed_swc_files[d.name] = list(d.glob("**/*FIXED_PARENT_CONNECTIONS.swc"))
    original_swc_files[d.name] = [Path(str(f).replace("_FIXED_PARENT_CONNECTIONS", "")) for f in fixed_swc_files[d.name]]
for key in original_swc_files.keys():
    if len(original_swc_files[key]) != len(fixed_swc_files[key]):
        print(f"Warning: {key} has {len(original_swc_files[key])} original files but {len(fixed_swc_files[key])} fixed files.")
        
tif_files = {}
for d in tif_root.iterdir():
    if not d.is_dir():
        continue
    tif_files[d.name] = list(d.glob("**/*.tif"))

leave_out_folders = {
    'e_checked6_chick_uw',
    'e_checked6_zebrafish_larve_RGC_UW',
    'p_checked6_mouse_korea',
    'p_checked6_mouse_ugoettingen',
    'p_checked6_silkmoth_utokyo'
    }

gold166_lookup_path = swc_root.parent / "lookup_gold166.csv"
# load the gold166 lookup table
gold166_lookup = pd.read_csv(gold166_lookup_path)

original_swc_metadata = _build_original_metadata_map(
original_files_by_dataset=original_swc_files,
lookup_df=gold166_lookup,
leave_out_keys=leave_out_folders,
)

swc_paths, img_paths = _build_qc_swc_img_lists(original_swc_files, tif_files)
if not swc_paths:
    raise RuntimeError('No SWC/image pairs found')

idx = 60
swc = Path(swc_paths[idx])
img = Path(img_paths[idx])
meta = original_swc_metadata.get(str(swc), {})

fig = plot_qc_overview(
    original_swc_file=swc,
    fixed_swc_file=_to_fixed_swc_path(swc),
    img_file=img,
    crop_size=17,
    seed=42,
    figsize=(22, 18),
    dataset_name=meta.get('dataset_clean'),
    original_file_name=meta.get('file_name', swc.name),
    file_id=meta.get('file_id'),
    included_in_final=meta.get('included'),
    swc_zoom=1.25,
)

# out = Path('/home/brysongray/neurotrack/outputs/data_QC_6-18/_single_test_qc_v7.png')
# out.parent.mkdir(parents=True, exist_ok=True)
# fig.savefig(out, dpi=200, bbox_inches='tight')
# plt.close(fig)
# print(f'saved={out}')
# print(f'swc={swc}')
# print(f'img={img}')

#%%
if __name__ == '__main__':

    swc_root = Path("/home/brysongray/data/neurotrack_data/gold166/gold166_original")
    tif_root = Path("/home/brysongray/data/neurotrack_data/gold166/gold166_converted")

    original_swc_files = {}
    fixed_swc_files = {}
    for d in swc_root.iterdir():
        if not d.is_dir():
            continue
        fixed_swc_files[d.name] = list(d.glob("**/*FIXED_PARENT_CONNECTIONS.swc"))
        original_swc_files[d.name] = [Path(str(f).replace("_FIXED_PARENT_CONNECTIONS", "")) for f in fixed_swc_files[d.name]]
    for key in original_swc_files.keys():
        if len(original_swc_files[key]) != len(fixed_swc_files[key]):
            print(f"Warning: {key} has {len(original_swc_files[key])} original files but {len(fixed_swc_files[key])} fixed files.")
            
    tif_files = {}
    for d in tif_root.iterdir():
        if not d.is_dir():
            continue
        tif_files[d.name] = list(d.glob("**/*.tif"))

    leave_out_folders = {
        'e_checked6_chick_uw',
        'e_checked6_zebrafish_larve_RGC_UW',
        'p_checked6_mouse_korea',
        'p_checked6_mouse_ugoettingen',
        'p_checked6_silkmoth_utokyo'
        }
    
    gold166_lookup_path = swc_root.parent / "lookup_gold166.csv"
    # load the gold166 lookup table
    gold166_lookup = pd.read_csv(gold166_lookup_path)

    original_swc_metadata = _build_original_metadata_map(
    original_files_by_dataset=original_swc_files,
    lookup_df=gold166_lookup,
    leave_out_keys=leave_out_folders,
)

    output_root = Path('/home/brysongray/neurotrack/outputs/data_QC_6-20')
    swc_files, img_files_ordered = _build_qc_swc_img_lists(original_swc_files, tif_files)

    if not swc_files:
        raise RuntimeError('No SWC/image pairs found for QC generation.')

    print(f'Prepared {len(swc_files)} SWC/image pairs for QC generation.')

    generate_all_qc_pages(
        original_swc_paths=swc_files,
        swc_metadata=original_swc_metadata,
        img_paths=img_files_ordered,
        output_dir=output_root,
        crop_size=17,
        base_seed=42,
        figsize=(22, 18),
        dpi=200,
        compile_pdf=True,
    )

# %%
