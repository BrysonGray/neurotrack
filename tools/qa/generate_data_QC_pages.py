#%%
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import tifffile as tf
import os
import shutil
import subprocess

from neurotrack.data import loading as load
#%%

data_root = Path('/home/brysongray/data/neurotrack_data/gold166/gold166_cropped/')

# Load SWC files and corresponding image files, ensuring they are ordered correctly

swc_files = list(data_root.glob('**/*.swc'))
img_files = [f for f in data_root.glob('**/*.tif') if f.is_file()]

# order the image files the same as the swc files
img_files_ordered = []
for swc_file in swc_files:
    split_key = '.v3dpbd' if 'v3dpbd' in swc_file.stem else '.v3draw'
    img_file = [f for f in img_files if f.stem == swc_file.stem.split(split_key)[0]]
    if len(img_file) == 0:
        print(f"No image file found for {swc_file.stem}")
    elif len(img_file) > 1:
        print(f"Multiple image files found for {swc_file.stem}: {[f.stem for f in img_file]}. Using the first one.")
        img_files_ordered.append(img_file[0])
    else:
        img_files_ordered.append(img_file[0])

for swc_file, img_file in zip(swc_files, img_files_ordered):
    print(f"{swc_file.stem} : {img_file.stem}")

data_converted_path = Path('/home/brysongray/data/neurotrack_data/gold166/gold166_converted/')
tiff_name_to_dataset = {f.name: f.parent.parent.name for f in data_converted_path.rglob('*.tif') if f.is_file()}


def plot_qc_overview(swc_file, img_file, crop_size=50, seed=None, figsize=(20, 12)):
    """
    Create a QC overview plot with whole image MIP on the left and 4 random subregions on the right.
    
    Parameters:
    -----------
    swc_file : Path or str
        Path to the SWC file
    img_file : Path or str
        Path to the TIFF image file
    crop_size : int, optional
        Half-size of the cropped region for subregions (default: 50)
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
    swc_list = load.swc(swc_file)
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
    
    # Main grid: 3 rows x 2 columns.
    # Left image spans full height; right subregion panel uses a taller middle row
    # and wider right column so equal-aspect subregion panels can be larger.
    gs_main = GridSpec(
        3,
        2,
        figure=fig,
        width_ratios=[0.9, 1.25],
        height_ratios=[0.5, 1.4, 0.5],
        wspace=0.04,
        hspace=0.0,
    )
    
    # Left panel: Whole image MIP along z-axis with SWC overlay
    ax_whole = fig.add_subplot(gs_main[:, 0])
    ax_whole.imshow(img.max(axis=0), cmap='gray')
    ax_whole.axis('off')
    
    # Right side layout with explicit spacer rows/column so vertical and horizontal
    # subregion gaps are tightly controlled and comparable.
    # Rows: [row1, spacer, row2, spacer, row3]
    # Cols: [triplet A (3 cols), spacer, triplet B (3 cols)]
    gs_right = gs_main[1, 1].subgridspec(
        5,
        7,
        width_ratios=[1, 1, 1, 0.06, 1, 1, 1],
        height_ratios=[1, 0.03, 1, 0.03, 1],
        wspace=0.01,
        hspace=0.01,
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
                
                c1, c2 = coord_pairs[mip_idx]
                ax.plot([parent_coords[c1], child_coords[c1]], 
                       [parent_coords[c2], child_coords[c2]], 
                       'r-', linewidth=1.0, alpha=0.8)
    
    fig.subplots_adjust(left=0.02, right=0.985, bottom=0.02, top=0.955)

    border_colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown']
    border_pad = 0.0015
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

    fig.suptitle(f'QC Overview: {Path(img_file).stem}\n\
                 from dataset {tiff_name_to_dataset[Path(img_file).name]}', fontsize=24, y=0.985)
    
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
        # show two figures per page
        lines.extend([
            r'\begin{figure}[H]',
            r'\centering',
            rf'\includegraphics[width=\textwidth, keepaspectratio]{{{fig_pair[0]}}}',
            r'\end{figure}',
            r'\begin{figure}[H]',
            r'\centering',
            rf'\includegraphics[width=\textwidth, keepaspectratio]{{{fig_pair[1] if len(fig_pair) > 1 else ""}}}',
            r'\end{figure}',
            r'\clearpage',
        ])

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
    swc_paths,
    img_paths,
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
    n_pairs = min(len(swc_paths), len(img_paths))
    print(f'Generating QC figures for {n_pairs} image/SWC pairs...')

    for idx, (swc_path, img_path) in enumerate(zip(swc_paths, img_paths)):
        swc_path = Path(swc_path)
        img_path = Path(img_path)

        seed = None if base_seed is None else base_seed + idx
        fig = plot_qc_overview(
            swc_file=swc_path,
            img_file=img_path,
            crop_size=crop_size,
            seed=seed,
            figsize=figsize,
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
if __name__ == '__main__':
    output_root = Path('/home/brysongray/neurotrack/outputs/data_QC')
    generate_all_qc_pages(
        swc_paths=swc_files,
        img_paths=img_files_ordered,
        output_dir=output_root,
        crop_size=50,
        base_seed=42,
        figsize=(20, 12),
        dpi=200,
        compile_pdf=True,
    )
