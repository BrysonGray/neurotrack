# Tree Postprocessing Integration

This document describes the integration of tree processing functions from `data_prep/tree.py` into the new refactored neurotrack package.

## Functionality Added

### Core Functions (`neurotrack.data.tree`)

1. **`restructure_neuron_tree(paths)`** - Main function for restructuring neuron trees
   - Splits paths into sections based on intersections
   - Calculates stream lengths hierarchically
   - Restitches sections for optimal representation

2. **`remove_soma(swc_list, max_radius=7.0)`** - Remove soma based on radius threshold
   - Walks through SWC tree structure
   - Removes nodes with radius above threshold
   - Returns cleaned SWC data and seed points

3. **Supporting functions:**
   - `split_paths_into_sections()` - Split paths at intersection points
   - `get_all_stream_lengths()` - Calculate section lengths and build adjacency graph
   - `get_hierarchical_streams()` - Order streams by length
   - `restitch_sections()` - Combine sections into final paths

### CLI Integration

Added new `data postprocess` command with the following options:

```bash
neurotrack data postprocess --help
```

**Arguments:**
- `--input-dir, -i`: Directory containing inference results (.npz files)
- `--output-dir, -o`: Output directory for processed data
- `--restructure`: Apply tree restructuring to neuron paths
- `--remove-soma`: Remove soma from neuron trees  
- `--max-radius`: Maximum radius for soma removal (default: 7.0)
- `--save-swc`: Save results as SWC files
- `--verbose, -v`: Enable verbose output

## Usage Examples

### Command Line Usage

1. **Basic postprocessing with restructuring:**
```bash
neurotrack data postprocess \
    --input-dir /path/to/inference/results \
    --output-dir /path/to/output \
    --restructure \
    --verbose
```

2. **Remove soma and save as SWC:**
```bash
neurotrack data postprocess \
    --input-dir /path/to/inference/results \
    --output-dir /path/to/output \
    --remove-soma \
    --max-radius 5.0 \
    --save-swc \
    --verbose
```

3. **Full postprocessing pipeline:**
```bash
neurotrack data postprocess \
    --input-dir /path/to/inference/results \
    --output-dir /path/to/output \
    --restructure \
    --remove-soma \
    --max-radius 7.0 \
    --save-swc \
    --verbose
```

### Python API Usage

```python
from neurotrack.data.tree import restructure_neuron_tree, remove_soma
import numpy as np

# Load your inference results
data = np.load('inference_results.npz', allow_pickle=True)
paths = data['paths']

# Restructure neuron tree
restructured_paths = restructure_neuron_tree(paths)
print(f"Restructured {len(paths)} paths into {len(restructured_paths)} sections")

# For SWC data, remove soma
swc_data = [...]  # Your SWC data
cleaned_swc, seeds = remove_soma(swc_data, max_radius=7.0, verbose=True)
print(f"Removed soma, found {len(seeds)} seed points")
```

## File Processing

The CLI command processes .npz files containing inference results:

1. **Input:** .npz files with 'paths' data from neuron tracing inference
2. **Processing:** Applies requested tree operations (restructuring, soma removal)
3. **Output:** 
   - Updated .npz files with processed paths
   - Optional: Individual SWC files for each path (with `--save-swc`)

## Integration Details

### Module Structure
- **Location:** `neurotrack/data/tree.py`
- **Imports:** Available via `from neurotrack.data import restructure_neuron_tree`
- **CLI Handler:** `handle_postprocess_trees()` in `neurotrack/cli/data.py`

### Dependencies Added
- Enhanced `SWCLoader.build_edge_list()` static method for graph construction
- Updated data module exports to include tree functions

### Error Handling
- Validates input/output directories
- Gracefully handles missing or invalid data
- Provides verbose logging for debugging
- Continues processing if individual files fail

This integration provides a clean, CLI-accessible way to apply the sophisticated tree processing algorithms from the original codebase to inference results, making it easy to incorporate into automated pipelines.
