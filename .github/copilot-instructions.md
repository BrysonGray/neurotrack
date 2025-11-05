# Neurotrack AI Agent Instructions

## Project Overview
Neurotrack is a reinforcement learning system for automated neuron tracing from microscopy images. It uses two neural networks: a **Soft Actor-Critic (SAC) agent** for sequential path tracing and a **ResNet branch classifier** for detecting branch points.

## Current Architecture Status
The project is actively undergoing modernization with a focus on:
- **Modular, class-based design** with clear separation of concerns
- **Type-safe configuration management** with validation
- **Sections-based data flow** where all drawing methods accept sections (not segments)
- **Advanced post-processing** including simplex noise and foreground/background separation
- **Backward compatibility** with legacy interfaces during transition

## Package Structure
```
neurotrack/
├── neurotrack/              # New modular package (evolving)
│   ├── core/               # Core utilities (config, geometry, coordinates)
│   ├── data/               # Data handling (loaders, formats)
│   ├── training/           # Training components (memory buffers)
│   ├── inference/          # Inference pipeline (placeholder)
│   └── visualization/      # Visualization tools (placeholder)
├── data_prep/              # Legacy data processing (being modernized)
├── models/                 # Neural network models
├── environments/           # RL environments
├── scripts/                # Example scripts
├── tests/                  # Test suite
└── configs/                # Configuration files
```

## Core Modules

### Drawing and Visualization (`data_prep.draw`) - **RECENTLY REFACTORED**
Modern class-based neuron rendering with advanced post-processing:
```python
from data_prep.draw import NeuronRenderer, DrawingConfig, GifConfig

# All methods now accept SECTIONS (not segments)
sections = {section_id: segments_array}  # Key architectural pattern

renderer = NeuronRenderer(rng=np.random.default_rng())
config = DrawingConfig(
    width=3.0, noise=0.1, rgb=True,
    # New post-processing parameters
    foreground_mean=0.8, foreground_std=0.1,
    background_mean=0.2, background_std=0.05,
    mask_threshold=0.1, simplex_scale=10.0
)
img = renderer.draw_neuron(sections, shape, config)
```

**Key Features:**
- **Sections-based API**: All methods accept `sections` dictionary, not raw segments
- **Internal consolidation**: `_consolidate_segments()` method handles sections → segments conversion
- **Advanced post-processing**: Foreground/background separation with configurable means/stds
- **Simplex noise**: Multi-octave noise generation for realistic image textures
- **Type-safe configuration**: `DrawingConfig` with validation and sensible defaults
- **Backward compatibility**: Legacy functions still available during transition

**Post-processing Pipeline:**
1. **Foreground/background masking**: Uses `mask_threshold` to separate regions
2. **Simplex noise generation**: Multi-octave noise with configurable `simplex_scale`
3. **Differential scaling**: Separate `foreground_mean/std` and `background_mean/std`
4. **Brightness clamping**: Results kept in [0,1] range with optional brightness variation

### Configuration Management (`neurotrack.core.config`)
All training uses type-safe configuration with validation:
```python
from neurotrack.core.config import ConfigManager, TrainingConfig
config = ConfigManager.load_training_config("configs/train_sac_gold166.json")
ConfigManager.validate_config(config)  # Validates paths and parameters
```

### Data Loading (`neurotrack.data.loaders`)
Unified data loading with clear interfaces:
```python
from neurotrack.data.loaders import TIFFLoader, SWCLoader
tiff_loader = TIFFLoader(pixelsize=[1.0, 1.0, 1.0], downsample_factor=1.0)
swc_loader = SWCLoader(rotate=False, verbose=True)
```

### Memory Buffers (`neurotrack.training.memory`)
Clean replay buffer implementations:
```python
from neurotrack.training.memory import create_replay_buffer
buffer = create_replay_buffer('prioritized', capacity=100000, obs_shape=(2,35,35,35), action_shape=(3,))
```

### Core Utilities (`neurotrack.core`)
- `geometry.py` - 3D interpolation, inhomogeneity correction, spatial operations
- `coordinates.py` - Coordinate transformations, scaling, voxel/world conversions
- `config.py` - Type-safe configuration management with validation

## Migration Status

### ✅ Completed (Current Phase)
- **draw.py refactor**: Complete class-based rewrite with NeuronRenderer, DrawingConfig, GifConfig
- **Sections-based architecture**: All drawing methods now accept sections, not segments
- **Advanced post-processing**: Simplex noise, foreground/background separation, configurable means/stds
- **Backward compatibility**: Legacy convenience functions maintained during transition
- New package structure created
- Core configuration management with type safety
- Data loaders for TIFF and SWC formats
- Format-specific utilities (SWC parsing, TIFF processing)
- Memory buffers for RL training
- Geometric and coordinate utilities
- Basic test suite structure
- Example scripts demonstrating new APIs

### 🔄 Legacy Modules (Still Available)
Original modules remain functional during transition:
- `bin/` - Command-line scripts (to be replaced by unified CLI)
- `data_prep/` - Original data processing (draw.py modernized, others being migrated to `neurotrack.data`)
- `models/` - Neural network models (to be reorganized)
- `solvers/` - Training algorithms (to be moved to `neurotrack.training`)
- `environments/` - RL environments (to be moved to `neurotrack.environments`)

## Development Patterns

### Sections-First Architecture
**Critical**: All draw methods now accept `sections`, not raw `segments`:
```python
# Correct format - sections dictionary
sections = {
    0: segments_array_1,  # section_id: segments for that section
    1: segments_array_2,
    # ...
}
renderer.draw_neuron(sections, shape, config)
renderer.draw_density(sections, shape)
renderer.draw_section_labels(sections, shape)

# Internal consolidation happens automatically via _consolidate_segments()
```

### Import Strategy
Use new package imports when available:
```python
# Preferred (new structure)
from neurotrack.core.config import TrainingConfig
from neurotrack.data.loaders import TIFFLoader
from neurotrack.training.memory import ReplayBuffer

# Drawing (modernized)
from data_prep.draw import NeuronRenderer, DrawingConfig, GifConfig

# Legacy (during transition)
from data_prep.load import swc, tiff
from memory.buffer import ReplayBuffer
```

### Configuration-Driven Development
Always use the configuration system:
```python
config = TrainingConfig(step_size=2.0, alpha=1.0, batch_size=256)
ConfigManager.validate_config(config)  # Catches errors early
```

### Device Management
Consistent CUDA handling across modules:
```python
from neurotrack.core.config import get_device
device = get_device()  # Returns "cuda:0" or "cpu"
```

## Next Phase Priorities
1. **Models reorganization** - Move neural networks to `neurotrack.models`
2. **Training integration** - Combine SAC and classifier training
3. **Environment migration** - Move RL environments to new structure
4. **CLI unification** - Replace bin scripts with `neurotrack` command
5. **Inference pipeline** - Complete inference module implementation

## Testing Strategy
Use the established test structure:
```bash
pytest tests/unit/           # Unit tests for individual components
pytest tests/integration/    # Integration tests for workflows
```

## Key Files for Reference
- `scripts/train_sac_example.py` - Shows new training setup pattern
- `scripts/data_preparation_example.py` - Demonstrates data processing workflow
- `tests/unit/` - Examples of testing patterns for each module
