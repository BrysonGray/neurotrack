# Resume-from-Checkpoint Functionality

## Overview

The neurotrack training system now supports comprehensive checkpoint functionality for both SAC and classifier training, allowing users to pause and resume training seamlessly.

## Features

### 1. **Automatic Checkpoint Saving**
- Configurable checkpoint intervals (episodes for SAC, epochs for classifier)
- Automatic cleanup of old checkpoints (keeps only the N most recent)
- Standardized checkpoint file naming

### 2. **Resume from Checkpoint**
- Command-line support via `--resume` flag
- Configuration file support via `checkpoint_path` parameter
- Automatic state restoration (models, optimizers, training progress)

### 3. **Standardized Configuration**
- All configs now use `learning_rate` parameter (no more `lr`)
- Unified checkpoint parameters across all training types
- Backward compatibility maintained for legacy configs

## Configuration Parameters

### Base Training Config
```json
{
    "checkpoint_path": null,              // Path to resume from (null = start fresh)
    "save_checkpoints": true,             // Enable/disable checkpoint saving
    "checkpoint_interval": 100,           // Save every N episodes/epochs
    "keep_last_n_checkpoints": 5          // Number of recent checkpoints to keep
}
```

### SAC Training Config
```json
{
    "name": "sac_training",
    "outdir": "/path/to/output",
    "learning_rate": 0.001,               // Standardized parameter name
    "n_episodes": 5000,
    "checkpoint_interval": 100,           // Save every 100 episodes
    // ... other SAC parameters
}
```

### Classifier Training Config
```json
{
    "name": "classifier_training",
    "outdir": "/path/to/output", 
    "learning_rate": 0.001,               // Standardized parameter name
    "epochs": 100,
    "checkpoint_interval": 10,            // Save every 10 epochs
    // ... other classifier parameters
}
```

## Usage Examples

### Command Line Usage

#### SAC Training with Resume
```bash
# Start fresh training
neurotrack train sac --config configs/train_sac_gold166.json

# Resume from checkpoint
neurotrack train sac --config configs/train_sac_gold166.json --resume /path/to/checkpoint.pt

# Dry run (validate config without training)
neurotrack train sac --config configs/train_sac_gold166.json --dry-run
```

#### Classifier Training with Resume
```bash
# Start fresh training
neurotrack train classifier --config configs/train_classifier_example.json

# Resume from checkpoint  
neurotrack train classifier --config configs/train_classifier_example.json --resume /path/to/checkpoint.pt

# Dry run (validate config without training)
neurotrack train classifier --config configs/train_classifier_example.json --dry-run
```

### Programmatic Usage

#### SAC Training
```python
from neurotrack.core.config import ConfigManager
from neurotrack.training.sac_trainer import SACTrainer

# Load config and set checkpoint path
config = ConfigManager.load_training_config("sac_config.json", "auto")
config.checkpoint_path = "/path/to/checkpoint.pt"  # Optional: set resume path

# Train with checkpointing
trainer = SACTrainer(config)
trainer.train()  # Automatically handles resume if checkpoint_path is set
```

#### Classifier Training
```python
from neurotrack.core.config import ConfigManager  
from neurotrack.training.classifier_trainer import ClassifierTrainer

# Load config and set checkpoint path
config = ConfigManager.load_training_config("classifier_config.json", "auto")
config.checkpoint_path = "/path/to/checkpoint.pt"  # Optional: set resume path

# Train with checkpointing
trainer = ClassifierTrainer(config)
trainer.train()  # Automatically handles resume if checkpoint_path is set
```

## Checkpoint File Structure

### SAC Checkpoints
```python
{
    'episode': 1000,                      // Current episode
    'config': {...},                      // Training configuration
    'actor_state_dict': {...},            // Actor model state
    'Q1_state_dict': {...},               // Q1 critic state
    'Q2_state_dict': {...},               // Q2 critic state
    'Q1_target_state_dict': {...},        // Q1 target state
    'Q2_target_state_dict': {...},        // Q2 target state
    'log_alpha': tensor(...),             // Temperature parameter
    'actor_optimizer_state_dict': {...},  // Actor optimizer state
    'Q1_optimizer_state_dict': {...},     // Q1 optimizer state
    'Q2_optimizer_state_dict': {...},     // Q2 optimizer state
    'log_alpha_optimizer_state_dict': {...}, // Alpha optimizer state
    'timestamp': '2025-08-23T10:30:00'    // Save timestamp
}
```

### Classifier Checkpoints
```python
{
    'epoch': 50,                          // Current epoch
    'config': {...},                      // Training configuration
    'model_state_dict': {...},            // Model state
    'optimizer_state_dict': {...},        // Optimizer state
    'loss': 0.123,                        // Current loss value
    'timestamp': '2025-08-23T10:30:00'    // Save timestamp
}
```

## File Naming Convention

- **SAC**: `{name}_step_{episode}.pt` (e.g., `gold166_branching_step_1000.pt`)
- **Classifier**: `{name}_epoch_{epoch}.pt` (e.g., `classifier_training_epoch_50.pt`)
- **Latest**: `{name}_latest.pt` (most recent checkpoint)

## Benefits

### 1. **Robustness**
- Training can survive system crashes, power outages, etc.
- Long training runs can be safely interrupted and resumed
- Experimentation with different hyperparameters from specific checkpoints

### 2. **Efficiency**
- No need to restart long training runs from scratch
- Can pause training to use resources for other tasks
- Easy to create training pipelines with staged training

### 3. **Reproducibility**
- Exact training state can be restored
- Training history is preserved in checkpoint metadata
- Consistent results when resuming from same checkpoint

## Implementation Notes

- Checkpoint saving is automatic based on the configured interval
- Old checkpoints are automatically cleaned up to save disk space
- Resume functionality works seamlessly with both CLI and programmatic interfaces
- All training state is preserved (models, optimizers, episode/epoch counters)
- The system maintains backward compatibility with existing configurations
