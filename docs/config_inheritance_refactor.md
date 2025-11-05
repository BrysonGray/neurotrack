# Configuration Inheritance Refactor

## Overview

The training configuration system has been refactored to use inheritance, providing better code organization, type safety, and extensibility.

## New Structure

### Base Class: `TrainingConfig`
- Contains common attributes shared by all training types:
  - `name`: Training run name
  - `outdir`: Output directory
  - `batch_size`: Training batch size
  - `learning_rate`: Learning rate

### Subclasses

#### `SACTrainingConfig(TrainingConfig)`
- Inherits from `TrainingConfig`
- Overrides: `batch_size=256`, uses `lr` instead of `learning_rate`
- Adds SAC-specific parameters: `img_path`, `n_episodes`, `tau`, etc.

#### `ClassifierTrainingConfig(TrainingConfig)`
- Inherits from `TrainingConfig`
- Overrides: `batch_size=32`
- Adds classifier-specific parameters: `data_dir`, `epochs`, `channels`, etc.

## Benefits

### 1. **Shared Common Attributes**
```python
# Both configs have these attributes from the base class
sac_config.name
classifier_config.name
sac_config.outdir
classifier_config.outdir
```

### 2. **Polymorphic Behavior**
```python
def process_config(config: TrainingConfig):
    """Works with any training config type"""
    return f"Processing {config.name} -> {config.outdir}"

process_config(sac_config)       # Works
process_config(classifier_config) # Works
```

### 3. **Hierarchical Validation**
```python
# Base validation is automatically called for all subclasses
ConfigManager.validate_config(sac_config)        # Validates base + SAC-specific
ConfigManager.validate_config(classifier_config) # Validates base + classifier-specific
```

### 4. **Automatic Type Detection**
```python
# ConfigManager can auto-detect config type from content
config = ConfigManager.load_training_config("config.json", "auto")
# Returns SACTrainingConfig or ClassifierTrainingConfig based on file content
```

## Backward Compatibility

- All existing code continues to work unchanged
- Existing config files load correctly
- Legacy methods are still available
- Type checking is improved without breaking changes

## Usage Examples

### Loading Configs
```python
# Explicit loading
sac_config = ConfigManager.load_sac_training_config("sac_config.json")
classifier_config = ConfigManager.load_classifier_training_config("classifier_config.json")

# Auto-detection
config = ConfigManager.load_training_config("some_config.json", "auto")
```

### Polymorphic Functions
```python
def print_summary(config: TrainingConfig, duration: float):
    """Works with any training config"""
    print(f"Training {config.name} completed in {duration}s")
    print(f"Output saved to: {config.outdir}")
```

### Type-Safe Validation
```python
# Validates common fields + type-specific fields
ConfigManager.validate_config(config)
```

## Testing Results

All tests pass successfully:
- ✅ Inheritance relationships work correctly
- ✅ Polymorphic behavior functions properly  
- ✅ Automatic type detection works
- ✅ Validation inheritance operates correctly
- ✅ Existing configs load without issues
- ✅ CLI integration works seamlessly

## Migration Path

No migration required! The refactor maintains full backward compatibility while providing new capabilities for future development.
