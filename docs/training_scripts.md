# Training Scripts Documentation

This document provides instructions on how to use the two main scripts for training the Soft Actor-Critic (SAC) neuron tracking model: `bin/setup_sac_train_v1.py` and `bin/sac_train.py`.

## 1. Setting up the Training Data (`bin/setup_sac_train_v1.py`)

This script is used to generate and prepare the training data. It can generate synthetic neuron data or process existing SWC and image files.

### Usage

```bash
python bin/setup_sac_train_v1.py --output_dir <path_to_output_dir> [options]
```

### Arguments

| Argument | Type | Required | Default | Description |
| :--- | :--- | :--- | :--- | :--- |
| `--output_dir` | str | Yes | - | Path to the directory where the generated data will be saved. |
| `--swc_dir` | str | No | - | Path to a directory containing SWC files (if using existing data). |
| `--image_dir` | str | No | - | Path to a directory containing TIFF images (if using existing data). |
| `--remove_soma` | flag | No | False | If set, removes the soma from the SWC files before processing. |
| `--rng_seed` | int | No | 1 | Random seed for the data generator. |
| `--complexity_range` | float | No | 0.0 1.0 | Range of drawing complexities to use (min max). |
| `--morphology` | str | No | "all" | Complexity of neuron morphology to generate (e.g., 'simple', 'moderate', 'complex', 'full', or 'all'). |
| `--subtrees_per_swc` | int | No | 1 | Number of subtrees to draw per SWC file. |
| `--dataset_size` | int | No | 100 | Number of synthetic neurons to generate if no SWC directory is provided. |

### Examples

**Generate 100 synthetic neurons:**

```bash
python bin/setup_sac_train_v1.py --output_dir ./training_data --dataset_size 100
```

**Process existing SWC files:**

```bash
python bin/setup_sac_train_v1.py --swc_dir ./raw_data/swc --output_dir ./training_data --subtrees_per_swc 5
```

---

## 2. Training the SAC Model (`bin/sac_train.py`)

This script trains the Soft Actor-Critic (SAC) model using the data prepared by the setup script. It requires a JSON configuration file to specify training parameters.

### Usage

```bash
python bin/sac_train.py --json <path_to_config.json>
```

### Configuration File (JSON)

The configuration file should contain the following parameters:

| Parameter | Type | Description | Default (if omitted) |
| :--- | :--- | :--- | :--- |
| `data_dir` | str | Path to the input data directory (created by `setup_sac_train_v1.py`). | **Required** |
| `outdir` | str | Directory to save output models. | **Required** |
| `name` | str | Name for the training session. | **Required** |
| `step_size` | float | Step size for the environment. | 1.0 |
| `step_width` | float | Step width for the environment. | 1.0 |
| `batchsize` | int | Batch size for training. | 256 |
| `tau` | float | Soft update parameter for target networks. | 0.005 |
| `gamma` | float | Discount factor for future rewards. | 0.99 |
| `lr` | float | Learning rate for optimizers. | 0.001 |
| `alpha` | float | Weight applied to the accuracy component of reward. | 1.0 |
| `beta` | float | Weight applied to the reward prior. | 1e-3 |
| `friction` | float | Weight applied to the friction component of reward. | 1e-4 |
| `n_episodes` | int | Number of training episodes. | 100 |
| `init_temperature` | float | Initial temperature for SAC entropy. | 0.005 |
| `target_entropy` | float | Target entropy for SAC. | 0.0 |
| `classifier_weights` | str | Path to pre-trained classifier weights (optional). | - |
| `sac_weights` | str | Path to pre-trained SAC model weights (optional). | - |

### Example Configuration (`config.json`)

```json
{
    "data_dir": "./training_data",
    "outdir": "./results",
    "name": "sac_experiment_1",
    "n_episodes": 1000,
    "batchsize": 64,
    "lr": 0.0003
}
```

### Example Command

```bash
python bin/sac_train.py --json config.json
```

Notes: 
- The current working config file is configs/train_sac_dynamic_complexity.json
- Logs are saved in the project directory to the logs folder. They consist of GIFs of the paths as they are drawn and CSV files of episode info.