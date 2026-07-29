# Neurotrack

## Features

- Train a 3D CNN tracing policy with behavior cloning (BC) and optional DAgger rounds.
- Run interactive tracing sessions with GUI-based seed selection and per-image navigation.
- Perform automated tracing from selected seeds using trained policy checkpoints.
- Manually revise predicted traces in the GUI and iterate on reconstructions.
- Post-process reconstructions (resampling, smoothing, merging, branch filtering).
- Evaluate predictions against ground-truth SWC files and export session reports.

## Overview

Neurotrack is a neuron tracing toolkit for 3D microscopy volumes built around a behavior cloning pipeline with DAgger fine-tuning.

The current tracing policy is a deterministic 3D CNN that predicts the next step vector directly from local 3D image context.
Training starts with BC warmstart and can continue with DAgger rounds that aggregate expert labels on policy-visited states.

In addition to training and batch inference, Neurotrack provides an interactive GUI workflow for practical reconstruction work:

1. Select or edit seed points in orthogonal 3D views.
2. Run automated tracing from selected seeds.
3. Manually revise reconstructed paths.
4. Run post-processing on predictions.
5. Evaluate against reference SWC and export metrics.

Legacy SAC utilities remain in the repository for compatibility and comparison experiments, but the primary pipeline is BC + DAgger.

## Pipeline

### 1) Train policy (BC / DAgger)

Use JSON configs under `configs/training` with the BC training CLI:

```bash
python -m neurotrack.cli.run_bc_train -i configs/training/train_dagger_example.json
```

This training path supports:

- Warmstart behavior cloning
- Configurable DAgger rounds and beta schedules (linear, exponential, adaptive)
- Replay-buffer based aggregation across rounds
- Per-round logging and checkpointing

### 2) Run interactive tracing GUI

Launch the interactive tracing session:

```bash
python -m neurotrack.cli.interactive_tracing -c configs/inference/test_example_inference.json
```

Or provide paths directly:

```bash
python -m neurotrack.cli.interactive_tracing \
	--img_dir /path/to/images \
	--seeds_input /path/to/seeds.json \
```

The GUI supports seed selection, trace-all or per-image tracing, model selection, path revision, post-processing controls, and evaluation/report export.

### 3) Batch inference / evaluation

Use the inference pipeline CLI with a JSON config:

```bash
python -m neurotrack.cli.run_inference -i configs/inference/test_example_inference.json
```

## Requirements

This codebase is developed and tested with Python 3.12.8.

Install dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt
```

PyTorch can run on CPU or GPU. A GPU is recommended for faster training and inference, but is not required.

## Demo

An example inference, postprocessing, and evaluation workflow is available in the Jupyter notebook `notebooks/inference_pipeline_demo.ipynb`.

### Before running the demo, download the necessary data by following these steps:
### 1) Download the trained model weights from Huggingface.co
```bash
pip install "huggingface_hub[cli]"
```
Navigate to the neurotrack root directory.
```bash
cd /path/to/neurotrack
```
```bash
hf download brysongray/NeuroTrack --local-dir ./neurotrack_data/model_weights
```
### 2) Download and unzip the NeuroTrack data zip file

```bash
curl -L  https://api.figshare.com/v2/file/download/67098257 -o neurotrack_data.zip && unzip neurotrack_data.zip -d ./neurotrack_data/
```



## Example neuron tracking results
Example neuron traces of simulated image volumes from real neuron reconstructions obtained from neuromorpho.org.
Gray surface is the true neuron, red surface is the result of automated tracing.

![hippo](https://media.giphy.com/media/ZEcTAh6nwSrbRMdMDS/giphy.gif)

![hippo](https://media.giphy.com/media/7Vu79HIrW4XROOVX03/giphy.gif)

![hippo](https://media.giphy.com/media/Qlkpz2BHlb1bjyHEmo/giphy.gif)

