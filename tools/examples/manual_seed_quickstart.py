#!/usr/bin/env python
"""
Quick start guide for manual seed selection.

This is a minimal example showing how to use the new manual seed feature.
"""


# Set matplotlib backend BEFORE importing any neurotrack modules
import matplotlib
matplotlib.use('TkAgg')  # Use interactive backend

from neurotrack.data.datasets import NeuronPatchDataset
from neurotrack.environments.neuron_tracking_environment import NeuronTrackingEnvironment

# Step 1: Create your dataset
print("Creating dataset...")
dataset = NeuronPatchDataset(
    swc_dir="/home/brysongray/data/neurotrack_data/gold166/gold166_cleaned/all/morphology",
    img_dir="/home/brysongray/data/neurotrack_data/gold166/gold166_cleaned/all/images",
    crop_size=128,
    patches_per_image=10,
    crop_patches=False  # Use full images for manual seed selection
)

# Step 2: Create environment with manual_seed=True
print("\nCreating environment with manual seed selection...")
env = NeuronTrackingEnvironment(
    dataset=dataset,
    step_size=4.0,
    step_width=4.0,
    manual_seed=True  # THIS IS THE KEY PARAMETER!
)

# Step 3: Reset to trigger seed selection
# This will open the interactive seed selector
env.reset()
obs = env.get_state()

# Step 4: Check the result
print("\n" + "="*70)
print("SUCCESS!")
print(f"Seed position (z, y, x): {env.paths[0][-1]}")
print(f"Observation shape: {obs.shape}")
print("="*70)

# Now you can use the environment as normal
# For example, take some steps:
print("\nTaking a few random steps...")
for i in range(3):
    import torch
    action = torch.randn(3)  # Random action
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"  Step {i+1}: reward = {reward.item():.3f}, terminated = {terminated}, truncated = {truncated}")

print("\nDone! The environment is ready to use.")
