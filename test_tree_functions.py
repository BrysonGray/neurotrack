#!/usr/bin/env python3
"""
Test script for the neurotrack tree postprocessing functionality.
"""

import sys
import os
import numpy as np
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, '/home/brysongray/neurotrack')

from neurotrack.data.tree import restructure_neuron_tree, remove_soma

def test_tree_functions():
    """Test the tree processing functions with synthetic data."""
    
    print("Testing tree processing functions...")
    
    # Create synthetic path data
    # Path 1: Main trunk
    path1 = np.array([
        [0, 0, 0, 1.0],  # coordinates + radius
        [1, 0, 0, 1.0],
        [2, 0, 0, 1.0],
        [3, 0, 0, 1.0],
        [4, 0, 0, 1.0],
    ])
    
    # Path 2: Branch from path1
    path2 = np.array([
        [2, 0, 0, 1.0],  # starts from path1
        [2, 1, 0, 1.0],
        [2, 2, 0, 1.0],
    ])
    
    # Path 3: Another branch
    path3 = np.array([
        [3, 0, 0, 1.0],  # starts from path1
        [3, 0, 1, 1.0],
        [3, 0, 2, 1.0],
        [3, 0, 3, 1.0],
    ])
    
    paths = [path1, path2, path3]
    
    print(f"Input paths: {len(paths)} paths")
    for i, path in enumerate(paths):
        print(f"  Path {i+1}: {len(path)} points")
    
    # Test restructure_neuron_tree
    print("\nTesting restructure_neuron_tree...")
    try:
        restructured = restructure_neuron_tree(paths)
        print(f"Restructured into {len(restructured)} sections")
        for section_id, section in restructured.items():
            print(f"  Section {section_id}: {len(section)} points")
    except Exception as e:
        print(f"Error in restructure_neuron_tree: {e}")
    
    # Test remove_soma with synthetic SWC data
    print("\nTesting remove_soma...")
    try:
        # Create synthetic SWC data
        # Format: [id, type, x, y, z, radius, parent]
        swc_data = [
            [1, 1, 0, 0, 0, 10.0, -1],  # soma (large radius)
            [2, 2, 1, 0, 0, 5.0, 1],   # large dendrite
            [3, 2, 2, 0, 0, 3.0, 2],   # medium dendrite
            [4, 2, 3, 0, 0, 1.0, 3],   # normal dendrite
            [5, 2, 4, 0, 0, 1.0, 4],   # normal dendrite
            [6, 2, 5, 0, 0, 1.0, 5],   # normal dendrite
        ]
        
        print(f"Input SWC data: {len(swc_data)} nodes")
        
        processed_swc, seeds = remove_soma(swc_data, max_radius=7.0, verbose=True)
        
        print(f"After soma removal: {len(processed_swc)} nodes, {len(seeds)} seeds")
        print(f"Seeds: {seeds}")
        
    except Exception as e:
        print(f"Error in remove_soma: {e}")
    
    print("\nTree processing tests completed!")

if __name__ == "__main__":
    test_tree_functions()
