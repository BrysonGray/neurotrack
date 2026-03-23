#!/usr/bin/env python
"""
Test fixtures and utilities for neuron_data tests.

Author: Bryson Gray
2024
"""

import numpy as np
import tempfile
import torch
from pathlib import Path
from typing import List, Tuple, Dict
import csv


def create_sample_swc_data() -> List[Tuple]:
    """Create sample SWC data for testing."""
    # SWC format: (node_id, node_type, x, y, z, radius, parent_id)
    swc_data = [
        (1, 1, 0.0, 0.0, 0.0, 2.0, -1),    # Soma
        (2, 3, 5.0, 0.0, 0.0, 1.0, 1),     # Dendrite
        (3, 3, 10.0, 0.0, 0.0, 1.0, 2),    # Dendrite
        (4, 3, 15.0, 0.0, 0.0, 1.0, 3),    # Dendrite
        (5, 3, 20.0, 0.0, 0.0, 1.0, 4),    # Dendrite
        (6, 3, 10.0, 5.0, 0.0, 1.0, 3),    # Branch from node 3
        (7, 3, 10.0, 10.0, 0.0, 1.0, 6),   # Continue branch
        (8, 3, 10.0, 15.0, 0.0, 1.0, 7),   # End of branch
        (9, 3, 15.0, 5.0, 0.0, 1.0, 4),    # Another branch from node 4
        (10, 3, 15.0, 10.0, 0.0, 1.0, 9),  # Continue second branch
    ]
    return swc_data


def create_linear_swc_data(num_nodes: int = 10) -> List[Tuple]:
    """Create linear SWC data without branches for testing."""
    swc_data = []
    for i in range(num_nodes):
        node_id = i + 1
        node_type = 1 if i == 0 else 3  # Soma for first, dendrite for rest
        x = float(i * 2)  # Space nodes 2 units apart
        y = 0.0
        z = 0.0
        radius = 2.0 if i == 0 else 1.0
        parent_id = -1 if i == 0 else i  # Parent is previous node
        
        swc_data.append((node_id, node_type, x, y, z, radius, parent_id))
    
    return swc_data


def create_complex_swc_data() -> List[Tuple]:
    """Create complex branched SWC data for testing."""
    swc_data = [
        (1, 1, 0.0, 0.0, 0.0, 3.0, -1),     # Soma
        
        # Main trunk
        (2, 3, 5.0, 0.0, 0.0, 2.0, 1),
        (3, 3, 10.0, 0.0, 0.0, 2.0, 2),
        (4, 3, 15.0, 0.0, 0.0, 2.0, 3),
        (5, 3, 20.0, 0.0, 0.0, 2.0, 4),
        (6, 3, 25.0, 0.0, 0.0, 1.5, 5),
        
        # First branch from node 3
        (7, 3, 10.0, 5.0, 0.0, 1.5, 3),
        (8, 3, 10.0, 10.0, 0.0, 1.0, 7),
        (9, 3, 10.0, 15.0, 0.0, 1.0, 8),
        (10, 3, 5.0, 15.0, 0.0, 1.0, 9),
        
        # Second branch from node 4
        (11, 3, 15.0, 5.0, 0.0, 1.5, 4),
        (12, 3, 20.0, 10.0, 0.0, 1.0, 11),
        (13, 3, 25.0, 15.0, 0.0, 1.0, 12),
        
        # Sub-branch from branch 1
        (14, 3, 10.0, 12.0, 5.0, 1.0, 8),
        (15, 3, 10.0, 12.0, 10.0, 1.0, 14),
        
        # Third main branch from node 5
        (16, 3, 20.0, -5.0, 0.0, 1.5, 5),
        (17, 3, 20.0, -10.0, 0.0, 1.0, 16),
        (18, 3, 25.0, -15.0, 0.0, 1.0, 17),
    ]
    return swc_data


def create_mock_edge_list(swc_data: List[Tuple]) -> Dict[int, List[int]]:
    """Create undirected edge list from SWC data."""
    edge_list = {}
    
    # Initialize all nodes
    for node in swc_data:
        node_id = node[0]
        edge_list[node_id] = []
    
    # Add edges based on parent-child relationships
    for node in swc_data:
        node_id = node[0]
        parent_id = node[6]
        
        if parent_id != -1:
            # Add bidirectional edge
            edge_list[node_id].append(parent_id)
            edge_list[parent_id].append(node_id)
    
    return edge_list


def create_test_image(shape: Tuple[int, ...] = (32, 32, 32)) -> np.ndarray:
    """Create test image with some structure."""
    image = np.zeros(shape, dtype=np.uint8)
    
    # Add some structure - a line through the middle
    if len(shape) == 3:
        z, y, x = shape
        mid_z, mid_y = z // 2, y // 2
        # Draw a line from one side to the other
        for i in range(x):
            image[mid_z, mid_y, i] = 255
            # Add some branching
            if i > x // 2:
                for j in range(max(0, mid_y - 3), min(y, mid_y + 4)):
                    image[mid_z, j, i] = 128
    
    return image


def create_test_dataset_csv(temp_dir: str, num_entries: int = 5) -> str:
    """Create a test CSV file with dataset entries."""
    csv_path = Path(temp_dir) / "test_dataset.csv"
    
    entries = []
    for i in range(num_entries):
        complexity_level = i / (num_entries - 1)  # 0.0 to 1.0
        if complexity_level < 0.33:
            morphology = "simple"
        elif complexity_level < 0.67:
            morphology = "moderate"
        else:
            morphology = "complex"
        
        entry = {
            'swc_path': f'/fake/path/neuron_{i:03d}.swc',
            'img_path': f'/fake/path/neuron_{i:03d}.tif' if i % 2 == 0 else '',
            'artifact_level': complexity_level,
            'morphology': morphology
        }
        entries.append(entry)
    
    with open(csv_path, 'w', newline='') as f:
        fieldnames = ['swc_path', 'img_path', 'artifact_level', 'morphology']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(entries)
    
    return str(csv_path)


def create_test_swc_file(temp_dir: str, swc_data: List[Tuple], filename: str = "test_neuron.swc") -> str:
    """Create a test SWC file on disk."""
    swc_path = Path(temp_dir) / filename
    
    with open(swc_path, 'w') as f:
        f.write("# Test SWC file\n")
        f.write("# node_id type x y z radius parent_id\n")
        for node in swc_data:
            f.write(f"{node[0]} {node[1]} {node[2]:.3f} {node[3]:.3f} {node[4]:.3f} {node[5]:.3f} {node[6]}\n")
    
    return str(swc_path)


class MockRenderer:
    """Mock renderer for testing without actual drawing functionality."""
    
    def __init__(self, rng=None):
        self.rng = rng or np.random.default_rng()
    
    def draw_neuron(self, sections, shape, config):
        """Mock draw_neuron that returns a tensor with predictable values."""
        result = MockDrawResult()
        result.data = torch.ones(shape) * 0.5  # Medium intensity
        return result
    
    def draw_density(self, sections, shape):
        """Mock draw_density that returns a binary mask."""
        result = MockDrawResult()
        result.data = torch.zeros(shape)
        # Add some positive values in the center
        if len(shape) == 3:
            z, y, x = shape
            result.data[z//2-1:z//2+2, y//2-1:y//2+2, x//2-1:x//2+2] = 1.0
        return result
    
    def draw_section_labels(self, sections, shape):
        """Mock draw_section_labels that returns labeled sections."""
        result = MockDrawResult()
        result.data = torch.zeros(shape, dtype=torch.long)
        return result


class MockDrawResult:
    """Mock result object for draw operations."""
    
    def __init__(self):
        self.data = None


def create_mock_sections(num_sections: int = 3, points_per_section: int = 5) -> Dict[int, np.ndarray]:
    """Create mock sections data for testing."""
    sections = {}
    
    for section_id in range(num_sections):
        # Create points along a path
        points = []
        for i in range(points_per_section):
            x = section_id * 10.0 + i * 2.0
            y = i * 1.0
            z = section_id * 1.0
            points.append([x, y, z])
        
        sections[section_id] = np.array(points)
    
    return sections


# Pytest fixtures
import pytest


@pytest.fixture
def sample_swc_data():
    """Fixture providing sample SWC data."""
    return create_sample_swc_data()


@pytest.fixture
def linear_swc_data():
    """Fixture providing linear SWC data."""
    return create_linear_swc_data()


@pytest.fixture
def complex_swc_data():
    """Fixture providing complex branched SWC data."""
    return create_complex_swc_data()


@pytest.fixture
def test_image():
    """Fixture providing a test image."""
    return create_test_image()


@pytest.fixture
def mock_renderer():
    """Fixture providing a mock renderer."""
    return MockRenderer()


@pytest.fixture
def mock_sections():
    """Fixture providing mock sections data."""
    return create_mock_sections()


@pytest.fixture
def temp_dataset_csv():
    """Fixture providing a temporary dataset CSV file."""
    with tempfile.TemporaryDirectory() as temp_dir:
        csv_path = create_test_dataset_csv(temp_dir)
        yield csv_path


@pytest.fixture
def temp_swc_file():
    """Fixture providing a temporary SWC file."""
    with tempfile.TemporaryDirectory() as temp_dir:
        swc_data = create_sample_swc_data()
        swc_path = create_test_swc_file(temp_dir, swc_data)
        yield swc_path
