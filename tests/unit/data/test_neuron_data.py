#!/usr/bin/env python
"""
Unit tests for neuron_data.py module.

Author: Bryson Gray
2024
"""

import csv
import json
import numpy as np
import os
import pytest
import tempfile
import torch
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import sys

# Add parent directories to path for imports
test_file = Path(__file__)
project_root = test_file.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from neurotrack.data.neuron_data import (
    DrawingComplexityConfig,
    Dataset,
    DataLoader,
    DataGenerator,
    create_neuron_data_components
)


class TestDrawingComplexityConfig:
    """Test the DrawingComplexityConfig class."""
    
    def test_default_initialization(self):
        """Test default initialization of DrawingComplexityConfig."""
        config = DrawingComplexityConfig()
        
        # Check default ranges are properly set
        assert config.width_range == (1.0, 5.0)
        assert config.blur_range == (0.5, 3.0)
        assert config.foreground_mean_range == (0.6, 0.9)
        assert config.foreground_std_range == (0.05, 0.2)
        assert config.background_mean_range == (0.1, 0.4)
        assert config.background_std_range == (0.02, 0.1)
        assert config.mask_threshold_range == (0.05, 0.2)
        assert config.simplex_scale_range == (5.0, 20.0)
        assert config.simplex_amplitude_range == (0.5, 1.5)
        assert config.vignette_magnitude_range == (0.0, 0.5)
    
    def test_custom_initialization(self):
        """Test custom initialization of DrawingComplexityConfig."""
        config = DrawingComplexityConfig(
            width_range=(2.0, 6.0),
            blur_range=(1.0, 4.0)
        )
        
        assert config.width_range == (2.0, 6.0)
        assert config.blur_range == (1.0, 4.0)
        # Other values should remain default
        assert config.foreground_mean_range == (0.6, 0.9)
    
    @patch('neurotrack.data.neuron_data.draw')
    def test_interpolate_config_minimum_complexity(self, mock_draw):
        """Test interpolation at minimum complexity (0.0)."""
        config = DrawingComplexityConfig()
        mock_draw_config = Mock()
        mock_draw.DrawingConfig.return_value = mock_draw_config
        
        result = config.interpolate_config(0.0)
        
        # Check that DrawingConfig was called with min values
        mock_draw.DrawingConfig.assert_called_once()
        call_args = mock_draw.DrawingConfig.call_args
        
        # Verify key parameters are at minimum end
        assert call_args.kwargs['width'] == 1.0  # min of width_range
        assert call_args.kwargs['blur'] == 0.5   # min of blur_range
        assert call_args.kwargs['rgb'] is False
        
    @patch('neurotrack.data.neuron_data.draw')
    def test_interpolate_config_maximum_complexity(self, mock_draw):
        """Test interpolation at maximum complexity (1.0)."""
        config = DrawingComplexityConfig()
        mock_draw_config = Mock()
        mock_draw.DrawingConfig.return_value = mock_draw_config
        
        result = config.interpolate_config(1.0)
        
        # Check that DrawingConfig was called with max values
        mock_draw.DrawingConfig.assert_called_once()
        call_args = mock_draw.DrawingConfig.call_args
        
        # Verify key parameters are at maximum end
        assert call_args.kwargs['width'] == 5.0  # max of width_range
        assert call_args.kwargs['blur'] == 3.0   # max of blur_range
        
    @patch('neurotrack.data.neuron_data.draw')
    def test_interpolate_config_mid_complexity(self, mock_draw):
        """Test interpolation at mid complexity (0.5)."""
        config = DrawingComplexityConfig()
        mock_draw_config = Mock()
        mock_draw.DrawingConfig.return_value = mock_draw_config
        
        result = config.interpolate_config(0.5)
        
        # Check that DrawingConfig was called with mid values
        mock_draw.DrawingConfig.assert_called_once()
        call_args = mock_draw.DrawingConfig.call_args
        
        # Verify key parameters are at middle values
        assert call_args.kwargs['width'] == 3.0  # mid of width_range (1.0 + 0.5 * 4.0)
        assert call_args.kwargs['blur'] == 1.75  # mid of blur_range (0.5 + 0.5 * 2.5)
    
    @patch('neurotrack.data.neuron_data.draw')
    def test_interpolate_config_clamps_input(self, mock_draw):
        """Test that complexity input is clamped to [0.0, 1.0]."""
        config = DrawingComplexityConfig()
        mock_draw_config = Mock()
        mock_draw.DrawingConfig.return_value = mock_draw_config
        
        # Test negative value clamped to 0
        config.interpolate_config(-0.5)
        call_args_neg = mock_draw.DrawingConfig.call_args
        
        mock_draw.DrawingConfig.reset_mock()
        
        # Test value > 1 clamped to 1
        config.interpolate_config(1.5)
        call_args_pos = mock_draw.DrawingConfig.call_args
        
        # Both should be clamped appropriately
        assert call_args_neg.kwargs['width'] == 1.0  # Min value
        assert call_args_pos.kwargs['width'] == 5.0  # Max value


class TestDataset:
    """Test the Dataset class."""
    
    def test_empty_initialization(self):
        """Test initialization with no parameters creates empty dataset."""
        with pytest.warns(UserWarning, match="No valid data_dir or csv_path provided"):
            dataset = Dataset()
        
        assert len(dataset) == 0
        assert dataset.entries == []
    
    def test_initialization_with_nonexistent_paths(self):
        """Test initialization with non-existent paths."""
        with pytest.warns(UserWarning, match="No valid data_dir or csv_path provided"):
            dataset = Dataset(data_dir="/nonexistent/path", csv_path="/nonexistent/file.csv")
        
        assert len(dataset) == 0
    
    def test_scan_directory(self):
        """Test scanning directory for SWC files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create mock SWC files
            swc1 = temp_path / "neuron1.swc"
            swc2 = temp_path / "neuron2.swc"
            swc1.touch()
            swc2.touch()
            
            # Create corresponding image file for neuron1
            img1 = temp_path / "neuron1_image.tif"
            img1.touch()
            
            dataset = Dataset(data_dir=temp_dir)
            
            assert len(dataset) == 2
            
            # Check that entries were created
            swc_paths = [entry['swc_path'] for entry in dataset.entries]
            assert str(swc1) in swc_paths
            assert str(swc2) in swc_paths
            
            # Check that image file was found for neuron1
            entry1 = next(entry for entry in dataset.entries if 'neuron1' in entry['swc_path'])
            assert entry1['img_path'] == str(img1)
            
            # Check that neuron2 has no image
            entry2 = next(entry for entry in dataset.entries if 'neuron2' in entry['swc_path'])
            assert entry2['img_path'] is None
            
            # Check complexity values are assigned
            for entry in dataset.entries:
                assert 0.0 <= entry['artifact_level'] <= 0.5
                assert entry['morphology'] in ['simple', 'moderate', 'complex']
    
    def test_load_from_csv(self):
        """Test loading dataset from CSV file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            writer = csv.DictWriter(f, fieldnames=['swc_path', 'img_path', 'artifact_level', 'morphology'])
            writer.writeheader()
            writer.writerow({
                'swc_path': '/path/to/neuron1.swc',
                'img_path': '/path/to/neuron1.tif',
                'artifact_level': 0.3,
                'morphology': 'moderate'
            })
            writer.writerow({
                'swc_path': '/path/to/neuron2.swc',
                'img_path': '',
                'artifact_level': 0.8,
                'morphology': 'complex'
            })
            csv_path = f.name
        
        try:
            dataset = Dataset(csv_path=csv_path)
            
            assert len(dataset) == 2
            
            entry1 = dataset[0]
            assert entry1['swc_path'] == '/path/to/neuron1.swc'
            assert entry1['img_path'] == '/path/to/neuron1.tif'
            assert entry1['artifact_level'] == 0.3
            assert entry1['morphology'] == 'moderate'
            
            entry2 = dataset[1]
            assert entry2['swc_path'] == '/path/to/neuron2.swc'
            assert entry2['img_path'] is None  # Empty string becomes None
            assert entry2['artifact_level'] == 0.8
            assert entry2['morphology'] == 'complex'
        
        finally:
            os.unlink(csv_path)
    
    def test_load_from_csv_missing_columns(self):
        """Test loading from CSV with missing required columns."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            writer = csv.DictWriter(f, fieldnames=['swc_path'])  # Missing required columns
            writer.writeheader()
            writer.writerow({'swc_path': '/path/to/neuron1.swc'})
            csv_path = f.name
        
        try:
            with pytest.raises(ValueError, match="CSV must contain columns"):
                Dataset(csv_path=csv_path)
        finally:
            os.unlink(csv_path)
    
    def test_save_to_csv(self):
        """Test saving dataset to CSV file."""
        dataset = Dataset()
        dataset.entries = [
            {
                'swc_path': '/path/to/neuron1.swc',
                'img_path': '/path/to/neuron1.tif',
                'artifact_level': 0.3,
                'morphology': 'moderate'
            },
            {
                'swc_path': '/path/to/neuron2.swc',
                'img_path': None,
                'artifact_level': 0.8,
                'morphology': 'complex'
            }
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            csv_path = f.name
        
        try:
            dataset.save_to_csv(csv_path)
            
            # Read back and verify
            loaded_dataset = Dataset(csv_path=csv_path)
            assert len(loaded_dataset) == 2
            
            entry1 = loaded_dataset[0]
            assert entry1['swc_path'] == '/path/to/neuron1.swc'
            assert entry1['artifact_level'] == 0.3
            assert entry1['morphology'] == 'moderate'
            
        finally:
            os.unlink(csv_path)
    
    def test_get_complexity_distribution(self):
        """Test getting complexity distribution from dataset."""
        dataset = Dataset()
        dataset.entries = [
            {'morphology': 'simple', 'artifact_level': 0.1},
            {'morphology': 'simple', 'artifact_level': 0.2},
            {'morphology': 'moderate', 'artifact_level': 0.5},
            {'morphology': 'complex', 'artifact_level': 0.8},
        ]
        
        distribution = dataset.get_complexity_distribution()
        
        # Check morphology distribution
        assert distribution['morphology_distribution']['simple'] == 2
        assert distribution['morphology_distribution']['moderate'] == 1
        assert distribution['morphology_distribution']['complex'] == 1
        
        # Check artifact level statistics
        artifact_stats = distribution['artifact_stats']
        assert artifact_stats['mean'] == 0.4  # (0.1+0.2+0.5+0.8)/4
        assert artifact_stats['min'] == 0.1
        assert artifact_stats['max'] == 0.8
    
    def test_getitem_and_len(self):
        """Test indexing and length methods."""
        dataset = Dataset()
        dataset.entries = [
            {'swc_path': 'neuron1.swc', 'artifact_level': 0.1, 'morphology': 'simple'},
            {'swc_path': 'neuron2.swc', 'artifact_level': 0.5, 'morphology': 'moderate'}
        ]
        
        assert len(dataset) == 2
        assert dataset[0]['swc_path'] == 'neuron1.swc'
        assert dataset[1]['swc_path'] == 'neuron2.swc'


class TestDataLoader:
    """Test the DataLoader class."""
    
    def create_test_dataset(self):
        """Create a test dataset with known complexity values."""
        dataset = Dataset()
        dataset.entries = [
            {'swc_path': 'simple.swc', 'artifact_level': 0.1, 'morphology': 'simple'},
            {'swc_path': 'moderate.swc', 'artifact_level': 0.5, 'morphology': 'moderate'},
            {'swc_path': 'complex.swc', 'artifact_level': 0.9, 'morphology': 'complex'},
        ]
        return dataset
    
    def test_initialization(self):
        """Test DataLoader initialization."""
        dataset = self.create_test_dataset()
        dataloader = DataLoader(dataset, complexity=0.3)
        
        assert dataloader.dataset == dataset
        assert dataloader.complexity == 0.3
        assert dataloader.current_idx == 0
        assert len(dataloader.weights) == len(dataset)
    
    def test_initialization_empty_dataset(self):
        """Test DataLoader with empty dataset."""
        dataset = Dataset()
        dataloader = DataLoader(dataset, complexity=0.5)
        
        assert len(dataloader.weights) == 0
    
    def test_set_complexity(self):
        """Test updating complexity parameter."""
        dataset = self.create_test_dataset()
        dataloader = DataLoader(dataset, complexity=0.0)
        
        # Change complexity
        dataloader.set_complexity(0.8)
        assert dataloader.complexity == 0.8
        
        # Test clamping
        dataloader.set_complexity(-0.5)
        assert dataloader.complexity == 0.0
        
        dataloader.set_complexity(1.5)
        assert dataloader.complexity == 1.0
    
    def test_sampling_weights_low_complexity(self):
        """Test that sampling weights favor simple neurons for low complexity."""
        dataset = self.create_test_dataset()
        dataloader = DataLoader(dataset, complexity=0.1)
        
        # For low complexity, simple neurons should have higher weights
        simple_idx = 0  # simple neuron
        complex_idx = 2  # complex neuron
        
        assert dataloader.weights[simple_idx] > dataloader.weights[complex_idx]
    
    def test_sampling_weights_high_complexity(self):
        """Test that sampling weights favor complex neurons for high complexity."""
        dataset = self.create_test_dataset()
        dataloader = DataLoader(dataset, complexity=0.9)
        
        # For high complexity, complex neurons should have higher weights
        simple_idx = 0  # simple neuron
        complex_idx = 2  # complex neuron
        
        assert dataloader.weights[complex_idx] > dataloader.weights[simple_idx]
    
    def test_sample(self):
        """Test sampling from dataset."""
        dataset = self.create_test_dataset()
        dataloader = DataLoader(dataset, complexity=0.5)
        
        # Set random seed for reproducible test
        np.random.seed(42)
        
        sample = dataloader.sample()
        assert sample in dataset.entries
        assert 'swc_path' in sample
        assert 'morphology' in sample
    
    def test_sample_empty_dataset(self):
        """Test sampling from empty dataset raises error."""
        dataset = Dataset()
        dataloader = DataLoader(dataset, complexity=0.5)
        
        with pytest.raises(ValueError, match="Dataset is empty"):
            dataloader.sample()
    
    def test_iteration(self):
        """Test iterating through dataloader."""
        dataset = self.create_test_dataset()
        dataloader = DataLoader(dataset, complexity=0.5)
        
        items = list(dataloader)
        assert len(items) == len(dataset)
        
        # Check all items were returned in order
        for i, item in enumerate(items):
            assert item == dataset[i]
    
    def test_iterator_protocol(self):
        """Test iterator protocol methods."""
        dataset = self.create_test_dataset()
        dataloader = DataLoader(dataset, complexity=0.5)
        
        # Test iter returns self
        assert iter(dataloader) is dataloader
        
        # Test manual iteration
        assert dataloader.current_idx == 0
        item1 = next(dataloader)
        assert dataloader.current_idx == 1
        assert item1 == dataset[0]
        
        item2 = next(dataloader)
        assert dataloader.current_idx == 2
        assert item2 == dataset[1]
        
        item3 = next(dataloader)
        assert dataloader.current_idx == 3
        assert item3 == dataset[2]
        
        # Should raise StopIteration
        with pytest.raises(StopIteration):
            next(dataloader)


class TestDataGenerator:
    """Test the DataGenerator class."""
    
    def test_initialization_default(self):
        """Test DataGenerator initialization with defaults."""
        generator = DataGenerator()
        
        assert generator.cache_dir is None
        assert generator.patch_size == 64
        assert isinstance(generator.complexity_config, DrawingComplexityConfig)
        assert generator.rng is not None
        assert generator.renderer is not None
    
    def test_initialization_with_parameters(self):
        """Test DataGenerator initialization with custom parameters."""
        with tempfile.TemporaryDirectory() as temp_dir:
            custom_config = DrawingComplexityConfig(width_range=(2.0, 6.0))
            
            generator = DataGenerator(
                cache_dir=temp_dir,
                patch_size=32,
                complexity_config=custom_config,
                rng_seed=12345
            )
            
            assert generator.cache_dir == Path(temp_dir)
            assert generator.patch_size == 32
            assert generator.complexity_config == custom_config
            assert generator.cache_dir.exists()
    
    @patch('neurotrack.data.neuron_data.load')
    def test_load_files_swc_only(self, mock_load):
        """Test loading SWC file without image."""
        generator = DataGenerator()
        mock_swc_data = [
            (1, 1, 0.0, 0.0, 0.0, 1.0, -1),
            (2, 3, 1.0, 0.0, 0.0, 1.0, 1)
        ]
        mock_load.swc.return_value = mock_swc_data
        
        swc_list, img_tensor = generator.load_files("test.swc")
        
        mock_load.swc.assert_called_once_with("test.swc")
        assert swc_list == mock_swc_data
        assert img_tensor is None
    
    @patch('neurotrack.data.neuron_data.load')
    @patch('neurotrack.data.neuron_data.tf')
    @patch('neurotrack.data.neuron_data.os.path.exists')
    def test_load_files_with_image(self, mock_exists, mock_tf, mock_load):
        """Test loading SWC file with corresponding image."""
        generator = DataGenerator()
        mock_swc_data = [(1, 1, 0.0, 0.0, 0.0, 1.0, -1)]
        mock_load.swc.return_value = mock_swc_data
        mock_exists.return_value = True
        
        # Mock 3D image data
        mock_img_array = np.random.randint(0, 255, (64, 64, 64), dtype=np.uint8)
        mock_tf.imread.return_value = mock_img_array
        
        swc_list, img_tensor = generator.load_files("test.swc", "test.tif")
        
        mock_load.swc.assert_called_once_with("test.swc")
        mock_tf.imread.assert_called_once_with("test.tif")
        assert swc_list == mock_swc_data
        assert img_tensor is not None
        assert img_tensor.shape == (1, 64, 64, 64)  # Channel dimension added
        assert torch.allclose(img_tensor, torch.from_numpy(mock_img_array.astype(np.float32) / 255.0))
    
    def test_complexity_to_category(self):
        """Test complexity to category conversion."""
        generator = DataGenerator()
        
        assert generator._complexity_to_category(0.1) == "simple"
        assert generator._complexity_to_category(0.32) == "simple"
        assert generator._complexity_to_category(0.4) == "moderate"
        assert generator._complexity_to_category(0.66) == "moderate"
        assert generator._complexity_to_category(0.8) == "complex"
        assert generator._complexity_to_category(1.0) == "complex"
    
    def test_complexity_to_mode(self):
        """Test complexity to mode conversion."""
        generator = DataGenerator()
        
        assert generator._complexity_to_mode(0.1) == "no_branch"
        assert generator._complexity_to_mode(0.32) == "no_branch"
        assert generator._complexity_to_mode(0.4) == "one_branch"
        assert generator._complexity_to_mode(0.66) == "one_branch"
        assert generator._complexity_to_mode(0.8) == "any_branch"
        assert generator._complexity_to_mode(1.0) == "any_branch"
    
    @patch('neurotrack.data.neuron_data.load')
    def test_generate_reward_mask(self, mock_load):
        """Test reward mask generation."""
        generator = DataGenerator()
        
        # Mock subtree data
        subtree = [
            (1, 1, 0.0, 0.0, 0.0, 1.0, -1),
            (2, 3, 1.0, 0.0, 0.0, 1.0, 1),
            (3, 3, 2.0, 0.0, 0.0, 1.0, 2)
        ]
        
        # Mock parse_swc to return sections
        mock_sections = {0: np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]])}
        mock_load.parse_swc.return_value = (mock_sections, None)
        
        # Mock renderer.draw_density
        mock_density = Mock()
        mock_density.data = torch.zeros(32, 32, 32)
        generator.renderer.draw_density = Mock(return_value=mock_density)
        
        shape = (32, 32, 32)
        reward_mask = generator.generate_reward_mask(subtree, shape)
        
        mock_load.parse_swc.assert_called_once_with(subtree)
        generator.renderer.draw_density.assert_called_once_with(mock_sections, shape)
        assert torch.equal(reward_mask, torch.zeros(32, 32, 32))
    
    @patch('neurotrack.data.neuron_data.load')
    def test_simulate_neuron_image(self, mock_load):
        """Test simulated neuron image generation."""
        generator = DataGenerator()
        
        # Mock subtree data
        subtree = [
            (1, 1, 0.0, 0.0, 0.0, 1.0, -1),
            (2, 3, 1.0, 0.0, 0.0, 1.0, 1)
        ]
        
        # Mock parse_swc to return sections
        mock_sections = {0: np.array([[0, 0, 0], [1, 0, 0]])}
        mock_load.parse_swc.return_value = (mock_sections, None)
        
        # Mock renderer.draw_neuron
        mock_image = Mock()
        mock_image.data = torch.ones(32, 32, 32)
        generator.renderer.draw_neuron = Mock(return_value=mock_image)
        
        shape = (32, 32, 32)
        complexity = 0.5
        
        result = generator.simulate_neuron_image(subtree, shape, complexity)
        
        mock_load.parse_swc.assert_called_once_with(subtree)
        generator.renderer.draw_neuron.assert_called_once()
        assert torch.equal(result, torch.ones(32, 32, 32))
    
    def test_save_data(self):
        """Test saving generated data to disk."""
        with tempfile.TemporaryDirectory() as temp_dir:
            generator = DataGenerator()
            
            # Create mock data
            data = {
                'image': torch.ones(1, 32, 32, 32),
                'reward_mask': torch.zeros(32, 32, 32),
                'subtree': [
                    (1, 1, 0.0, 0.0, 0.0, 1.0, -1),
                    (2, 3, 1.0, 0.0, 0.0, 1.0, 1)
                ],
                'metadata': {
                    'source': 'simulated',
                    'complexity': (0.5, 'moderate')
                }
            }
            
            # Mock tifffile.imwrite
            with patch('neurotrack.data.neuron_data.tf.imwrite') as mock_imwrite:
                generator.save_data(data, temp_dir, prefix="test")
                
                # Check that files were written
                assert mock_imwrite.call_count == 2  # Image and reward mask
                
                # Check SWC file was created
                swc_path = Path(temp_dir) / "test_subtree.swc"
                assert swc_path.exists()
                
                # Check metadata file was created
                meta_path = Path(temp_dir) / "test_metadata.json"
                assert meta_path.exists()
                
                # Verify metadata content
                with open(meta_path) as f:
                    metadata = json.load(f)
                    assert metadata['source'] == 'simulated'
                    assert metadata['complexity'] == [0.5, 'moderate']  # Lists in JSON


class TestConvenienceFunctions:
    """Test convenience functions."""
    
    @patch('neurotrack.data.neuron_data.Dataset')
    @patch('neurotrack.data.neuron_data.DataLoader')
    @patch('neurotrack.data.neuron_data.DataGenerator')
    def test_create_neuron_data_components(self, mock_generator_class, mock_loader_class, mock_dataset_class):
        """Test create_neuron_data_components function."""
        # Setup mocks
        mock_dataset = Mock()
        mock_loader = Mock()
        mock_generator = Mock()
        
        mock_dataset_class.return_value = mock_dataset
        mock_loader_class.return_value = mock_loader
        mock_generator_class.return_value = mock_generator
        
        custom_complexity_config = DrawingComplexityConfig(width_range=(2.0, 6.0))
        
        # Call function
        dataset, dataloader, data_generator = create_neuron_data_components(
            data_dir="/test/data",
            complexity=0.3,
            complexity_config=custom_complexity_config,
            cache_dir="/test/cache",
            rng_seed=42
        )
        
        # Verify components were created with correct parameters
        mock_dataset_class.assert_called_once_with(data_dir="/test/data")
        mock_loader_class.assert_called_once_with(dataset=mock_dataset, complexity=0.3)
        mock_generator_class.assert_called_once_with(
            cache_dir="/test/cache",
            complexity_config=custom_complexity_config,
            rng_seed=42
        )
        
        # Verify return values
        assert dataset == mock_dataset
        assert dataloader == mock_loader
        assert data_generator == mock_generator


# Integration test with mocked dependencies
class TestDataGeneratorIntegration:
    """Integration tests for DataGenerator with mocked dependencies."""
    
    @patch('neurotrack.data.neuron_data.load')
    @patch('neurotrack.data.neuron_data.tf')
    @patch('neurotrack.data.neuron_data.os.path.exists')
    def test_generate_data_simulated(self, mock_exists, mock_tf, mock_load):
        """Test full data generation pipeline for simulated data."""
        generator = DataGenerator(rng_seed=42)
        
        # Mock entry without image (simulated)
        entry = {
            'swc_path': 'test.swc',
            'img_path': None,
            'artifact_level': 0.3,
            'morphology': 'moderate'
        }
        
        # Mock SWC data - simple linear chain for "no_branch" mode
        # Format: (id, type, x, y, z, radius, parent_id)
        mock_swc_data = [
            (1, 3, 0.0, 0.0, 0.0, 1.0, -1),  # Root (endpoint)
            (2, 3, 1.0, 0.0, 0.0, 1.0, 1),
            (3, 3, 2.0, 0.0, 0.0, 1.0, 2),
            (4, 3, 3.0, 0.0, 0.0, 1.0, 3),
            (5, 3, 4.0, 0.0, 0.0, 1.0, 4),
            (6, 3, 5.0, 0.0, 0.0, 1.0, 5),  # Endpoint
        ]
        mock_load.swc.return_value = mock_swc_data
        
        # Mock undirected_edge_list - simple linear chain
        edge_list = {
            1: [2],           # Endpoint: connects to 2 only
            2: [1, 3],        # Linear: connects to 1 and 3
            3: [2, 4],        # Linear: connects to 2 and 4
            4: [3, 5],        # Linear: connects to 3 and 5
            5: [4, 6],        # Linear: connects to 4 and 6
            6: [5],           # Endpoint: connects to 5 only
        }
        mock_load.undirected_edge_list.return_value = edge_list
        
        # Mock parse_swc
        mock_sections = {0: np.array([[float(i), 0, 0] for i in range(5)])}
        mock_load.parse_swc.return_value = (mock_sections, None)
        
        # Mock renderer methods
        mock_image = Mock()
        mock_image.data = torch.ones(25, 20, 20)
        generator.renderer.draw_neuron = Mock(return_value=mock_image)
        
        mock_density = Mock()
        mock_density.data = torch.zeros(25, 20, 20)
        generator.renderer.draw_density = Mock(return_value=mock_density)
        
        # Generate data - use complexity=0.2 for "no_branch" mode which is simpler
        results = generator.generate_data(entry, num_subtrees=1, num_edges=4, complexity=0.2)
        
        # Verify results
        assert len(results) == 1
        result = results[0]
        
        assert 'image' in result
        assert 'subtree' in result
        assert 'reward_mask' in result
        assert 'metadata' in result
        
        metadata = result['metadata']
        assert metadata['source'] == 'simulated'
        assert metadata['original_swc'] == 'test.swc'
        assert metadata['simulation_complexity'] == 0.2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
