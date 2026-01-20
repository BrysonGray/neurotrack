"""
Main testing pipeline for neuron tracing model evaluation.
"""

import json
import os
import shutil
import traceback
from pathlib import Path
from typing import Optional, Dict, List, Any
import numpy as np
import torch
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from data_prep import tree, save, load, NeuronPatchDataset
from environments.neuron_tracking_environment import NeuronTrackingEnvironment
from models.deeper_cnn import ConvNet
from solvers import sac
from neurotrack.testing import postprocess, evaluation


class TestingPipeline:
    """
    Complete testing pipeline for neuron tracing model evaluation.
    
    This class handles the full workflow:
    1. Loading configuration from JSON
    2. Running inference with trained models
    3. Post-processing reconstructed paths
    4. Evaluating against ground truth
    5. Saving results and metrics
    
    Parameters
    ----------
    config_path : str
        Path to JSON configuration file
        
    Attributes
    ----------
    config : dict
        Parsed configuration parameters
    results : list
        Per-neuron evaluation results
    """
    
    def __init__(self, config_path: str):
        """
        Initialize pipeline with JSON configuration file.
        
        Parameters
        ----------
        config_path : str
            Path to JSON configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
        self._validate_config()
        self.results = []
        
        # Set random seed if specified
        if 'rng_seed' in self.config and self.config['rng_seed'] is not None:
            np.random.seed(self.config['rng_seed'])
            torch.manual_seed(self.config['rng_seed'])
            if torch.cuda.is_available():
                torch.cuda.manual_seed(self.config['rng_seed'])
    
    def _load_config(self) -> dict:
        """
        Load and parse JSON configuration file.
        
        Returns
        -------
        dict
            Configuration parameters
            
        Raises
        ------
        FileNotFoundError
            If config file does not exist
        json.JSONDecodeError
            If JSON is malformed
        """
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            config = json.load(f)
        
        # Set default values for optional parameters
        defaults = {
            # Tracking parameters
            'step_size': 2.0,
            'step_width': 4.0,
            'alpha': 1.0,
            'beta': 1e-3,
            'friction': 1e-4,
            'section_masking': False,
            'repeat_starts': True,
            'rng_seed': None,
            'n_trials': 5,
            'classifier_weights': None,
            
            # Post-processing parameters
            'min_branch_length': 10.0,
            'smoothing_window': 5,
            'overlap_threshold': 0.8,
            
            # Evaluation parameters
            'distance_threshold': 2.0,
        }
        
        # Update config with defaults for missing keys
        for key, value in defaults.items():
            if key not in config:
                config[key] = value
        
        return config
    
    def _validate_config(self):
        """
        Validate configuration parameters.
        
        Raises
        ------
        ValueError
            If required parameters are missing or paths don't exist
        """
        # Required parameters
        required = [
            'img_dir',
            'swc_dir',
            'out_dir',
            'test_name',
            'sac_weights'
        ]
        
        missing = [key for key in required if key not in self.config]
        if missing:
            raise ValueError(f"Missing required config parameters: {missing}")
        
        # Validate paths exist
        paths_to_check = {
            'img_dir': self.config['img_dir'],
            'swc_dir': self.config['swc_dir'],
            'sac_weights': self.config['sac_weights']
        }
        
        if self.config.get('classifier_weights') is not None:
            paths_to_check['classifier_weights'] = self.config['classifier_weights']
        
        for name, path in paths_to_check.items():
            if not os.path.exists(path):
                raise ValueError(f"{name} does not exist: {path}")
        
        # Create output directory if it doesn't exist
        os.makedirs(self.config['out_dir'], exist_ok=True)
        
        print(f"Configuration validated successfully")
        print(f"  Test name: {self.config['test_name']}")
        print(f"  Output directory: {self.config['out_dir']}")
    
    def run(self) -> Dict[str, Any]:
        """
        Execute the complete testing pipeline.
        
        Returns
        -------
        dict
            Summary results including per-neuron metrics and aggregated statistics
        """
        print(f"\n{'='*60}")
        print(f"Starting Testing Pipeline: {self.config['test_name']}")
        print(f"{'='*60}\n")
        
        # Phase 1: Run inference
        print("Phase 1: Running inference...")
        inference_results = self._run_inference()
        
        # Phase 2: Post-process results
        print("\nPhase 2: Post-processing reconstructions...")
        postprocessed_results = self._postprocess_all(inference_results)
        self.postprocessed_results = postprocessed_results  # Store for later use
        
        # Phase 3: Evaluate against ground truth
        print("\nPhase 3: Evaluating against ground truth...")
        evaluation_results = self._evaluate_all(postprocessed_results)
        
        # Phase 4: Save results
        print("\nPhase 4: Saving results...")
        self._save_results(evaluation_results)
        
        # Compute summary statistics
        summary = self._compute_summary(evaluation_results)
        
        print(f"\n{'='*60}")
        print(f"Pipeline Complete!")
        print(f"{'='*60}")
        print(f"\nSummary Statistics:")
        print(f"  Neurons evaluated: {summary['n_neurons']}")
        print(f"  Mean bidirectional distance: {summary['mean_bidirectional_distance']:.4f}")
        print(f"  Mean directed divergence (pred→gt): {summary['mean_directed_pred_to_gt']:.4f}")
        print(f"  Mean directed divergence (gt→pred): {summary['mean_directed_gt_to_pred']:.4f}")
        
        return summary
    
    def _run_inference(self) -> List[Dict[str, Any]]:
        """
        Run inference on test images using trained SAC model.
        
        Returns
        -------
        list
            List of dicts containing inference results for each neuron
        """
        print(f"  Loading models...")
        
        DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        dtype = torch.float32
        
        # Load SAC actor model
        print(f"    Loading SAC actor from: {self.config['sac_weights']}")
        in_channels = 2  # Standard for NeuronTrackingEnvironment
        actor = ConvNet(chin=in_channels, chout=4)
        actor = actor.to(device=DEVICE, dtype=dtype)
        
        model_dicts = torch.load(self.config['sac_weights'], map_location=DEVICE)
        actor.load_state_dict(model_dicts['policy_state_dict'])
        actor.eval()
        print(f"    SAC actor loaded")
        
        # Load Q-network if n_trials > 1
        Q_net = None
        if self.config['n_trials'] > 1:
            print(f"    Loading Q-network for multi-trial selection...")
            Q_net = ConvNet(chin=in_channels+3, chout=1)
            Q_net = Q_net.to(device=DEVICE, dtype=dtype)
            Q_net.load_state_dict(model_dicts['Q1_state_dict'])
            Q_net.eval()
            print(f"    Q-network loaded")
        
        # Create dataset and environment
        print(f"  Creating dataset and environment...")
        patch_radius = 17  # Standard patch radius
        rng = np.random.default_rng(self.config.get('rng_seed', 0))
        
        # Create dataset
        dataset = NeuronPatchDataset(
            img_dir=self.config['img_dir'],
            swc_dir=self.config['swc_dir'],
            crop_size=128,
            patches_per_image=10,
            alpha=0.5,
            rng=rng,
            crop_patches=False  # Use full images during inference
        )
        
        # Create environment
        env = NeuronTrackingEnvironment(
            dataset=dataset,
            radius=patch_radius,
            step_size=self.config['step_size'],
            step_width=self.config['step_width'],
            max_len=10000,
            max_paths=1000,
            gamma=0.99,
            branching=False,  # Typically disabled during inference
            repeat_starts=self.config['repeat_starts'],
            start_idx=0
        )
        print(f"    Environment created with {len(dataset.img_files)} images")
        
        # Create inference output directory
        inference_outdir = os.path.join(self.config['out_dir'], 'inference_outputs')
        os.makedirs(inference_outdir, exist_ok=True)
        
        # Run inference
        print(f"  Running inference on {len(dataset.img_files)} images...")
        print(f"    n_trials: {self.config['n_trials']}")
        print(f"    step_size: {self.config['step_size']}")
        print(f"    step_width: {self.config['step_width']}")
        
        inference_results = sac.inference(
            env=env,
            actor=actor,
            outdir=inference_outdir,
            Q_net=Q_net,
            n_trials=self.config['n_trials'],
            show=False,  # Don't show during batch processing
            show_live=False,
            save_paths=True,  # Save raw outputs to disk as backup
            sync=False,
            stochastic=False  # Use deterministic policy (mean actions)
        )
        
        print(f"    Inference complete: {len(inference_results)} neurons processed")
        
        return inference_results
    
    def _postprocess_all(self, inference_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Apply post-processing to all reconstructed neuron paths.
        
        Parameters
        ----------
        inference_results : list
            Raw inference results from _run_inference()
            
        Returns
        -------
        list
            Post-processed results with cleaned paths
        """
        postprocessed_results = []
        
        print(f"  Processing {len(inference_results)} neurons...")
        
        for i, result in enumerate(inference_results):
            neuron_name = result['neuron_name']
            raw_paths = result['paths']
            
            print(f"    [{i+1}/{len(inference_results)}] {neuron_name}: {len(raw_paths)} raw paths")
            
            try:
                # Convert paths from list of lists to list of torch tensors for tree functions
                # Paths from inference are lists of numpy arrays with shape (N, 3) for [x, y, z]
                paths_as_tensors = []
                for path in raw_paths:
                    if isinstance(path, list):
                        # Convert list of numpy arrays to single tensor
                        path_tensor = torch.stack([torch.from_numpy(p) if isinstance(p, np.ndarray) else p for p in path])
                    elif isinstance(path, np.ndarray):
                        path_tensor = torch.from_numpy(path)
                    else:
                        path_tensor = path
                    paths_as_tensors.append(path_tensor)
                
                # Step 1: Restructure neuron tree (hierarchical branching structure)
                # This returns sections as dict of tensors
                sections = tree.restructure_neuron_tree(paths_as_tensors, input_type='paths')
                print(f"      → Restructured into {len(sections)} sections")
                
                # Step 2: Remove short branches
                # Convert sections to list of paths for post-processing
                paths = [v.cpu().numpy() if isinstance(v, torch.Tensor) else v 
                        for v in sections.values()]
                paths = postprocess.remove_short_paths(
                    paths, 
                    min_length=self.config['min_branch_length']
                )
                
                # Step 3: Smooth paths
                paths = postprocess.smooth_paths(
                    paths, 
                    window_size=self.config['smoothing_window']
                )
                
                # Step 4: Merge redundant/overlapping paths
                paths = postprocess.merge_redundant_paths(
                    paths, 
                    overlap_threshold=self.config['overlap_threshold']
                )
                
                # Step 5: Convert paths to torch tensors for SWC conversion
                # paths are numpy arrays with shape (N, 3) for coordinates only
                # Convert to list of torch tensors with shape (N, 3) for paths_to_swc
                processed_paths = [torch.from_numpy(path) if isinstance(path, np.ndarray) else path 
                                  for path in paths]
                print(f"      → Final: {len(processed_paths)} processed paths")
                
                # Step 6: Convert to SWC format
                # paths_to_swc expects list of tensors/arrays with shape (N, 3)
                swc_list = save.paths_to_swc(processed_paths)
                print(f"      → Converted to SWC: {len(swc_list)} nodes")
                
                # Store processed result
                postprocessed_results.append({
                    'neuron_name': neuron_name,
                    'swc_list': swc_list,
                    'processed_paths': processed_paths,
                    'n_raw_paths': len(raw_paths),
                    'n_processed_paths': len(processed_paths),
                    'n_swc_nodes': len(swc_list)
                })
                
            except Exception as e:
                print(f"      ✗ Error processing {neuron_name}: {e}")
                traceback.print_exc()
                # Store empty result to maintain alignment
                postprocessed_results.append({
                    'neuron_name': neuron_name,
                    'swc_list': [],
                    'processed_paths': [],
                    'n_raw_paths': len(raw_paths),
                    'n_processed_paths': 0,
                    'n_swc_nodes': 0,
                    'error': str(e)
                })
        
        print(f"  Post-processing complete")
        return postprocessed_results
    
    def _evaluate_all(self, postprocessed_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Evaluate all reconstructions against ground truth.
        
        Parameters
        ----------
        postprocessed_results : list
            Post-processed reconstruction results
            
        Returns
        -------
        list
            Evaluation metrics for each neuron
        """
        evaluation_results = []
        
        print(f"  Evaluating {len(postprocessed_results)} neurons against ground truth...")
        
        # Get list of ground truth SWC files
        gt_swc_dir = Path(self.config['swc_dir'])
        gt_swc_files = {f.stem: f for f in gt_swc_dir.glob("*.swc")}
        
        for i, result in enumerate(postprocessed_results):
            neuron_name = result['neuron_name']
            pred_swc = result['swc_list']
            
            # Extract base name from neuron path (may include directory structure)
            neuron_basename = Path(neuron_name).stem
            
            print(f"    [{i+1}/{len(postprocessed_results)}] {neuron_basename}")
            
            # Check if there was an error in post-processing
            if 'error' in result:
                print(f"      ✗ Skipping (post-processing failed): {result['error']}")
                evaluation_results.append({
                    'neuron_name': neuron_name,
                    'error': result['error'],
                    'skipped': True
                })
                continue
            
            # Find matching ground truth file
            gt_file = None
            if neuron_basename in gt_swc_files:
                gt_file = gt_swc_files[neuron_basename]
            else:
                # Try partial matching
                for gt_name, gt_path in gt_swc_files.items():
                    if neuron_basename in gt_name or gt_name in neuron_basename:
                        gt_file = gt_path
                        break
            
            if gt_file is None:
                print(f"      ✗ No ground truth found for {neuron_basename}")
                evaluation_results.append({
                    'neuron_name': neuron_name,
                    'error': 'No ground truth file found',
                    'skipped': True
                })
                continue
            
            try:
                # Load ground truth SWC
                gt_swc = load.swc(str(gt_file), verbose=False)
                print(f"      Ground truth: {len(gt_swc)} nodes, Predicted: {len(pred_swc)} nodes")
                
                # Check if predicted reconstruction is empty
                if len(pred_swc) == 0:
                    print(f"      ✗ Empty prediction")
                    evaluation_results.append({
                        'neuron_name': neuron_name,
                        'n_nodes_pred': 0,
                        'n_nodes_gt': len(gt_swc),
                        'error': 'Empty prediction',
                        'skipped': True
                    })
                    continue
                
                # Evaluate using existing functions
                metrics = evaluation.evaluate_reconstruction(
                    pred_swc, 
                    gt_swc, 
                    threshold=self.config['distance_threshold']
                )
                
                # Add metadata
                metrics['neuron_name'] = neuron_name
                metrics['gt_file'] = str(gt_file)
                metrics['skipped'] = False
                
                # Add post-processing statistics
                metrics['n_raw_paths'] = result['n_raw_paths']
                metrics['n_processed_paths'] = result['n_processed_paths']
                
                print(f"      → Bidirectional distance: {metrics['bidirectional_distance']:.4f}")
                print(f"      → Directed (pred→gt): {metrics['directed_div_pred_to_gt']:.4f}")
                print(f"      → Directed (gt→pred): {metrics['directed_div_gt_to_pred']:.4f}")
                
                evaluation_results.append(metrics)
                
            except Exception as e:
                print(f"      ✗ Evaluation error: {e}")
                evaluation_results.append({
                    'neuron_name': neuron_name,
                    'error': str(e),
                    'skipped': True
                })
        
        print(f"  Evaluation complete")
        return evaluation_results
    
    def _save_results(self, evaluation_results: List[Dict[str, Any]]):
        """
        Save evaluation results to disk.
        
        Parameters
        ----------
        evaluation_results : list
            Per-neuron evaluation metrics
        """
        out_dir = Path(self.config['out_dir'])
        
        # Create subdirectories
        swc_dir = out_dir / 'processed_swc'
        swc_dir.mkdir(exist_ok=True)
        
        # Save per-neuron metrics to CSV
        metrics_csv = out_dir / f"{self.config['test_name']}_metrics.csv"
        evaluation.save_evaluation_results(
            evaluation_results, 
            str(metrics_csv),
            summary_path=str(out_dir / f"{self.config['test_name']}_summary.json")
        )
        
        # Save processed SWC files
        print(f"  Saving processed SWC files to: {swc_dir}")
        saved_count = 0
        for result in evaluation_results:
            if result.get('skipped', False):
                continue
            
            neuron_name = result['neuron_name']
            neuron_basename = Path(neuron_name).stem
            
            # Get the corresponding swc_list from postprocessed results
            # We need to match by neuron_name
            swc_list = None
            for post_result in self.postprocessed_results:
                if post_result['neuron_name'] == neuron_name:
                    swc_list = post_result['swc_list']
                    break
            
            if swc_list is not None and len(swc_list) > 0:
                swc_file = swc_dir / f"{neuron_basename}_reconstructed.swc"
                save.write_swc(swc_list, str(swc_file))
                saved_count += 1
        
        print(f"    Saved {saved_count} SWC files")
        
        # Save copy of configuration used
        config_copy = out_dir / f"{self.config['test_name']}_config.json"
        shutil.copy(self.config_path, config_copy)
        print(f"  Saved configuration to: {config_copy}")
        
        print(f"  All results saved to: {out_dir}")
    
    def _compute_summary(self, evaluation_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compute aggregate statistics from evaluation results.
        
        Parameters
        ----------
        evaluation_results : list
            Per-neuron evaluation metrics
            
        Returns
        -------
        dict
            Summary statistics
        """
        # Filter out skipped results
        valid_results = [r for r in evaluation_results if not r.get('skipped', False)]
        
        if len(valid_results) == 0:
            return {
                'n_neurons': len(evaluation_results),
                'n_valid': 0,
                'n_skipped': len(evaluation_results),
                'mean_bidirectional_distance': 0.0,
                'mean_directed_pred_to_gt': 0.0,
                'mean_directed_gt_to_pred': 0.0
            }
        
        # Compute aggregate statistics
        bidirectional_dists = [r['bidirectional_distance'] for r in valid_results]
        directed_pred_to_gt = [r['directed_div_pred_to_gt'] for r in valid_results]
        directed_gt_to_pred = [r['directed_div_gt_to_pred'] for r in valid_results]
        
        summary = {
            'n_neurons': len(evaluation_results),
            'n_valid': len(valid_results),
            'n_skipped': len(evaluation_results) - len(valid_results),
            'mean_bidirectional_distance': float(np.mean(bidirectional_dists)),
            'std_bidirectional_distance': float(np.std(bidirectional_dists)),
            'median_bidirectional_distance': float(np.median(bidirectional_dists)),
            'mean_directed_pred_to_gt': float(np.mean(directed_pred_to_gt)),
            'std_directed_pred_to_gt': float(np.std(directed_pred_to_gt)),
            'mean_directed_gt_to_pred': float(np.mean(directed_gt_to_pred)),
            'std_directed_gt_to_pred': float(np.std(directed_gt_to_pred))
        }
        
        return summary
