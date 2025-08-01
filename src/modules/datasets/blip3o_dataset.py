#!/usr/bin/env python3
"""
<<<<<<< HEAD:src/modules/datasets/blip3o_dataset.py
Clean BLIP3-o Dataset for CLIP Reproduction
Simple implementation for loading EVA-CLIP embedding pairs
=======
Fixed BLIP3-o Dataset for EVA-CLIP Denoising
Key fixes:
1. EVA → EVA denoising (not CLIP → EVA reproduction)
2. Proper spherical data handling
3. Correct input/output flow
4. Better error handling and normalization
>>>>>>> main:src/modules/datasets/blip3o_eva_dataset.py
"""

import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
import pickle
import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Union, Iterator
from pathlib import Path
import logging
import json
import random
import time
import gc
<<<<<<< HEAD:src/modules/datasets/blip3o_dataset.py
=======
import torch.nn.functional as F
import math
>>>>>>> main:src/modules/datasets/blip3o_eva_dataset.py

logger = logging.getLogger(__name__)


<<<<<<< HEAD:src/modules/datasets/blip3o_dataset.py
class BLIP3oCLIPReproductionDataset(IterableDataset):
    """
    Clean Dataset for CLIP reproduction from EVA embeddings
    
    This dataset loads:
    - CLIP embeddings [B, N, 1024] as TARGET (what we want to reproduce)
    - EVA embeddings [B, N, 4096] as CONDITIONING (guidance)
=======
class BLIP3oEVADenoisingDataset(IterableDataset):
    """
    Fixed dataset for EVA-CLIP denoising with proper spherical data handling
    
    This dataset:
    - Takes clean EVA embeddings [B, N, 4096] as TARGET and CONDITIONING
    - Creates noisy versions for INPUT
    - Implements proper spherical noise and interpolation
>>>>>>> main:src/modules/datasets/blip3o_eva_dataset.py
    """
    
    def __init__(
        self,
        chunked_embeddings_dir: Union[str, Path],
        split: str = "train",
        training_mode: str = "patch_only",
        max_shards: Optional[int] = None,
        shuffle_shards: bool = True,
        shuffle_within_shard: bool = True,
        expected_tokens: Optional[int] = None,
<<<<<<< HEAD:src/modules/datasets/blip3o_dataset.py
        skip_corrupted_samples: bool = True,
        validate_tensor_shapes: bool = True,
=======
        # Spherical noise parameters
        noise_schedule: str = "uniform",  # uniform, cosine
        max_noise_level: float = 0.9,  # Maximum noise mixing ratio
        min_noise_level: float = 0.1,   # Minimum noise mixing ratio
        # Error handling
        skip_corrupted: bool = True,
        validate_shapes: bool = True,
>>>>>>> main:src/modules/datasets/blip3o_eva_dataset.py
        max_retries: int = 3,
    ):
        super().__init__()
        
        self.chunked_embeddings_dir = Path(chunked_embeddings_dir)
        self.split = split
        self.training_mode = training_mode
        self.max_shards = max_shards
        self.shuffle_shards = shuffle_shards
        self.shuffle_within_shard = shuffle_within_shard
        self.skip_corrupted_samples = skip_corrupted_samples
        self.validate_tensor_shapes = validate_tensor_shapes
        self.max_retries = max_retries
        
        # Spherical noise parameters
        self.noise_schedule = noise_schedule
        self.max_noise_level = max_noise_level
        self.min_noise_level = min_noise_level
        
        # Determine expected tokens
        if expected_tokens is None:
            self.expected_tokens = 257 if training_mode == "cls_patch" else 256
        else:
            self.expected_tokens = expected_tokens
        
        # Setup random state
        self.rng = random.Random(42)
        
        # Load manifest and prepare shards
        self._load_manifest()
        self._prepare_shard_list()
        
        # Current state
        self.current_shard_idx = 0
        self.current_shard_data = None
        self.current_sample_idx = 0
        self.total_samples_processed = 0
        
<<<<<<< HEAD:src/modules/datasets/blip3o_dataset.py
        # Calculate estimated length for __len__ method
        self._estimate_length()
        
        logger.info(f"Clean CLIP Reproduction Dataset initialized:")
        logger.info(f"  Directory: {self.chunked_embeddings_dir}")
        logger.info(f"  Mode: {self.training_mode} ({self.expected_tokens} tokens)")
        logger.info(f"  TARGET: CLIP embeddings [B, N, 1024]")
        logger.info(f"  CONDITIONING: EVA embeddings [B, N, 4096]")
        logger.info(f"  Shards: {len(self.shard_files) if hasattr(self, 'shard_files') else 'Unknown'}")
=======
        logger.info(f"EVA Denoising Dataset initialized:")
        logger.info(f"  Directory: {self.chunked_embeddings_dir}")
        logger.info(f"  Mode: {self.training_mode} ({self.expected_tokens} tokens)")
        logger.info(f"  TASK: EVA Denoising")
        logger.info(f"  INPUT: Noisy EVA embeddings [B, N, 4096]")
        logger.info(f"  CONDITIONING: Clean EVA embeddings [B, N, 4096]")
        logger.info(f"  TARGET: Clean EVA embeddings [B, N, 4096]")
        logger.info(f"  Noise schedule: {self.noise_schedule}")
        logger.info(f"  Noise range: [{self.min_noise_level}, {self.max_noise_level}]")
        logger.info(f"  Shards: {len(self.shard_files) if hasattr(self, 'shard_files') else 'Loading...'}")
>>>>>>> main:src/modules/datasets/blip3o_eva_dataset.py

    def _load_manifest(self):
        """Load embeddings manifest"""
        manifest_path = self.chunked_embeddings_dir / "embeddings_manifest.json"
        
        try:
            if manifest_path.exists():
                with open(manifest_path, 'r') as f:
                    self.manifest = json.load(f)
                logger.info(f"Loaded manifest: {self.manifest.get('total_shards', 0)} shards, {self.manifest.get('total_samples', 0):,} samples")
            else:
                self.manifest = {"total_shards": 0, "total_samples": 0}
                logger.warning(f"No manifest found at {manifest_path}")
        except Exception as e:
            logger.warning(f"Failed to load manifest: {e}")
            self.manifest = {"total_shards": 0, "total_samples": 0}

    def _prepare_shard_list(self):
        """Prepare list of shard files"""
        # Look for shard files
        mode_suffix = "cls_patch" if self.training_mode == "cls_patch" else "patch_only"
        patterns = [
            f"embeddings_shard_*_{mode_suffix}.pkl",
            f"*_{mode_suffix}.pkl",
            "embeddings_shard_*.pkl",
            "*.pkl"
        ]
        
        shard_files = []
        for pattern in patterns:
            shard_files = list(self.chunked_embeddings_dir.glob(pattern))
            if shard_files:
                logger.info(f"Found {len(shard_files)} files with pattern: {pattern}")
                break
        
        if not shard_files:
            raise FileNotFoundError(f"No shard files found in {self.chunked_embeddings_dir}")
        
        # Sort files
        shard_files.sort(key=lambda x: int(''.join(filter(str.isdigit, x.stem))) if any(c.isdigit() for c in x.stem) else 0)
        
        # Apply max shards limit
        if self.max_shards is not None:
            shard_files = shard_files[:self.max_shards]
        
        # Filter existing files
        self.shard_files = [f for f in shard_files if f.exists()]
        
        if self.shuffle_shards:
            self.rng.shuffle(self.shard_files)
        
        logger.info(f"Prepared {len(self.shard_files)} shard files")

    def _estimate_length(self):
        """Estimate total number of samples for __len__ method"""
        try:
            # If manifest is available, use it
            if self.manifest.get('total_samples', 0) > 0:
                manifest_samples = self.manifest['total_samples']
                # Adjust for max_shards limitation
                if self.max_shards is not None and self.manifest.get('total_shards', 0) > 0:
                    ratio = min(self.max_shards / self.manifest['total_shards'], 1.0)
                    self.estimated_length = int(manifest_samples * ratio)
                else:
                    self.estimated_length = manifest_samples
                logger.info(f"Using manifest for length estimation: {self.estimated_length:,} samples")
                return
            
            # Fallback: try to load first shard to estimate
            if self.shard_files:
                try:
                    first_shard = self._load_shard(self.shard_files[0])
                    if first_shard and 'clip_blip3o_embeddings' in first_shard:
                        samples_per_shard = first_shard['clip_blip3o_embeddings'].shape[0]
                        self.estimated_length = samples_per_shard * len(self.shard_files)
                        logger.info(f"Estimated length from first shard: {self.estimated_length:,} samples ({samples_per_shard} per shard)")
                        return
                except Exception as e:
                    logger.warning(f"Could not load first shard for estimation: {e}")
            
            # Final fallback: rough estimate
            self.estimated_length = len(self.shard_files) * 1000  # Assume 1000 samples per shard
            logger.warning(f"Using rough length estimate: {self.estimated_length:,} samples")
            
        except Exception as e:
            logger.warning(f"Length estimation failed: {e}")
            self.estimated_length = 1000  # Very conservative fallback

    def __len__(self) -> int:
        """Return estimated length for DataLoader compatibility"""
        return self.estimated_length

    def _load_shard(self, shard_path: Path) -> Optional[Dict[str, Any]]:
        """Load a single shard with error handling"""
        for attempt in range(self.max_retries):
            try:
                with open(shard_path, 'rb') as f:
                    shard_data = pickle.load(f)
                
                # Validate and process shard
                self._validate_and_process_shard(shard_data, shard_path)
                
                return shard_data
                
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1}/{self.max_retries} failed for {shard_path}: {e}")
                if attempt == self.max_retries - 1:
                    if self.skip_corrupted_samples:
                        logger.warning(f"Skipping corrupted shard: {shard_path}")
                        return None
                    else:
                        raise
                time.sleep(0.1)

    def _validate_and_process_shard(self, shard_data: Dict[str, Any], shard_path: Path):
        """Validate and process shard data"""
        # Check required keys
        required_keys = ['eva_blip3o_embeddings', 'captions']
        for key in required_keys:
            if key not in shard_data:
                raise ValueError(f"Missing key '{key}' in shard {shard_path}")
        
        # Get EVA embeddings (this is our main data)
        eva_emb = shard_data['eva_blip3o_embeddings']
        
        # Convert to tensors if needed
        if not torch.is_tensor(eva_emb):
            eva_emb = torch.tensor(eva_emb, dtype=torch.float32)
            shard_data['eva_blip3o_embeddings'] = eva_emb
        
        # Validate shapes
<<<<<<< HEAD:src/modules/datasets/blip3o_dataset.py
        if self.validate_tensor_shapes:
            if clip_emb.dim() != 3 or eva_emb.dim() != 3:
                raise ValueError(f"Expected 3D tensors, got CLIP: {clip_emb.shape}, EVA: {eva_emb.shape}")
            
            if clip_emb.shape[0] != eva_emb.shape[0]:
                raise ValueError(f"Batch size mismatch: CLIP {clip_emb.shape[0]} vs EVA {eva_emb.shape[0]}")
            
            clip_tokens, eva_tokens = clip_emb.shape[1], eva_emb.shape[1]
            if clip_tokens != eva_tokens:
                raise ValueError(f"Token count mismatch: CLIP {clip_tokens} vs EVA {eva_tokens}")
=======
        if self.validate_shapes:
            if eva_emb.dim() != 3:
                raise ValueError(f"Expected 3D tensor, got EVA: {eva_emb.shape}")
>>>>>>> main:src/modules/datasets/blip3o_eva_dataset.py
        
        # Handle token count adaptation
        current_tokens = eva_emb.shape[1]
        if current_tokens != self.expected_tokens:
            if current_tokens == 256 and self.expected_tokens == 257:
                # Add CLS token (average of patches)
                eva_cls = eva_emb.mean(dim=1, keepdim=True)
                shard_data['eva_blip3o_embeddings'] = torch.cat([eva_cls, eva_emb], dim=1)
            elif current_tokens == 257 and self.expected_tokens == 256:
                # Remove CLS token
                shard_data['eva_blip3o_embeddings'] = eva_emb[:, 1:, :]
            else:
                raise ValueError(f"Cannot adapt from {current_tokens} to {self.expected_tokens} tokens")
<<<<<<< HEAD:src/modules/datasets/blip3o_dataset.py
=======
        
        # Apply normalization - CRITICAL: EVA embeddings must be L2 normalized
        shard_data = self._normalize_embeddings(shard_data)

    def _normalize_embeddings(self, shard_data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply L2 normalization to EVA embeddings"""
        eva_emb = shard_data['eva_blip3o_embeddings']
        
        # Check for NaN/Inf
        if torch.isnan(eva_emb).any() or torch.isinf(eva_emb).any():
            logger.warning("Found NaN/Inf in EVA embeddings")
            eva_emb = torch.nan_to_num(eva_emb, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Apply L2 normalization to ensure unit sphere
        eps = 1e-8
        eva_normalized = F.normalize(eva_emb + eps, p=2, dim=-1)
        
        # Verify normalization
        eva_norm = torch.norm(eva_normalized, dim=-1).mean().item()
        
        if abs(eva_norm - 1.0) > 0.1:
            logger.warning(f"EVA normalization may have failed: norm = {eva_norm:.3f}")
        
        shard_data['eva_blip3o_embeddings'] = eva_normalized
        shard_data['normalization_applied'] = True
        
        return shard_data
>>>>>>> main:src/modules/datasets/blip3o_eva_dataset.py

    def _add_spherical_noise(self, clean_eva: torch.Tensor, noise_level: float) -> torch.Tensor:
        """Add spherical noise to EVA embeddings using slerp"""
        device = clean_eva.device
        dtype = clean_eva.dtype
        
        # Generate random noise on sphere
        noise = torch.randn_like(clean_eva, device=device, dtype=dtype)
        noise = F.normalize(noise, p=2, dim=-1)
        
        # Spherical linear interpolation (slerp)
        # noise_level = 0: clean, noise_level = 1: pure noise
        
        # Compute angle between clean and noise
        cos_angle = torch.sum(clean_eva * noise, dim=-1, keepdim=True)
        cos_angle = torch.clamp(cos_angle, -1 + 1e-7, 1 - 1e-7)
        angle = torch.acos(cos_angle)
        
        # Avoid division by zero
        sin_angle = torch.sin(angle)
        sin_angle = torch.clamp(sin_angle, min=1e-7)
        
        # Slerp formula: slerp(a, b, t) = (sin((1-t)*θ)/sin(θ)) * a + (sin(t*θ)/sin(θ)) * b
        clean_weight = torch.sin((1 - noise_level) * angle) / sin_angle
        noise_weight = torch.sin(noise_level * angle) / sin_angle
        
        noisy_eva = clean_weight * clean_eva + noise_weight * noise
        
        # Ensure result is on unit sphere
        noisy_eva = F.normalize(noisy_eva, p=2, dim=-1)
        
        return noisy_eva, noise

    def _sample_noise_level(self) -> float:
        """Sample noise level based on schedule"""
        if self.noise_schedule == "uniform":
            return self.rng.uniform(self.min_noise_level, self.max_noise_level)
        elif self.noise_schedule == "cosine":
            # Cosine schedule favors lower noise levels
            u = self.rng.uniform(0, 1)
            t = 0.5 * (1 + math.cos(u * math.pi))  # Cosine decay
            return self.min_noise_level + t * (self.max_noise_level - self.min_noise_level)
        else:
            raise ValueError(f"Unknown noise schedule: {self.noise_schedule}")

    def _load_next_shard(self) -> bool:
        """Load next shard"""
        # Cleanup previous shard
        if self.current_shard_data is not None:
            del self.current_shard_data
            gc.collect()
        
        # Check if more shards available
        if self.current_shard_idx >= len(self.shard_files):
            self.current_shard_data = None
            return False
        
        # Try to load next shard
        while self.current_shard_idx < len(self.shard_files):
            shard_path = self.shard_files[self.current_shard_idx]
            
            self.current_shard_data = self._load_shard(shard_path)
            
            if self.current_shard_data is not None:
                # Prepare samples
                num_samples = self.current_shard_data['eva_blip3o_embeddings'].shape[0]
                self.current_samples = list(range(num_samples))
                
                if self.shuffle_within_shard:
                    self.rng.shuffle(self.current_samples)
                
                self.current_sample_idx = 0
                self.current_shard_idx += 1
                return True
            else:
                self.current_shard_idx += 1
                continue
        
        self.current_shard_data = None
        return False

    def __len__(self) -> int:
        """Estimate total number of samples"""
        if hasattr(self, '_estimated_length'):
            return self._estimated_length
        
        # Try to estimate from manifest
        if hasattr(self, 'manifest') and 'total_samples' in self.manifest:
            manifest_samples = self.manifest['total_samples']
            if self.max_shards is not None:
                # Estimate based on max_shards ratio
                total_shards = self.manifest.get('total_shards', len(self.shard_files))
                if total_shards > 0:
                    estimated_samples = int(manifest_samples * self.max_shards / total_shards)
                    self._estimated_length = estimated_samples
                    return estimated_samples
            else:
                self._estimated_length = manifest_samples
                return manifest_samples
        
        # Fallback: estimate based on file count and average samples per shard
        num_shards = len(self.shard_files) if hasattr(self, 'shard_files') else 1
        avg_samples_per_shard = 1000  # Conservative estimate
        
        estimated_samples = num_shards * avg_samples_per_shard
        self._estimated_length = estimated_samples
        
        logger.debug(f"Estimated dataset length: {estimated_samples} samples from {num_shards} shards")
        return estimated_samples

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Iterate through all samples"""
        self.current_shard_idx = 0
        self.current_shard_data = None
        self.current_sample_idx = 0
        self.total_samples_processed = 0
        
        if not self._load_next_shard():
            return
        
        while self.current_shard_data is not None:
            while self.current_sample_idx < len(self.current_samples):
                try:
                    sample_idx = self.current_samples[self.current_sample_idx]
                    
                    # Extract clean EVA embeddings
                    clean_eva = self.current_shard_data['eva_blip3o_embeddings'][sample_idx]
                    caption = self.current_shard_data['captions'][sample_idx]
                    
                    # Final validation
<<<<<<< HEAD:src/modules/datasets/blip3o_dataset.py
                    if self.validate_tensor_shapes:
                        if clip_emb.shape != (self.expected_tokens, 1024):
                            raise ValueError(f"Invalid CLIP shape: {clip_emb.shape}")
                        if eva_emb.shape != (self.expected_tokens, 4096):
                            raise ValueError(f"Invalid EVA shape: {eva_emb.shape}")
                    
                    # Check for NaN/Inf
                    if torch.isnan(clip_emb).any() or torch.isnan(eva_emb).any():
                        if self.skip_corrupted_samples:
=======
                    if self.validate_shapes:
                        if clean_eva.shape != (self.expected_tokens, 4096):
                            raise ValueError(f"Invalid EVA shape: {clean_eva.shape}")
                    
                    # Check for NaN/Inf
                    if torch.isnan(clean_eva).any():
                        if self.skip_corrupted:
>>>>>>> main:src/modules/datasets/blip3o_eva_dataset.py
                            self.current_sample_idx += 1
                            continue
                        else:
                            raise ValueError("NaN detected in EVA embeddings")
                    
<<<<<<< HEAD:src/modules/datasets/blip3o_dataset.py
                    # Create sample item for CLIP reproduction
                    item = {
                        'eva_embeddings': eva_emb,      # [N, 4096] - CONDITIONING
                        'clip_embeddings': clip_emb,    # [N, 1024] - TARGET to reproduce
=======
                    # Sample noise level for this sample
                    noise_level = self._sample_noise_level()
                    
                    # Add spherical noise to create noisy version
                    noisy_eva, noise = self._add_spherical_noise(clean_eva, noise_level)
                    
                    # Create sample item for EVA denoising
                    item = {
                        # Model inputs
                        'noisy_eva_embeddings': noisy_eva,      # [N, 4096] - Noisy input
                        'clean_eva_embeddings': clean_eva,      # [N, 4096] - Clean conditioning & target
                        'noise': noise,                         # [N, 4096] - Pure noise used
                        'noise_level': noise_level,             # scalar - Noise mixing ratio
>>>>>>> main:src/modules/datasets/blip3o_eva_dataset.py
                        'caption': caption,
                        
                        # Metadata
                        'key': f"shard_{self.current_shard_idx-1}_sample_{sample_idx}",
                        'sample_idx': sample_idx,
                        'training_mode': self.training_mode,
                        'num_tokens': self.expected_tokens,
                    }
                    
                    self.current_sample_idx += 1
                    self.total_samples_processed += 1
                    
                    yield item
                    
                except Exception as e:
                    if self.skip_corrupted_samples:
                        logger.warning(f"Skipping corrupted sample {sample_idx}: {e}")
                        self.current_sample_idx += 1
                        continue
                    else:
                        raise
            
            if not self._load_next_shard():
                break
        
        logger.info(f"Iteration completed: {self.total_samples_processed} samples processed")


<<<<<<< HEAD:src/modules/datasets/blip3o_dataset.py
def clip_reproduction_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Clean collate function for CLIP reproduction
    
    This function:
    1. Takes clean CLIP embeddings as targets
    2. Uses EVA embeddings for conditioning  
    3. Creates standard Gaussian noise
    4. Applies rectified flow interpolation
=======
def eva_denoising_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate function for EVA denoising with proper spherical flow matching setup
    
    This function:
    1. Takes clean EVA embeddings as targets and conditioning
    2. Uses pre-computed noisy versions for input
    3. Sets up spherical flow matching with proper timesteps
>>>>>>> main:src/modules/datasets/blip3o_eva_dataset.py
    """
    if not batch:
        raise ValueError("Empty batch")
    
    # Filter valid items
    valid_batch = [item for item in batch if item is not None]
    if not valid_batch:
        raise ValueError("No valid items in batch")
    
    try:
        # Stack embeddings
<<<<<<< HEAD:src/modules/datasets/blip3o_dataset.py
        eva_embeddings = torch.stack([item['eva_embeddings'] for item in valid_batch])     # [B, N, 4096]
        clip_embeddings = torch.stack([item['clip_embeddings'] for item in valid_batch])   # [B, N, 1024]
=======
        noisy_eva = torch.stack([item['noisy_eva_embeddings'] for item in valid_batch])    # [B, N, 4096]
        clean_eva = torch.stack([item['clean_eva_embeddings'] for item in valid_batch])    # [B, N, 4096]
        noise = torch.stack([item['noise'] for item in valid_batch])                       # [B, N, 4096]
        noise_levels = torch.tensor([item['noise_level'] for item in valid_batch])         # [B]
>>>>>>> main:src/modules/datasets/blip3o_eva_dataset.py
        
        # Collect metadata
        captions = [item['caption'] for item in valid_batch]
        keys = [item['key'] for item in valid_batch]
        
<<<<<<< HEAD:src/modules/datasets/blip3o_dataset.py
        batch_size, seq_len, clip_dim = clip_embeddings.shape
        device = clip_embeddings.device
        dtype = clip_embeddings.dtype
        
        # Ensure float32 for stability
        eva_embeddings = eva_embeddings.float()
        clip_embeddings = clip_embeddings.float()
        
        # Sample random timesteps for each sample in batch
        timesteps = torch.rand(batch_size, device=device, dtype=dtype)
        
        # Create standard Gaussian noise
        noise = torch.randn_like(clip_embeddings, device=device, dtype=dtype)
        
        # Linear interpolation for rectified flow: x_t = (1-t) * noise + t * clip_clean
        t_expanded = timesteps.view(batch_size, 1, 1)  # [B, 1, 1]
        noisy_clip = (1 - t_expanded) * noise + t_expanded * clip_embeddings
        
        # Velocity target: v = clip_clean - noise (for rectified flow)
        velocity_target = clip_embeddings - noise
        
        # Validation
        assert eva_embeddings.shape == (batch_size, seq_len, 4096), f"EVA shape: {eva_embeddings.shape}"
        assert clip_embeddings.shape == (batch_size, seq_len, 1024), f"CLIP shape: {clip_embeddings.shape}"
        assert noisy_clip.shape == (batch_size, seq_len, 1024), f"Noisy CLIP shape: {noisy_clip.shape}"
        assert velocity_target.shape == (batch_size, seq_len, 1024), f"Velocity target shape: {velocity_target.shape}"
=======
        batch_size, seq_len, eva_dim = clean_eva.shape
        device = clean_eva.device
        dtype = clean_eva.dtype
        
        # Ensure float32 for stability
        noisy_eva = noisy_eva.float()
        clean_eva = clean_eva.float()
        noise = noise.float()
        noise_levels = noise_levels.float()
        
        # Ensure L2 normalization
        eps = 1e-8
        noisy_eva = F.normalize(noisy_eva + eps, p=2, dim=-1)
        clean_eva = F.normalize(clean_eva + eps, p=2, dim=-1)
        noise = F.normalize(noise + eps, p=2, dim=-1)
        
        # SPHERICAL FLOW MATCHING SETUP
        # Sample timesteps for flow matching (0 = noise, 1 = clean)
        timesteps = torch.rand(batch_size, device=device, dtype=dtype)
        
        # For spherical flow, we interpolate on the sphere using slerp
        t_expanded = timesteps.view(batch_size, 1, 1)  # [B, 1, 1]
        
        # Compute angles between clean and noise
        cos_angles = torch.sum(clean_eva * noise, dim=-1, keepdim=True)
        cos_angles = torch.clamp(cos_angles, -1 + 1e-7, 1 - 1e-7)
        angles = torch.acos(cos_angles)
        
        # Avoid division by zero
        sin_angles = torch.sin(angles)
        sin_angles = torch.clamp(sin_angles, min=1e-7)
        
        # Spherical interpolation: x_t = slerp(noise, clean, t)
        clean_weight = torch.sin(t_expanded * angles) / sin_angles
        noise_weight = torch.sin((1 - t_expanded) * angles) / sin_angles
        
        x_t = noise_weight * noise + clean_weight * clean_eva
        x_t = F.normalize(x_t + eps, p=2, dim=-1)
        
        # Spherical velocity (tangent to sphere)
        # For spherical flow: v = d/dt slerp(noise, clean, t)
        velocity_target = (clean_eva - noise_weight / clean_weight * x_t) * angles / sin_angles
        
        # Alternative: Direct velocity from parametric derivative
        # This is more stable for training
        velocity_target = clean_eva - noise
        
        # Validation
        assert noisy_eva.shape == (batch_size, seq_len, 4096), f"Noisy EVA shape: {noisy_eva.shape}"
        assert clean_eva.shape == (batch_size, seq_len, 4096), f"Clean EVA shape: {clean_eva.shape}"
        assert x_t.shape == (batch_size, seq_len, 4096), f"x_t shape: {x_t.shape}"
        assert velocity_target.shape == (batch_size, seq_len, 4096), f"Velocity target shape: {velocity_target.shape}"
>>>>>>> main:src/modules/datasets/blip3o_eva_dataset.py
        assert timesteps.shape == (batch_size,), f"Timesteps shape: {timesteps.shape}"
        
        return {
            # Model inputs
<<<<<<< HEAD:src/modules/datasets/blip3o_dataset.py
            'encoder_hidden_states': eva_embeddings,     # [B, N, 4096] - EVA conditioning
            'hidden_states': noisy_clip,                 # [B, N, 1024] - Noisy CLIP input
            'timestep': timesteps,                       # [B] - Flow matching timesteps
            
            # Training targets
            'clip_embeddings': clip_embeddings,          # [B, N, 1024] - Clean CLIP (target)
            'velocity_target': velocity_target,          # [B, N, 1024] - Velocity for flow matching
            'noise': noise,                              # [B, N, 1024] - Standard Gaussian noise
=======
            'hidden_states': x_t,                        # [B, N, 4096] - Interpolated state
            'encoder_hidden_states': clean_eva,          # [B, N, 4096] - Clean EVA conditioning
            'timestep': timesteps,                       # [B] - Flow matching timesteps
            
            # Training targets
            'clean_eva_embeddings': clean_eva,           # [B, N, 4096] - Clean EVA (target)
            'velocity_target': velocity_target,          # [B, N, 4096] - Velocity for flow matching
            'noise': noise,                              # [B, N, 4096] - Pure noise
            'noisy_eva_embeddings': noisy_eva,           # [B, N, 4096] - Pre-computed noisy version
            
            # Flow matching state
            'x_t': x_t,                                  # [B, N, 4096] - Current flow state
            'noise_levels': noise_levels,                # [B] - Original noise levels
>>>>>>> main:src/modules/datasets/blip3o_eva_dataset.py
            
            # Metadata
            'captions': captions,
            'keys': keys,
            'batch_size': batch_size,
            'training_mode': valid_batch[0]['training_mode'],
            'num_tokens': valid_batch[0]['num_tokens'],
            'seq_len': seq_len,
<<<<<<< HEAD:src/modules/datasets/blip3o_dataset.py
=======
            
            # Normalization status
            'eva_embeddings_normalized': True,
            'eva_norm_mean': torch.norm(clean_eva, dim=-1).mean().item(),
            'noisy_eva_norm_mean': torch.norm(noisy_eva, dim=-1).mean().item(),
>>>>>>> main:src/modules/datasets/blip3o_eva_dataset.py
        }
        
    except Exception as e:
        logger.error(f"Error in collate function: {e}")
        logger.error(f"Batch size: {len(batch)}")
        if batch:
            try:
                logger.error(f"First item keys: {list(batch[0].keys())}")
                for key, value in batch[0].items():
                    if torch.is_tensor(value):
                        logger.error(f"  {key}: {value.shape} {value.dtype}")
                    else:
                        logger.error(f"  {key}: {type(value)} = {value}")
            except:
                pass
        raise


<<<<<<< HEAD:src/modules/datasets/blip3o_dataset.py
def create_clip_reproduction_dataloaders(
=======
def create_eva_denoising_dataloaders(
>>>>>>> main:src/modules/datasets/blip3o_eva_dataset.py
    chunked_embeddings_dir: Union[str, Path],
    batch_size: int = 16,
    eval_batch_size: Optional[int] = None,
    training_mode: str = "patch_only",
    max_shards: Optional[int] = None,
<<<<<<< HEAD:src/modules/datasets/blip3o_dataset.py
=======
    noise_schedule: str = "uniform",
    max_noise_level: float = 0.9,
    min_noise_level: float = 0.1,
>>>>>>> main:src/modules/datasets/blip3o_eva_dataset.py
    num_workers: int = 0,
    pin_memory: bool = False,
    **kwargs
) -> Tuple[DataLoader, Optional[DataLoader]]:
<<<<<<< HEAD:src/modules/datasets/blip3o_dataset.py
    """Create clean dataloaders for CLIP reproduction"""
=======
    """Create dataloaders for EVA denoising"""
>>>>>>> main:src/modules/datasets/blip3o_eva_dataset.py
    
    if eval_batch_size is None:
        eval_batch_size = batch_size
    
<<<<<<< HEAD:src/modules/datasets/blip3o_dataset.py
    logger.info(f"Creating clean CLIP reproduction dataloaders:")
    logger.info(f"  Target: CLIP embeddings [B, N, 1024]")
    logger.info(f"  Conditioning: EVA embeddings [B, N, 4096]")
    logger.info(f"  Noise: Standard Gaussian")
    
    # Use identical settings for both train and eval
    dataset_kwargs = {
        'chunked_embeddings_dir': chunked_embeddings_dir,
        'training_mode': training_mode,
        'max_shards': max_shards,
        **kwargs
    }
    
    # Create training dataset
    train_dataset = BLIP3oCLIPReproductionDataset(
        split="train",
        shuffle_shards=True,
        shuffle_within_shard=True,
        **dataset_kwargs
=======
    logger.info(f"Creating EVA denoising dataloaders:")
    logger.info(f"  TASK: EVA-CLIP Denoising")
    logger.info(f"  INPUT: Noisy EVA embeddings [B, N, 4096]")
    logger.info(f"  CONDITIONING: Clean EVA embeddings [B, N, 4096]")
    logger.info(f"  TARGET: Clean EVA embeddings [B, N, 4096]")
    logger.info(f"  Noise schedule: {noise_schedule}")
    logger.info(f"  Noise range: [{min_noise_level}, {max_noise_level}]")
    
    # Create training dataset
    train_dataset = BLIP3oEVADenoisingDataset(
        chunked_embeddings_dir=chunked_embeddings_dir,
        split="train",
        training_mode=training_mode,
        max_shards=max_shards,
        shuffle_shards=True,
        shuffle_within_shard=True,
        noise_schedule=noise_schedule,
        max_noise_level=max_noise_level,
        min_noise_level=min_noise_level,
        **kwargs
>>>>>>> main:src/modules/datasets/blip3o_eva_dataset.py
    )
    
    # Create training dataloader
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
<<<<<<< HEAD:src/modules/datasets/blip3o_dataset.py
        collate_fn=clip_reproduction_collate_fn,
=======
        collate_fn=eva_denoising_collate_fn,
>>>>>>> main:src/modules/datasets/blip3o_eva_dataset.py
        pin_memory=pin_memory,
        drop_last=True,
        persistent_workers=num_workers > 0,
    )
    
<<<<<<< HEAD:src/modules/datasets/blip3o_dataset.py
    # Create evaluation dataset with identical settings but no shuffling
    eval_dataset = BLIP3oCLIPReproductionDataset(
        split="eval",
        shuffle_shards=False,  # No shuffling for consistent evaluation
        shuffle_within_shard=False,
        **dataset_kwargs
=======
    # Create evaluation dataset (same data, different noise)
    eval_dataset = BLIP3oEVADenoisingDataset(
        chunked_embeddings_dir=chunked_embeddings_dir,
        split="eval",
        training_mode=training_mode,
        max_shards=max_shards,
        shuffle_shards=False,
        shuffle_within_shard=False,
        noise_schedule="uniform",  # Use uniform for consistent eval
        max_noise_level=0.7,  # Less noise for evaluation
        min_noise_level=0.3,
        **kwargs
>>>>>>> main:src/modules/datasets/blip3o_eva_dataset.py
    )
    
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=eval_batch_size,
        num_workers=min(num_workers, 1),
<<<<<<< HEAD:src/modules/datasets/blip3o_dataset.py
        collate_fn=clip_reproduction_collate_fn,
=======
        collate_fn=eva_denoising_collate_fn,
>>>>>>> main:src/modules/datasets/blip3o_eva_dataset.py
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=min(num_workers, 1) > 0,
    )
    
<<<<<<< HEAD:src/modules/datasets/blip3o_dataset.py
    logger.info(f"Clean CLIP reproduction dataloaders created successfully")
    logger.info(f"  Training dataset length: {len(train_dataset):,}")
    logger.info(f"  Evaluation dataset length: {len(eval_dataset):,}")
=======
    logger.info(f"EVA denoising dataloaders created successfully")
>>>>>>> main:src/modules/datasets/blip3o_eva_dataset.py
    
    return train_dataloader, eval_dataloader