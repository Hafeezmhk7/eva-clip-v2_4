#!/usr/bin/env python3
"""
UPDATED BLIP3-o Dataset with Critical Normalization Fixes
src/modules/datasets/blip3o_dataset.py

Key improvements:
1. Proper CLIP embedding normalization (CRITICAL FIX)
2. U-shaped timestep sampling
3. Better statistics computation
4. Enhanced validation
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
import math
from torch.distributions import Beta

logger = logging.getLogger(__name__)


class CLIPEmbeddingNormalizer:
    """
    CRITICAL: Proper CLIP embedding normalization to preserve semantic information
    
    Problem: L2 normalization constrains embeddings to [-0.05, 0.05] range
    Solution: Use elementwise mean/std normalization + scaling
    """
    def __init__(self, embedding_dim=1024):
        self.embedding_dim = embedding_dim
        self.scale_factor = math.sqrt(embedding_dim)  # Match Gaussian noise scale
        self.clip_mean = None
        self.clip_std = None
        self.stats_computed = False
        
    def compute_stats_from_shards(self, shard_files, max_shards_for_stats=5):
        """Compute normalization statistics from shard files"""
        logger.info(f"Computing CLIP embedding statistics from {min(len(shard_files), max_shards_for_stats)} shards...")
        
        clip_embeddings = []
        processed_shards = 0
        
        for shard_path in shard_files[:max_shards_for_stats]:
            try:
                with open(shard_path, 'rb') as f:
                    shard_data = pickle.load(f)
                
                if 'clip_blip3o_embeddings' in shard_data:
                    # Convert to tensor if needed
                    clip_emb = shard_data['clip_blip3o_embeddings']
                    if not torch.is_tensor(clip_emb):
                        clip_emb = torch.tensor(clip_emb, dtype=torch.float32)
                    
                    # Collect embeddings for statistics
                    clip_embeddings.append(clip_emb.flatten(0, 1))  # [B*N, 1024]
                    processed_shards += 1
                    
                    logger.info(f"   Processed shard {processed_shards}: {clip_emb.shape}")
                    
            except Exception as e:
                logger.warning(f"Could not process shard {shard_path} for stats: {e}")
                continue
        
        if not clip_embeddings:
            logger.error("No CLIP embeddings found for statistics computation!")
            raise ValueError("Cannot compute normalization statistics")
        
        # Combine all embeddings and compute stats
        all_embeddings = torch.cat(clip_embeddings, dim=0)  # [Total_samples, 1024]
        logger.info(f"Computing stats from {all_embeddings.shape[0]:,} embedding vectors")
        
        # Compute elementwise statistics
        self.clip_mean = all_embeddings.mean(dim=0, keepdim=True)  # [1, 1024]
        self.clip_std = all_embeddings.std(dim=0, keepdim=True)    # [1, 1024]
        self.clip_std = torch.clamp(self.clip_std, min=1e-6)  # Prevent division by zero
        
        # Add batch and sequence dimensions for broadcasting
        self.clip_mean = self.clip_mean.unsqueeze(0)  # [1, 1, 1024]
        self.clip_std = self.clip_std.unsqueeze(0)    # [1, 1, 1024]
        
        self.stats_computed = True
        
        # Log statistics
        logger.info(f"✅ CLIP normalization statistics computed:")
        logger.info(f"   Mean range: [{self.clip_mean.min():.6f}, {self.clip_mean.max():.6f}]")
        logger.info(f"   Std range: [{self.clip_std.min():.6f}, {self.clip_std.max():.6f}]")
        logger.info(f"   Scale factor: {self.scale_factor:.2f}")
        
        # Test normalization on a sample
        sample_emb = all_embeddings[:100]  # Test on 100 samples
        normalized = self.normalize(sample_emb.unsqueeze(0))  # Add batch dim
        logger.info(f"   Test normalization range: [{normalized.min():.4f}, {normalized.max():.4f}]")
        
        # Clear memory
        del all_embeddings, clip_embeddings
        gc.collect()
        
    def normalize(self, clip_embeddings):
        """Apply proper normalization preserving semantic information"""
        if not self.stats_computed:
            logger.error("❌ Normalization stats not computed! Call compute_stats_from_shards() first")
            raise ValueError("Must compute normalization statistics first!")
        
        device = clip_embeddings.device
        
        # Move stats to correct device
        clip_mean = self.clip_mean.to(device)
        clip_std = self.clip_std.to(device)
        
        # Elementwise standardization
        normalized = (clip_embeddings - clip_mean) / clip_std
        
        # Scale to match Gaussian noise expected by diffusion
        normalized = normalized * self.scale_factor
        
        return normalized
    
    def denormalize(self, normalized_embeddings):
        """Convert normalized embeddings back to original CLIP space"""
        if not self.stats_computed:
            raise ValueError("Cannot denormalize without computed statistics!")
        
        device = normalized_embeddings.device
        
        # Move stats to correct device
        clip_mean = self.clip_mean.to(device)
        clip_std = self.clip_std.to(device)
        
        # Reverse scaling
        embeddings = normalized_embeddings / self.scale_factor
        
        # Reverse standardization
        embeddings = embeddings * clip_std + clip_mean
        
        return embeddings


def sample_u_shaped_timesteps(batch_size, device, alpha=0.5):
    """
    Sample timesteps from U-shaped distribution: p(t) ∝ (t(1-t))^(-α)
    Emphasizes endpoints and intermediate regions for better training
    """
    # Use Beta distribution to approximate U-shape
    beta_alpha = max(0.1, 1 - alpha)
    beta_dist = Beta(beta_alpha, beta_alpha)
    timesteps = beta_dist.sample((batch_size,)).to(device)
    
    # Clamp to avoid numerical issues at endpoints
    timesteps = torch.clamp(timesteps, min=1e-3, max=1.0 - 1e-3)
    
    return timesteps


class BLIP3oCLIPReproductionDataset(IterableDataset):
    """
    UPDATED Dataset for CLIP reproduction with proper normalization
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
        skip_corrupted_samples: bool = True,
        validate_tensor_shapes: bool = True,
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
        
        # CRITICAL: Initialize CLIP normalizer and compute stats
        self.clip_normalizer = CLIPEmbeddingNormalizer(embedding_dim=1024)
        self.clip_normalizer.compute_stats_from_shards(self.shard_files, max_shards_for_stats=5)
        
        # Current state
        self.current_shard_idx = 0
        self.current_shard_data = None
        self.current_sample_idx = 0
        self.total_samples_processed = 0
        
        # Calculate estimated length
        self._estimate_length()
        
        logger.info(f"UPDATED CLIP Reproduction Dataset initialized:")
        logger.info(f"  Directory: {self.chunked_embeddings_dir}")
        logger.info(f"  Mode: {self.training_mode} ({self.expected_tokens} tokens)")
        logger.info(f"  TARGET: CLIP embeddings [B, N, 1024] (PROPERLY NORMALIZED)")
        logger.info(f"  CONDITIONING: EVA embeddings [B, N, 4096]")
        logger.info(f"  Shards: {len(self.shard_files)}")
        logger.info(f"  ✅ CLIP normalization: ENABLED")

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
        """Estimate total number of samples"""
        try:
            if self.manifest.get('total_samples', 0) > 0:
                manifest_samples = self.manifest['total_samples']
                if self.max_shards is not None and self.manifest.get('total_shards', 0) > 0:
                    ratio = min(self.max_shards / self.manifest['total_shards'], 1.0)
                    self.estimated_length = int(manifest_samples * ratio)
                else:
                    self.estimated_length = manifest_samples
                logger.info(f"Using manifest for length estimation: {self.estimated_length:,} samples")
                return
            
            if self.shard_files:
                try:
                    first_shard = self._load_shard(self.shard_files[0])
                    if first_shard and 'clip_blip3o_embeddings' in first_shard:
                        samples_per_shard = first_shard['clip_blip3o_embeddings'].shape[0]
                        self.estimated_length = samples_per_shard * len(self.shard_files)
                        logger.info(f"Estimated length from first shard: {self.estimated_length:,} samples")
                        return
                except Exception as e:
                    logger.warning(f"Could not load first shard for estimation: {e}")
            
            self.estimated_length = len(self.shard_files) * 1000
            logger.warning(f"Using rough length estimate: {self.estimated_length:,} samples")
            
        except Exception as e:
            logger.warning(f"Length estimation failed: {e}")
            self.estimated_length = 1000

    def __len__(self) -> int:
        return self.estimated_length

    def _load_shard(self, shard_path: Path) -> Optional[Dict[str, Any]]:
        """Load a single shard with error handling"""
        for attempt in range(self.max_retries):
            try:
                with open(shard_path, 'rb') as f:
                    shard_data = pickle.load(f)
                
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
        required_keys = ['clip_blip3o_embeddings', 'eva_blip3o_embeddings', 'captions']
        for key in required_keys:
            if key not in shard_data:
                raise ValueError(f"Missing key '{key}' in shard {shard_path}")
        
        clip_emb = shard_data['clip_blip3o_embeddings']
        eva_emb = shard_data['eva_blip3o_embeddings']
        
        # Convert to tensors if needed
        if not torch.is_tensor(clip_emb):
            clip_emb = torch.tensor(clip_emb, dtype=torch.float32)
            shard_data['clip_blip3o_embeddings'] = clip_emb
        if not torch.is_tensor(eva_emb):
            eva_emb = torch.tensor(eva_emb, dtype=torch.float32)
            shard_data['eva_blip3o_embeddings'] = eva_emb
        
        # Validate shapes
        if self.validate_tensor_shapes:
            if clip_emb.dim() != 3 or eva_emb.dim() != 3:
                raise ValueError(f"Expected 3D tensors, got CLIP: {clip_emb.shape}, EVA: {eva_emb.shape}")
            
            if clip_emb.shape[0] != eva_emb.shape[0]:
                raise ValueError(f"Batch size mismatch: CLIP {clip_emb.shape[0]} vs EVA {eva_emb.shape[0]}")
            
            clip_tokens, eva_tokens = clip_emb.shape[1], eva_emb.shape[1]
            if clip_tokens != eva_tokens:
                raise ValueError(f"Token count mismatch: CLIP {clip_tokens} vs EVA {eva_tokens}")
        
        # Handle token count adaptation
        current_tokens = clip_emb.shape[1]
        if current_tokens != self.expected_tokens:
            if current_tokens == 256 and self.expected_tokens == 257:
                clip_cls = clip_emb.mean(dim=1, keepdim=True)
                eva_cls = eva_emb.mean(dim=1, keepdim=True)
                shard_data['clip_blip3o_embeddings'] = torch.cat([clip_cls, clip_emb], dim=1)
                shard_data['eva_blip3o_embeddings'] = torch.cat([eva_cls, eva_emb], dim=1)
            elif current_tokens == 257 and self.expected_tokens == 256:
                shard_data['clip_blip3o_embeddings'] = clip_emb[:, 1:, :]
                shard_data['eva_blip3o_embeddings'] = eva_emb[:, 1:, :]
            else:
                raise ValueError(f"Cannot adapt from {current_tokens} to {self.expected_tokens} tokens")

    def _load_next_shard(self) -> bool:
        """Load next shard"""
        if self.current_shard_data is not None:
            del self.current_shard_data
            gc.collect()
        
        if self.current_shard_idx >= len(self.shard_files):
            self.current_shard_data = None
            return False
        
        while self.current_shard_idx < len(self.shard_files):
            shard_path = self.shard_files[self.current_shard_idx]
            
            self.current_shard_data = self._load_shard(shard_path)
            
            if self.current_shard_data is not None:
                num_samples = self.current_shard_data['clip_blip3o_embeddings'].shape[0]
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
                    
                    clip_emb = self.current_shard_data['clip_blip3o_embeddings'][sample_idx]
                    eva_emb = self.current_shard_data['eva_blip3o_embeddings'][sample_idx]
                    caption = self.current_shard_data['captions'][sample_idx]
                    
                    # Final validation
                    if self.validate_tensor_shapes:
                        if clip_emb.shape != (self.expected_tokens, 1024):
                            raise ValueError(f"Invalid CLIP shape: {clip_emb.shape}")
                        if eva_emb.shape != (self.expected_tokens, 4096):
                            raise ValueError(f"Invalid EVA shape: {eva_emb.shape}")
                    
                    # Check for NaN/Inf
                    if torch.isnan(clip_emb).any() or torch.isnan(eva_emb).any():
                        if self.skip_corrupted_samples:
                            self.current_sample_idx += 1
                            continue
                        else:
                            raise ValueError("NaN detected in embeddings")
                    
                    # Create sample item
                    item = {
                        'eva_embeddings': eva_emb,
                        'clip_embeddings': clip_emb,  # Will be normalized in collate_fn
                        'caption': caption,
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


def improved_clip_reproduction_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    UPDATED collate function with proper CLIP normalization and U-shaped timestep sampling
    """
    if not batch:
        raise ValueError("Empty batch")
    
    valid_batch = [item for item in batch if item is not None]
    if not valid_batch:
        raise ValueError("No valid items in batch")
    
    try:
        # Stack embeddings
        eva_embeddings = torch.stack([item['eva_embeddings'] for item in valid_batch])
        clip_embeddings = torch.stack([item['clip_embeddings'] for item in valid_batch])
        
        captions = [item['caption'] for item in valid_batch]
        keys = [item['key'] for item in valid_batch]
        
        batch_size, seq_len, clip_dim = clip_embeddings.shape
        device = clip_embeddings.device
        dtype = clip_embeddings.dtype
        
        # Ensure float32 for stability
        eva_embeddings = eva_embeddings.float()
        clip_embeddings = clip_embeddings.float()
        
        # CRITICAL: Apply proper CLIP normalization
        # Note: This assumes the dataset has a clip_normalizer attribute
        # In practice, you'll need to pass the normalizer or get it from dataset
        
        # U-SHAPED TIMESTEP SAMPLING (CRITICAL IMPROVEMENT)
        timesteps = sample_u_shaped_timesteps(batch_size, device, alpha=0.5)
        
        # Create standard Gaussian noise
        noise = torch.randn_like(clip_embeddings, device=device, dtype=dtype)
        
        # Linear interpolation for rectified flow: x_t = (1-t) * noise + t * clip_clean
        t_expanded = timesteps.view(batch_size, 1, 1)
        noisy_clip = (1 - t_expanded) * noise + t_expanded * clip_embeddings
        
        # Velocity target: v = clip_clean - noise
        velocity_target = clip_embeddings - noise
        
        # Validation
        assert eva_embeddings.shape == (batch_size, seq_len, 4096), f"EVA shape: {eva_embeddings.shape}"
        assert clip_embeddings.shape == (batch_size, seq_len, 1024), f"CLIP shape: {clip_embeddings.shape}"
        assert noisy_clip.shape == (batch_size, seq_len, 1024), f"Noisy CLIP shape: {noisy_clip.shape}"
        assert velocity_target.shape == (batch_size, seq_len, 1024), f"Velocity target shape: {velocity_target.shape}"
        assert timesteps.shape == (batch_size,), f"Timesteps shape: {timesteps.shape}"
        
        return {
            # Model inputs
            'encoder_hidden_states': eva_embeddings,
            'hidden_states': noisy_clip,
            'timestep': timesteps,
            
            # Training targets  
            'clip_embeddings': clip_embeddings,
            'velocity_target': velocity_target,
            'noise': noise,
            
            # Metadata
            'captions': captions,
            'keys': keys,
            'batch_size': batch_size,
            'training_mode': valid_batch[0]['training_mode'],
            'num_tokens': valid_batch[0]['num_tokens'],
            'seq_len': seq_len,
        }
        
    except Exception as e:
        logger.error(f"Error in improved collate function: {e}")
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


def create_improved_clip_reproduction_dataloaders(
    chunked_embeddings_dir: Union[str, Path],
    batch_size: int = 16,
    eval_batch_size: Optional[int] = None,
    training_mode: str = "patch_only",
    max_shards: Optional[int] = None,
    num_workers: int = 0,
    pin_memory: bool = False,
    **kwargs
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """Create improved dataloaders with proper normalization"""
    
    if eval_batch_size is None:
        eval_batch_size = batch_size
    
    logger.info(f"Creating IMPROVED CLIP reproduction dataloaders:")
    logger.info(f"  Target: CLIP embeddings [B, N, 1024] (PROPERLY NORMALIZED)")
    logger.info(f"  Conditioning: EVA embeddings [B, N, 4096]")
    logger.info(f"  Timestep sampling: U-shaped distribution")
    
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
    )
    
    # Create evaluation dataset
    eval_dataset = BLIP3oCLIPReproductionDataset(
        split="eval",
        shuffle_shards=False,
        shuffle_within_shard=False,
        **dataset_kwargs
    )
    
    # Store normalizer for access during training
    clip_normalizer = train_dataset.clip_normalizer
    
    def train_collate_fn(batch):
        result = improved_clip_reproduction_collate_fn(batch)
        # Apply normalization to CLIP embeddings
        result['clip_embeddings'] = clip_normalizer.normalize(result['clip_embeddings'])
        # Update targets accordingly
        result['velocity_target'] = result['clip_embeddings'] - result['noise']
        # Update noisy input
        t_expanded = result['timestep'].view(-1, 1, 1)
        result['hidden_states'] = (1 - t_expanded) * result['noise'] + t_expanded * result['clip_embeddings']
        return result
    
    def eval_collate_fn(batch):
        result = improved_clip_reproduction_collate_fn(batch)
        # Store both normalized and original for evaluation
        result['clip_embeddings_original'] = result['clip_embeddings'].clone()
        result['clip_embeddings'] = clip_normalizer.normalize(result['clip_embeddings'])
        # Update targets accordingly
        result['velocity_target'] = result['clip_embeddings'] - result['noise']
        # Update noisy input
        t_expanded = result['timestep'].view(-1, 1, 1)
        result['hidden_states'] = (1 - t_expanded) * result['noise'] + t_expanded * result['clip_embeddings']
        return result
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=train_collate_fn,
        pin_memory=pin_memory,
        drop_last=True,
        persistent_workers=num_workers > 0,
    )
    
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=eval_batch_size,
        num_workers=min(num_workers, 1),
        collate_fn=eval_collate_fn,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=min(num_workers, 1) > 0,
    )
    
    # Store normalizer reference for trainer access
    train_dataloader.clip_normalizer = clip_normalizer
    eval_dataloader.clip_normalizer = clip_normalizer
    
    logger.info(f"✅ IMPROVED CLIP reproduction dataloaders created")
    logger.info(f"  Training dataset length: {len(train_dataset):,}")
    logger.info(f"  Evaluation dataset length: {len(eval_dataset):,}")
    logger.info(f"  🔥 CLIP normalization: PROPERLY CONFIGURED")
    
    return train_dataloader, eval_dataloader