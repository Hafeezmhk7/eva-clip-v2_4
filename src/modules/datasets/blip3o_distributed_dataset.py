"""
FIXED Distributed Dataset Implementation for BLIP3-o
src/modules/datasets/blip3o_distributed_dataset.py

MAJOR FIXES:
- Fixed iteration to prevent hanging
- Better distributed data sharding
- Improved error handling and progress tracking
- Fixed IterableDataset compatibility issues
"""

import torch
from torch.utils.data import DataLoader
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
from tqdm import tqdm

# Import base dataset
from .blip3o_dataset import (
    BLIP3oCLIPReproductionDataset,
    clip_reproduction_collate_fn,
    sample_u_shaped_timesteps
)

logger = logging.getLogger(__name__)


class DistributedBLIP3oCLIPReproductionDataset(BLIP3oCLIPReproductionDataset):
    """
    FIXED: Distributed version of BLIP3-o dataset with proper shard distribution
    
    Major fixes:
    - Ensures proper iteration without hanging
    - Better distributed sampling
    - Progress tracking for debugging
    - Fixed IterableDataset handling
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
        simple_scale_factor: float = 1.0,
        
        # Distributed-specific parameters
        world_size: Optional[int] = None,
        rank: Optional[int] = None,
        distributed_seed: int = 42,
        progress_tracking: bool = True,  # NEW: Enable progress tracking
    ):
        
        # Set distributed parameters
        if torch.distributed.is_initialized():
            self.world_size = world_size or torch.distributed.get_world_size()
            self.rank = rank or torch.distributed.get_rank()
            self.is_distributed = True
        else:
            self.world_size = 1
            self.rank = 0
            self.is_distributed = False
        
        self.distributed_seed = distributed_seed
        self.progress_tracking = progress_tracking
        
        # Initialize base dataset first
        super().__init__(
            chunked_embeddings_dir=chunked_embeddings_dir,
            split=split,
            training_mode=training_mode,
            max_shards=max_shards,
            shuffle_shards=shuffle_shards,
            shuffle_within_shard=shuffle_within_shard,
            expected_tokens=expected_tokens,
            skip_corrupted_samples=skip_corrupted_samples,
            validate_tensor_shapes=validate_tensor_shapes,
            max_retries=max_retries,
            simple_scale_factor=simple_scale_factor,
        )
        
        # FIXED: Setup distributed shards properly
        if self.is_distributed:
            self._setup_distributed_shards()
        
        # Track iteration state
        self._iteration_count = 0
        self._last_progress_time = time.time()
        
        if self.rank == 0:
            logger.info(f"✅ Distributed BLIP3-o dataset initialized:")
            logger.info(f"  World size: {self.world_size}")
            logger.info(f"  Rank: {self.rank}")
            logger.info(f"  Shards per rank: {len(self.shard_files)}")
            logger.info(f"  Is distributed: {self.is_distributed}")
            logger.info(f"  Progress tracking: {self.progress_tracking}")

    def _setup_distributed_shards(self):
        """FIXED: Distribute shards across ranks for balanced loading"""
        
        if not self.is_distributed or self.world_size == 1:
            return
        
        # Store original shard files
        all_shard_files = self.shard_files.copy()
        
        # FIXED: More robust shard distribution
        rank_shard_files = []
        for i, shard_file in enumerate(all_shard_files):
            if i % self.world_size == self.rank:
                rank_shard_files.append(shard_file)
        
        self.shard_files = rank_shard_files
        
        # FIXED: Better length estimation for distributed setting
        if len(all_shard_files) > 0:
            total_shards = len(all_shard_files)
            shards_per_rank = len(rank_shard_files)
            
            if hasattr(self, 'estimated_length'):
                # Adjust estimated length proportionally
                original_length = self.estimated_length
                self.estimated_length = max(1, int(original_length * shards_per_rank / total_shards))
            else:
                # Fallback estimation
                self.estimated_length = max(1, shards_per_rank * 1000)
        
        logger.info(f"[Rank {self.rank}] Assigned {len(rank_shard_files)}/{len(all_shard_files)} shards")
        logger.info(f"[Rank {self.rank}] Estimated length: {self.estimated_length:,} samples")

    def _prepare_shard_list(self):
        """FIXED: Prepare list of shard files with distributed-aware shuffling"""
        
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
                if self.rank == 0:
                    logger.info(f"Found {len(shard_files)} files with pattern: {pattern}")
                break
        
        if not shard_files:
            raise FileNotFoundError(f"No shard files found in {self.chunked_embeddings_dir}")
        
        # Sort files consistently across all ranks
        shard_files.sort(key=lambda x: int(''.join(filter(str.isdigit, x.stem))) if any(c.isdigit() for c in x.stem) else 0)
        
        # Apply max shards limit before distribution
        if self.max_shards is not None:
            shard_files = shard_files[:self.max_shards]
        
        # Filter existing files
        shard_files = [f for f in shard_files if f.exists()]
        
        # FIXED: Shuffle consistently across ranks if requested
        if self.shuffle_shards:
            # Use same random seed across all ranks for consistent shuffling
            shuffle_rng = random.Random(self.distributed_seed)
            shuffle_rng.shuffle(shard_files)
        
        self.shard_files = shard_files
        
        if self.rank == 0:
            logger.info(f"Prepared {len(self.shard_files)} shard files for distributed processing")

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """FIXED: Iterate through all samples with progress tracking"""
        # Reset iteration state
        self.current_shard_idx = 0
        self.current_shard_data = None
        self.current_sample_idx = 0
        self.total_samples_processed = 0
        self._iteration_count = 0
        self._last_progress_time = time.time()
        
        # Initialize progress tracking
        if self.progress_tracking and self.rank == 0:
            logger.info(f"Starting iteration over {len(self.shard_files)} shards")
        
        # FIXED: Ensure we can load at least one shard
        if not self._load_next_shard():
            logger.warning(f"[Rank {self.rank}] No valid shards found!")
            return
        
        samples_yielded = 0
        max_samples = getattr(self, 'max_samples_per_epoch', None) or float('inf')
        
        while self.current_shard_data is not None and samples_yielded < max_samples:
            while self.current_sample_idx < len(self.current_samples) and samples_yielded < max_samples:
                try:
                    sample_idx = self.current_samples[self.current_sample_idx]
                    
                    clip_emb = self.current_shard_data['clip_blip3o_embeddings'][sample_idx]
                    eva_emb = self.current_shard_data['eva_blip3o_embeddings'][sample_idx]
                    caption = self.current_shard_data['captions'][sample_idx]
                    
                    # Apply simple scaling if specified (data-independent)
                    if self.simple_scale_factor != 1.0:
                        clip_emb = clip_emb * self.simple_scale_factor
                    
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
                        'clip_embeddings': clip_emb,  # Raw CLIP embeddings (no normalization)
                        'caption': caption,
                        'key': f"rank{self.rank}_shard_{self.current_shard_idx-1}_sample_{sample_idx}",
                        'sample_idx': sample_idx,
                        'training_mode': self.training_mode,
                        'num_tokens': self.expected_tokens,
                        'rank': self.rank,  # For debugging
                    }
                    
                    self.current_sample_idx += 1
                    self.total_samples_processed += 1
                    self._iteration_count += 1
                    samples_yielded += 1
                    
                    # FIXED: Progress tracking without flooding logs
                    if self.progress_tracking and self._iteration_count % 100 == 0:
                        current_time = time.time()
                        if current_time - self._last_progress_time > 30:  # Log every 30 seconds
                            logger.info(f"[Rank {self.rank}] Processed {self.total_samples_processed} samples, "
                                      f"current shard: {self.current_shard_idx-1}/{len(self.shard_files)}")
                            self._last_progress_time = current_time
                    
                    yield item
                    
                except Exception as e:
                    if self.skip_corrupted_samples:
                        logger.warning(f"[Rank {self.rank}] Skipping corrupted sample {sample_idx}: {e}")
                        self.current_sample_idx += 1
                        continue
                    else:
                        raise
            
            # Load next shard
            if not self._load_next_shard():
                break
        
        if self.progress_tracking:
            logger.info(f"[Rank {self.rank}] Iteration completed: {self.total_samples_processed} samples processed")


def create_distributed_dataloaders(
    chunked_embeddings_dir: Union[str, Path],
    world_size: int,
    rank: int,
    batch_size: int = 16,
    eval_batch_size: Optional[int] = None,
    training_mode: str = "patch_only",
    max_shards: Optional[int] = None,
    num_workers: int = 0,
    pin_memory: bool = False,
    simple_scale_factor: float = 1.0,
    distributed_seed: int = 42,
    drop_last: bool = True,
    progress_tracking: bool = True,
    **kwargs
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """
    FIXED: Create distributed dataloaders for FSDP training
    
    IMPORTANT: IterableDataset cannot use external samplers.
    Distribution is handled internally by the dataset.
    """
    
    if eval_batch_size is None:
        eval_batch_size = batch_size
    
    if rank == 0:
        logger.info(f"Creating distributed CLIP reproduction dataloaders:")
        logger.info(f"  World size: {world_size}")
        logger.info(f"  Rank: {rank}")
        logger.info(f"  Batch size per GPU: {batch_size}")
        logger.info(f"  Target: CLIP embeddings [B, N, 1024] (RAW)")
        logger.info(f"  Conditioning: EVA embeddings [B, N, 4096]")
        logger.info(f"  Simple scale factor: {simple_scale_factor}")
        logger.info(f"  Distributed seed: {distributed_seed}")
        logger.info(f"  Progress tracking: {progress_tracking}")
    
    dataset_kwargs = {
        'chunked_embeddings_dir': chunked_embeddings_dir,
        'training_mode': training_mode,
        'max_shards': max_shards,
        'simple_scale_factor': simple_scale_factor,
        'world_size': world_size,
        'rank': rank,
        'distributed_seed': distributed_seed,
        'progress_tracking': progress_tracking,
        **kwargs
    }
    
    # Create distributed training dataset
    train_dataset = DistributedBLIP3oCLIPReproductionDataset(
        split="train",
        shuffle_shards=True,
        shuffle_within_shard=True,
        **dataset_kwargs
    )
    
    # Create distributed evaluation dataset
    eval_dataset = DistributedBLIP3oCLIPReproductionDataset(
        split="eval",
        shuffle_shards=False,
        shuffle_within_shard=False,
        **dataset_kwargs
    )
    
    # FIXED: No samplers for IterableDataset - distribution handled internally
    # Also using minimal num_workers to avoid hanging
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,  # Shuffling handled by dataset
        sampler=None,   # CRITICAL: No sampler for IterableDataset
        num_workers=0,  # FIXED: Use 0 workers to avoid hanging in distributed setting
        collate_fn=clip_reproduction_collate_fn,
        pin_memory=pin_memory,
        drop_last=drop_last,
        persistent_workers=False,  # FIXED: Disable for stability
        timeout=0,  # FIXED: No timeout for stability
    )
    
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=eval_batch_size,
        shuffle=False,  # No shuffling for eval
        sampler=None,   # CRITICAL: No sampler for IterableDataset
        num_workers=0,  # FIXED: Use 0 workers to avoid hanging
        collate_fn=clip_reproduction_collate_fn,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=False,  # FIXED: Disable for stability
        timeout=0,  # FIXED: No timeout for stability
    )
    
    if rank == 0:
        logger.info(f"✅ Distributed dataloaders created:")
        logger.info(f"  Training dataset length (per rank): {len(train_dataset):,}")
        logger.info(f"  Evaluation dataset length (per rank): {len(eval_dataset):,}")
        logger.info(f"  Total effective batch size: {batch_size * world_size}")
        logger.info(f"  CLIP normalization: DISABLED")
        logger.info(f"  🔧 FIX APPLIED: No samplers, no workers for stability")
        
        # FIXED: Simple dataloader test without hanging
        try:
            logger.info("Testing dataloader...")
            dataloader_iter = iter(train_dataloader)
            test_batch = next(dataloader_iter)
            
            logger.info(f"✅ Distributed dataloader test successful:")
            logger.info(f"  Batch size: {test_batch.get('batch_size', 'unknown')}")
            logger.info(f"  CLIP embeddings shape: {test_batch['clip_embeddings'].shape}")
            logger.info(f"  EVA embeddings shape: {test_batch['encoder_hidden_states'].shape}")
            
            # Check CLIP embedding range (should be raw)
            sample_clip = test_batch['clip_embeddings']
            clip_range = (sample_clip.min().item(), sample_clip.max().item())
            logger.info(f"  Raw CLIP range: [{clip_range[0]:.3f}, {clip_range[1]:.3f}]")
            
            # Show effect of scaling if applied
            if simple_scale_factor != 1.0:
                logger.info(f"  Scale factor applied: {simple_scale_factor}")
                
        except Exception as e:
            logger.error(f"❌ Distributed dataloader test failed: {e}")
            # Don't raise here - continue with training, test will happen during training
    
    return train_dataloader, eval_dataloader


def create_distributed_clip_reproduction_dataloaders(
    chunked_embeddings_dir: Union[str, Path],
    world_size: int,
    rank: int,
    batch_size: int = 16,
    eval_batch_size: Optional[int] = None,
    training_mode: str = "patch_only",
    max_shards: Optional[int] = None,
    num_workers: int = 0,
    pin_memory: bool = False,
    simple_scale_factor: float = 1.0,
    **kwargs
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """
    Factory function for creating distributed CLIP reproduction dataloaders
    
    This is the main entry point for distributed dataloader creation,
    providing the same interface as the single-GPU version but with
    distributed sampling support.
    """
    
    return create_distributed_dataloaders(
        chunked_embeddings_dir=chunked_embeddings_dir,
        world_size=world_size,
        rank=rank,
        batch_size=batch_size,
        eval_batch_size=eval_batch_size,
        training_mode=training_mode,
        max_shards=max_shards,
        num_workers=num_workers,
        pin_memory=pin_memory,
        simple_scale_factor=simple_scale_factor,
        **kwargs
    )