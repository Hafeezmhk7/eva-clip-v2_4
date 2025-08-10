"""
FIXED Distributed Dataset Implementation for BLIP3-o
src/modules/datasets/blip3o_distributed_dataset.py

FIXES:
- Proper IterableDataset compatibility (no external samplers)
- Better distributed data sharding
- Improved error handling
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
    
    Ensures that data is properly distributed across GPUs for FSDP training
    while maintaining all existing functionality from the base dataset.
    
    CRITICAL FIXES:
    - Properly distributes shards across ranks
    - No external samplers needed (IterableDataset handles distribution internally)
    - Deterministic shuffling across ranks
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
        
        # Modify shard distribution for distributed training
        if self.is_distributed:
            self._setup_distributed_shards()
        
        if self.rank == 0:
            logger.info(f"Distributed BLIP3-o dataset initialized:")
            logger.info(f"  World size: {self.world_size}")
            logger.info(f"  Rank: {self.rank}")
            logger.info(f"  Shards per rank: {len(self.shard_files)}")
            logger.info(f"  Is distributed: {self.is_distributed}")

    def _setup_distributed_shards(self):
        """FIXED: Distribute shards across ranks for balanced loading"""
        
        if not self.is_distributed or self.world_size == 1:
            return
        
        # Store original shard files
        all_shard_files = self.shard_files.copy()
        
        # Distribute shards round-robin across ranks
        rank_shard_files = []
        for i, shard_file in enumerate(all_shard_files):
            if i % self.world_size == self.rank:
                rank_shard_files.append(shard_file)
        
        self.shard_files = rank_shard_files
        
        # Update estimated length based on rank assignment
        if len(all_shard_files) > 0:
            total_shards = len(all_shard_files)
            shards_per_rank = len(rank_shard_files)
            
            # Adjust estimated length proportionally
            original_length = self.estimated_length
            self.estimated_length = int(original_length * shards_per_rank / total_shards)
        
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
        
        logger.info(f"Prepared {len(self.shard_files)} shard files for distributed processing")


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
    
    dataset_kwargs = {
        'chunked_embeddings_dir': chunked_embeddings_dir,
        'training_mode': training_mode,
        'max_shards': max_shards,
        'simple_scale_factor': simple_scale_factor,
        'world_size': world_size,
        'rank': rank,
        'distributed_seed': distributed_seed,
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
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,  # Shuffling handled by dataset
        sampler=None,   # CRITICAL: No sampler for IterableDataset
        num_workers=num_workers,
        collate_fn=clip_reproduction_collate_fn,
        pin_memory=pin_memory,
        drop_last=drop_last,
        persistent_workers=num_workers > 0,
    )
    
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=eval_batch_size,
        shuffle=False,  # No shuffling for eval
        sampler=None,   # CRITICAL: No sampler for IterableDataset
        num_workers=min(num_workers, 1),
        collate_fn=clip_reproduction_collate_fn,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=min(num_workers, 1) > 0,
    )
    
    if rank == 0:
        logger.info(f"✅ Distributed dataloaders created:")
        logger.info(f"  Training dataset length (per rank): {len(train_dataset):,}")
        logger.info(f"  Evaluation dataset length (per rank): {len(eval_dataset):,}")
        logger.info(f"  Training batches (per rank): {len(train_dataloader):,}")
        logger.info(f"  Evaluation batches (per rank): {len(eval_dataloader):,}")
        logger.info(f"  Total effective batch size: {batch_size * world_size}")
        logger.info(f"  CLIP normalization: DISABLED")
        logger.info(f"  🔧 FIX APPLIED: No samplers used with IterableDataset")
        
        # Test dataloader
        try:
            test_batch = next(iter(train_dataloader))
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
            raise
    
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


class DistributedDataLoaderMetrics:
    """Utility class for tracking distributed dataloader metrics"""
    
    def __init__(self, world_size: int, rank: int):
        self.world_size = world_size
        self.rank = rank
        self.batch_counts = []
        self.sample_counts = []
        self.load_times = []
    
    def record_batch(self, batch_size: int, load_time: float):
        """Record metrics for a batch"""
        self.batch_counts.append(1)
        self.sample_counts.append(batch_size)
        self.load_times.append(load_time)
    
    def get_metrics(self) -> Dict[str, float]:
        """Get aggregated metrics"""
        if not self.batch_counts:
            return {}
        
        return {
            'total_batches': sum(self.batch_counts),
            'total_samples': sum(self.sample_counts),
            'avg_batch_size': np.mean(self.sample_counts),
            'avg_load_time': np.mean(self.load_times),
            'samples_per_second': sum(self.sample_counts) / sum(self.load_times) if sum(self.load_times) > 0 else 0,
            'rank': self.rank,
            'world_size': self.world_size,
        }
    
    def sync_metrics_across_ranks(self) -> Dict[str, float]:
        """Synchronize metrics across all ranks"""
        local_metrics = self.get_metrics()
        
        if not torch.distributed.is_initialized():
            return local_metrics
        
        # Convert metrics to tensors for synchronization
        metrics_to_sync = [
            'total_batches', 'total_samples', 'avg_batch_size', 
            'avg_load_time', 'samples_per_second'
        ]
        
        synced_metrics = {}
        for metric in metrics_to_sync:
            if metric in local_metrics:
                tensor = torch.tensor(float(local_metrics[metric]), device='cuda')
                torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
                synced_metrics[f'total_{metric}'] = tensor.item()
                synced_metrics[f'avg_{metric}'] = tensor.item() / self.world_size
        
        synced_metrics['world_size'] = self.world_size
        return synced_metrics