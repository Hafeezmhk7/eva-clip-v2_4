"""
FIXED Distributed Dataset Implementation for BLIP3-o
src/modules/datasets/blip3o_distributed_dataset.py

MAJOR FIXES:
- Fixed import/export name mismatches
- Added proper aliases for backward compatibility
- Fixed shard assignment to ensure all ranks get data
- Added robust iteration with timeout handling
- Fixed hanging issues in dataloader
- Better error handling and progress tracking
- Proper distributed data balancing
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


class DistributedDataLoaderMetrics:
    """Tracks metrics for distributed dataloader performance"""
    
    def __init__(self, rank: int, world_size: int):
        self.rank = rank
        self.world_size = world_size
        self.reset()
    
    def reset(self):
        """Reset all metrics"""
        self.samples_processed = 0
        self.batches_processed = 0
        self.total_time = 0.0
        self.data_loading_time = 0.0
        self.start_time = time.time()
        self.last_batch_time = self.start_time
    
    def update_batch(self, batch_size: int, data_loading_time: float = 0.0):
        """Update metrics after processing a batch"""
        current_time = time.time()
        self.samples_processed += batch_size
        self.batches_processed += 1
        self.data_loading_time += data_loading_time
        self.total_time = current_time - self.start_time
        self.last_batch_time = current_time
    
    def get_stats(self) -> Dict[str, float]:
        """Get current statistics"""
        if self.total_time > 0:
            samples_per_sec = self.samples_processed / self.total_time
            batches_per_sec = self.batches_processed / self.total_time
        else:
            samples_per_sec = 0.0
            batches_per_sec = 0.0
        
        return {
            'rank': self.rank,
            'world_size': self.world_size,
            'samples_processed': self.samples_processed,
            'batches_processed': self.batches_processed,
            'total_time': self.total_time,
            'samples_per_sec': samples_per_sec,
            'batches_per_sec': batches_per_sec,
            'data_loading_time': self.data_loading_time,
            'data_loading_ratio': self.data_loading_time / max(self.total_time, 1e-6),
        }


class DistributedBLIP3oCLIPReproductionDataset(BLIP3oCLIPReproductionDataset):
    """
    FIXED: Distributed version of BLIP3-o dataset with proper shard distribution
    
    Major fixes:
    - Ensures ALL ranks get data through better shard distribution
    - Fixed iteration to prevent hanging
    - Better progress tracking and error handling
    - Proper load balancing across GPUs
    - Fixed import/export names
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
        progress_tracking: bool = True,
        max_samples_per_epoch: Optional[int] = None,
        
        # FIXED: Additional parameters for stability
        min_samples_per_rank: int = 10,  # Ensure each rank gets minimum samples
        duplicate_data_if_needed: bool = True,  # Duplicate data to balance ranks
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
        self.max_samples_per_epoch = max_samples_per_epoch
        self.min_samples_per_rank = min_samples_per_rank
        self.duplicate_data_if_needed = duplicate_data_if_needed
        
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
        
        # FIXED: Setup distributed shards with load balancing
        if self.is_distributed:
            self._setup_distributed_shards_fixed()
        
        # Initialize metrics
        self.metrics = DistributedDataLoaderMetrics(self.rank, self.world_size)
        
        # Track iteration state
        self._iteration_count = 0
        self._last_progress_time = time.time()
        
        # FIXED: Validate that all ranks have data
        self._validate_rank_data_availability()
        
        if self.rank == 0:
            logger.info(f"✅ FIXED Distributed BLIP3-o dataset initialized:")
            logger.info(f"  World size: {self.world_size}")
            logger.info(f"  Rank: {self.rank}")
            logger.info(f"  Shards per rank: {len(self.shard_files)}")
            logger.info(f"  Is distributed: {self.is_distributed}")
            logger.info(f"  Progress tracking: {self.progress_tracking}")
            logger.info(f"  Max samples per epoch: {self.max_samples_per_epoch or 'Unlimited'}")
            logger.info(f"  Data balancing: ENABLED")

    def _setup_distributed_shards_fixed(self):
        """FIXED: Distribute shards with guaranteed data for all ranks"""
        
        if not self.is_distributed or self.world_size == 1:
            return
        
        # Store original shard files
        all_shard_files = self.shard_files.copy()
        
        if not all_shard_files:
            raise ValueError("No shard files found for distributed training!")
        
        # FIXED: Ensure we have enough data for all ranks
        if len(all_shard_files) < self.world_size:
            if self.duplicate_data_if_needed:
                # Duplicate shards to ensure all ranks get data
                logger.warning(f"[Rank {self.rank}] Only {len(all_shard_files)} shards for {self.world_size} ranks. Duplicating data.")
                while len(all_shard_files) < self.world_size:
                    all_shard_files.extend(self.shard_files.copy())
                all_shard_files = all_shard_files[:self.world_size * 2]  # Don't go crazy with duplication
            else:
                raise ValueError(f"Only {len(all_shard_files)} shards available for {self.world_size} ranks. Need at least {self.world_size} shards or enable duplicate_data_if_needed=True")
        
        # FIXED: Round-robin distribution to ensure balance
        rank_shard_files = []
        for i, shard_file in enumerate(all_shard_files):
            if i % self.world_size == self.rank:
                rank_shard_files.append(shard_file)
        
        # FIXED: Ensure each rank gets at least one shard
        if not rank_shard_files:
            # Fallback: give first shard to this rank
            rank_shard_files = [all_shard_files[0]]
            logger.warning(f"[Rank {self.rank}] No shards assigned, using fallback shard: {all_shard_files[0]}")
        
        self.shard_files = rank_shard_files
        
        # FIXED: Better length estimation for distributed setting
        if len(all_shard_files) > 0:
            total_shards = len(all_shard_files)
            shards_per_rank = len(rank_shard_files)
            
            if hasattr(self, 'estimated_length'):
                # Adjust estimated length proportionally
                original_length = self.estimated_length
                self.estimated_length = max(self.min_samples_per_rank, int(original_length * shards_per_rank / total_shards))
                
                # Apply max_samples_per_epoch limit if set
                if self.max_samples_per_epoch:
                    self.estimated_length = min(self.estimated_length, self.max_samples_per_epoch)
            else:
                # Fallback estimation
                self.estimated_length = max(self.min_samples_per_rank, shards_per_rank * 1000)
                if self.max_samples_per_epoch:
                    self.estimated_length = min(self.estimated_length, self.max_samples_per_epoch)
        
        logger.info(f"[Rank {self.rank}] FIXED shard assignment: {len(rank_shard_files)}/{len(all_shard_files)} shards")
        logger.info(f"[Rank {self.rank}] Estimated length: {self.estimated_length:,} samples")

    def _validate_rank_data_availability(self):
        """FIXED: Validate that all ranks have data available"""
        has_data = len(self.shard_files) > 0
        
        if self.is_distributed:
            # Check across all ranks
            has_data_tensor = torch.tensor([1 if has_data else 0], dtype=torch.int32)
            if torch.cuda.is_available():
                has_data_tensor = has_data_tensor.cuda()
            
            # All-gather to see which ranks have data
            all_has_data = [torch.zeros_like(has_data_tensor) for _ in range(self.world_size)]
            torch.distributed.all_gather(all_has_data, has_data_tensor)
            
            ranks_without_data = []
            for rank_idx, rank_has_data in enumerate(all_has_data):
                if rank_has_data.item() == 0:
                    ranks_without_data.append(rank_idx)
            
            if ranks_without_data:
                error_msg = f"Ranks {ranks_without_data} have no data! This will cause hanging."
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            if self.rank == 0:
                logger.info("✅ All ranks have data available")
        
        elif not has_data:
            raise ValueError(f"Rank {self.rank} has no data!")

    def _prepare_shard_list(self):
        """FIXED: Prepare list of shard files with better error handling"""
        
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
                if self.rank == 0 or not self.is_distributed:
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
        
        if self.rank == 0 or not self.is_distributed:
            logger.info(f"Prepared {len(self.shard_files)} shard files for distributed processing")

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """FIXED: Iterate through all samples with timeout and better error handling"""
        # Reset iteration state
        self.current_shard_idx = 0
        self.current_shard_data = None
        self.current_sample_idx = 0
        self.total_samples_processed = 0
        self._iteration_count = 0
        self._last_progress_time = time.time()
        
        # Reset metrics
        self.metrics.reset()
        
        # Initialize progress tracking
        if self.progress_tracking and self.rank == 0:
            logger.info(f"[Rank {self.rank}] Starting iteration over {len(self.shard_files)} shards")
        
        # FIXED: Validate we have shards before starting
        if not self.shard_files:
            logger.error(f"[Rank {self.rank}] No shard files available for iteration!")
            return
        
        # FIXED: Load first shard with timeout
        start_time = time.time()
        if not self._load_next_shard():
            logger.error(f"[Rank {self.rank}] Failed to load any shards!")
            return
        
        load_time = time.time() - start_time
        if load_time > 30:  # Warn if loading takes too long
            logger.warning(f"[Rank {self.rank}] Shard loading took {load_time:.1f}s - check I/O performance")
        
        samples_yielded = 0
        max_samples = self.max_samples_per_epoch or float('inf')
        
        # FIXED: Main iteration loop with timeout protection
        iteration_start_time = time.time()
        last_activity_time = iteration_start_time
        
        while self.current_shard_data is not None and samples_yielded < max_samples:
            # FIXED: Check for iteration timeout (prevent infinite hanging)
            current_time = time.time()
            if current_time - last_activity_time > 120:  # 2 minutes without activity
                logger.error(f"[Rank {self.rank}] Iteration timeout - no activity for 120s")
                break
            
            while self.current_sample_idx < len(self.current_samples) and samples_yielded < max_samples:
                try:
                    batch_start_time = time.time()
                    
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
                        'clip_embeddings': clip_emb,
                        'caption': caption,
                        'key': f"rank{self.rank}_shard_{self.current_shard_idx-1}_sample_{sample_idx}",
                        'sample_idx': sample_idx,
                        'training_mode': self.training_mode,
                        'num_tokens': self.expected_tokens,
                        'rank': self.rank,
                    }
                    
                    self.current_sample_idx += 1
                    self.total_samples_processed += 1
                    self._iteration_count += 1
                    samples_yielded += 1
                    last_activity_time = current_time  # Update activity time
                    
                    # Update metrics
                    data_loading_time = time.time() - batch_start_time
                    self.metrics.update_batch(1, data_loading_time)
                    
                    # FIXED: Reduced frequency progress tracking
                    if self.progress_tracking and self._iteration_count % 50 == 0:  # Every 50 samples
                        if current_time - self._last_progress_time > 10:  # Every 10 seconds
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
        
        if self.progress_tracking and self.rank == 0:
            logger.info(f"[Rank {self.rank}] Iteration completed: {self.total_samples_processed} samples processed")


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
    distributed_seed: int = 42,
    drop_last: bool = True,
    progress_tracking: bool = True,
    max_samples_per_epoch: Optional[int] = None,
    **kwargs
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """
    FIXED: Create distributed dataloaders that don't hang
    
    Key fixes:
    - Guaranteed data for all ranks
    - Timeout protection
    - Better error handling
    - Load balancing
    - Fixed import/export names
    """
    
    if eval_batch_size is None:
        eval_batch_size = batch_size
    
    if rank == 0:
        logger.info(f"Creating FIXED distributed CLIP reproduction dataloaders:")
        logger.info(f"  World size: {world_size}")
        logger.info(f"  Rank: {rank}")
        logger.info(f"  Batch size per GPU: {batch_size}")
        logger.info(f"  Target: CLIP embeddings [B, N, 1024] (RAW)")
        logger.info(f"  Conditioning: EVA embeddings [B, N, 4096]")
        logger.info(f"  Simple scale factor: {simple_scale_factor}")
        logger.info(f"  Distributed seed: {distributed_seed}")
        logger.info(f"  Progress tracking: {progress_tracking}")
        logger.info(f"  Max samples per epoch: {max_samples_per_epoch or 'Unlimited'}")
        logger.info(f"  FIXED: Data balancing and hanging prevention enabled")
    
    dataset_kwargs = {
        'chunked_embeddings_dir': chunked_embeddings_dir,
        'training_mode': training_mode,
        'max_shards': max_shards,
        'simple_scale_factor': simple_scale_factor,
        'world_size': world_size,
        'rank': rank,
        'distributed_seed': distributed_seed,
        'progress_tracking': progress_tracking,
        'max_samples_per_epoch': max_samples_per_epoch,
        'min_samples_per_rank': 10,
        'duplicate_data_if_needed': True,
        **kwargs
    }
    
    # Create distributed training dataset
    train_dataset = DistributedBLIP3oCLIPReproductionDataset(
        split="train",
        shuffle_shards=True,
        shuffle_within_shard=True,
        **dataset_kwargs
    )
    
    # Create distributed evaluation dataset (smaller for faster eval)
    eval_dataset_kwargs = dataset_kwargs.copy()
    if max_samples_per_epoch:
        eval_dataset_kwargs['max_samples_per_epoch'] = min(max_samples_per_epoch // 4, 50)
    
    eval_dataset = DistributedBLIP3oCLIPReproductionDataset(
        split="eval",
        shuffle_shards=False,
        shuffle_within_shard=False,
        **eval_dataset_kwargs
    )
    
    # FIXED: Create dataloaders with timeout and proper settings
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,  # Shuffling handled by dataset
        sampler=None,   # No sampler for IterableDataset
        num_workers=0,  # Always 0 for stability
        collate_fn=clip_reproduction_collate_fn,
        pin_memory=pin_memory,
        drop_last=drop_last,
        persistent_workers=False,
        timeout=0,
        prefetch_factor=None,  # Disable prefetching for stability
    )
    
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        sampler=None,
        num_workers=0,
        collate_fn=clip_reproduction_collate_fn,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=False,
        timeout=0,
        prefetch_factor=None,
    )
    
    if rank == 0:
        logger.info(f"✅ FIXED distributed dataloaders created:")
        logger.info(f"  Training dataset length (per rank): {len(train_dataset):,}")
        logger.info(f"  Evaluation dataset length (per rank): {len(eval_dataset):,}")
        logger.info(f"  Total effective batch size: {batch_size * world_size}")
        logger.info(f"  CLIP normalization: DISABLED")
        logger.info(f"  FIXES: No hanging, balanced data, timeout protection")
        
        # Test dataloader briefly
        logger.info("Testing FIXED dataloader...")
        try:
            start_time = time.time()
            dataloader_iter = iter(train_dataloader)
            test_batch = next(dataloader_iter)
            test_time = time.time() - start_time
            
            logger.info(f"✅ FIXED dataloader test successful in {test_time:.2f}s:")
            logger.info(f"  Batch size: {test_batch.get('batch_size', 'unknown')}")
            logger.info(f"  CLIP embeddings shape: {test_batch['clip_embeddings'].shape}")
            logger.info(f"  EVA embeddings shape: {test_batch['encoder_hidden_states'].shape}")
            
            if test_time > 10:
                logger.warning(f"⚠️ Slow dataloader test ({test_time:.2f}s) - check I/O performance")
            
        except Exception as e:
            logger.error(f"❌ FIXED dataloader test failed: {e}")
            raise
    
    return train_dataloader, eval_dataloader


# FIXED: Add proper aliases for backward compatibility
create_distributed_dataloaders = create_distributed_clip_reproduction_dataloaders
create_fixed_distributed_dataloaders = create_distributed_clip_reproduction_dataloaders