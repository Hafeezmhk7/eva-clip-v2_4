#!/usr/bin/env python3
"""
Robust COCO Collate Function with Comprehensive Error Handling
robust_coco_collate.py

This file contains the robust collate function and helper utilities
for handling COCO evaluation data with extensive error handling.
"""

import torch
import torch.nn.functional as F
import logging
import numpy as np
from typing import Dict, Any, List, Optional, Union

logger = logging.getLogger(__name__)


def sample_u_shaped_timesteps(batch_size: int, device: torch.device, alpha: float = 0.5) -> torch.Tensor:
    """Sample timesteps from U-shaped distribution for better training dynamics"""
    from torch.distributions import Beta
    
    beta_alpha = max(0.1, 1 - alpha)
    beta_dist = Beta(beta_alpha, beta_alpha)
    timesteps = beta_dist.sample((batch_size,)).to(device)
    
    # Clamp to avoid numerical issues at endpoints
    timesteps = torch.clamp(timesteps, min=1e-3, max=1.0 - 1e-3)
    
    return timesteps


def validate_tensor_item(item: Dict[str, Any], item_idx: int) -> bool:
    """Validate a single item from the batch"""
    required_keys = ['eva_embeddings', 'clip_embeddings', 'caption', 'key', 'training_mode', 'num_tokens']
    
    # Check required keys
    missing_keys = [key for key in required_keys if key not in item]
    if missing_keys:
        logger.error(f"Item {item_idx} missing keys: {missing_keys}")
        return False
    
    # Check tensor types and shapes
    eva_emb = item['eva_embeddings']
    clip_emb = item['clip_embeddings']
    
    if not torch.is_tensor(eva_emb):
        logger.error(f"Item {item_idx}: eva_embeddings is not a tensor: {type(eva_emb)}")
        return False
    
    if not torch.is_tensor(clip_emb):
        logger.error(f"Item {item_idx}: clip_embeddings is not a tensor: {type(clip_emb)}")
        return False
    
    # Check tensor dimensions
    if eva_emb.dim() != 2:
        logger.error(f"Item {item_idx}: eva_embeddings wrong dims: {eva_emb.shape} (expected 2D)")
        return False
    
    if clip_emb.dim() != 2:
        logger.error(f"Item {item_idx}: clip_embeddings wrong dims: {clip_emb.shape} (expected 2D)")
        return False
    
    # Check sequence length consistency
    if eva_emb.shape[0] != clip_emb.shape[0]:
        logger.error(f"Item {item_idx}: sequence length mismatch: EVA {eva_emb.shape[0]} vs CLIP {clip_emb.shape[0]}")
        return False
    
    # Check embedding dimensions
    if eva_emb.shape[1] != 4096:
        logger.error(f"Item {item_idx}: eva_embeddings wrong dim: {eva_emb.shape[1]} (expected 4096)")
        return False
    
    if clip_emb.shape[1] != 1024:
        logger.error(f"Item {item_idx}: clip_embeddings wrong dim: {clip_emb.shape[1]} (expected 1024)")
        return False
    
    # Check for NaN/Inf
    if torch.isnan(eva_emb).any():
        logger.error(f"Item {item_idx}: NaN in eva_embeddings")
        return False
    
    if torch.isnan(clip_emb).any():
        logger.error(f"Item {item_idx}: NaN in clip_embeddings")
        return False
    
    if torch.isinf(eva_emb).any():
        logger.error(f"Item {item_idx}: Inf in eva_embeddings")
        return False
    
    if torch.isinf(clip_emb).any():
        logger.error(f"Item {item_idx}: Inf in clip_embeddings")
        return False
    
    return True


def debug_batch_info(batch: List[Dict[str, Any]]) -> None:
    """Print detailed debug information about the batch"""
    logger.info(f"🔍 DEBUG: Batch analysis")
    logger.info(f"   Batch size: {len(batch)}")
    
    for i, item in enumerate(batch[:3]):  # Only show first 3 items
        if item is None:
            logger.info(f"   Item {i}: None")
            continue
        
        logger.info(f"   Item {i}:")
        logger.info(f"     Keys: {list(item.keys())}")
        
        for key, value in item.items():
            if torch.is_tensor(value):
                logger.info(f"     {key}: {value.shape} {value.dtype}")
            else:
                logger.info(f"     {key}: {type(value)} = {str(value)[:50]}...")


def robust_coco_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    ROBUST collate function with extensive error handling for COCO data
    
    This function handles:
    1. Empty or None batches
    2. Invalid tensor shapes
    3. Missing required keys
    4. NaN/Inf values
    5. Dtype mismatches
    6. Device inconsistencies
    """
    
    # Phase 1: Basic validation
    try:
        if not batch:
            raise ValueError("Empty batch provided to collate function")
        
        logger.debug(f"Processing batch of size {len(batch)}")
        
        # Filter out None items
        valid_items = []
        for i, item in enumerate(batch):
            if item is None:
                logger.warning(f"Skipping None item at index {i}")
                continue
            valid_items.append(item)
        
        if not valid_items:
            raise ValueError("No valid items in batch after filtering None items")
        
        logger.debug(f"Valid items after filtering: {len(valid_items)}")
        
    except Exception as e:
        logger.error(f"❌ Phase 1 error (basic validation): {e}")
        debug_batch_info(batch)
        raise
    
    # Phase 2: Item validation
    try:
        validated_items = []
        for i, item in enumerate(valid_items):
            if validate_tensor_item(item, i):
                validated_items.append(item)
            else:
                logger.warning(f"Skipping invalid item at index {i}")
        
        if not validated_items:
            raise ValueError("No valid items after tensor validation")
        
        valid_batch = validated_items
        logger.debug(f"Items after validation: {len(valid_batch)}")
        
    except Exception as e:
        logger.error(f"❌ Phase 2 error (item validation): {e}")
        debug_batch_info(valid_items)
        raise
    
    # Phase 3: Tensor stacking
    try:
        logger.debug("Attempting to stack tensors...")
        
        # Extract tensors with shape logging
        eva_tensors = []
        clip_tensors = []
        
        for i, item in enumerate(valid_batch):
            eva_emb = item['eva_embeddings']
            clip_emb = item['clip_embeddings']
            
            # Ensure consistent dtype
            eva_emb = eva_emb.float()
            clip_emb = clip_emb.float()
            
            eva_tensors.append(eva_emb)
            clip_tensors.append(clip_emb)
            
            if i == 0:  # Log first item details
                logger.debug(f"First item tensor details:")
                logger.debug(f"  EVA: {eva_emb.shape} {eva_emb.dtype} device={eva_emb.device}")
                logger.debug(f"  CLIP: {clip_emb.shape} {clip_emb.dtype} device={clip_emb.device}")
        
        # Stack tensors
        try:
            eva_embeddings = torch.stack(eva_tensors, dim=0)
            clip_embeddings = torch.stack(clip_tensors, dim=0)
        except RuntimeError as e:
            logger.error(f"❌ Tensor stacking failed: {e}")
            logger.error("Tensor shapes in batch:")
            for i, (eva, clip) in enumerate(zip(eva_tensors, clip_tensors)):
                logger.error(f"  Item {i}: EVA {eva.shape}, CLIP {clip.shape}")
            raise
        
        logger.debug(f"Stacked tensors successfully:")
        logger.debug(f"  EVA embeddings: {eva_embeddings.shape}")
        logger.debug(f"  CLIP embeddings: {clip_embeddings.shape}")
        
    except Exception as e:
        logger.error(f"❌ Phase 3 error (tensor stacking): {e}")
        raise
    
    # Phase 4: Extract metadata
    try:
        captions = []
        keys = []
        
        for item in valid_batch:
            caption = item.get('caption', '')
            key = item.get('key', f'unknown_{len(captions)}')
            
            # Ensure caption is string
            if not isinstance(caption, str):
                caption = str(caption)
            
            # Ensure key is string
            if not isinstance(key, str):
                key = str(key)
            
            captions.append(caption)
            keys.append(key)
        
        logger.debug(f"Extracted metadata: {len(captions)} captions, {len(keys)} keys")
        
    except Exception as e:
        logger.error(f"❌ Phase 4 error (metadata extraction): {e}")
        raise
    
    # Phase 5: Shape validation and flow matching setup
    try:
        # Get and validate shapes
        batch_size, seq_len, clip_dim = clip_embeddings.shape
        eva_batch_size, eva_seq_len, eva_dim = eva_embeddings.shape
        
        # Cross-validate shapes
        if batch_size != eva_batch_size:
            raise ValueError(f"Batch size mismatch: CLIP {batch_size} vs EVA {eva_batch_size}")
        
        if seq_len != eva_seq_len:
            raise ValueError(f"Sequence length mismatch: CLIP {seq_len} vs EVA {eva_seq_len}")
        
        if clip_dim != 1024:
            raise ValueError(f"Expected CLIP embedding dim 1024, got {clip_dim}")
        
        if eva_dim != 4096:
            raise ValueError(f"Expected EVA embedding dim 4096, got {eva_dim}")
        
        # Get device and dtype
        device = clip_embeddings.device
        dtype = torch.float32  # Force consistent dtype
        
        # Ensure float32 for stability
        eva_embeddings = eva_embeddings.to(dtype)
        clip_embeddings = clip_embeddings.to(dtype)
        
        logger.debug(f"Shape validation passed:")
        logger.debug(f"  Batch size: {batch_size}")
        logger.debug(f"  Sequence length: {seq_len}")
        logger.debug(f"  Device: {device}")
        logger.debug(f"  Dtype: {dtype}")
        
    except Exception as e:
        logger.error(f"❌ Phase 5 error (shape validation): {e}")
        raise
    
    # Phase 6: Flow matching components
    try:
        # Sample timesteps
        timesteps = sample_u_shaped_timesteps(batch_size, device, alpha=0.5)
        
        # Create noise
        noise = torch.randn_like(clip_embeddings, device=device, dtype=dtype)
        
        # Linear interpolation for rectified flow: x_t = (1-t) * noise + t * clip_clean
        t_expanded = timesteps.view(batch_size, 1, 1)
        noisy_clip = (1 - t_expanded) * noise + t_expanded * clip_embeddings
        
        # Velocity target: v = clip_clean - noise
        velocity_target = clip_embeddings - noise
        
        logger.debug(f"Flow matching components created:")
        logger.debug(f"  Timesteps: {timesteps.shape}")
        logger.debug(f"  Noise: {noise.shape}")
        logger.debug(f"  Noisy CLIP: {noisy_clip.shape}")
        logger.debug(f"  Velocity target: {velocity_target.shape}")
        
    except Exception as e:
        logger.error(f"❌ Phase 6 error (flow matching setup): {e}")
        raise
    
    # Phase 7: Final validation and result construction
    try:
        # Final shape assertions
        assert eva_embeddings.shape == (batch_size, seq_len, 4096), f"EVA shape: {eva_embeddings.shape}"
        assert clip_embeddings.shape == (batch_size, seq_len, 1024), f"CLIP shape: {clip_embeddings.shape}"
        assert noisy_clip.shape == (batch_size, seq_len, 1024), f"Noisy CLIP shape: {noisy_clip.shape}"
        assert velocity_target.shape == (batch_size, seq_len, 1024), f"Velocity target shape: {velocity_target.shape}"
        assert timesteps.shape == (batch_size,), f"Timesteps shape: {timesteps.shape}"
        
        # Check for NaN/Inf in final tensors
        tensors_to_check = {
            'eva_embeddings': eva_embeddings,
            'clip_embeddings': clip_embeddings,
            'noisy_clip': noisy_clip,
            'velocity_target': velocity_target,
            'timesteps': timesteps,
            'noise': noise
        }
        
        for name, tensor in tensors_to_check.items():
            if torch.isnan(tensor).any():
                raise ValueError(f"NaN detected in {name}")
            if torch.isinf(tensor).any():
                raise ValueError(f"Inf detected in {name}")
        
        # Construct result dictionary
        result = {
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
        
        logger.debug("✅ Collate function completed successfully")
        return result
        
    except Exception as e:
        logger.error(f"❌ Phase 7 error (final validation): {e}")
        raise
    
    except Exception as e:
        logger.error(f"❌ Unexpected error in robust collate function: {e}")
        logger.error(f"   Batch info:")
        logger.error(f"     Original batch size: {len(batch) if batch else 0}")
        logger.error(f"     Valid items: {len(valid_batch) if 'valid_batch' in locals() else 'unknown'}")
        
        # Try to provide more debug info
        if 'valid_batch' in locals() and valid_batch:
            try:
                logger.error(f"   First valid item keys: {list(valid_batch[0].keys())}")
                for key, value in valid_batch[0].items():
                    if torch.is_tensor(value):
                        logger.error(f"     {key}: {value.shape} {value.dtype}")
                    else:
                        logger.error(f"     {key}: {type(value)}")
            except:
                logger.error("   Could not inspect first valid item")
        
        raise


def create_robust_coco_dataloader(dataset, batch_size: int = 4, num_workers: int = 0):
    """
    Create a robust dataloader for COCO evaluation with error handling
    """
    from torch.utils.data import DataLoader
    
    def error_handling_collate_fn(batch):
        """Wrapper for robust collate with additional error context"""
        try:
            return robust_coco_collate_fn(batch)
        except Exception as e:
            logger.error(f"❌ Collate function failed for batch")
            logger.error(f"   Error: {e}")
            logger.error(f"   Batch size: {len(batch) if batch else 0}")
            
            # Log dataset info if available
            if hasattr(dataset, 'num_samples'):
                logger.error(f"   Dataset size: {dataset.num_samples}")
            if hasattr(dataset, 'clip_embeddings'):
                logger.error(f"   Dataset CLIP shape: {dataset.clip_embeddings.shape}")
            if hasattr(dataset, 'eva_embeddings'):
                logger.error(f"   Dataset EVA shape: {dataset.eva_embeddings.shape}")
            
            raise
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=error_handling_collate_fn,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=False,  # Disable pin_memory for stability
    )


if __name__ == "__main__":
    # Test the collate function
    print("Testing robust COCO collate function...")
    
    # Create dummy test data
    batch = []
    for i in range(3):
        item = {
            'eva_embeddings': torch.randn(256, 4096),
            'clip_embeddings': torch.randn(256, 1024),
            'caption': f'Test caption {i}',
            'key': f'test_key_{i}',
            'training_mode': 'patch_only',
            'num_tokens': 256,
        }
        batch.append(item)
    
    try:
        result = robust_coco_collate_fn(batch)
        print("✅ Test passed!")
        print(f"Result keys: {list(result.keys())}")
        print(f"EVA embeddings shape: {result['encoder_hidden_states'].shape}")
        print(f"CLIP embeddings shape: {result['clip_embeddings'].shape}")
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()