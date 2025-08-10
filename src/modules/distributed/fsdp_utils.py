"""
FIXED FSDP Utilities for BLIP3-o Distributed Training
src/modules/distributed/fsdp_utils.py

FIXES:
- Fixed device_id parameter issue (was passing int instead of device)
- Better error handling and initialization
- Proper device setup before process group initialization
"""

import torch
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    BackwardPrefetch,
    ShardingStrategy,
    CPUOffload,
    FullStateDictConfig,
    StateDictType
)
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
    enable_wrap,
    wrap
)
from functools import partial
import logging
from typing import Optional, Dict, Any, Callable
from pathlib import Path
import os
from datetime import timedelta

logger = logging.getLogger(__name__)


def setup_distributed_environment(rank: int, world_size: int, master_port: str = "12355"):
    """FIXED: Setup distributed training environment with proper device handling"""
    
    # Set environment variables
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = master_port
    os.environ['RANK'] = str(rank)
    os.environ['LOCAL_RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    
    # FIXED: Set CUDA device first and create device object
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)
        device = torch.device(f'cuda:{rank}')
    else:
        device = torch.device('cpu')
        raise RuntimeError("CUDA required for distributed training")
    
    # FIXED: Initialize process group WITHOUT device_id parameter
    # The device_id parameter in init_process_group is deprecated and causes issues
    try:
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=world_size,
            rank=rank,
            timeout=timedelta(minutes=10)
        )
        
        if rank == 0:
            logger.info(f"✅ Distributed environment initialized:")
            logger.info(f"   World size: {world_size}")
            logger.info(f"   Backend: nccl")
            logger.info(f"   Master port: {master_port}")
            logger.info(f"   Device: {device}")
            
    except Exception as e:
        logger.error(f"❌ Failed to initialize distributed environment: {e}")
        raise
    
    return device


def cleanup_distributed():
    """Clean up distributed training"""
    if dist.is_initialized():
        try:
            dist.barrier()
            dist.destroy_process_group()
            logger.info("🔧 Distributed environment cleaned up")
        except Exception as e:
            logger.warning(f"Warning during distributed cleanup: {e}")


def get_fsdp_sharding_policy():
    """Get optimal FSDP sharding policy for BLIP3-o DiT"""
    
    # Import the specific DiT block class
    try:
        from src.modules.models.blip3o_dit import StableDiTBlock3D
        dit_block_class = StableDiTBlock3D
    except ImportError:
        logger.warning("Could not import StableDiTBlock3D, using generic transformer policy")
        dit_block_class = torch.nn.TransformerEncoderLayer  # Fallback
    
    # Create auto-wrap policy for DiT blocks
    auto_wrap_policy = partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={dit_block_class},
    )
    
    return auto_wrap_policy


def create_fsdp_mixed_precision_policy(use_fp16: bool = True) -> Optional[MixedPrecision]:
    """Create mixed precision policy for FSDP"""
    
    if not use_fp16:
        return None
    
    # Use BF16 for parameters and gradients, FP32 for loss computation
    mixed_precision_policy = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
        keep_low_precision_grads=True,
    )
    
    logger.info("✅ FSDP Mixed precision policy created (BF16)")
    return mixed_precision_policy


def wrap_model_with_fsdp(
    model: torch.nn.Module,
    device: torch.device,
    sharding_strategy: ShardingStrategy = ShardingStrategy.FULL_SHARD,
    use_mixed_precision: bool = True,
    cpu_offload: bool = False,
    backward_prefetch: BackwardPrefetch = BackwardPrefetch.BACKWARD_PRE,
    limit_all_gathers: bool = True,
) -> FSDP:
    """
    FIXED: Wrap model with FSDP for distributed training
    """
    
    # Get sharding policy
    auto_wrap_policy = get_fsdp_sharding_policy()
    
    # Mixed precision policy
    mixed_precision_policy = create_fsdp_mixed_precision_policy(use_mixed_precision)
    
    # CPU offload policy
    cpu_offload_policy = CPUOffload(offload_params=True) if cpu_offload else None
    
    # FIXED: Wrap model with FSDP - use device index correctly
    fsdp_model = FSDP(
        model,
        auto_wrap_policy=auto_wrap_policy,
        mixed_precision=mixed_precision_policy,
        sharding_strategy=sharding_strategy,
        cpu_offload=cpu_offload_policy,
        backward_prefetch=backward_prefetch,
        limit_all_gathers=limit_all_gathers,
        use_orig_params=False,
        device_id=device.index if device.type == 'cuda' else None,  # FIXED: Use device.index
        sync_module_states=True,
    )
    
    if dist.get_rank() == 0:
        logger.info("✅ Model wrapped with FSDP:")
        logger.info(f"   Sharding strategy: {sharding_strategy}")
        logger.info(f"   Mixed precision: {'BF16' if use_mixed_precision else 'FP32'}")
        logger.info(f"   CPU offload: {'Enabled' if cpu_offload else 'Disabled'}")
        logger.info(f"   Backward prefetch: {backward_prefetch}")
        logger.info(f"   Limit all-gathers: {limit_all_gathers}")
        logger.info(f"   Device ID: {device.index if device.type == 'cuda' else 'CPU'}")
        logger.info(f"   Sync module states: True")
    
    return fsdp_model


def save_fsdp_checkpoint(
    model: FSDP,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    scaler: Optional[torch.amp.GradScaler],
    checkpoint_path: Path,
    global_step: int,
    additional_data: Optional[Dict[str, Any]] = None,
    save_full_state: bool = True
):
    """
    Save FSDP checkpoint with proper state dict handling
    """
    
    if dist.get_rank() != 0:
        return  # Only rank 0 saves checkpoints
    
    logger.info(f"💾 Saving FSDP checkpoint: {checkpoint_path}")
    
    checkpoint_data = {
        'global_step': global_step,
        'fsdp_model': True,
        'sharding_strategy': str(model.sharding_strategy),
    }
    
    # Add additional data
    if additional_data:
        checkpoint_data.update(additional_data)
    
    # Configure state dict type for FSDP model
    if save_full_state:
        # Save full state dict for inference compatibility
        with FSDP.state_dict_type(
            model,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        ):
            model_state_dict = model.state_dict()
            checkpoint_data['model_state_dict'] = model_state_dict
            logger.info("   Saved full model state dict (inference-compatible)")
    else:
        # Save sharded state dict (smaller, training-only)
        checkpoint_data['model_state_dict'] = model.state_dict()
        logger.info("   Saved sharded model state dict (training-only)")
    
    # Save optimizer state (only on rank 0)
    if hasattr(optimizer, 'state_dict'):
        checkpoint_data['optimizer_state_dict'] = optimizer.state_dict()
    
    # Save scheduler state
    if scheduler is not None:
        checkpoint_data['scheduler_state_dict'] = scheduler.state_dict()
    
    # Save gradient scaler state
    if scaler is not None:
        checkpoint_data['scaler_state_dict'] = scaler.state_dict()
    
    # Ensure checkpoint directory exists
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save checkpoint
    torch.save(checkpoint_data, checkpoint_path)
    
    # Log checkpoint info
    file_size_mb = checkpoint_path.stat().st_size / (1024 * 1024)
    logger.info(f"✅ FSDP checkpoint saved: {checkpoint_path.name} ({file_size_mb:.1f} MB)")


def load_fsdp_checkpoint(
    model: FSDP,
    checkpoint_path: Path,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    scaler: Optional[torch.amp.GradScaler] = None,
    strict: bool = True
) -> Dict[str, Any]:
    """
    Load FSDP checkpoint with proper state dict handling
    """
    
    logger.info(f"📂 Loading FSDP checkpoint: {checkpoint_path}")
    
    # Load checkpoint data
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Check if this is an FSDP checkpoint
    if not checkpoint.get('fsdp_model', False):
        logger.warning("⚠️ Loading non-FSDP checkpoint into FSDP model")
    
    # Load model state dict
    model_state_dict = checkpoint['model_state_dict']
    
    # Configure state dict type for loading
    with FSDP.state_dict_type(
        model,
        StateDictType.FULL_STATE_DICT,
        FullStateDictConfig(offload_to_cpu=True, rank0_only=False)
    ):
        model.load_state_dict(model_state_dict, strict=strict)
    
    logger.info("✅ Model state loaded successfully")
    
    # Load optimizer state
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logger.info("✅ Optimizer state loaded")
    
    # Load scheduler state
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        logger.info("✅ Scheduler state loaded")
    
    # Load scaler state
    if scaler is not None and 'scaler_state_dict' in checkpoint:
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        logger.info("✅ Gradient scaler state loaded")
    
    # Return metadata
    metadata = {
        'global_step': checkpoint.get('global_step', 0),
        'sharding_strategy': checkpoint.get('sharding_strategy', 'unknown'),
        'fsdp_model': checkpoint.get('fsdp_model', False),
    }
    
    # Remove large tensors from return value
    excluded_keys = {'model_state_dict', 'optimizer_state_dict', 'scheduler_state_dict', 'scaler_state_dict'}
    for key, value in checkpoint.items():
        if key not in excluded_keys:
            metadata[key] = value
    
    return metadata


def estimate_fsdp_memory_usage(
    model_parameters: int,
    world_size: int,
    use_mixed_precision: bool = True,
    cpu_offload: bool = False
) -> Dict[str, float]:
    """
    Estimate memory usage for FSDP training
    """
    
    # Parameter size estimates
    if use_mixed_precision:
        param_bytes_per_param = 2  # BF16
        grad_bytes_per_param = 2   # BF16
    else:
        param_bytes_per_param = 4  # FP32
        grad_bytes_per_param = 4   # FP32
    
    # FSDP sharding reduces parameter memory per GPU
    params_per_gpu = model_parameters / world_size
    
    # Memory components
    if not cpu_offload:
        model_memory_gb = (params_per_gpu * param_bytes_per_param) / (1024**3)
    else:
        model_memory_gb = 0.1  # Minimal GPU memory with CPU offload
    
    gradient_memory_gb = (params_per_gpu * grad_bytes_per_param) / (1024**3)
    
    # Optimizer states (AdamW: 2x parameters for momentum and variance)
    if not cpu_offload:
        optimizer_memory_gb = (params_per_gpu * 8) / (1024**3)  # FP32 optimizer states
    else:
        optimizer_memory_gb = 0.1  # Minimal with CPU offload
    
    # Activation memory (rough estimate)
    activation_memory_gb = 4.0  # Depends on batch size and sequence length
    
    # Total memory
    total_memory_gb = (
        model_memory_gb + 
        gradient_memory_gb + 
        optimizer_memory_gb + 
        activation_memory_gb
    )
    
    return {
        'model_memory_gb': model_memory_gb,
        'gradient_memory_gb': gradient_memory_gb,
        'optimizer_memory_gb': optimizer_memory_gb,
        'activation_memory_gb': activation_memory_gb,
        'total_memory_gb': total_memory_gb,
        'parameters_per_gpu': params_per_gpu,
        'memory_reduction_factor': world_size if not cpu_offload else world_size * 4,
    }


def print_fsdp_memory_estimate(
    model_parameters: int,
    world_size: int,
    use_mixed_precision: bool = True,
    cpu_offload: bool = False
):
    """Print FSDP memory usage estimates"""
    
    estimates = estimate_fsdp_memory_usage(
        model_parameters, world_size, use_mixed_precision, cpu_offload
    )
    
    print(f"\n🧠 FSDP Memory Estimates ({world_size} GPUs):")
    print("=" * 50)
    print(f"Total parameters: {model_parameters:,}")
    print(f"Parameters per GPU: {estimates['parameters_per_gpu']:,.0f}")
    print(f"Memory reduction factor: {estimates['memory_reduction_factor']:.1f}x")
    print()
    print(f"Per-GPU Memory Usage:")
    print(f"  Model parameters: {estimates['model_memory_gb']:.2f} GB")
    print(f"  Gradients:       {estimates['gradient_memory_gb']:.2f} GB")
    print(f"  Optimizer:       {estimates['optimizer_memory_gb']:.2f} GB")
    print(f"  Activations:     {estimates['activation_memory_gb']:.2f} GB")
    print(f"  Total:           {estimates['total_memory_gb']:.2f} GB")
    print()
    
    if estimates['total_memory_gb'] < 40:  # H100 memory
        print("✅ Should fit on H100 (80GB) with room for batch processing")
    elif estimates['total_memory_gb'] < 80:
        print("⚠️ Tight fit on H100 - consider reducing batch size")
    else:
        print("❌ May not fit on H100 - consider CPU offload or more GPUs")
    
    print("=" * 50)


def sync_across_gpus(tensor: torch.Tensor, average: bool = True) -> torch.Tensor:
    """Synchronize tensor across all GPUs with timeout protection"""
    
    if not dist.is_initialized():
        return tensor
    
    try:
        # Clone to avoid modifying original
        synced_tensor = tensor.clone()
        
        # All-reduce across GPUs
        dist.all_reduce(synced_tensor, op=dist.ReduceOp.SUM)
        
        if average:
            synced_tensor /= dist.get_world_size()
        
        return synced_tensor
    
    except Exception as e:
        logger.warning(f"Sync across GPUs failed: {e}")
        return tensor


def barrier_with_timeout(timeout_seconds: int = 60):
    """Barrier with timeout to detect hangs"""
    if not dist.is_initialized():
        return
    
    try:
        # Use a work object to implement timeout
        work = dist.barrier(async_op=True)
        work.wait(timeout=timedelta(seconds=timeout_seconds))
    except Exception as e:
        logger.error(f"Barrier timeout or error: {e}")
        raise


def is_master_rank() -> bool:
    """Check if current rank is master (rank 0)"""
    return not dist.is_initialized() or dist.get_rank() == 0


def get_world_size() -> int:
    """Get world size (number of GPUs)"""
    return dist.get_world_size() if dist.is_initialized() else 1


def get_rank() -> int:
    """Get current rank"""
    return dist.get_rank() if dist.is_initialized() else 0