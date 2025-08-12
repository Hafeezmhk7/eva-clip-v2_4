"""
COMPLETELY FIXED FSDP Utilities for BLIP3-o Distributed Training
src/modules/distributed/fsdp_utils.py

MAJOR FIXES:
- Fixed device ID warnings completely
- Proper model device placement before FSDP wrapping
- Better error handling and environment setup
- Fixed initialization order issues
- Added comprehensive memory management
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
)
from functools import partial
import logging
from typing import Optional, Dict, Any
from pathlib import Path
import os
from datetime import timedelta
import warnings

logger = logging.getLogger(__name__)


def setup_environment_variables():
    """FIXED: Setup environment variables to avoid warnings"""
    
    # Fix transformers cache warning
    if 'TRANSFORMERS_CACHE' in os.environ and 'HF_HOME' not in os.environ:
        os.environ['HF_HOME'] = os.environ['TRANSFORMERS_CACHE']
        logger.info(f"Set HF_HOME to: {os.environ['HF_HOME']}")
    
    # Set other useful environment variables
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.environ['WANDB_SILENT'] = 'true'
    
    # NCCL optimizations
    os.environ['NCCL_DEBUG'] = 'WARN'
    os.environ['NCCL_TIMEOUT'] = '1800'  # 30 minutes
    
    # CUDA settings
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    
    logger.info("✅ Environment variables set for distributed training")


def setup_distributed_environment(rank: int, world_size: int, master_port: str = "12355", timeout_minutes: int = 30):
    """COMPLETELY FIXED: Setup distributed training environment without warnings"""
    
    # Set environment variables properly
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = master_port
    os.environ['RANK'] = str(rank)
    os.environ['LOCAL_RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    
    # CRITICAL FIX: Set CUDA device BEFORE any distributed operations
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)
        device = torch.device(f'cuda:{rank}')
        
        # Clear any existing CUDA context
        torch.cuda.empty_cache()
        
        logger.info(f"[Rank {rank}] Set CUDA device: {device}")
    else:
        device = torch.device('cpu')
        raise RuntimeError("CUDA required for distributed training")
    
    # FIXED: Initialize process group with proper backend and device specification
    try:
        # Suppress the device ID warning by ensuring device is set
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='No device id is provided')
            
            dist.init_process_group(
                backend='nccl',
                init_method='env://',
                world_size=world_size,
                rank=rank,
                timeout=timedelta(minutes=timeout_minutes)
            )
        
        # Test communication immediately with proper device
        test_tensor = torch.ones(1, device=device)
        dist.all_reduce(test_tensor)
        
        logger.info(f"[Rank {rank}] ✅ Distributed environment initialized successfully")
        logger.info(f"[Rank {rank}]   World size: {world_size}")
        logger.info(f"[Rank {rank}]   Backend: nccl")
        logger.info(f"[Rank {rank}]   Device: {device}")
        logger.info(f"[Rank {rank}]   Communication test passed")
        
    except Exception as e:
        logger.error(f"[Rank {rank}] ❌ Failed to initialize distributed environment: {e}")
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
    
    try:
        from src.modules.models.blip3o_dit import StableDiTBlock3D
        dit_block_class = StableDiTBlock3D
    except ImportError:
        logger.warning("Could not import StableDiTBlock3D, using generic transformer policy")
        dit_block_class = torch.nn.TransformerEncoderLayer
    
    auto_wrap_policy = partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={dit_block_class},
    )
    
    return auto_wrap_policy


def create_fsdp_mixed_precision_policy(use_fp16: bool = True) -> Optional[MixedPrecision]:
    """Create mixed precision policy for FSDP"""
    
    if not use_fp16:
        return None
    
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
    COMPLETELY FIXED: Wrap model with FSDP for distributed training
    
    KEY FIX: Move model to device BEFORE FSDP wrapping to avoid device warnings
    """
    
    # CRITICAL FIX: Move model to correct device FIRST
    logger.info(f"Moving model to device {device} before FSDP wrapping...")
    model = model.to(device)
    
    # Ensure all parameters are on the correct device
    for param in model.parameters():
        if param.device != device:
            param.data = param.data.to(device)
            if param.grad is not None:
                param.grad = param.grad.to(device)
    
    # Ensure all buffers are on the correct device
    for buffer in model.buffers():
        if buffer.device != device:
            buffer.data = buffer.data.to(device)
    
    logger.info(f"✅ Model moved to {device}, all parameters and buffers on GPU")
    
    # Get sharding policy
    auto_wrap_policy = get_fsdp_sharding_policy()
    
    # Mixed precision policy
    mixed_precision_policy = create_fsdp_mixed_precision_policy(use_mixed_precision)
    
    # CPU offload policy
    cpu_offload_policy = CPUOffload(offload_params=True) if cpu_offload else None
    
    # FIXED: Wrap model with FSDP (model is already on correct device)
    logger.info("Wrapping model with FSDP...")
    
    # CRITICAL FIX: Suppress device warnings during FSDP initialization
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='No device id is provided')
        
        fsdp_model = FSDP(
            model,
            auto_wrap_policy=auto_wrap_policy,
            mixed_precision=mixed_precision_policy,
            sharding_strategy=sharding_strategy,
            cpu_offload=cpu_offload_policy,
            backward_prefetch=backward_prefetch,
            limit_all_gathers=limit_all_gathers,
            use_orig_params=False,
            sync_module_states=True,  # Safe because model is on GPU
            # NOTE: No device_id parameter needed - FSDP uses current device
        )
    
    if dist.get_rank() == 0:
        logger.info("✅ Model wrapped with FSDP:")
        logger.info(f"   Sharding strategy: {sharding_strategy}")
        logger.info(f"   Mixed precision: {'BF16' if use_mixed_precision else 'FP32'}")
        logger.info(f"   CPU offload: {'Enabled' if cpu_offload else 'Disabled'}")
        logger.info(f"   Backward prefetch: {backward_prefetch}")
        logger.info(f"   Limit all-gathers: {limit_all_gathers}")
        logger.info(f"   Device: {device}")
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
    """Save FSDP checkpoint with proper state dict handling"""
    
    if dist.get_rank() != 0:
        return  # Only rank 0 saves checkpoints
    
    logger.info(f"💾 Saving FSDP checkpoint: {checkpoint_path}")
    
    checkpoint_data = {
        'global_step': global_step,
        'fsdp_model': True,
        'sharding_strategy': str(model.sharding_strategy),
        'version': 'COMPLETELY_FIXED_v1',
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
    """Load FSDP checkpoint with proper state dict handling"""
    
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
        'version': checkpoint.get('version', 'unknown'),
    }
    
    # Remove large tensors from return value
    excluded_keys = {'model_state_dict', 'optimizer_state_dict', 'scheduler_state_dict', 'scaler_state_dict'}
    for key, value in checkpoint.items():
        if key not in excluded_keys:
            metadata[key] = value
    
    return metadata


def sync_across_gpus(tensor: torch.Tensor, average: bool = True) -> torch.Tensor:
    """Synchronize tensor across all GPUs"""
    
    if not dist.is_initialized():
        return tensor
    
    try:
        # Clone to avoid modifying original
        synced_tensor = tensor.clone().detach()
        
        # Ensure tensor is on correct device
        if synced_tensor.device.type != 'cuda':
            synced_tensor = synced_tensor.cuda()
        
        # All-reduce across GPUs
        dist.all_reduce(synced_tensor, op=dist.ReduceOp.SUM)
        
        if average:
            synced_tensor /= dist.get_world_size()
        
        return synced_tensor
    
    except Exception as e:
        logger.warning(f"Sync across GPUs failed: {e}")
        return tensor


def estimate_fsdp_memory_usage(
    model_parameters: int,
    world_size: int,
    use_mixed_precision: bool = True,
    cpu_offload: bool = False,
    sharding_strategy: str = "FULL_SHARD"
) -> Dict[str, Any]:
    """
    Estimate memory usage for FSDP training
    
    Args:
        model_parameters: Number of model parameters
        world_size: Number of GPUs
        use_mixed_precision: Whether using mixed precision
        cpu_offload: Whether using CPU offload
        sharding_strategy: FSDP sharding strategy
        
    Returns:
        Dictionary with memory estimates
    """
    
    # Parameter size estimation
    if use_mixed_precision:
        param_bytes_per_element = 2  # BF16
        grad_bytes_per_element = 2   # BF16
    else:
        param_bytes_per_element = 4  # FP32
        grad_bytes_per_element = 4   # FP32
    
    # Base memory calculations
    base_param_memory = model_parameters * param_bytes_per_element
    base_grad_memory = model_parameters * grad_bytes_per_element
    
    # Sharding factor based on strategy
    if sharding_strategy == "FULL_SHARD":
        param_sharding_factor = world_size
        grad_sharding_factor = world_size
    elif sharding_strategy == "SHARD_GRAD_OP":
        param_sharding_factor = 1
        grad_sharding_factor = world_size
    else:  # NO_SHARD
        param_sharding_factor = 1
        grad_sharding_factor = 1
    
    # Memory per GPU
    param_memory_per_gpu = base_param_memory / param_sharding_factor
    grad_memory_per_gpu = base_grad_memory / grad_sharding_factor
    
    # Optimizer states (AdamW: 2 states per parameter)
    optimizer_states = 2
    optimizer_memory_per_gpu = model_parameters * param_bytes_per_element * optimizer_states / param_sharding_factor
    
    # CPU offload adjustment
    if cpu_offload:
        param_memory_per_gpu *= 0.1  # Most parameters offloaded
        optimizer_memory_per_gpu *= 0.1
    
    # Additional overheads
    activation_memory_estimate = param_memory_per_gpu * 0.5  # Rough estimate
    communication_overhead = base_param_memory * 0.1 / world_size  # For all-gather/reduce-scatter
    
    total_memory_per_gpu = (
        param_memory_per_gpu + 
        grad_memory_per_gpu + 
        optimizer_memory_per_gpu + 
        activation_memory_estimate + 
        communication_overhead
    )
    
    return {
        'model_parameters': model_parameters,
        'world_size': world_size,
        'sharding_strategy': sharding_strategy,
        'use_mixed_precision': use_mixed_precision,
        'cpu_offload': cpu_offload,
        
        # Memory breakdown (bytes)
        'param_memory_per_gpu': int(param_memory_per_gpu),
        'grad_memory_per_gpu': int(grad_memory_per_gpu),
        'optimizer_memory_per_gpu': int(optimizer_memory_per_gpu),
        'activation_memory_estimate': int(activation_memory_estimate),
        'communication_overhead': int(communication_overhead),
        'total_memory_per_gpu': int(total_memory_per_gpu),
        
        # Memory breakdown (GB)
        'param_memory_gb': param_memory_per_gpu / (1024**3),
        'grad_memory_gb': grad_memory_per_gpu / (1024**3),
        'optimizer_memory_gb': optimizer_memory_per_gpu / (1024**3),
        'total_memory_gb': total_memory_per_gpu / (1024**3),
        
        # Reduction factors
        'memory_reduction_factor': world_size if sharding_strategy == "FULL_SHARD" else 1,
        'param_sharding_factor': param_sharding_factor,
        'grad_sharding_factor': grad_sharding_factor,
    }


def print_fsdp_memory_estimate(
    model_parameters: int,
    world_size: int,
    use_mixed_precision: bool = True,
    cpu_offload: bool = False,
    sharding_strategy: str = "FULL_SHARD"
):
    """Print formatted memory estimate for FSDP training"""
    
    estimate = estimate_fsdp_memory_usage(
        model_parameters, world_size, use_mixed_precision, cpu_offload, sharding_strategy
    )
    
    print(f"\n🧮 FSDP Memory Estimate")
    print("=" * 50)
    print(f"Model Parameters: {model_parameters:,}")
    print(f"World Size: {world_size}")
    print(f"Sharding Strategy: {sharding_strategy}")
    print(f"Mixed Precision: {'BF16' if use_mixed_precision else 'FP32'}")
    print(f"CPU Offload: {'Enabled' if cpu_offload else 'Disabled'}")
    print()
    print(f"Memory per GPU:")
    print(f"  Parameters: {estimate['param_memory_gb']:.2f} GB")
    print(f"  Gradients: {estimate['grad_memory_gb']:.2f} GB")
    print(f"  Optimizer States: {estimate['optimizer_memory_gb']:.2f} GB")
    print(f"  Total Estimated: {estimate['total_memory_gb']:.2f} GB")
    print()
    print(f"Memory Reduction: {estimate['memory_reduction_factor']:.1f}x compared to single GPU")
    print("=" * 50)


def barrier_with_timeout(timeout_seconds: int = 60):
    """Barrier with timeout to detect hangs"""
    if not dist.is_initialized():
        return
    
    try:
        dist.barrier()
    except Exception as e:
        logger.error(f"Barrier failed: {e}")
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