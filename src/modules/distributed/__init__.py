"""
Distributed Training Module for BLIP3-o
src/modules/distributed/__init__.py

This module provides FSDP (Fully Sharded Data Parallel) support for BLIP3-o training,
enabling efficient multi-GPU training with memory optimization and scaling capabilities.
"""

import logging
import torch

logger = logging.getLogger(__name__)

# Import availability flags
FSDP_UTILS_AVAILABLE = False
DISTRIBUTED_TRAINER_AVAILABLE = False
DISTRIBUTED_EXTRACTION_AVAILABLE = False

# Store imported components
_imported_components = {}

# =============================================================================
# FSDP UTILITIES IMPORTS
# =============================================================================
try:
    from .fsdp_utils import (
        setup_distributed_environment,
        cleanup_distributed,
        wrap_model_with_fsdp,
        save_fsdp_checkpoint,
        load_fsdp_checkpoint,
        estimate_fsdp_memory_usage,
        print_fsdp_memory_estimate,
        sync_across_gpus,
        is_master_rank,
        get_world_size,
        get_rank,
        get_fsdp_sharding_policy,
        create_fsdp_mixed_precision_policy
    )
    FSDP_UTILS_AVAILABLE = True
    _imported_components.update({
        'setup_distributed_environment': setup_distributed_environment,
        'cleanup_distributed': cleanup_distributed,
        'wrap_model_with_fsdp': wrap_model_with_fsdp,
        'save_fsdp_checkpoint': save_fsdp_checkpoint,
        'load_fsdp_checkpoint': load_fsdp_checkpoint,
        'estimate_fsdp_memory_usage': estimate_fsdp_memory_usage,
        'print_fsdp_memory_estimate': print_fsdp_memory_estimate,
        'sync_across_gpus': sync_across_gpus,
        'is_master_rank': is_master_rank,
        'get_world_size': get_world_size,
        'get_rank': get_rank,
        'get_fsdp_sharding_policy': get_fsdp_sharding_policy,
        'create_fsdp_mixed_precision_policy': create_fsdp_mixed_precision_policy,
    })
    logger.info("✅ FSDP utilities loaded successfully")
except ImportError as e:
    logger.warning(f"⚠️ Failed to import FSDP utilities: {e}")

# =============================================================================
# DISTRIBUTED TRAINER IMPORTS
# =============================================================================
try:
    from src.modules.trainers.blip3o_distributed_trainer import (
        BLIP3oDistributedTrainer,
        create_distributed_clip_trainer
    )
    DISTRIBUTED_TRAINER_AVAILABLE = True
    _imported_components.update({
        'BLIP3oDistributedTrainer': BLIP3oDistributedTrainer,
        'create_distributed_clip_trainer': create_distributed_clip_trainer,
    })
    logger.info("✅ Distributed trainer loaded successfully")
except ImportError as e:
    logger.warning(f"⚠️ Failed to import distributed trainer: {e}")

# =============================================================================
# DISTRIBUTED EXTRACTION IMPORTS  
# =============================================================================
try:
    from src.modules.extract_embeddings_distributed import (
        distribute_tar_files,
        consolidate_gpu_outputs,
        create_distributed_manifest
    )
    DISTRIBUTED_EXTRACTION_AVAILABLE = True
    _imported_components.update({
        'distribute_tar_files': distribute_tar_files,
        'consolidate_gpu_outputs': consolidate_gpu_outputs,
        'create_distributed_manifest': create_distributed_manifest,
    })
    logger.info("✅ Distributed extraction utilities loaded successfully")
except ImportError as e:
    logger.warning(f"⚠️ Failed to import distributed extraction utilities: {e}")

# =============================================================================
# DISTRIBUTED DATASET IMPORTS
# =============================================================================
try:
    from src.modules.datasets.blip3o_distributed_dataset import (
        DistributedBLIP3oCLIPReproductionDataset,
        create_distributed_dataloaders,
        create_distributed_clip_reproduction_dataloaders,
        DistributedSamplerWithSeed,
        DistributedDataLoaderMetrics
    )
    DISTRIBUTED_DATASET_AVAILABLE = True
    _imported_components.update({
        'DistributedBLIP3oCLIPReproductionDataset': DistributedBLIP3oCLIPReproductionDataset,
        'create_distributed_dataloaders': create_distributed_dataloaders,
        'create_distributed_clip_reproduction_dataloaders': create_distributed_clip_reproduction_dataloaders,
        'DistributedSamplerWithSeed': DistributedSamplerWithSeed,
        'DistributedDataLoaderMetrics': DistributedDataLoaderMetrics,
    })
    logger.info("✅ Distributed dataset utilities loaded successfully")
except ImportError as e:
    logger.warning(f"⚠️ Failed to import distributed dataset utilities: {e}")

# =============================================================================
# EXPORT ALL COMPONENTS
# =============================================================================

# Main availability flags
__all__ = [
    # Availability flags
    "FSDP_UTILS_AVAILABLE",
    "DISTRIBUTED_TRAINER_AVAILABLE", 
    "DISTRIBUTED_EXTRACTION_AVAILABLE",
    "DISTRIBUTED_DATASET_AVAILABLE",
]

# Add available components to exports
if FSDP_UTILS_AVAILABLE:
    __all__.extend([
        "setup_distributed_environment", "cleanup_distributed",
        "wrap_model_with_fsdp", "save_fsdp_checkpoint", "load_fsdp_checkpoint",
        "estimate_fsdp_memory_usage", "print_fsdp_memory_estimate",
        "sync_across_gpus", "is_master_rank", "get_world_size", "get_rank",
        "get_fsdp_sharding_policy", "create_fsdp_mixed_precision_policy"
    ])

if DISTRIBUTED_TRAINER_AVAILABLE:
    __all__.extend([
        "BLIP3oDistributedTrainer", "create_distributed_clip_trainer"
    ])

if DISTRIBUTED_EXTRACTION_AVAILABLE:
    __all__.extend([
        "distribute_tar_files", "consolidate_gpu_outputs", "create_distributed_manifest"
    ])

if DISTRIBUTED_DATASET_AVAILABLE:
    __all__.extend([
        "DistributedBLIP3oCLIPReproductionDataset", "create_distributed_dataloaders",
        "create_distributed_clip_reproduction_dataloaders", "DistributedSamplerWithSeed",
        "DistributedDataLoaderMetrics"
    ])

# Make imported components available at module level
locals().update(_imported_components)

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def check_distributed_environment():
    """Check if distributed training environment is properly set up"""
    
    status = {
        'fsdp_utils': FSDP_UTILS_AVAILABLE,
        'distributed_trainer': DISTRIBUTED_TRAINER_AVAILABLE,
        'distributed_extraction': DISTRIBUTED_EXTRACTION_AVAILABLE,
        'distributed_dataset': DISTRIBUTED_DATASET_AVAILABLE,
        'cuda_available': torch.cuda.is_available(),
        'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'pytorch_distributed': hasattr(torch, 'distributed'),
        'nccl_available': torch.distributed.is_nccl_available() if hasattr(torch, 'distributed') else False,
    }
    
    # Check if all distributed components are available
    distributed_components_available = all([
        status['fsdp_utils'],
        status['distributed_trainer'], 
        status['distributed_extraction'],
        status['distributed_dataset']
    ])
    
    # Check if system supports distributed training
    system_supports_distributed = all([
        status['cuda_available'],
        status['cuda_device_count'] > 1,
        status['pytorch_distributed'],
        status['nccl_available']
    ])
    
    ready_for_distributed = distributed_components_available and system_supports_distributed
    
    return {
        'component_status': status,
        'distributed_components_available': distributed_components_available,
        'system_supports_distributed': system_supports_distributed,
        'ready_for_distributed_training': ready_for_distributed,
        'missing_components': [name for name, available in {
            'fsdp_utils': status['fsdp_utils'],
            'distributed_trainer': status['distributed_trainer'],
            'distributed_extraction': status['distributed_extraction'],
            'distributed_dataset': status['distributed_dataset']
        }.items() if not available],
        'system_requirements': {
            'cuda_devices': status['cuda_device_count'],
            'nccl_available': status['nccl_available'],
            'pytorch_distributed': status['pytorch_distributed']
        }
    }

def print_distributed_status():
    """Print detailed distributed training status"""
    print("🚀 BLIP3-o Distributed Training Environment Status")
    print("=" * 60)
    
    status = check_distributed_environment()
    
    print("📦 Component Status:")
    component_mapping = {
        'fsdp_utils': 'FSDP Utilities',
        'distributed_trainer': 'Distributed Trainer',
        'distributed_extraction': 'Multi-GPU Extraction',
        'distributed_dataset': 'Distributed Dataset'
    }
    
    for component, description in component_mapping.items():
        available = status['component_status'].get(component, False)
        status_icon = "✅" if available else "❌"
        print(f"  {status_icon} {description}")
    
    print(f"\n🖥️ System Requirements:")
    sys_status = status['system_requirements']
    cuda_available = status['component_status']['cuda_available']
    cuda_count = sys_status['cuda_devices']
    
    print(f"  {'✅' if cuda_available else '❌'} CUDA Available: {cuda_available}")
    print(f"  {'✅' if cuda_count > 1 else '❌'} CUDA Devices: {cuda_count}")
    print(f"  {'✅' if sys_status['pytorch_distributed'] else '❌'} PyTorch Distributed: {sys_status['pytorch_distributed']}")
    print(f"  {'✅' if sys_status['nccl_available'] else '❌'} NCCL Backend: {sys_status['nccl_available']}")
    
    print(f"\n📊 Overall Status:")
    if status['ready_for_distributed_training']:
        print("  🎉 Ready for distributed training!")
        print("  🚀 All components and system requirements available")
        print(f"  ⚡ Can use up to {cuda_count} GPUs with FSDP")
    else:
        print("  ⚠️ Not ready for distributed training")
        
        if not status['distributed_components_available']:
            print("  📦 Missing components:")
            for component in status['missing_components']:
                print(f"     • {component}")
        
        if not status['system_supports_distributed']:
            print("  🖥️ System limitations:")
            if not cuda_available:
                print("     • CUDA not available")
            if cuda_count <= 1:
                print(f"     • Need >1 GPU, found {cuda_count}")
            if not sys_status['nccl_available']:
                print("     • NCCL backend not available")
    
    print(f"\n💡 Usage Examples:")
    if status['ready_for_distributed_training']:
        print("  # Multi-GPU embedding extraction:")
        print("  sbatch job_scripts/extract_embeddings_distributed.job")
        print("")
        print("  # FSDP distributed training:")
        print("  sbatch job_scripts/train_blip3o_fsdp.job")
        print("")
        print("  # Manual distributed training:")
        print("  torchrun --nproc_per_node=4 train_dit_distributed.py \\")
        print("    --chunked_embeddings_dir /path/to/embeddings \\")
        print("    --output_dir ./checkpoints --distributed")
    else:
        print("  # Single-GPU training (fallback):")
        print("  python train_dit.py --chunked_embeddings_dir /path/to/embeddings")
        print("")
        print("  # Single-GPU extraction:")
        print("  python src/modules/extract_embeddings_g.py")
    
    print("=" * 60)

def get_recommended_fsdp_config(model_parameters: int, available_gpus: int) -> dict:
    """Get recommended FSDP configuration based on model size and available GPUs"""
    
    # Memory estimates per GPU (in GB)
    gpu_memory_gb = 80  # H100 default
    
    # Estimate memory usage
    if FSDP_UTILS_AVAILABLE:
        memory_estimates = estimate_fsdp_memory_usage(
            model_parameters, available_gpus, use_mixed_precision=True
        )
        estimated_memory_per_gpu = memory_estimates['total_memory_gb']
    else:
        # Rough estimate if FSDP utils not available
        estimated_memory_per_gpu = (model_parameters * 16) / (available_gpus * 1e9)  # Rough estimate
    
    # Determine optimal configuration
    config = {
        'world_size': min(available_gpus, 8),  # Max 8 GPUs for single node
        'sharding_strategy': 'FULL_SHARD',
        'mixed_precision': True,
        'cpu_offload': False,
        'batch_size_per_gpu': 32,
    }
    
    # Adjust based on memory requirements
    if estimated_memory_per_gpu > gpu_memory_gb * 0.9:
        config['cpu_offload'] = True
        config['batch_size_per_gpu'] = 16
        config['sharding_strategy'] = 'FULL_SHARD'
    elif estimated_memory_per_gpu > gpu_memory_gb * 0.7:
        config['batch_size_per_gpu'] = 24
    elif estimated_memory_per_gpu < gpu_memory_gb * 0.3:
        config['batch_size_per_gpu'] = 48
        if available_gpus <= 2:
            config['sharding_strategy'] = 'SHARD_GRAD_OP'  # Less aggressive sharding
    
    # Model size specific adjustments
    if model_parameters > 1e9:  # >1B parameters
        config['cpu_offload'] = True
        config['batch_size_per_gpu'] = min(config['batch_size_per_gpu'], 16)
    elif model_parameters < 100e6:  # <100M parameters
        config['sharding_strategy'] = 'SHARD_GRAD_OP'
        config['batch_size_per_gpu'] = min(config['batch_size_per_gpu'] * 2, 64)
    
    return config

# =============================================================================
# INITIALIZATION
# =============================================================================

# Run environment check on import
_env_status = check_distributed_environment()

# Log initialization status
if _env_status['ready_for_distributed_training']:
    logger.info("🎉 BLIP3-o distributed training environment fully initialized!")
    logger.info(f"⚡ Ready for FSDP training with {_env_status['system_requirements']['cuda_devices']} GPUs")
else:
    missing_info = []
    if not _env_status['distributed_components_available']:
        missing_info.append(f"missing components: {_env_status['missing_components']}")
    if not _env_status['system_supports_distributed']:
        missing_info.append("system requirements not met")
    
    logger.warning(f"⚠️ Partial distributed environment initialization")
    if missing_info:
        logger.warning(f"   Issues: {', '.join(missing_info)}")

# Export environment status for external access
DISTRIBUTED_ENVIRONMENT_STATUS = _env_status

# Cleanup
del _env_status, _imported_components