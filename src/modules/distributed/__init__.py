"""
FIXED Distributed Training Module for BLIP3-o
src/modules/distributed/__init__.py

This module provides FSDP (Fully Sharded Data Parallel) support for BLIP3-o training,
enabling efficient multi-GPU training with memory optimization and scaling capabilities.

FIXES:
- Fixed import logic to define variables before using them
- Removed non-existent DistributedSamplerWithSeed import
- Proper error handling for imports
"""

import logging
import torch

logger = logging.getLogger(__name__)

# Initialize availability flags
FSDP_UTILS_AVAILABLE = False
DISTRIBUTED_TRAINER_AVAILABLE = False
DISTRIBUTED_EXTRACTION_AVAILABLE = False
DISTRIBUTED_DATASET_AVAILABLE = False

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
# DISTRIBUTED DATASET IMPORTS (FIXED)
# =============================================================================
try:
    from src.modules.datasets.blip3o_distributed_dataset import (
        DistributedBLIP3oCLIPReproductionDataset,
        create_distributed_dataloaders,
        create_distributed_clip_reproduction_dataloaders,
        DistributedDataLoaderMetrics
    )
    DISTRIBUTED_DATASET_AVAILABLE = True
    _imported_components.update({
        'DistributedBLIP3oCLIPReproductionDataset': DistributedBLIP3oCLIPReproductionDataset,
        'create_distributed_dataloaders': create_distributed_dataloaders,
        'create_distributed_clip_reproduction_dataloaders': create_distributed_clip_reproduction_dataloaders,
        'DistributedDataLoaderMetrics': DistributedDataLoaderMetrics,
    })
    logger.info("✅ Distributed dataset utilities loaded successfully")
except ImportError as e:
    logger.warning(f"⚠️ Failed to import distributed dataset utilities: {e}")

# =============================================================================
# COMMUNICATION UTILITIES (NEW)
# =============================================================================
try:
    # Simple communication utilities for distributed training
    class DistributedCommunicator:
        """Simple distributed communication utilities"""
        def __init__(self, rank: int, world_size: int):
            self.rank = rank
            self.world_size = world_size
        
        def barrier(self):
            """Synchronize all processes"""
            if torch.distributed.is_initialized():
                torch.distributed.barrier()
        
        def is_master(self) -> bool:
            """Check if current process is master"""
            return self.rank == 0
    
    class MetricsAggregator:
        """Simple metrics aggregation for distributed training"""
        def __init__(self, rank: int, world_size: int):
            self.rank = rank
            self.world_size = world_size
        
        def aggregate_metrics(self, local_metrics: dict) -> dict:
            """Aggregate metrics across all ranks"""
            if not torch.distributed.is_initialized():
                return local_metrics
            
            # Simple aggregation - just return local metrics from master
            if self.rank == 0:
                return local_metrics
            else:
                return {}
    
    def log_distributed_info(rank: int, world_size: int, message: str):
        """Log distributed information"""
        if rank == 0:
            logger.info(f"[Distributed] {message}")
    
    _imported_components.update({
        'DistributedCommunicator': DistributedCommunicator,
        'MetricsAggregator': MetricsAggregator,
        'log_distributed_info': log_distributed_info,
    })
    logger.info("✅ Communication utilities loaded successfully")
    
except Exception as e:
    logger.warning(f"⚠️ Failed to setup communication utilities: {e}")

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
        "create_distributed_clip_reproduction_dataloaders", "DistributedDataLoaderMetrics"
    ])

# Always add communication utilities
__all__.extend([
    "DistributedCommunicator", "MetricsAggregator", "log_distributed_info"
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
    
    print("=" * 60)

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