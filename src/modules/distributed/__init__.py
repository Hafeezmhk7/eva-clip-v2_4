"""
FIXED Distributed Training Module for BLIP3-o
src/modules/distributed/__init__.py

MAJOR FIXES:
- Simplified imports to avoid circular dependencies
- Better error handling for missing components
- Fixed import/export name consistency
- Added proper aliases for backward compatibility
"""

import logging
import torch

logger = logging.getLogger(__name__)

# Initialize availability flags
FSDP_UTILS_AVAILABLE = False
DISTRIBUTED_TRAINER_AVAILABLE = False
DISTRIBUTED_DATASET_AVAILABLE = False
COMMUNICATION_UTILS_AVAILABLE = False

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
        sync_across_gpus,
        is_master_rank,
        get_world_size,
        get_rank,
        setup_environment_variables,
        create_fsdp_mixed_precision_policy,
        estimate_fsdp_memory_usage,
        print_fsdp_memory_estimate
    )
    FSDP_UTILS_AVAILABLE = True
    _imported_components.update({
        'setup_distributed_environment': setup_distributed_environment,
        'cleanup_distributed': cleanup_distributed,
        'wrap_model_with_fsdp': wrap_model_with_fsdp,
        'save_fsdp_checkpoint': save_fsdp_checkpoint,
        'load_fsdp_checkpoint': load_fsdp_checkpoint,
        'sync_across_gpus': sync_across_gpus,
        'is_master_rank': is_master_rank,
        'get_world_size': get_world_size,
        'get_rank': get_rank,
        'setup_environment_variables': setup_environment_variables,
        'create_fsdp_mixed_precision_policy': create_fsdp_mixed_precision_policy,
        'estimate_fsdp_memory_usage': estimate_fsdp_memory_usage,
        'print_fsdp_memory_estimate': print_fsdp_memory_estimate,
    })
    logger.info("✅ FSDP utilities loaded successfully")
except ImportError as e:
    logger.warning(f"⚠️ Failed to import FSDP utilities: {e}")

# =============================================================================
# DISTRIBUTED TRAINER IMPORTS - SIMPLIFIED
# =============================================================================
try:
    from src.modules.trainers.blip3o_distributed_trainer import (
        BLIP3oDistributedTrainer,
        create_distributed_clip_trainer,
    )
    DISTRIBUTED_TRAINER_AVAILABLE = True
    _imported_components.update({
        'BLIP3oDistributedTrainer': BLIP3oDistributedTrainer,
        'create_distributed_clip_trainer': create_distributed_clip_trainer,
        # Add aliases for backward compatibility
        'FixedBLIP3oDistributedTrainer': BLIP3oDistributedTrainer,
        'create_fixed_distributed_clip_trainer': create_distributed_clip_trainer,
    })
    logger.info("✅ Distributed trainer loaded successfully")
except ImportError as e:
    logger.warning(f"⚠️ Failed to import distributed trainer: {e}")

# =============================================================================
# DISTRIBUTED DATASET IMPORTS - SIMPLIFIED  
# =============================================================================
try:
    from src.modules.datasets.blip3o_distributed_dataset import (
        DistributedBLIP3oCLIPReproductionDataset,
        create_distributed_clip_reproduction_dataloaders,
    )
    DISTRIBUTED_DATASET_AVAILABLE = True
    _imported_components.update({
        'DistributedBLIP3oCLIPReproductionDataset': DistributedBLIP3oCLIPReproductionDataset,
        'create_distributed_clip_reproduction_dataloaders': create_distributed_clip_reproduction_dataloaders,
        # Add aliases for backward compatibility
        'create_distributed_dataloaders': create_distributed_clip_reproduction_dataloaders,
        'create_fixed_distributed_dataloaders': create_distributed_clip_reproduction_dataloaders,
    })
    logger.info("✅ Distributed dataset utilities loaded successfully")
except ImportError as e:
    logger.warning(f"⚠️ Failed to import distributed dataset utilities: {e}")

# =============================================================================
# SIMPLE COMMUNICATION UTILITIES
# =============================================================================
try:
    class SimpleCommunicator:
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
    
    def log_distributed_info(rank: int, world_size: int, message: str):
        """Log distributed information"""
        if rank == 0:
            logger.info(f"[Distributed {world_size}GPUs] {message}")
    
    def wait_for_everyone(rank: int, world_size: int, message: str = "Synchronizing"):
        """Wait for all ranks to reach this point"""
        if rank == 0:
            logger.info(f"⏳ {message} - waiting for all {world_size} ranks...")
        
        if torch.distributed.is_initialized():
            try:
                torch.distributed.barrier()
                if rank == 0:
                    logger.info(f"✅ All {world_size} ranks synchronized")
            except Exception as e:
                logger.warning(f"Synchronization failed on rank {rank}: {e}")
    
    def check_distributed_setup(rank: int, world_size: int) -> dict:
        """Check distributed training setup"""
        import os
        return {
            'rank': rank,
            'world_size': world_size,
            'is_distributed': torch.distributed.is_initialized(),
            'backend': torch.distributed.get_backend() if torch.distributed.is_initialized() else None,
            'master_addr': os.environ.get('MASTER_ADDR', 'unknown'),
            'master_port': os.environ.get('MASTER_PORT', 'unknown'),
        }
    
    COMMUNICATION_UTILS_AVAILABLE = True
    _imported_components.update({
        'SimpleCommunicator': SimpleCommunicator,
        'log_distributed_info': log_distributed_info,
        'wait_for_everyone': wait_for_everyone,
        'check_distributed_setup': check_distributed_setup,
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
    "DISTRIBUTED_DATASET_AVAILABLE",
    "COMMUNICATION_UTILS_AVAILABLE",
]

# Add available components to exports
if FSDP_UTILS_AVAILABLE:
    __all__.extend([
        "setup_distributed_environment", "cleanup_distributed",
        "wrap_model_with_fsdp", "save_fsdp_checkpoint", "load_fsdp_checkpoint",
        "sync_across_gpus", "is_master_rank", "get_world_size", "get_rank",
        "setup_environment_variables", "create_fsdp_mixed_precision_policy",
        "estimate_fsdp_memory_usage", "print_fsdp_memory_estimate"
    ])

if DISTRIBUTED_TRAINER_AVAILABLE:
    __all__.extend([
        "BLIP3oDistributedTrainer", "create_distributed_clip_trainer",
        "FixedBLIP3oDistributedTrainer", "create_fixed_distributed_clip_trainer"
    ])

if DISTRIBUTED_DATASET_AVAILABLE:
    __all__.extend([
        "DistributedBLIP3oCLIPReproductionDataset", 
        "create_distributed_clip_reproduction_dataloaders",
        "create_distributed_dataloaders", "create_fixed_distributed_dataloaders"
    ])

if COMMUNICATION_UTILS_AVAILABLE:
    __all__.extend([
        "SimpleCommunicator", "log_distributed_info", 
        "wait_for_everyone", "check_distributed_setup"
    ])

# Make imported components available at module level
locals().update(_imported_components)

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def check_distributed_environment():
    """Check if distributed training environment is set up correctly"""
    
    status = {
        'fsdp_utils': FSDP_UTILS_AVAILABLE,
        'distributed_trainer': DISTRIBUTED_TRAINER_AVAILABLE,
        'distributed_dataset': DISTRIBUTED_DATASET_AVAILABLE,
        'communication_utils': COMMUNICATION_UTILS_AVAILABLE,
        'cuda_available': torch.cuda.is_available(),
        'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'pytorch_distributed': hasattr(torch, 'distributed'),
        'nccl_available': torch.distributed.is_nccl_available() if hasattr(torch, 'distributed') else False,
    }
    
    # Check if core components are available
    core_components_available = all([
        status['fsdp_utils'],
        status['distributed_trainer'], 
        status['distributed_dataset'],
    ])
    
    # Check if system supports distributed training
    system_supports_distributed = all([
        status['cuda_available'],
        status['cuda_device_count'] > 1,
        status['pytorch_distributed'],
        status['nccl_available']
    ])
    
    ready_for_distributed = core_components_available and system_supports_distributed
    
    return {
        'component_status': status,
        'core_components_available': core_components_available,
        'system_supports_distributed': system_supports_distributed,
        'ready_for_distributed_training': ready_for_distributed,
        'missing_components': [name for name, available in {
            'fsdp_utils': status['fsdp_utils'],
            'distributed_trainer': status['distributed_trainer'],
            'distributed_dataset': status['distributed_dataset'],
        }.items() if not available],
    }

def print_distributed_status():
    """Print distributed training status"""
    print("🚀 BLIP3-o Distributed Training Environment Status")
    print("=" * 60)
    
    status = check_distributed_environment()
    
    print("📦 Component Status:")
    component_mapping = {
        'fsdp_utils': 'FSDP Utilities',
        'distributed_trainer': 'Distributed Trainer',
        'distributed_dataset': 'Distributed Dataset',
        'communication_utils': 'Communication Utils'
    }
    
    for component, description in component_mapping.items():
        available = status['component_status'].get(component, False)
        status_icon = "✅" if available else "❌"
        print(f"  {status_icon} {description}")
    
    print(f"\n🖥️ System Requirements:")
    sys_status = status['component_status']
    cuda_available = sys_status['cuda_available']
    cuda_count = sys_status['cuda_device_count']
    
    print(f"  {'✅' if cuda_available else '❌'} CUDA Available: {cuda_available}")
    print(f"  {'✅' if cuda_count > 1 else '❌'} CUDA Devices: {cuda_count}")
    print(f"  {'✅' if sys_status['pytorch_distributed'] else '❌'} PyTorch Distributed: {sys_status['pytorch_distributed']}")
    print(f"  {'✅' if sys_status['nccl_available'] else '❌'} NCCL Backend: {sys_status['nccl_available']}")
    
    print(f"\n📊 Overall Status:")
    if status['ready_for_distributed_training']:
        print("  🎉 Ready for distributed training!")
        print(f"  ⚡ Can use up to {cuda_count} GPUs with FSDP")
    else:
        print("  ⚠️ Not ready for distributed training")
        if not status['core_components_available']:
            print("  📦 Missing components:")
            for component in status['missing_components']:
                print(f"     • {component}")
        if not status['system_supports_distributed']:
            print("  🖥️ System limitations:")
            if not cuda_available:
                print("     • CUDA not available")
            if cuda_count <= 1:
                print(f"     • Need >1 GPU, found {cuda_count}")
    
    print("=" * 60)

# =============================================================================
# INITIALIZATION
# =============================================================================

# Run environment check on import
_env_status = check_distributed_environment()

# Log initialization status
if _env_status['ready_for_distributed_training']:
    logger.info("🎉 BLIP3-o distributed training environment ready!")
    logger.info(f"⚡ Can use {_env_status['component_status']['cuda_device_count']} GPUs")
elif _env_status['core_components_available']:
    logger.info("✅ BLIP3-o distributed components loaded successfully")
    logger.info("⚠️ System may not support multi-GPU training")
else:
    logger.warning(f"⚠️ Partial environment initialization")
    logger.warning(f"Missing: {_env_status['missing_components']}")

# Export environment status
DISTRIBUTED_ENVIRONMENT_STATUS = _env_status

# Cleanup
del _env_status, _imported_components