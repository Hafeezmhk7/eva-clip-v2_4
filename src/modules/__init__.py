# src/modules/__init__.py
"""
FIXED BLIP3-o Modules - Updated for Distributed Training
src/modules/__init__.py

FIXES:
- Proper import handling with correct aliases
- Better error handling for missing components
- Compatibility with distributed trainer
"""

import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# Add current directory to path for imports
current_dir = Path(__file__).parent.parent.parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

# Import availability flags
MODEL_AVAILABLE = False
LOSS_AVAILABLE = False
TRAINER_AVAILABLE = False
DATASET_AVAILABLE = False
CONFIG_AVAILABLE = False

# Store imported components
_imported_components = {}

# =============================================================================
# MODEL IMPORTS (src/modules/models/blip3o_dit.py)
# =============================================================================
try:
    from src.modules.models.blip3o_dit import (
        ImprovedBLIP3oCLIPDiTModel,
        BLIP3oCLIPDiTModel,  # Alias for compatibility
        BLIP3oCLIPDiTConfig, 
        create_improved_clip_reproduction_model,
        create_clip_reproduction_model  # Alias for compatibility
    )
    MODEL_AVAILABLE = True
    _imported_components.update({
        'ImprovedBLIP3oCLIPDiTModel': ImprovedBLIP3oCLIPDiTModel,
        'BLIP3oCLIPDiTModel': BLIP3oCLIPDiTModel,  # FIXED: Proper alias
        'BLIP3oCLIPDiTConfig': BLIP3oCLIPDiTConfig,
        'create_improved_clip_reproduction_model': create_improved_clip_reproduction_model,
        'create_clip_reproduction_model': create_clip_reproduction_model,
    })
    logger.info("✅ CLIP DiT model loaded successfully")
except ImportError as e:
    logger.warning(f"⚠️ Failed to import model: {e}")

# =============================================================================
# LOSS IMPORTS (src/modules/losses/blip3o_fm_loss.py)
# =============================================================================
try:
    from src.modules.losses.blip3o_fm_loss import (
        SemanticPreservingFlowMatchingLoss,
        BLIP3oCLIPFlowMatchingLoss,  # Alias for compatibility
        create_clip_reproduction_loss
    )
    LOSS_AVAILABLE = True
    _imported_components.update({
        'SemanticPreservingFlowMatchingLoss': SemanticPreservingFlowMatchingLoss,
        'BLIP3oCLIPFlowMatchingLoss': BLIP3oCLIPFlowMatchingLoss,  # FIXED: Proper alias
        'create_clip_reproduction_loss': create_clip_reproduction_loss,
    })
    logger.info("✅ Flow matching loss loaded successfully")
except ImportError as e:
    logger.warning(f"⚠️ Failed to import loss: {e}")

# =============================================================================
# TRAINER IMPORTS (src/modules/trainers/blip3o_trainer.py) 
# =============================================================================
try:
    from src.modules.trainers.blip3o_trainer import (
        BLIP3oCLIPTrainer,
        create_clip_trainer
    )
    TRAINER_AVAILABLE = True
    _imported_components.update({
        'BLIP3oCLIPTrainer': BLIP3oCLIPTrainer,
        'create_clip_trainer': create_clip_trainer,
    })
    logger.info("✅ CLIP trainer loaded successfully")
except ImportError as e:
    logger.warning(f"⚠️ Failed to import trainer: {e}")

# =============================================================================
# DATASET IMPORTS (src/modules/datasets/blip3o_dataset.py)
# =============================================================================
try:
    from src.modules.datasets.blip3o_dataset import (
        create_clip_reproduction_dataloaders,
        BLIP3oCLIPReproductionDataset,
        clip_reproduction_collate_fn
    )
    DATASET_AVAILABLE = True
    _imported_components.update({
        'create_clip_reproduction_dataloaders': create_clip_reproduction_dataloaders,
        'BLIP3oCLIPReproductionDataset': BLIP3oCLIPReproductionDataset,
        'clip_reproduction_collate_fn': clip_reproduction_collate_fn,
    })
    logger.info("✅ CLIP datasets loaded successfully")
except ImportError as e:
    logger.warning(f"⚠️ Failed to import dataset: {e}")

# =============================================================================
# CONFIG IMPORTS (src/modules/config/blip3o_config.py)
# =============================================================================
try:
    from src.modules.config.blip3o_config import (
        get_blip3o_clip_config,
        create_config_from_args,
        BLIP3oCLIPDiTConfig,
        FlowMatchingConfig,
        TrainingConfig,
        EvaluationConfig
    )
    CONFIG_AVAILABLE = True
    _imported_components.update({
        'get_blip3o_clip_config': get_blip3o_clip_config,
        'create_config_from_args': create_config_from_args,
        'BLIP3oCLIPDiTConfig': BLIP3oCLIPDiTConfig,
        'FlowMatchingConfig': FlowMatchingConfig,
        'TrainingConfig': TrainingConfig,
        'EvaluationConfig': EvaluationConfig,
    })
    logger.info("✅ Configuration loaded successfully")
except ImportError as e:
    logger.warning(f"⚠️ Failed to import config: {e}")

# =============================================================================
# EXPORT ALL COMPONENTS
# =============================================================================

# Main availability flags
__all__ = [
    # Availability flags
    "MODEL_AVAILABLE",
    "LOSS_AVAILABLE", 
    "TRAINER_AVAILABLE",
    "DATASET_AVAILABLE",
    "CONFIG_AVAILABLE",
]

# Add available components to exports
if MODEL_AVAILABLE:
    __all__.extend([
        "ImprovedBLIP3oCLIPDiTModel", 
        "BLIP3oCLIPDiTModel", 
        "BLIP3oCLIPDiTConfig", 
        "create_improved_clip_reproduction_model",
        "create_clip_reproduction_model"
    ])

if LOSS_AVAILABLE:
    __all__.extend([
        "SemanticPreservingFlowMatchingLoss",
        "BLIP3oCLIPFlowMatchingLoss", 
        "create_clip_reproduction_loss"
    ])

if TRAINER_AVAILABLE:
    __all__.extend(["BLIP3oCLIPTrainer", "create_clip_trainer"])

if DATASET_AVAILABLE:
    __all__.extend(["create_clip_reproduction_dataloaders", "BLIP3oCLIPReproductionDataset", "clip_reproduction_collate_fn"])

if CONFIG_AVAILABLE:
    __all__.extend([
        "get_blip3o_clip_config", "create_config_from_args", "BLIP3oCLIPDiTConfig",
        "FlowMatchingConfig", "TrainingConfig", "EvaluationConfig"
    ])

# Make imported components available at module level
locals().update(_imported_components)

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def check_environment():
    """Check if all required components are available"""
    status = {
        'model': MODEL_AVAILABLE,
        'loss': LOSS_AVAILABLE,
        'trainer': TRAINER_AVAILABLE,
        'dataset': DATASET_AVAILABLE,
        'config': CONFIG_AVAILABLE,
    }
    
    all_available = all(status.values())
    
    if all_available:
        logger.info("🎉 All CLIP reproduction components loaded successfully!")
    else:
        missing = [name for name, available in status.items() if not available]
        logger.warning(f"⚠️ Missing components: {missing}")
    
    return {
        'component_status': status,
        'all_available': all_available,
        'missing_components': [name for name, available in status.items() if not available],
        'available_components': [name for name, available in status.items() if available],
    }

def print_environment_status():
    """Print detailed environment status"""
    print("🔍 FIXED BLIP3-o CLIP Reproduction Environment Status")
    print("=" * 60)
    
    status = check_environment()
    
    print("📄 File Mapping:")
    file_mapping = {
        'model': 'src/modules/models/blip3o_dit.py',
        'loss': 'src/modules/losses/blip3o_fm_loss.py',
        'trainer': 'src/modules/trainers/blip3o_trainer.py',
        'dataset': 'src/modules/datasets/blip3o_dataset.py',
        'config': 'src/modules/config/blip3o_config.py',
    }
    
    for component, filename in file_mapping.items():
        available = status['component_status'].get(component, False)
        status_icon = "✅" if available else "❌"
        print(f"  {status_icon} {component}: {filename}")
    
    print(f"\n📊 Component Status:")
    for component, available in status['component_status'].items():
        status_icon = "✅" if available else "❌"
        print(f"  {status_icon} {component.capitalize()}: {'Available' if available else 'Not Available'}")
    
    if status['all_available']:
        print(f"\n🎉 All components available! Ready for distributed training.")
        print(f"🎯 Task: EVA [4096] → CLIP [1024] reproduction")
        print(f"🌊 Method: FSDP distributed training with BLIP3-o DiT")
        print(f"🔧 FIXES APPLIED: Proper aliases and IterableDataset compatibility")
    else:
        print(f"\n⚠️ Missing components: {', '.join(status['missing_components'])}")
        print(f"Available components: {', '.join(status['available_components'])}")
    
    print("=" * 60)

# =============================================================================
# INITIALIZATION
# =============================================================================

# Run environment check on import
_env_status = check_environment()

# Log initialization status
if _env_status['all_available']:
    logger.info("🎉 FIXED BLIP3-o CLIP reproduction modules fully initialized!")
    logger.info("🎯 Ready for distributed EVA → CLIP reproduction training")
    logger.info("🔧 All compatibility fixes applied")
else:
    logger.warning(f"⚠️ Partial initialization. Missing: {_env_status['missing_components']}")

# Export environment status for external access
ENVIRONMENT_STATUS = _env_status

# Cleanup
del _env_status, _imported_components