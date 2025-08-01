# src/modules/__init__.py
"""
<<<<<<< HEAD
Clean BLIP3-o Modules for CLIP Reproduction
Simplified module initialization
=======
BLIP3-o Modules - Updated for Spherical EVA Denoising
src/modules/__init__.py

Main entry point for all BLIP3-o modules including spherical EVA denoising components
>>>>>>> main
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

<<<<<<< HEAD
# Store imported components
_imported_components = {}

# =============================================================================
# MODEL IMPORTS (src/modules/models/blip3o_dit.py)
# =============================================================================
try:
    from src.modules.models.blip3o_dit import (
        BLIP3oCLIPDiTModel,
        BLIP3oCLIPDiTConfig, 
        create_clip_reproduction_model
    )
    MODEL_AVAILABLE = True
    _imported_components.update({
        'BLIP3oCLIPDiTModel': BLIP3oCLIPDiTModel,
        'BLIP3oCLIPDiTConfig': BLIP3oCLIPDiTConfig,
        'create_clip_reproduction_model': create_clip_reproduction_model,
    })
    logger.info("✅ CLIP DiT model loaded successfully")
except ImportError as e:
    logger.warning(f"⚠️ Failed to import model: {e}")
=======
# Import availability flags for spherical EVA denoising components
SPHERICAL_EVA_MODEL_AVAILABLE = False
SPHERICAL_EVA_LOSS_AVAILABLE = False
SPHERICAL_EVA_TRAINER_AVAILABLE = False
SPHERICAL_EVA_DATASET_AVAILABLE = False

# Import availability flags for legacy EVA reproduction components
EVA_MODEL_AVAILABLE = False
EVA_LOSS_AVAILABLE = False
EVA_TRAINER_AVAILABLE = False
EVA_DATASET_AVAILABLE = False



# Try importing NEW spherical EVA denoising components (MAIN COMPONENTS)
try:
    from .models.blip3o_eva_dit import (
        SphericalEVADiTModel, 
        SphericalEVADiTConfig, 
        create_spherical_eva_model
    )
    SPHERICAL_EVA_MODEL_AVAILABLE = True
    logger.info("✅ Spherical EVA denoising model loaded successfully")
except ImportError as e:
    logger.error(f"❌ Spherical EVA model import failed: {e}")
>>>>>>> main

# =============================================================================
# LOSS IMPORTS (src/modules/losses/blip3o_fm_loss.py)
# =============================================================================
try:
<<<<<<< HEAD
    from src.modules.losses.blip3o_fm_loss import (
        BLIP3oCLIPFlowMatchingLoss,
        create_clip_reproduction_loss
    )
    LOSS_AVAILABLE = True
    _imported_components.update({
        'BLIP3oCLIPFlowMatchingLoss': BLIP3oCLIPFlowMatchingLoss,
        'create_clip_reproduction_loss': create_clip_reproduction_loss,
    })
    logger.info("✅ Flow matching loss loaded successfully")
except ImportError as e:
    logger.warning(f"⚠️ Failed to import loss: {e}")
=======
    from .losses.blip3o_eva_loss import (
        SphericalFlowMatchingLoss, 
        create_spherical_flow_loss
    )
    SPHERICAL_EVA_LOSS_AVAILABLE = True
    logger.info("✅ Spherical flow matching loss loaded successfully")
except ImportError as e:
    logger.error(f"❌ Spherical flow loss import failed: {e}")
>>>>>>> main

# =============================================================================
# TRAINER IMPORTS (src/modules/trainers/blip3o_trainer.py) 
# =============================================================================
try:
<<<<<<< HEAD
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
=======
    from .trainers.blip3o_eva_trainer import (
        SphericalEVATrainer, 
        create_spherical_eva_trainer
    )
    SPHERICAL_EVA_TRAINER_AVAILABLE = True
    logger.info("✅ Spherical EVA trainer loaded successfully")
except ImportError as e:
    logger.error(f"❌ Spherical EVA trainer import failed: {e}")
>>>>>>> main

# =============================================================================
# DATASET IMPORTS (src/modules/datasets/blip3o_dataset.py)
# =============================================================================
try:
<<<<<<< HEAD
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
=======
    from .datasets.blip3o_eva_dataset import (
        create_eva_denoising_dataloaders, 
        BLIP3oEVADenoisingDataset,
        eva_denoising_collate_fn
    )
    SPHERICAL_EVA_DATASET_AVAILABLE = True
    logger.info("✅ EVA denoising dataset loaded successfully")
except ImportError as e:
    logger.error(f"❌ EVA denoising dataset import failed: {e}")



# Export main components
>>>>>>> main
__all__ = [
    # Availability flags
    "MODEL_AVAILABLE",
    "LOSS_AVAILABLE", 
    "TRAINER_AVAILABLE",
    "DATASET_AVAILABLE",
    "CONFIG_AVAILABLE",
<<<<<<< HEAD
]

# Add available components to exports
if MODEL_AVAILABLE:
    __all__.extend(["BLIP3oCLIPDiTModel", "BLIP3oCLIPDiTConfig", "create_clip_reproduction_model"])

if LOSS_AVAILABLE:
    __all__.extend(["BLIP3oCLIPFlowMatchingLoss", "create_clip_reproduction_loss"])

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
=======
    "SPHERICAL_EVA_MODEL_AVAILABLE",
    "SPHERICAL_EVA_LOSS_AVAILABLE",
    "SPHERICAL_EVA_TRAINER_AVAILABLE",
    "SPHERICAL_EVA_DATASET_AVAILABLE",
    "EVA_MODEL_AVAILABLE",
    "EVA_LOSS_AVAILABLE",
    "EVA_TRAINER_AVAILABLE",
    "EVA_DATASET_AVAILABLE",
]



# NEW spherical EVA denoising components (MAIN EXPORTS)
if SPHERICAL_EVA_MODEL_AVAILABLE:
    __all__.extend(["SphericalEVADiTModel", "SphericalEVADiTConfig", "create_spherical_eva_model"])

if SPHERICAL_EVA_LOSS_AVAILABLE:
    __all__.extend(["SphericalFlowMatchingLoss", "create_spherical_flow_loss"])

if SPHERICAL_EVA_TRAINER_AVAILABLE:
    __all__.extend(["SphericalEVATrainer", "create_spherical_eva_trainer"])

if SPHERICAL_EVA_DATASET_AVAILABLE:
    __all__.extend(["create_eva_denoising_dataloaders", "BLIP3oEVADenoisingDataset", "eva_denoising_collate_fn"])

# # Legacy EVA reproduction components
# if EVA_MODEL_AVAILABLE:
#     __all__.extend(["BLIP3oEVADiTModel", "BLIP3oEVADiTConfig", "create_eva_reproduction_model"])

# if EVA_LOSS_AVAILABLE:
#     __all__.extend(["BLIP3oEVAFlowMatchingLoss", "create_eva_reproduction_loss"])

# if EVA_TRAINER_AVAILABLE:
#     __all__.extend(["BLIP3oEVATrainer", "create_eva_training_args"])

# if EVA_DATASET_AVAILABLE:
#     __all__.extend(["create_eva_reproduction_dataloaders", "BLIP3oEVAReproductionDataset"])
>>>>>>> main

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
    
<<<<<<< HEAD
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
    print("🔍 Clean BLIP3-o CLIP Reproduction Environment Status")
    print("=" * 60)
    
    status = check_environment()
    
    print("📄 File Mapping:")
    file_mapping = {
        'model': 'src/modules/models/blip3o_dit.py',
        'loss': 'src/modules/losses/blip3o_fm_loss.py',
        'trainer': 'src/modules/trainers/blip3o_trainer.py',
        'dataset': 'src/modules/datasets/blip3o_dataset.py',
        'config': 'src/modules/config/blip3o_config.py',
=======
    spherical_eva_status = {
        'spherical_model': SPHERICAL_EVA_MODEL_AVAILABLE,
        'spherical_loss': SPHERICAL_EVA_LOSS_AVAILABLE,
        'spherical_trainer': SPHERICAL_EVA_TRAINER_AVAILABLE,
        'spherical_dataset': SPHERICAL_EVA_DATASET_AVAILABLE,
    }
    
    legacy_eva_status = {
        'legacy_eva_model': EVA_MODEL_AVAILABLE,
        'legacy_eva_loss': EVA_LOSS_AVAILABLE,
        'legacy_eva_trainer': EVA_TRAINER_AVAILABLE,
        'legacy_eva_dataset': EVA_DATASET_AVAILABLE,
    }
    
    all_original_available = all(original_status.values())
    all_spherical_eva_available = all(spherical_eva_status.values())
    all_legacy_eva_available = all(legacy_eva_status.values())
    
    if all_original_available:
        logger.info("🎉 All original BLIP3-o components loaded successfully!")
    else:
        missing = [name for name, available in original_status.items() if not available]
        logger.warning(f"⚠️ Missing original components: {missing}")
    
    if all_spherical_eva_available:
        logger.info("🎉 All spherical EVA denoising components loaded successfully!")
    else:
        missing = [name for name, available in spherical_eva_status.items() if not available]
        logger.error(f"❌ Missing spherical EVA components: {missing}")
    
    if all_legacy_eva_available:
        logger.info("✅ All legacy EVA reproduction components loaded successfully!")
    else:
        missing = [name for name, available in legacy_eva_status.items() if not available]
        logger.warning(f"⚠️ Missing legacy EVA components: {missing}")
    
    return {
        'original': original_status,
        'spherical_eva_denoising': spherical_eva_status,
        'legacy_eva_reproduction': legacy_eva_status,
        'all_original_available': all_original_available,
        'all_spherical_eva_available': all_spherical_eva_available,
        'all_legacy_eva_available': all_legacy_eva_available,
    }


def get_version_info():
    """Get version and component information"""
    return {
        'blip3o_implementation': 'spherical_eva_denoising_v1',
        'main_task': 'spherical_eva_denoising',
        'original_components': {
            'model': MODEL_AVAILABLE,
            'loss': LOSS_AVAILABLE,
            'trainer': TRAINER_AVAILABLE,
            'dataset': DATASET_AVAILABLE,
            'config': CONFIG_AVAILABLE,
        },
        'spherical_eva_components': {
            'model': SPHERICAL_EVA_MODEL_AVAILABLE,
            'loss': SPHERICAL_EVA_LOSS_AVAILABLE,
            'trainer': SPHERICAL_EVA_TRAINER_AVAILABLE,
            'dataset': SPHERICAL_EVA_DATASET_AVAILABLE,
        },
        'legacy_eva_components': {
            'model': EVA_MODEL_AVAILABLE,
            'loss': EVA_LOSS_AVAILABLE,
            'trainer': EVA_TRAINER_AVAILABLE,
            'dataset': EVA_DATASET_AVAILABLE,
        },
        'features': [
            'spherical_flow_matching',
            'eva_denoising',
            'unit_hypersphere_constraints',
            'slerp_interpolation',
            'cross_attention_conditioning',
            'proper_gradient_flow',
            'spherical_evaluation_metrics',
            'dit_architecture_optimized',
        ]
>>>>>>> main
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
        print(f"\n🎉 All components available! Ready for training.")
        print(f"🎯 Task: EVA [4096] → CLIP [1024] reproduction")
        print(f"🌊 Method: Clean rectified flow matching with BLIP3-o DiT")
    else:
        print(f"\n⚠️ Missing components: {', '.join(status['missing_components'])}")
        print(f"Available components: {', '.join(status['available_components'])}")
    
    print("=" * 60)

# =============================================================================
# INITIALIZATION
# =============================================================================

<<<<<<< HEAD
# Run environment check on import
_env_status = check_environment()

# Log initialization status
if _env_status['all_available']:
    logger.info("🎉 Clean BLIP3-o CLIP reproduction modules fully initialized!")
    logger.info("🎯 Ready for EVA → CLIP reproduction training")
else:
    logger.warning(f"⚠️ Partial initialization. Missing: {_env_status['missing_components']}")
=======
def get_recommended_components():
    """Get recommended components for different tasks"""
    return {
        'spherical_eva_denoising': {
            'description': 'NEW: Spherical EVA-CLIP denoising with proper flow matching',
            'input': 'Noisy EVA embeddings [B, N, 4096]',
            'conditioning': 'Clean EVA embeddings [B, N, 4096]', 
            'output': 'Clean EVA embeddings [B, N, 4096]',
            'components': {
                'model': 'SphericalEVADiTModel',
                'loss': 'SphericalFlowMatchingLoss', 
                'trainer': 'SphericalEVATrainer',
                'dataset': 'BLIP3oEVADenoisingDataset',
                'dataloaders': 'create_eva_denoising_dataloaders',
            },
            'available': all([
                SPHERICAL_EVA_MODEL_AVAILABLE,
                SPHERICAL_EVA_LOSS_AVAILABLE, 
                SPHERICAL_EVA_TRAINER_AVAILABLE,
                SPHERICAL_EVA_DATASET_AVAILABLE
            ]),
            'recommended': True,
        },
        'legacy_eva_reproduction': {
            'description': 'LEGACY: EVA-CLIP reproduction (may have issues)',
            'input': 'Noisy EVA embeddings [B, N, 4096]',
            'conditioning': 'CLIP embeddings [B, N, 1024]',
            'output': 'EVA embeddings [B, N, 4096]',
            'components': {
                'model': 'BLIP3oEVADiTModel',
                'loss': 'BLIP3oEVAFlowMatchingLoss',
                'trainer': 'BLIP3oEVATrainer',
                'dataset': 'BLIP3oEVAReproductionDataset',
                'dataloaders': 'create_eva_reproduction_dataloaders',
            },
            'available': all([
                EVA_MODEL_AVAILABLE,
                EVA_LOSS_AVAILABLE,
                EVA_TRAINER_AVAILABLE, 
                EVA_DATASET_AVAILABLE
            ]),
            'recommended': False,
        },
        'original_blip3o': {
            'description': 'Original BLIP3-o implementation',
            'components': {
                'model': 'BLIP3oPatchDiTModel',
                'loss': 'BLIP3oFlowMatchingLoss',
                'trainer': 'BLIP3oTrainer', 
                'dataset': 'BLIP3oEmbeddingDataset',
                'dataloaders': 'create_blip3o_dataloaders',
            },
            'available': all([
                MODEL_AVAILABLE,
                LOSS_AVAILABLE,
                TRAINER_AVAILABLE,
                DATASET_AVAILABLE
            ]),
            'recommended': False,
        }
    }


# Initialize on import
_status = check_environment()

# Priority messaging
if _status['all_spherical_eva_available']:
    logger.info("🎉 SPHERICAL EVA DENOISING components ready! (RECOMMENDED)")
    logger.info("  Input: Noisy EVA embeddings [B, N, 4096]")
    logger.info("  Conditioning: Clean EVA embeddings [B, N, 4096]")
    logger.info("  Output: Clean EVA embeddings [B, N, 4096]")
    logger.info("  Features: Spherical flow matching, proper gradient flow, slerp interpolation")
else:
    logger.error("❌ SPHERICAL EVA DENOISING components missing!")
    logger.error("  Please ensure the following files are present:")
    logger.error("    - src/modules/models/spherical_eva_dit.py")
    logger.error("    - src/modules/losses/spherical_flow_loss.py")
    logger.error("    - src/modules/trainers/spherical_eva_trainer.py")
    logger.error("    - src/modules/datasets/eva_denoising_dataset.py")

if not _status['all_original_available']:
    logger.warning("⚠️ Some original BLIP3-o components failed to load. Check individual imports.")

if not _status['all_legacy_eva_available']:
    logger.warning("⚠️ Some legacy EVA reproduction components failed to load.")

# Final status
if _status['all_spherical_eva_available']:
    logger.info("🎯 READY: Use spherical EVA denoising for your project!")
    logger.info("  Use: create_spherical_eva_model, create_spherical_flow_loss, etc.")
elif _status['all_legacy_eva_available']:
    logger.warning("⚠️ FALLBACK: Only legacy EVA reproduction available (may have issues)")
else:
    logger.error("❌ CRITICAL: No EVA denoising components available!")
>>>>>>> main

# Export environment status for external access
ENVIRONMENT_STATUS = _env_status

<<<<<<< HEAD
# Cleanup
del _env_status, _imported_components
=======
# =============================================================================
# Sub-module __init__.py files
# =============================================================================

# src/modules/models/__init__.py
"""
BLIP3-o Models Module - Updated for Spherical EVA Denoising
src/modules/models/__init__.py
"""

# src/modules/losses/__init__.py  
"""
BLIP3-o Losses Module - Updated for Spherical EVA Denoising
src/modules/losses/__init__.py
"""

# src/modules/trainers/__init__.py
"""
BLIP3-o Trainers Module - Updated for Spherical EVA Denoising  
src/modules/trainers/__init__.py
"""

# src/modules/datasets/__init__.py
"""
BLIP3-o Datasets Module - Updated for Spherical EVA Denoising
src/modules/datasets/__init__.py
"""
>>>>>>> main
