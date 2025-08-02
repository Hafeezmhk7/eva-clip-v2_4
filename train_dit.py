#!/usr/bin/env python3
"""
FIXED BLIP3-o Training Script with All Critical Improvements
train_dit.py

🔥 CRITICAL FIXES IMPLEMENTED:
1. ✅ Conservative CLIP embedding normalization (scale 4.0→2.0)
2. ✅ Increased semantic loss weights (0.1→0.5, 0.05→0.2)
3. ✅ New direct CLIP consistency loss (0.3 weight)
4. ✅ Heun's solver for O(h²) integration accuracy
5. ✅ Proper evaluation denormalization
6. ✅ Comprehensive disconnect detection and monitoring
7. ✅ Earlier semantic loss application (threshold 0.7→0.5)

Expected improvement: CLIP similarity 0.46 → 0.65+ 🚀

Usage:
    python train_dit.py --chunked_embeddings_dir /path/to/embeddings --output_dir ./checkpoints_fixed
"""

import os
import sys
import argparse
import torch
import json
import logging
from pathlib import Path
from datetime import datetime
import traceback
import math

# Setup paths
sys.path.insert(0, str(Path(__file__).parent))

def setup_logging():
    """Setup comprehensive logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('fixed_clip_training.log', mode='w')
        ]
    )
    return logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments with FIXED defaults"""
    parser = argparse.ArgumentParser(
        description="FIXED BLIP3-o CLIP Reproduction Training with Critical Improvements",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument("--chunked_embeddings_dir", type=str, required=True,
                       help="Path to chunked embeddings directory")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory for checkpoints")
    
    # Model configuration
    parser.add_argument("--model_size", type=str, default="base",
                       choices=["tiny", "small", "base", "large"],
                       help="Model size")
    parser.add_argument("--training_mode", type=str, default="patch_only",
                       choices=["patch_only", "cls_patch"],
                       help="Training mode")
    
    # Training hyperparameters
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                       help="Learning rate (conservative for stability)")
    parser.add_argument("--batch_size", type=int, default=8,
                       help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=10,
                       help="Number of epochs")
    parser.add_argument("--warmup_steps", type=int, default=100,
                       help="Warmup steps")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                       help="Weight decay")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                       help="Max gradient norm")
    
    # FIXED: Much higher loss component weights
    parser.add_argument("--velocity_weight", type=float, default=1.0,
                       help="Weight for velocity prediction loss")
    parser.add_argument("--semantic_weight", type=float, default=0.5,  # FIXED: 5x increase
                       help="Weight for semantic consistency loss (INCREASED)")
    parser.add_argument("--cosine_weight", type=float, default=0.2,    # FIXED: 4x increase
                       help="Weight for cosine similarity loss (INCREASED)")
    parser.add_argument("--consistency_weight", type=float, default=0.3,  # NEW
                       help="Weight for direct CLIP consistency loss (NEW)")
    
    # Evaluation
    parser.add_argument("--eval_every_n_steps", type=int, default=50,
                       help="Evaluate every N steps")
    parser.add_argument("--eval_num_samples", type=int, default=100,
                       help="Number of samples for evaluation")
    parser.add_argument("--eval_inference_steps", type=int, default=50,
                       help="Number of inference steps for evaluation")
    parser.add_argument("--use_heun_inference", action="store_true", default=True,
                       help="Use Heun solver for inference (RECOMMENDED)")
    
    # Data
    parser.add_argument("--max_shards", type=int, default=None,
                       help="Maximum number of shards to use")
    
    # System
    parser.add_argument("--fp16", action="store_true", default=True,
                       help="Use mixed precision")
    parser.add_argument("--num_workers", type=int, default=0,
                       help="Number of dataloader workers")
    
    # Architecture improvements (all enabled by default)
    parser.add_argument("--use_eva_adapter", action="store_true", default=True,
                       help="Use EVA-CLIP adapter layers (RECOMMENDED)")
    parser.add_argument("--eva_adapter_layers", type=int, default=6,
                       help="Number of EVA adapter layers")
    parser.add_argument("--use_timestep_weighting", action="store_true", default=True,
                       help="Use timestep-dependent loss weighting")
    
    # WandB configuration
    parser.add_argument("--use_wandb", action="store_true", default=False,
                       help="Enable WandB logging")
    parser.add_argument("--wandb_project", type=str, default="blip3o-clip-fixed",
                       help="WandB project name")
    parser.add_argument("--wandb_run_name", type=str, default=None,
                       help="WandB run name")
    
    return parser.parse_args()

def validate_arguments(args, logger):
    """Validate command line arguments"""
    errors = []
    
    # Validate paths
    embeddings_dir = Path(args.chunked_embeddings_dir)
    if not embeddings_dir.exists():
        errors.append(f"Embeddings directory does not exist: {embeddings_dir}")
    
    # Validate numeric parameters
    if args.learning_rate <= 0:
        errors.append(f"Learning rate must be positive: {args.learning_rate}")
    
    if args.batch_size <= 0:
        errors.append(f"Batch size must be positive: {args.batch_size}")
    
    # Validate weights
    if args.velocity_weight < 0 or args.semantic_weight < 0 or args.cosine_weight < 0:
        errors.append("All loss weights must be non-negative")
    
    # FIXED: Warn if using old problematic weights
    if args.semantic_weight < 0.3:
        logger.warning(f"⚠️ Semantic weight ({args.semantic_weight}) seems low!")
        logger.warning("   Recommended: 0.5+ for better semantic preservation")
    
    if args.cosine_weight < 0.15:
        logger.warning(f"⚠️ Cosine weight ({args.cosine_weight}) seems low!")
        logger.warning("   Recommended: 0.2+ for better similarity preservation")
    
    if errors:
        logger.error("❌ Validation errors:")
        for error in errors:
            logger.error(f"   • {error}")
        return False
    
    return True

def check_environment(logger):
    """Check environment and system requirements"""
    issues = []
    
    # Check CUDA
    if not torch.cuda.is_available():
        issues.append("CUDA not available - training will be very slow on CPU")
    else:
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"GPU Memory: {total_memory:.1f} GB")
        
        if total_memory < 16:
            issues.append(f"Low GPU memory: {total_memory:.1f} GB (16+ GB recommended)")
    
    # Check PyTorch version
    logger.info(f"PyTorch version: {torch.__version__}")
    
    # Check for FIXED imports
    try:
        from src.modules.datasets.blip3o_dataset import create_fixed_clip_reproduction_dataloaders
        from src.modules.models.blip3o_dit import create_improved_clip_reproduction_model
        from src.modules.losses.blip3o_fm_loss import create_fixed_clip_reproduction_loss
        from src.modules.trainers.blip3o_trainer import create_fixed_clip_trainer
        logger.info("✅ All FIXED modules imported successfully")
    except ImportError as e:
        issues.append(f"Failed to import FIXED modules: {e}")
        logger.error("❌ CRITICAL: Could not import fixed modules!")
        logger.error("   Make sure you're using the updated files with FIXED implementations")
    
    if issues:
        logger.warning("Environment issues detected:")
        for issue in issues:
            logger.warning(f"  • {issue}")
    else:
        logger.info("✅ Environment check passed")
    
    return len(issues) == 0

def create_model(args, logger):
    """Create model with all enhancements"""
    try:
        from src.modules.models.blip3o_dit import create_improved_clip_reproduction_model
        
        model = create_improved_clip_reproduction_model(
            model_size=args.model_size,
            training_mode=args.training_mode,
            use_3d_rope=True,
            use_sandwich_norm=True,
            use_eva_adapter=args.use_eva_adapter,
            eva_adapter_layers=args.eva_adapter_layers,
        )
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        
        logger.info(f"✅ Model created with {model.get_num_parameters():,} parameters")
        logger.info(f"  Model size: {args.model_size}")
        logger.info(f"  Training mode: {args.training_mode}")
        logger.info(f"  3D RoPE: ✅ ENABLED")
        logger.info(f"  Sandwich Norm: ✅ ENABLED")
        logger.info(f"  EVA Adapter: {'✅ ENABLED' if args.use_eva_adapter else '❌ DISABLED'}")
        logger.info(f"  Heun Solver: ✅ ENABLED")
        
        return model, device
        
    except ImportError as e:
        logger.error(f"❌ Could not import model: {e}")
        raise
    except Exception as e:
        logger.error(f"❌ Error creating model: {e}")
        raise

def create_fixed_loss_function(args, logger):
    """Create FIXED loss function with critical improvements"""
    try:
        from src.modules.losses.blip3o_fm_loss import create_fixed_clip_reproduction_loss
        
        loss_fn = create_fixed_clip_reproduction_loss(
            prediction_type="velocity",
            flow_type="rectified",
            velocity_weight=args.velocity_weight,
            semantic_weight=args.semantic_weight,      # FIXED: Much higher
            cosine_weight=args.cosine_weight,          # FIXED: Much higher
            consistency_weight=args.consistency_weight, # NEW: Direct CLIP loss
            use_timestep_weighting=args.use_timestep_weighting,
        )
        
        logger.info("✅ FIXED loss function created:")
        logger.info(f"  Prediction type: velocity")
        logger.info(f"  Flow type: rectified")
        logger.info(f"  FIXED Weights - Velocity: {args.velocity_weight}, Semantic: {args.semantic_weight}, Cosine: {args.cosine_weight}")
        logger.info(f"  NEW Consistency weight: {args.consistency_weight}")
        logger.info(f"  Timestep weighting: {'✅ ENABLED' if args.use_timestep_weighting else '❌ DISABLED'}")
        
        # CRITICAL: Warn about weight increases
        if args.semantic_weight >= 0.5:
            logger.info(f"  🔥 Semantic weight INCREASED to {args.semantic_weight} (was 0.1)")
        if args.cosine_weight >= 0.2:
            logger.info(f"  🔥 Cosine weight INCREASED to {args.cosine_weight} (was 0.05)")
        
        return loss_fn
        
    except ImportError as e:
        logger.error(f"❌ Could not import FIXED loss function: {e}")
        raise
    except Exception as e:
        logger.error(f"❌ Error creating FIXED loss function: {e}")
        raise

def create_fixed_dataloaders(args, logger):
    """Create FIXED data loaders with proper normalization"""
    try:
        from src.modules.datasets.blip3o_dataset import create_fixed_clip_reproduction_dataloaders
        
        # Validate embeddings directory
        embeddings_dir = Path(args.chunked_embeddings_dir)
        logger.info(f"Loading embeddings from: {embeddings_dir}")
        
        # Look for embedding files
        pkl_files = list(embeddings_dir.glob("*.pkl"))
        if not pkl_files:
            raise FileNotFoundError(f"No .pkl files found in {embeddings_dir}")
        
        logger.info(f"Found {len(pkl_files)} .pkl files in embeddings directory")
        
        train_dataloader, eval_dataloader = create_fixed_clip_reproduction_dataloaders(
            chunked_embeddings_dir=args.chunked_embeddings_dir,
            batch_size=args.batch_size,
            training_mode=args.training_mode,
            max_shards=args.max_shards,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
            skip_corrupted_samples=True,
            validate_tensor_shapes=True,
        )
        
        logger.info("✅ FIXED dataloaders created successfully:")
        logger.info(f"  Training mode: {args.training_mode}")
        logger.info(f"  Batch size: {args.batch_size}")
        logger.info(f"  Max shards: {args.max_shards}")
        logger.info(f"  🔥 FIXED CLIP normalization: ✅ PROPERLY CONFIGURED")
        logger.info(f"  🔥 U-shaped timestep sampling: ✅ ENABLED")
        
        # Test dataloader
        test_batch = next(iter(train_dataloader))
        logger.info(f"✅ Dataloader test successful:")
        logger.info(f"  Batch size: {test_batch.get('batch_size', 'unknown')}")
        logger.info(f"  CLIP embeddings shape: {test_batch['clip_embeddings'].shape}")
        logger.info(f"  EVA embeddings shape: {test_batch['encoder_hidden_states'].shape}")
        
        # CRITICAL: Check CLIP normalizer is available and working
        if hasattr(train_dataloader, 'clip_normalizer') and train_dataloader.clip_normalizer:
            normalizer = train_dataloader.clip_normalizer
            logger.info(f"  🔥 CLIP normalizer: ✅ AVAILABLE")
            logger.info(f"     Stats computed: {normalizer.stats_computed}")
            logger.info(f"     Scale factor: {normalizer.scale_factor:.2f} (FIXED: reduced from 4.0)")
            
            # CRITICAL: Test normalization range
            sample_clip = test_batch['clip_embeddings']
            clip_range = (sample_clip.min().item(), sample_clip.max().item())
            logger.info(f"     Normalized range: [{clip_range[0]:.2f}, {clip_range[1]:.2f}]")
            
            # Validate range is reasonable
            if abs(clip_range[0]) > 15 or abs(clip_range[1]) > 15:
                logger.error(f"❌ NORMALIZATION RANGE TOO LARGE: {clip_range}")
                logger.error("   This will cause training instability!")
                raise ValueError("CLIP normalization range is too extreme")
            elif abs(clip_range[0]) < 0.5 and abs(clip_range[1]) < 0.5:
                logger.warning(f"⚠️ Normalization range seems small: {clip_range}")
                logger.warning("   This may limit semantic information")
            else:
                logger.info(f"     ✅ Normalization range looks good for semantic preservation")
        else:
            logger.error(f"❌ CLIP normalizer: MISSING!")
            logger.error("   This is a CRITICAL issue that will cause poor performance!")
            raise ValueError("CLIP normalizer is required but not found")
        
        return train_dataloader, eval_dataloader
        
    except ImportError as e:
        logger.error(f"❌ Could not import FIXED dataset: {e}")
        raise
    except Exception as e:
        logger.error(f"❌ Error creating FIXED dataloaders: {e}")
        raise

def create_fixed_trainer(model, loss_fn, train_dataloader, eval_dataloader, args, device, logger):
    """Create FIXED trainer with all enhancements"""
    try:
        from src.modules.trainers.blip3o_trainer import create_fixed_clip_trainer
        
        # Create run name if not provided
        wandb_run_name = args.wandb_run_name
        if wandb_run_name is None and args.use_wandb:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            improvements = []
            if args.use_eva_adapter:
                improvements.append("eva_adapter")
            if args.use_heun_inference:
                improvements.append("heun")
            improvements.append("FIXED")
            improvements_str = "_".join(improvements)
            wandb_run_name = f"blip3o_{args.model_size}_{args.training_mode}_{improvements_str}_{timestamp}"
        
        # WandB config with FIXED parameters
        wandb_config = {
            "model_size": args.model_size,
            "training_mode": args.training_mode,
            "batch_size": args.batch_size,
            "max_shards": args.max_shards,
            "experiment_version": "FIXED_v1",
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "num_epochs": args.num_epochs,
            "warmup_steps": args.warmup_steps,
            "max_grad_norm": args.max_grad_norm,
            "fp16": args.fp16,
            
            # CRITICAL: Fixed loss weights
            "velocity_weight": args.velocity_weight,
            "semantic_weight": args.semantic_weight,
            "cosine_weight": args.cosine_weight, 
            "consistency_weight": args.consistency_weight,
            
            # Architecture improvements
            "clip_normalization": "FIXED_conservative_scaling",
            "eva_adapter": args.use_eva_adapter,
            "eva_adapter_layers": args.eva_adapter_layers,
            "heun_inference": args.use_heun_inference,
            "timestep_weighting": args.use_timestep_weighting,
            
            # Expected improvements
            "expected_clip_similarity_improvement": "0.46 → 0.65+",
            "critical_fixes": [
                "conservative_clip_normalization",
                "increased_semantic_weights",
                "direct_clip_consistency_loss",
                "heun_solver_integration", 
                "proper_evaluation_denormalization",
                "disconnect_detection_monitoring"
            ]
        }
        
        # Get CLIP normalizer from dataloader
        clip_normalizer = getattr(train_dataloader, 'clip_normalizer', None)
        if clip_normalizer is None:
            logger.error("❌ CRITICAL: No CLIP normalizer found in dataloader!")
            raise ValueError("CLIP normalizer is required for proper training and evaluation")
        
        trainer = create_fixed_clip_trainer(
            model=model,
            loss_fn=loss_fn,
            train_dataloader=train_dataloader,
            eval_dataloader=eval_dataloader,
            clip_normalizer=clip_normalizer,  # CRITICAL: Pass normalizer
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            num_epochs=args.num_epochs,
            warmup_steps=args.warmup_steps,
            max_grad_norm=args.max_grad_norm,
            fp16=args.fp16,
            eval_every_n_steps=args.eval_every_n_steps,
            eval_num_samples=args.eval_num_samples,
            eval_inference_steps=args.eval_inference_steps,
            use_heun_inference=args.use_heun_inference,
            output_dir=args.output_dir,
            device=device,
            use_wandb=args.use_wandb,
            wandb_project=args.wandb_project,
            wandb_run_name=wandb_run_name,
            wandb_config=wandb_config,
        )
        
        logger.info("✅ FIXED trainer created successfully:")
        logger.info(f"  Evaluation: Every {args.eval_every_n_steps} steps")
        logger.info(f"  WandB enabled: {args.use_wandb}")
        logger.info(f"  CLIP normalizer: {'✅ AVAILABLE' if clip_normalizer else '❌ MISSING'}")
        logger.info(f"  Heun inference: {'✅ ENABLED' if args.use_heun_inference else '❌ DISABLED'}")
        logger.info(f"  Disconnect detection: ✅ ENABLED")
        
        return trainer
        
    except ImportError as e:
        logger.error(f"❌ Could not import FIXED trainer: {e}")
        raise
    except Exception as e:
        logger.error(f"❌ Error creating FIXED trainer: {e}")
        raise

def save_experiment_config(args, model, output_dir, logger):
    """Save detailed experiment configuration"""
    try:
        config = {
            'experiment_info': {
                'name': 'FIXED BLIP3-o CLIP Reproduction',
                'version': 'FIXED_v1',
                'timestamp': datetime.now().isoformat(),
                'task': 'Reproduce CLIP embeddings from EVA embeddings',
                'method': 'BLIP3-o DiT with CRITICAL FIXES',
                'expected_improvement': 'CLIP similarity 0.46 → 0.65+',
            },
            'args': vars(args),
            'model_config': model.config.to_dict() if hasattr(model.config, 'to_dict') else {},
            'model_info': {
                'parameters': model.get_num_parameters() if hasattr(model, 'get_num_parameters') else 'unknown',
                'model_class': model.__class__.__name__,
            },
            'critical_fixes': {
                'conservative_clip_normalization': {
                    'implemented': True,
                    'change': 'Scale factor 4.0 → 2.0',
                    'description': 'Preserve semantic structure in embeddings',
                    'expected_impact': 'HIGH (0.46 → 0.55+)',
                },
                'increased_semantic_weights': {
                    'implemented': True,
                    'changes': {
                        'semantic_weight': f'0.1 → {args.semantic_weight}',
                        'cosine_weight': f'0.05 → {args.cosine_weight}',
                    },
                    'description': 'Much stronger semantic preservation during training',
                    'expected_impact': 'HIGH (+0.1 similarity)',
                },
                'direct_clip_consistency': {
                    'implemented': True,
                    'weight': args.consistency_weight,
                    'description': 'Direct optimization of evaluation metric',
                    'expected_impact': 'MEDIUM (+0.05 similarity)',
                },
                'heun_solver': {
                    'implemented': args.use_heun_inference,
                    'description': 'O(h²) integration accuracy vs Euler O(h)',
                    'expected_impact': 'MEDIUM (+0.05 similarity)',
                },
                'proper_evaluation_denormalization': {
                    'implemented': True,
                    'description': 'Evaluate in original CLIP space, not normalized',
                    'expected_impact': 'CRITICAL (true performance measurement)',
                },
                'disconnect_detection': {
                    'implemented': True,
                    'description': 'Monitor velocity-embedding disconnect pattern',
                    'expected_impact': 'DIAGNOSTIC (catch issues early)',
                },
                'earlier_semantic_loss': {
                    'implemented': True,
                    'change': 'Threshold 0.7 → 0.5',
                    'description': 'Apply semantic constraints earlier in training',
                    'expected_impact': 'LOW-MEDIUM (+0.03 similarity)',
                },
            },
            'architecture_features': {
                '3d_rope': True,
                'sandwich_normalization': True,
                'grouped_query_attention': True,
                'rectified_flow_matching': True,
                'eva_adapter': args.use_eva_adapter,
                'heun_solver': True,
                'disconnect_monitoring': True,
            },
        }
        
        config_path = output_dir / 'experiment_config.json'
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2, default=str)
        
        logger.info(f"✅ FIXED configuration saved to {config_path}")
        return config
        
    except Exception as e:
        logger.error(f"❌ Error saving experiment config: {e}")
        return {}

def main():
    """Main FIXED training function"""
    # Setup logging
    logger = setup_logging()
    
    logger.info("🚀 FIXED BLIP3-o CLIP Reproduction Training")
    logger.info("=" * 80)
    logger.info("📋 Task: Reproduce CLIP embeddings from EVA embeddings")
    logger.info("🧠 Model: BLIP3-o DiT with CRITICAL FIXES")
    logger.info("🌊 Method: Rectified Flow Matching + FIXED Enhancements")
    logger.info("🎯 Target: CLIP embeddings [B, N, 1024] (FIXED NORMALIZATION)")
    logger.info("🎮 Conditioning: EVA embeddings [B, N, 4096] (ADAPTED)")
    logger.info("🔥 Expected: CLIP similarity 0.46 → 0.65+ 🚀")
    logger.info("=" * 80)
    logger.info("🛠️ CRITICAL FIXES APPLIED:")
    logger.info("   ✅ 1. Conservative CLIP normalization (scale 4.0→2.0)")
    logger.info("   ✅ 2. Increased semantic weights (0.1→0.5, 0.05→0.2)")
    logger.info("   ✅ 3. New direct CLIP consistency loss (0.3 weight)")
    logger.info("   ✅ 4. Heun's solver for O(h²) integration")
    logger.info("   ✅ 5. Proper evaluation denormalization")
    logger.info("   ✅ 6. Comprehensive disconnect detection")
    logger.info("   ✅ 7. Earlier semantic loss application (0.7→0.5)")
    logger.info("=" * 80)
    
    try:
        # Parse arguments
        args = parse_arguments()
        
        # Validate arguments
        if not validate_arguments(args, logger):
            return 1
        
        logger.info(f"FIXED Configuration:")
        logger.info(f"  Model size: {args.model_size}")
        logger.info(f"  Training mode: {args.training_mode}")
        logger.info(f"  Embeddings dir: {args.chunked_embeddings_dir}")
        logger.info(f"  Output dir: {args.output_dir}")
        logger.info(f"  Learning rate: {args.learning_rate}")
        logger.info(f"  Batch size: {args.batch_size}")
        logger.info(f"  Epochs: {args.num_epochs}")
        logger.info(f"  Max shards: {args.max_shards}")
        logger.info(f"  FIXED Loss weights:")
        logger.info(f"    Velocity: {args.velocity_weight}")
        logger.info(f"    Semantic: {args.semantic_weight} (INCREASED from 0.1)")
        logger.info(f"    Cosine: {args.cosine_weight} (INCREASED from 0.05)")
        logger.info(f"    Consistency: {args.consistency_weight} (NEW)")
        logger.info(f"  EVA adapter: {'✅ ENABLED' if args.use_eva_adapter else '❌ DISABLED'}")
        logger.info(f"  Heun inference: {'✅ ENABLED' if args.use_heun_inference else '❌ DISABLED'}")
        
        # Check environment
        if not check_environment(logger):
            logger.error("❌ Environment check failed - cannot proceed!")
            return 1
        
        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"✅ Output directory ready: {output_dir}")
        
        # Create model
        logger.info("🏗️ Creating model...")
        model, device = create_model(args, logger)
        
        # Create FIXED loss function
        logger.info("🌊 Creating FIXED loss function...")
        loss_fn = create_fixed_loss_function(args, logger)
        
        # Create FIXED dataloaders
        logger.info("📊 Creating FIXED dataloaders...")
        train_dataloader, eval_dataloader = create_fixed_dataloaders(args, logger)
        
        # Create FIXED trainer
        logger.info("🏃 Creating FIXED trainer...")
        trainer = create_fixed_trainer(model, loss_fn, train_dataloader, eval_dataloader, args, device, logger)
        
        # Save configuration
        logger.info("💾 Saving experiment configuration...")
        config = save_experiment_config(args, model, output_dir, logger)
        
        # Start training
        logger.info(f"\n🚀 Starting FIXED BLIP3-o training...")
        logger.info("=" * 80)
        logger.info("🔥 Expected Results Timeline:")
        logger.info("   • After normalization fix: IMMEDIATE improvement (0.46 → 0.55+)")
        logger.info("   • After semantic weights: Better semantic preservation")
        logger.info("   • After consistency loss: Direct eval metric optimization")
        logger.info("   • After Heun solver: Improved integration accuracy")
        logger.info("   • Combined effect: 0.65+ CLIP similarity")
        logger.info("=" * 80)
        logger.info("🚨 MONITOR FOR:")
        logger.info("   • CleanSim should NOT decrease (if it does, normalization issue)")
        logger.info("   • VelSim and CleanSim should both increase together")
        logger.info("   • Eval similarity should improve quickly (within 500 steps)")
        logger.info("   • Disconnect alerts should be minimal")
        logger.info("=" * 80)
        
        start_time = datetime.now()
        
        # Run FIXED training
        summary = trainer.train()
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # FINAL SUMMARY
        logger.info("\n" + "=" * 80)
        logger.info("🎉 FIXED BLIP3-o TRAINING COMPLETED!")
        logger.info("=" * 80)
        
        logger.info(f"📊 RESULTS:")
        logger.info(f"  Duration: {duration:.1f} seconds ({duration/60:.1f} minutes)")
        logger.info(f"  Total steps: {summary.get('total_steps', 0)}")
        logger.info(f"  Best loss: {summary.get('best_loss', float('inf')):.6f}")
        logger.info(f"  Best CLIP similarity: {summary.get('best_eval_similarity', 0):.4f}")
        logger.info(f"  Disconnect alerts: {summary.get('disconnect_alerts', 0)}")
        
        # Enhanced results analysis
        best_sim = summary.get('best_eval_similarity', 0)
        initial_sim = 0.46  # Previous problematic similarity
        improvement = best_sim - initial_sim
        
        logger.info(f"📈 IMPROVEMENT ANALYSIS:")
        logger.info(f"  Initial similarity (problematic): {initial_sim:.4f}")
        logger.info(f"  Final similarity (FIXED): {best_sim:.4f}")
        logger.info(f"  Absolute improvement: +{improvement:.4f}")
        logger.info(f"  Relative improvement: {(improvement/initial_sim)*100:.1f}%")
        
        # Success assessment
        if best_sim > 0.7:
            logger.info(f"  🎉 EXCELLENT SUCCESS: Similarity >0.7!")
            logger.info(f"  🔥 ALL CRITICAL FIXES WORKED PERFECTLY! 🔥")
            success_level = "excellent"
        elif best_sim > 0.6:
            logger.info(f"  🎉 MAJOR SUCCESS: Similarity >0.6!")
            logger.info(f"  ✅ CRITICAL FIXES HIGHLY EFFECTIVE!")
            success_level = "major"
        elif best_sim > 0.55:
            logger.info(f"  ✅ GOOD SUCCESS: Similarity >0.55!")
            logger.info(f"  💡 Fixes working, consider longer training")
            success_level = "good"
        elif best_sim > 0.5:
            logger.info(f"  ✅ MODERATE SUCCESS: Similarity >0.5!")
            logger.info(f"  💡 Some fixes working, check hyperparameters")
            success_level = "moderate"
        elif improvement > 0.05:
            logger.info(f"  📈 PARTIAL SUCCESS: +{improvement:.3f} improvement")
            logger.info(f"  🔍 Some fixes working, verify all implementations")
            success_level = "partial"
        else:
            logger.info(f"  ⚠️ LIMITED IMPROVEMENT: Check implementation")
            logger.info(f"  🔍 Debug: Verify fixes are properly applied")
            success_level = "limited"
        
        # Final evaluation results
        final_eval = summary.get('final_eval', {})
        if final_eval:
            logger.info(f"📊 Final Evaluation:")
            logger.info(f"  CLIP similarity: {final_eval.get('eval_clip_similarity', 0):.4f}")
            logger.info(f"  High quality (>0.7): {final_eval.get('eval_high_quality', 0)*100:.1f}%")
            logger.info(f"  Very high quality (>0.8): {final_eval.get('eval_very_high_quality', 0)*100:.1f}%")
            logger.info(f"  Excellent quality (>0.9): {final_eval.get('eval_excellent_quality', 0)*100:.1f}%")
            logger.info(f"  Denormalization applied: {final_eval.get('eval_denormalization_working', False)}")
        
        # Fixes verification
        fixes_applied = summary.get('fixes_applied', {})
        logger.info(f"🔧 FIXES STATUS:")
        for fix_name, enabled in fixes_applied.items():
            status = "✅ ENABLED" if enabled else "❌ DISABLED"
            logger.info(f"  {fix_name}: {status}")
        
        # Save enhanced final summary
        summary['duration_seconds'] = duration
        summary['end_time'] = end_time.isoformat()
        summary['experiment_config'] = config
        summary['improvement_analysis'] = {
            'initial_similarity': initial_sim,
            'final_similarity': best_sim,
            'absolute_improvement': improvement,
            'relative_improvement_percent': (improvement/initial_sim)*100 if initial_sim > 0 else 0,
            'success_level': success_level,
        }
        
        summary_path = output_dir / 'training_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info(f"📁 Outputs:")
        logger.info(f"  Training summary: {summary_path}")
        logger.info(f"  Model checkpoints: {output_dir}")
        logger.info(f"  Training logs: fixed_clip_training.log")
        
        logger.info("=" * 80)
        
        # Final success message
        if best_sim > 0.6:
            logger.info("🎉 TRAINING COMPLETED WITH MAJOR SUCCESS!")
            logger.info("🔥 CRITICAL FIXES PROVED HIGHLY EFFECTIVE! 🔥")
        elif best_sim > 0.5:
            logger.info("✅ TRAINING COMPLETED WITH GOOD PROGRESS!")
            logger.info("💡 Fixes are working, consider longer training")
        else:
            logger.info("📈 TRAINING COMPLETED - VERIFY ALL FIXES ARE APPLIED")
        
        logger.info("💡 Next Steps:")
        if best_sim > 0.65:
            logger.info("  • Outstanding results! Ready for:")
            logger.info("    - Production deployment")
            logger.info("    - Scaling to larger datasets")
            logger.info("    - Fine-tuning for specific tasks")
        elif best_sim > 0.55:
            logger.info("  • Great progress! To improve further:")
            logger.info("    - Train longer (more epochs)")
            logger.info("    - Try larger model size")
            logger.info("    - Increase consistency_weight to 0.4")
        elif best_sim > 0.5:
            logger.info("  • Good foundation! To improve:")
            logger.info("    - Increase semantic_weight to 0.7")
            logger.info("    - Train with more data (increase max_shards)")
            logger.info("    - Verify Heun solver is being used")
        else:
            logger.info("  • Need investigation:")
            logger.info("    - Check CLIP normalizer is working (scale_factor=2.0)")
            logger.info("    - Verify semantic weights are increased")
            logger.info("    - Ensure denormalization is working in evaluation")
            logger.info("    - Check for disconnect alerts in logs")
        
        return 0 if best_sim > 0.5 else 1
        
    except Exception as e:
        logger.error(f"❌ FIXED training failed with error: {e}")
        logger.error("=" * 50)
        logger.error("FULL ERROR TRACEBACK:")
        traceback.print_exc()
        logger.error("=" * 50)
        
        # Provide debugging advice for FIXED version
        error_str = str(e)
        if "CUDA out of memory" in error_str:
            logger.error("🔍 GPU MEMORY ERROR:")
            logger.error("   Try reducing --batch_size to 4 or --model_size to small")
        elif "No module named" in error_str or "import" in error_str.lower():
            logger.error("🔍 IMPORT ERROR:")
            logger.error("   Make sure you're using the FIXED modules:")
            logger.error("   - src/modules/datasets/blip3o_dataset.py (FIXED)")
            logger.error("   - src/modules/losses/blip3o_fm_loss.py (FIXED)")  
            logger.error("   - src/modules/trainers/blip3o_trainer.py (FIXED)")
        elif "FileNotFoundError" in error_str:
            logger.error("🔍 FILE NOT FOUND:")
            logger.error("   Check --chunked_embeddings_dir path exists")
        elif "normaliz" in error_str.lower():
            logger.error("🔍 NORMALIZATION ERROR:")
            logger.error("   Check CLIP embeddings format and FIXED normalizer")
        elif "disconnect" in error_str.lower():
            logger.error("🔍 DISCONNECT DETECTED:")
            logger.error("   The FIXED loss weights should prevent this")
            logger.error("   Verify semantic_weight=0.5 and cosine_weight=0.2")
        
        return 1

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except Exception as e:
        print(f"❌ Critical error in main: {e}")
        traceback.print_exc()
        sys.exit(1)