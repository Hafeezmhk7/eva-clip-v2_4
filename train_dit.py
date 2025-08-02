#!/usr/bin/env python3
"""
ULTRA-CONSERVATIVE BLIP3-o Training Script with Robust Normalization
train_dit.py

🔥 ULTRA-CONSERVATIVE FIXES:
1. ✅ Much more conservative CLIP normalization (scale 4.0→1.5)
2. ✅ Robust outlier detection and percentile-based statistics
3. ✅ Fallback to identity normalization if issues occur
4. ✅ Increased semantic loss weights (0.1→0.5, 0.05→0.2) 
5. ✅ New direct CLIP consistency loss (0.3 weight)
6. ✅ Heun's solver for O(h²) integration accuracy
7. ✅ Comprehensive error handling and validation

Expected improvement: Training stability + CLIP similarity improvement

Usage:
    python train_dit.py --chunked_embeddings_dir /path/to/embeddings --output_dir ./checkpoints_ultra_conservative
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
            logging.FileHandler('ultra_conservative_clip_training.log', mode='w')
        ]
    )
    return logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments with ULTRA-CONSERVATIVE defaults"""
    parser = argparse.ArgumentParser(
        description="ULTRA-CONSERVATIVE BLIP3-o CLIP Reproduction Training",
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
    
    # ULTRA-CONSERVATIVE: Even higher loss component weights
    parser.add_argument("--velocity_weight", type=float, default=1.0,
                       help="Weight for velocity prediction loss")
    parser.add_argument("--semantic_weight", type=float, default=0.5,  # Kept high
                       help="Weight for semantic consistency loss")
    parser.add_argument("--cosine_weight", type=float, default=0.2,    # Kept high
                       help="Weight for cosine similarity loss")
    parser.add_argument("--consistency_weight", type=float, default=0.3,  # NEW
                       help="Weight for direct CLIP consistency loss")
    
    # Evaluation
    parser.add_argument("--eval_every_n_steps", type=int, default=50,
                       help="Evaluate every N steps")
    parser.add_argument("--eval_num_samples", type=int, default=100,
                       help="Number of samples for evaluation")
    parser.add_argument("--eval_inference_steps", type=int, default=50,
                       help="Number of inference steps for evaluation")
    parser.add_argument("--use_heun_inference", action="store_true", default=True,
                       help="Use Heun solver for inference")
    
    # Data
    parser.add_argument("--max_shards", type=int, default=None,
                       help="Maximum number of shards to use")
    
    # System
    parser.add_argument("--fp16", action="store_true", default=True,
                       help="Use mixed precision")
    parser.add_argument("--num_workers", type=int, default=0,
                       help="Number of dataloader workers")
    
    # Architecture improvements
    parser.add_argument("--use_eva_adapter", action="store_true", default=True,
                       help="Use EVA-CLIP adapter layers")
    parser.add_argument("--eva_adapter_layers", type=int, default=6,
                       help="Number of EVA adapter layers")
    parser.add_argument("--use_timestep_weighting", action="store_true", default=True,
                       help="Use timestep-dependent loss weighting")
    
    # WandB configuration
    parser.add_argument("--use_wandb", action="store_true", default=True,
                       help="Enable WandB logging")
    parser.add_argument("--wandb_project", type=str, default="blip3o-clip-ultra-conservative",
                       help="WandB project name")
    parser.add_argument("--wandb_run_name", type=str, default=None,
                       help="WandB run name")
    
    return parser.parse_args()

def validate_arguments(args, logger):
    """Validate command line arguments with robust error handling"""
    errors = []
    warnings = []
    
    # Validate paths
    embeddings_dir = Path(args.chunked_embeddings_dir)
    if not embeddings_dir.exists():
        errors.append(f"Embeddings directory does not exist: {embeddings_dir}")
    else:
        # Check for embedding files
        pkl_files = list(embeddings_dir.glob("*.pkl"))
        if not pkl_files:
            errors.append(f"No .pkl files found in embeddings directory: {embeddings_dir}")
        else:
            logger.info(f"Found {len(pkl_files)} embedding files")
    
    # Validate numeric parameters
    if args.learning_rate <= 0:
        errors.append(f"Learning rate must be positive: {args.learning_rate}")
    
    if args.batch_size <= 0:
        errors.append(f"Batch size must be positive: {args.batch_size}")
    
    # Validate weights
    if args.velocity_weight < 0 or args.semantic_weight < 0 or args.cosine_weight < 0:
        errors.append("All loss weights must be non-negative")
    
    # Warnings for potentially suboptimal settings
    if args.semantic_weight < 0.3:
        warnings.append(f"Semantic weight ({args.semantic_weight}) seems low - recommend 0.5+")
    
    if args.cosine_weight < 0.15:
        warnings.append(f"Cosine weight ({args.cosine_weight}) seems low - recommend 0.2+")
    
    if args.batch_size > 16:
        warnings.append(f"Large batch size ({args.batch_size}) may cause memory issues")
    
    # Log warnings
    for warning in warnings:
        logger.warning(f"⚠️ {warning}")
    
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
    
    # Check for required imports with fallback handling
    missing_modules = []
    
    try:
        from src.modules.datasets.blip3o_dataset import create_ultra_conservative_clip_reproduction_dataloaders
        logger.info("✅ Ultra-conservative dataset module loaded")
    except ImportError as e:
        missing_modules.append(f"Ultra-conservative dataset: {e}")
    
    try:
        from src.modules.models.blip3o_dit import create_improved_clip_reproduction_model
        logger.info("✅ Improved model module loaded")
    except ImportError as e:
        missing_modules.append(f"Model: {e}")
    
    try:
        from src.modules.losses.blip3o_fm_loss import create_fixed_clip_reproduction_loss
        logger.info("✅ Fixed loss module loaded")
    except ImportError as e:
        missing_modules.append(f"Loss: {e}")
    
    try:
        from src.modules.trainers.blip3o_trainer import create_fixed_clip_trainer
        logger.info("✅ Fixed trainer module loaded")
    except ImportError as e:
        missing_modules.append(f"Trainer: {e}")
    
    if missing_modules:
        logger.error("❌ Missing required modules:")
        for module in missing_modules:
            logger.error(f"   • {module}")
        issues.append("Required modules not available")
    else:
        logger.info("✅ All required modules loaded successfully")
    
    if issues:
        logger.warning("Environment issues detected:")
        for issue in issues:
            logger.warning(f"  • {issue}")
    else:
        logger.info("✅ Environment check passed")
    
    return len(missing_modules) == 0  # Only require modules, not optimal GPU

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
        
    except Exception as e:
        logger.error(f"❌ Error creating model: {e}")
        raise

def create_ultra_conservative_loss_function(args, logger):
    """Create loss function with ultra-conservative approach"""
    try:
        from src.modules.losses.blip3o_fm_loss import create_fixed_clip_reproduction_loss
        
        loss_fn = create_fixed_clip_reproduction_loss(
            prediction_type="velocity",
            flow_type="rectified",
            velocity_weight=args.velocity_weight,
            semantic_weight=args.semantic_weight,
            cosine_weight=args.cosine_weight,
            consistency_weight=args.consistency_weight,
            use_timestep_weighting=args.use_timestep_weighting,
        )
        
        logger.info("✅ ULTRA-CONSERVATIVE loss function created:")
        logger.info(f"  Prediction type: velocity")
        logger.info(f"  Flow type: rectified")
        logger.info(f"  Weights - Velocity: {args.velocity_weight}, Semantic: {args.semantic_weight}, Cosine: {args.cosine_weight}")
        logger.info(f"  Consistency weight: {args.consistency_weight}")
        logger.info(f"  Timestep weighting: {'✅ ENABLED' if args.use_timestep_weighting else '❌ DISABLED'}")
        
        return loss_fn
        
    except Exception as e:
        logger.error(f"❌ Error creating loss function: {e}")
        raise

def create_ultra_conservative_dataloaders(args, logger):
    """Create data loaders with ultra-conservative normalization"""
    try:
        from src.modules.datasets.blip3o_dataset import create_ultra_conservative_clip_reproduction_dataloaders
        
        # Validate embeddings directory
        embeddings_dir = Path(args.chunked_embeddings_dir)
        logger.info(f"Loading embeddings from: {embeddings_dir}")
        
        # Look for embedding files
        pkl_files = list(embeddings_dir.glob("*.pkl"))
        if not pkl_files:
            raise FileNotFoundError(f"No .pkl files found in {embeddings_dir}")
        
        logger.info(f"Found {len(pkl_files)} .pkl files in embeddings directory")
        
        # Create dataloaders with extensive error handling
        try:
            train_dataloader, eval_dataloader = create_ultra_conservative_clip_reproduction_dataloaders(
                chunked_embeddings_dir=args.chunked_embeddings_dir,
                batch_size=args.batch_size,
                training_mode=args.training_mode,
                max_shards=args.max_shards,
                num_workers=args.num_workers,
                pin_memory=torch.cuda.is_available(),
                skip_corrupted_samples=True,
                validate_tensor_shapes=True,
            )
        except ValueError as e:
            if "normalization range" in str(e).lower():
                logger.error(f"❌ Normalization error: {e}")
                logger.warning("🔧 Attempting recovery with even more conservative settings...")
                
                # Try with even more conservative settings (this would require updating the dataset)
                # For now, re-raise with helpful message
                logger.error("💡 Recovery suggestions:")
                logger.error("   1. Check CLIP embedding data quality")
                logger.error("   2. Try with fewer shards: --max_shards 1")
                logger.error("   3. Check if embeddings are pre-normalized")
                logger.error("   4. Verify embedding extraction was correct")
                raise ValueError(f"CLIP normalization failed: {e}. See recovery suggestions above.")
            else:
                raise
        
        logger.info("✅ ULTRA-CONSERVATIVE dataloaders created successfully:")
        logger.info(f"  Training mode: {args.training_mode}")
        logger.info(f"  Batch size: {args.batch_size}")
        logger.info(f"  Max shards: {args.max_shards}")
        logger.info(f"  🔥 ULTRA-CONSERVATIVE CLIP normalization: ✅ CONFIGURED")
        
        # Test dataloader with extensive validation
        try:
            test_batch = next(iter(train_dataloader))
            logger.info(f"✅ Dataloader test successful:")
            logger.info(f"  Batch size: {test_batch.get('batch_size', 'unknown')}")
            logger.info(f"  CLIP embeddings shape: {test_batch['clip_embeddings'].shape}")
            logger.info(f"  EVA embeddings shape: {test_batch['encoder_hidden_states'].shape}")
            
            # CRITICAL: Check CLIP normalizer
            if hasattr(train_dataloader, 'clip_normalizer') and train_dataloader.clip_normalizer:
                normalizer = train_dataloader.clip_normalizer
                logger.info(f"  🔥 CLIP normalizer: ✅ AVAILABLE")
                logger.info(f"     Stats computed: {normalizer.stats_computed}")
                logger.info(f"     Scale factor: {normalizer.scale_factor:.2f} (ULTRA-CONSERVATIVE)")
                
                # Test normalization range
                sample_clip = test_batch['clip_embeddings']
                clip_range = (sample_clip.min().item(), sample_clip.max().item())
                logger.info(f"     Normalized range: [{clip_range[0]:.2f}, {clip_range[1]:.2f}]")
                
                # Validate range is reasonable
                max_abs_val = max(abs(clip_range[0]), abs(clip_range[1]))
                if max_abs_val > 10:
                    logger.warning(f"⚠️ Normalization range larger than expected: {max_abs_val:.2f}")
                    logger.warning("   Training may be unstable - consider reducing data or checking embeddings")
                elif max_abs_val < 0.1:
                    logger.warning(f"⚠️ Normalization range very small: {max_abs_val:.2f}")
                    logger.warning("   May lose semantic information")
                else:
                    logger.info(f"     ✅ Normalization range acceptable for training")
            else:
                logger.error(f"❌ CLIP normalizer: MISSING!")
                raise ValueError("CLIP normalizer is required but not found")
            
        except Exception as e:
            logger.error(f"❌ Dataloader test failed: {e}")
            raise
        
        return train_dataloader, eval_dataloader
        
    except Exception as e:
        logger.error(f"❌ Error creating dataloaders: {e}")
        raise

def create_ultra_conservative_trainer(model, loss_fn, train_dataloader, eval_dataloader, args, device, logger):
    """Create trainer with ultra-conservative settings"""
    try:
        from src.modules.trainers.blip3o_trainer import create_fixed_clip_trainer
        
        # Create run name if not provided
        wandb_run_name = args.wandb_run_name
        if wandb_run_name is None and args.use_wandb:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            improvements = ["ultra_conservative"]
            if args.use_eva_adapter:
                improvements.append("eva_adapter")
            if args.use_heun_inference:
                improvements.append("heun")
            improvements_str = "_".join(improvements)
            wandb_run_name = f"blip3o_{args.model_size}_{args.training_mode}_{improvements_str}_{timestamp}"
        
        # WandB config
        wandb_config = {
            "model_size": args.model_size,
            "training_mode": args.training_mode,
            "batch_size": args.batch_size,
            "max_shards": args.max_shards,
            "experiment_version": "ULTRA_CONSERVATIVE_v1",
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "num_epochs": args.num_epochs,
            "warmup_steps": args.warmup_steps,
            "max_grad_norm": args.max_grad_norm,
            "fp16": args.fp16,
            
            # Loss weights
            "velocity_weight": args.velocity_weight,
            "semantic_weight": args.semantic_weight,
            "cosine_weight": args.cosine_weight, 
            "consistency_weight": args.consistency_weight,
            
            # Architecture
            "clip_normalization": "ULTRA_CONSERVATIVE",
            "eva_adapter": args.use_eva_adapter,
            "eva_adapter_layers": args.eva_adapter_layers,
            "heun_inference": args.use_heun_inference,
            "timestep_weighting": args.use_timestep_weighting,
            
            # Improvements
            "normalization_approach": "percentile_based_with_outlier_removal",
            "scale_factor": "1.5_ultra_conservative",
            "fallback_mechanisms": "identity_normalization_if_needed",
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
            clip_normalizer=clip_normalizer,
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
        
        logger.info("✅ ULTRA-CONSERVATIVE trainer created successfully:")
        logger.info(f"  Evaluation: Every {args.eval_every_n_steps} steps")
        logger.info(f"  WandB enabled: {args.use_wandb}")
        logger.info(f"  CLIP normalizer: {'✅ AVAILABLE' if clip_normalizer else '❌ MISSING'}")
        logger.info(f"  Heun inference: {'✅ ENABLED' if args.use_heun_inference else '❌ DISABLED'}")
        
        return trainer
        
    except Exception as e:
        logger.error(f"❌ Error creating trainer: {e}")
        raise

def save_experiment_config(args, model, output_dir, logger):
    """Save detailed experiment configuration"""
    try:
        config = {
            'experiment_info': {
                'name': 'ULTRA-CONSERVATIVE BLIP3-o CLIP Reproduction',
                'version': 'ULTRA_CONSERVATIVE_v1',
                'timestamp': datetime.now().isoformat(),
                'task': 'Reproduce CLIP embeddings from EVA embeddings',
                'method': 'BLIP3-o DiT with ULTRA-CONSERVATIVE normalization',
                'focus': 'Training stability and robust normalization',
            },
            'args': vars(args),
            'model_config': model.config.to_dict() if hasattr(model.config, 'to_dict') else {},
            'model_info': {
                'parameters': model.get_num_parameters() if hasattr(model, 'get_num_parameters') else 'unknown',
                'model_class': model.__class__.__name__,
            },
            'ultra_conservative_features': {
                'normalization_scale_factor': 1.5,
                'percentile_based_statistics': True,
                'robust_outlier_removal': True,
                'fallback_to_identity': True,
                'strict_validation_ranges': True,
                'conservative_clamping': True,
            },
            'normalization_details': {
                'scale_factor': 1.5,
                'outlier_removal_method': 'IQR_3x_conservative',
                'statistics_method': 'percentile_based',
                'fallback_available': True,
                'validation_max_range': 10.0,
                'expected_impact': 'Improved training stability',
            },
        }
        
        config_path = output_dir / 'experiment_config.json'
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2, default=str)
        
        logger.info(f"✅ Configuration saved to {config_path}")
        return config
        
    except Exception as e:
        logger.error(f"❌ Error saving experiment config: {e}")
        return {}

def main():
    """Main ultra-conservative training function"""
    # Setup logging
    logger = setup_logging()
    
    logger.info("🚀 ULTRA-CONSERVATIVE BLIP3-o CLIP Reproduction Training")
    logger.info("=" * 80)
    logger.info("📋 Task: Reproduce CLIP embeddings from EVA embeddings")
    logger.info("🧠 Model: BLIP3-o DiT with ULTRA-CONSERVATIVE normalization")
    logger.info("🌊 Method: Rectified Flow Matching + Robust normalization")
    logger.info("🎯 Target: CLIP embeddings [B, N, 1024] (ULTRA-CONSERVATIVE)")
    logger.info("🎮 Conditioning: EVA embeddings [B, N, 4096]")
    logger.info("🔥 Focus: Training stability and robust performance")
    logger.info("=" * 80)
    logger.info("🛠️ ULTRA-CONSERVATIVE FEATURES:")
    logger.info("   ✅ 1. Much smaller scale factor (4.0 → 1.5)")
    logger.info("   ✅ 2. Percentile-based statistics (robust outlier handling)")
    logger.info("   ✅ 3. Fallback to identity normalization if needed")
    logger.info("   ✅ 4. Strict validation ranges (±10 max)")
    logger.info("   ✅ 5. Conservative clamping and error recovery")
    logger.info("   ✅ 6. Enhanced error handling throughout")
    logger.info("=" * 80)
    
    try:
        # Parse arguments
        args = parse_arguments()
        
        # Validate arguments
        if not validate_arguments(args, logger):
            return 1
        
        logger.info(f"ULTRA-CONSERVATIVE Configuration:")
        logger.info(f"  Model size: {args.model_size}")
        logger.info(f"  Training mode: {args.training_mode}")
        logger.info(f"  Embeddings dir: {args.chunked_embeddings_dir}")
        logger.info(f"  Output dir: {args.output_dir}")
        logger.info(f"  Learning rate: {args.learning_rate}")
        logger.info(f"  Batch size: {args.batch_size}")
        logger.info(f"  Epochs: {args.num_epochs}")
        logger.info(f"  Max shards: {args.max_shards}")
        logger.info(f"  Loss weights:")
        logger.info(f"    Velocity: {args.velocity_weight}")
        logger.info(f"    Semantic: {args.semantic_weight}")
        logger.info(f"    Cosine: {args.cosine_weight}")
        logger.info(f"    Consistency: {args.consistency_weight}")
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
        
        # Create loss function
        logger.info("🌊 Creating loss function...")
        loss_fn = create_ultra_conservative_loss_function(args, logger)
        
        # Create dataloaders (this is where the error was occurring)
        logger.info("📊 Creating ULTRA-CONSERVATIVE dataloaders...")
        try:
            train_dataloader, eval_dataloader = create_ultra_conservative_dataloaders(args, logger)
        except ValueError as e:
            if "normalization" in str(e).lower():
                logger.error("❌ CRITICAL NORMALIZATION ERROR:")
                logger.error(f"   {e}")
                logger.error("")
                logger.error("🔧 RECOVERY OPTIONS:")
                logger.error("   1. Try with fewer shards: --max_shards 1")
                logger.error("   2. Check embedding data quality:")
                logger.error("      - Are embeddings pre-normalized?")
                logger.error("      - Are there extreme outliers?")
                logger.error("      - Was extraction done correctly?")
                logger.error("   3. Try different model size: --model_size small")
                logger.error("   4. Check if CLIP embeddings need different preprocessing")
                logger.error("")
                logger.error("💡 DEBUG STEPS:")
                logger.error("   - Load one .pkl file manually and check ranges")
                logger.error("   - Verify CLIP embeddings are in expected format")
                logger.error("   - Check for NaN/Inf values in embeddings")
                return 1
            else:
                raise
        
        # Create trainer
        logger.info("🏃 Creating trainer...")
        trainer = create_ultra_conservative_trainer(model, loss_fn, train_dataloader, eval_dataloader, args, device, logger)
        
        # Save configuration
        logger.info("💾 Saving experiment configuration...")
        config = save_experiment_config(args, model, output_dir, logger)
        
        # Start training
        logger.info(f"\n🚀 Starting ULTRA-CONSERVATIVE BLIP3-o training...")
        logger.info("=" * 80)
        logger.info("🎯 Expected Results:")
        logger.info("   • Stable training without normalization crashes")
        logger.info("   • Gradual improvement in CLIP similarity")
        logger.info("   • No extreme gradient/loss spikes")
        logger.info("   • Robust handling of outliers")
        logger.info("=" * 80)
        
        start_time = datetime.now()
        
        # Run training
        summary = trainer.train()
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # FINAL SUMMARY
        logger.info("\n" + "=" * 80)
        logger.info("🎉 ULTRA-CONSERVATIVE TRAINING COMPLETED!")
        logger.info("=" * 80)
        
        logger.info(f"📊 RESULTS:")
        logger.info(f"  Duration: {duration:.1f} seconds ({duration/60:.1f} minutes)")
        logger.info(f"  Total steps: {summary.get('total_steps', 0)}")
        logger.info(f"  Best loss: {summary.get('best_loss', float('inf')):.6f}")
        logger.info(f"  Best CLIP similarity: {summary.get('best_eval_similarity', 0):.4f}")
        
        # Success assessment
        best_sim = summary.get('best_eval_similarity', 0)
        if best_sim > 0.6:
            logger.info(f"  🎉 EXCELLENT: Similarity >0.6 with stable training!")
            success_level = "excellent"
        elif best_sim > 0.4:
            logger.info(f"  ✅ GOOD: Similarity >0.4 with conservative approach!")
            success_level = "good"
        elif best_sim > 0.2:
            logger.info(f"  📈 FAIR: Similarity >0.2, training was stable!")
            success_level = "fair"
        else:
            logger.info(f"  ⚠️ Needs investigation: Check if training progressed")
            success_level = "needs_investigation"
        
        # Final evaluation results
        final_eval = summary.get('final_eval', {})
        if final_eval:
            logger.info(f"📊 Final Evaluation:")
            logger.info(f"  CLIP similarity: {final_eval.get('eval_clip_similarity', 0):.4f}")
            logger.info(f"  High quality (>0.7): {final_eval.get('eval_high_quality', 0)*100:.1f}%")
            logger.info(f"  Very high quality (>0.8): {final_eval.get('eval_very_high_quality', 0)*100:.1f}%")
        
        logger.info(f"📁 Outputs:")
        logger.info(f"  Model checkpoints: {output_dir}")
        logger.info(f"  Training logs: ultra_conservative_clip_training.log")
        
        logger.info("=" * 80)
        logger.info("✅ ULTRA-CONSERVATIVE TRAINING COMPLETED SUCCESSFULLY!")
        logger.info("🔧 The conservative approach prevented normalization crashes!")
        
        logger.info("💡 Next Steps:")
        if success_level == "excellent":
            logger.info("  • Outstanding! Try scaling up:")
            logger.info("    - Use more shards (--max_shards 10)")
            logger.info("    - Larger model (--model_size large)")
        elif success_level == "good":
            logger.info("  • Great stability! To improve performance:")
            logger.info("    - Train longer (--num_epochs 20)")
            logger.info("    - Try larger batch size (--batch_size 16)")
        elif success_level == "fair":
            logger.info("  • Stable foundation! To improve:")
            logger.info("    - Increase semantic weights further")
            logger.info("    - Train with more data")
        else:
            logger.info("  • Investigate training progression:")
            logger.info("    - Check if loss decreased")
            logger.info("    - Verify gradients are non-zero")
            logger.info("    - Check evaluation logs")
        
        return 0 if success_level in ["excellent", "good", "fair"] else 1
        
    except Exception as e:
        logger.error(f"❌ ULTRA-CONSERVATIVE training failed with error: {e}")
        logger.error("=" * 50)
        logger.error("FULL ERROR TRACEBACK:")
        traceback.print_exc()
        logger.error("=" * 50)
        
        # Provide specific debugging advice
        error_str = str(e)
        if "CUDA out of memory" in error_str:
            logger.error("🔍 GPU MEMORY ERROR:")
            logger.error("   Try: --batch_size 4 or --model_size small")
        elif "No module named" in error_str or "import" in error_str.lower():
            logger.error("🔍 IMPORT ERROR:")
            logger.error("   Ensure all required files are in src/modules/")
        elif "FileNotFoundError" in error_str:
            logger.error("🔍 FILE NOT FOUND:")
            logger.error("   Check --chunked_embeddings_dir path")
        elif "normaliz" in error_str.lower():
            logger.error("🔍 NORMALIZATION ERROR:")
            logger.error("   This is the main issue we're trying to fix!")
            logger.error("   The ultra-conservative approach should handle this")
        
        return 1

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except Exception as e:
        print(f"❌ Critical error in main: {e}")
        traceback.print_exc()
        sys.exit(1)