#!/usr/bin/env python3
"""
BLIP3-o Training Script WITHOUT CLIP Normalization
train_dit.py

CHANGES:
1. Removed all references to CLIP normalization
2. Works directly with raw CLIP embeddings
3. Simplified training pipeline without normalization concerns
4. Optional simple scaling factor (data-independent)

Usage:
    python train_dit.py --chunked_embeddings_dir /path/to/embeddings --output_dir ./checkpoints_no_norm
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
            logging.FileHandler('blip3o_training_no_norm.log', mode='w')
        ]
    )
    return logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="BLIP3-o CLIP Reproduction Training (NO NORMALIZATION)",
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
                       help="Learning rate")
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
    
    # Loss component weights
    parser.add_argument("--velocity_weight", type=float, default=1.0,
                       help="Weight for velocity prediction loss")
    parser.add_argument("--semantic_weight", type=float, default=0.5,
                       help="Weight for semantic consistency loss")
    parser.add_argument("--cosine_weight", type=float, default=0.2,
                       help="Weight for cosine similarity loss")
    parser.add_argument("--consistency_weight", type=float, default=0.3,
                       help="Weight for direct CLIP consistency loss")
    
    # Data-independent scaling (optional)
    parser.add_argument("--simple_scale_factor", type=float, default=1.0,
                       help="Simple data-independent scaling factor for CLIP embeddings")
    
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
    parser.add_argument("--wandb_project", type=str, default="blip3o-clip-no-norm",
                       help="WandB project name")
    parser.add_argument("--wandb_run_name", type=str, default="3shard_sf1",
                       help="WandB run name")
    
    return parser.parse_args()

def validate_arguments(args, logger):
    """Validate command line arguments"""
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
    
    # Validate scaling factor
    if args.simple_scale_factor <= 0:
        errors.append(f"Scale factor must be positive: {args.simple_scale_factor}")
    
    # Warnings for potentially suboptimal settings
    if args.semantic_weight < 0.3:
        warnings.append(f"Semantic weight ({args.semantic_weight}) seems low - recommend 0.5+")
    
    if args.cosine_weight < 0.15:
        warnings.append(f"Cosine weight ({args.cosine_weight}) seems low - recommend 0.2+")
    
    if args.batch_size > 16:
        warnings.append(f"Large batch size ({args.batch_size}) may cause memory issues")
    
    if args.simple_scale_factor != 1.0:
        warnings.append(f"Using simple scaling factor: {args.simple_scale_factor}")
    
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
    
    # Check for required imports
    missing_modules = []
    
    try:
        from src.modules.datasets.blip3o_dataset import create_clip_reproduction_dataloaders
        logger.info("✅ Dataset module loaded (NO NORMALIZATION)")
    except ImportError as e:
        missing_modules.append(f"Dataset: {e}")
    
    try:
        from src.modules.models.blip3o_dit import create_improved_clip_reproduction_model
        logger.info("✅ Model module loaded")
    except ImportError as e:
        missing_modules.append(f"Model: {e}")
    
    try:
        from src.modules.losses.blip3o_fm_loss import create_clip_reproduction_loss
        logger.info("✅ Loss module loaded (NO NORMALIZATION)")
    except ImportError as e:
        missing_modules.append(f"Loss: {e}")
    
    try:
        from src.modules.trainers.blip3o_trainer import create_clip_trainer
        logger.info("✅ Trainer module loaded (NO NORMALIZATION)")
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
    
    return len(missing_modules) == 0

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

def create_loss_function(args, logger):
    """Create loss function without normalization"""
    try:
        from src.modules.losses.blip3o_fm_loss import create_clip_reproduction_loss
        
        loss_fn = create_clip_reproduction_loss(
            prediction_type="velocity",
            flow_type="rectified",
            velocity_weight=args.velocity_weight,
            semantic_weight=args.semantic_weight,
            cosine_weight=args.cosine_weight,
            consistency_weight=args.consistency_weight,
            use_timestep_weighting=args.use_timestep_weighting,
        )
        
        logger.info("✅ Loss function created (NO NORMALIZATION):")
        logger.info(f"  Prediction type: velocity")
        logger.info(f"  Flow type: rectified")
        logger.info(f"  Weights - Velocity: {args.velocity_weight}, Semantic: {args.semantic_weight}, Cosine: {args.cosine_weight}")
        logger.info(f"  Consistency weight: {args.consistency_weight}")
        logger.info(f"  Timestep weighting: {'✅ ENABLED' if args.use_timestep_weighting else '❌ DISABLED'}")
        logger.info(f"  Normalization: DISABLED")
        
        return loss_fn
        
    except Exception as e:
        logger.error(f"❌ Error creating loss function: {e}")
        raise

def create_dataloaders(args, logger):
    """Create data loaders without normalization"""
    try:
        from src.modules.datasets.blip3o_dataset import create_clip_reproduction_dataloaders
        
        # Validate embeddings directory
        embeddings_dir = Path(args.chunked_embeddings_dir)
        logger.info(f"Loading embeddings from: {embeddings_dir}")
        
        # Look for embedding files
        pkl_files = list(embeddings_dir.glob("*.pkl"))
        if not pkl_files:
            raise FileNotFoundError(f"No .pkl files found in {embeddings_dir}")
        
        logger.info(f"Found {len(pkl_files)} .pkl files in embeddings directory")
        
        # Create dataloaders
        train_dataloader, eval_dataloader = create_clip_reproduction_dataloaders(
            chunked_embeddings_dir=args.chunked_embeddings_dir,
            batch_size=args.batch_size,
            training_mode=args.training_mode,
            max_shards=args.max_shards,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
            simple_scale_factor=args.simple_scale_factor,
            skip_corrupted_samples=True,
            validate_tensor_shapes=True,
        )
        
        logger.info("✅ Dataloaders created successfully (NO NORMALIZATION):")
        logger.info(f"  Training mode: {args.training_mode}")
        logger.info(f"  Batch size: {args.batch_size}")
        logger.info(f"  Max shards: {args.max_shards}")
        logger.info(f"  Simple scale factor: {args.simple_scale_factor}")
        logger.info(f"  CLIP normalization: DISABLED")
        
        # Test dataloader
        try:
            test_batch = next(iter(train_dataloader))
            logger.info(f"✅ Dataloader test successful:")
            logger.info(f"  Batch size: {test_batch.get('batch_size', 'unknown')}")
            logger.info(f"  CLIP embeddings shape: {test_batch['clip_embeddings'].shape}")
            logger.info(f"  EVA embeddings shape: {test_batch['encoder_hidden_states'].shape}")
            
            # Check CLIP embedding range (should be raw)
            sample_clip = test_batch['clip_embeddings']
            clip_range = (sample_clip.min().item(), sample_clip.max().item())
            logger.info(f"  Raw CLIP range: [{clip_range[0]:.3f}, {clip_range[1]:.3f}]")
            
            # Show effect of scaling if applied
            if args.simple_scale_factor != 1.0:
                logger.info(f"  Scale factor applied: {args.simple_scale_factor}")
                
        except Exception as e:
            logger.error(f"❌ Dataloader test failed: {e}")
            raise
        
        return train_dataloader, eval_dataloader
        
    except Exception as e:
        logger.error(f"❌ Error creating dataloaders: {e}")
        raise

def create_trainer(model, loss_fn, train_dataloader, eval_dataloader, args, device, logger):
    """Create trainer without normalization"""
    try:
        from src.modules.trainers.blip3o_trainer import create_clip_trainer
        
        # Create run name if not provided
        wandb_run_name = args.wandb_run_name
        if wandb_run_name is None and args.use_wandb:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            improvements = ["no_norm"]
            if args.use_eva_adapter:
                improvements.append("eva_adapter")
            if args.use_heun_inference:
                improvements.append("heun")
            if args.simple_scale_factor != 1.0:
                improvements.append(f"scale_{args.simple_scale_factor}")
            improvements_str = "_".join(improvements)
            wandb_run_name = f"blip3o_{args.model_size}_{args.training_mode}_{improvements_str}_{timestamp}"
        
        # WandB config
        wandb_config = {
            "model_size": args.model_size,
            "training_mode": args.training_mode,
            "batch_size": args.batch_size,
            "max_shards": args.max_shards,
            "experiment_version": "NO_NORMALIZATION_v1",
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
            "clip_normalization": "DISABLED",
            "simple_scale_factor": args.simple_scale_factor,
            "eva_adapter": args.use_eva_adapter,
            "eva_adapter_layers": args.eva_adapter_layers,
            "heun_inference": args.use_heun_inference,
            "timestep_weighting": args.use_timestep_weighting,
            
            # Approach
            "normalization_approach": "DISABLED",
            "working_space": "raw_clip_embeddings",
            "data_dependent_stats": False,
        }
        
        trainer = create_clip_trainer(
            model=model,
            loss_fn=loss_fn,
            train_dataloader=train_dataloader,
            eval_dataloader=eval_dataloader,
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
        
        logger.info("✅ Trainer created successfully (NO NORMALIZATION):")
        logger.info(f"  Evaluation: Every {args.eval_every_n_steps} steps")
        logger.info(f"  WandB enabled: {args.use_wandb}")
        logger.info(f"  Heun inference: {'✅ ENABLED' if args.use_heun_inference else '❌ DISABLED'}")
        logger.info(f"  Normalization: DISABLED")
        
        return trainer
        
    except Exception as e:
        logger.error(f"❌ Error creating trainer: {e}")
        raise

def save_experiment_config(args, model, output_dir, logger):
    """Save detailed experiment configuration"""
    try:
        config = {
            'experiment_info': {
                'name': 'BLIP3-o CLIP Reproduction WITHOUT Normalization',
                'version': 'NO_NORMALIZATION_v1',
                'timestamp': datetime.now().isoformat(),
                'task': 'Reproduce CLIP embeddings from EVA embeddings',
                'method': 'BLIP3-o DiT without CLIP normalization',
                'focus': 'Training without data-dependent normalization',
            },
            'args': vars(args),
            'model_config': model.config.to_dict() if hasattr(model.config, 'to_dict') else {},
            'model_info': {
                'parameters': model.get_num_parameters() if hasattr(model, 'get_num_parameters') else 'unknown',
                'model_class': model.__class__.__name__,
            },
            'normalization_info': {
                'clip_normalization': 'DISABLED',
                'working_space': 'raw_clip_embeddings',
                'simple_scale_factor': args.simple_scale_factor,
                'data_dependent_stats': False,
                'advantages': [
                    'No dependency on training data statistics',
                    'Simpler training and evaluation pipeline',
                    'No risk of normalization-related crashes',
                    'Direct work with original CLIP space'
                ],
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
    """Main training function without normalization"""
    # Setup logging
    logger = setup_logging()
    
    logger.info("🚀 BLIP3-o CLIP Reproduction Training (NO NORMALIZATION)")
    logger.info("=" * 80)
    logger.info("📋 Task: Reproduce CLIP embeddings from EVA embeddings")
    logger.info("🧠 Model: BLIP3-o DiT WITHOUT CLIP normalization")
    logger.info("🌊 Method: Rectified Flow Matching with raw embeddings")
    logger.info("🎯 Target: CLIP embeddings [B, N, 1024] (RAW)")
    logger.info("🎮 Conditioning: EVA embeddings [B, N, 4096]")
    logger.info("🔑 Focus: Training without data-dependent normalization")
    logger.info("=" * 80)
    logger.info("🛠️ KEY CHANGES:")
    logger.info("   ✅ 1. No CLIP normalization/denormalization")
    logger.info("   ✅ 2. Work directly with raw CLIP embeddings")
    logger.info("   ✅ 3. No dependency on training data statistics")
    logger.info("   ✅ 4. Simplified training and evaluation pipeline")
    logger.info("   ✅ 5. Optional simple data-independent scaling")
    logger.info("=" * 80)
    
    try:
        # Parse arguments
        args = parse_arguments()
        
        # Validate arguments
        if not validate_arguments(args, logger):
            return 1
        
        logger.info(f"Configuration (NO NORMALIZATION):")
        logger.info(f"  Model size: {args.model_size}")
        logger.info(f"  Training mode: {args.training_mode}")
        logger.info(f"  Embeddings dir: {args.chunked_embeddings_dir}")
        logger.info(f"  Output dir: {args.output_dir}")
        logger.info(f"  Learning rate: {args.learning_rate}")
        logger.info(f"  Batch size: {args.batch_size}")
        logger.info(f"  Epochs: {args.num_epochs}")
        logger.info(f"  Max shards: {args.max_shards}")
        logger.info(f"  Simple scale factor: {args.simple_scale_factor}")
        logger.info(f"  Loss weights:")
        logger.info(f"    Velocity: {args.velocity_weight}")
        logger.info(f"    Semantic: {args.semantic_weight}")
        logger.info(f"    Cosine: {args.cosine_weight}")
        logger.info(f"    Consistency: {args.consistency_weight}")
        logger.info(f"  EVA adapter: {'✅ ENABLED' if args.use_eva_adapter else '❌ DISABLED'}")
        logger.info(f"  Heun inference: {'✅ ENABLED' if args.use_heun_inference else '❌ DISABLED'}")
        logger.info(f"  Normalization: DISABLED")
        
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
        loss_fn = create_loss_function(args, logger)
        
        # Create dataloaders
        logger.info("📊 Creating dataloaders (NO NORMALIZATION)...")
        train_dataloader, eval_dataloader = create_dataloaders(args, logger)
        
        # Create trainer
        logger.info("🏃 Creating trainer...")
        trainer = create_trainer(model, loss_fn, train_dataloader, eval_dataloader, args, device, logger)
        
        # Save configuration
        logger.info("💾 Saving experiment configuration...")
        config = save_experiment_config(args, model, output_dir, logger)
        
        # Start training
        logger.info(f"\n🚀 Starting BLIP3-o training (NO NORMALIZATION)...")
        logger.info("=" * 80)
        logger.info("🎯 Expected Results:")
        logger.info("   • Simplified training without normalization concerns")
        logger.info("   • Direct work with original CLIP embedding space")
        logger.info("   • No dependency on training data statistics")
        logger.info("   • Easier debugging and evaluation")
        logger.info("=" * 80)
        
        start_time = datetime.now()
        
        # Run training
        summary = trainer.train()
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # FINAL SUMMARY
        logger.info("\n" + "=" * 80)
        logger.info("🎉 TRAINING COMPLETED (NO NORMALIZATION)!")
        logger.info("=" * 80)
        
        logger.info(f"📊 RESULTS:")
        logger.info(f"  Duration: {duration:.1f} seconds ({duration/60:.1f} minutes)")
        logger.info(f"  Total steps: {summary.get('total_steps', 0)}")
        logger.info(f"  Best loss: {summary.get('best_loss', float('inf')):.6f}")
        logger.info(f"  Best CLIP similarity: {summary.get('best_eval_similarity', 0):.4f}")
        
        # Success assessment
        best_sim = summary.get('best_eval_similarity', 0)
        if best_sim > 0.6:
            logger.info(f"  🎉 EXCELLENT: Similarity >0.6 without normalization!")
            success_level = "excellent"
        elif best_sim > 0.4:
            logger.info(f"  ✅ GOOD: Similarity >0.4 with simplified approach!")
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
            logger.info(f"📊 Final Evaluation (RAW CLIP space):")
            logger.info(f"  CLIP similarity: {final_eval.get('eval_clip_similarity', 0):.4f}")
            logger.info(f"  High quality (>0.7): {final_eval.get('eval_high_quality', 0)*100:.1f}%")
            logger.info(f"  Very high quality (>0.8): {final_eval.get('eval_very_high_quality', 0)*100:.1f}%")
        
        logger.info(f"📁 Outputs:")
        logger.info(f"  Model checkpoints: {output_dir}")
        logger.info(f"  Training logs: blip3o_training_no_norm.log")
        
        logger.info("=" * 80)
        logger.info("✅ TRAINING COMPLETED SUCCESSFULLY (NO NORMALIZATION)!")
        logger.info("🔑 Working directly with raw CLIP embeddings!")
        
        logger.info("💡 Next Steps:")
        if success_level == "excellent":
            logger.info("  • Outstanding! Try scaling up:")
            logger.info("    - Use more shards (--max_shards 10)")
            logger.info("    - Larger model (--model_size large)")
        elif success_level == "good":
            logger.info("  • Great results! To improve performance:")
            logger.info("    - Train longer (--num_epochs 20)")
            logger.info("    - Try larger batch size (--batch_size 16)")
        elif success_level == "fair":
            logger.info("  • Stable foundation! To improve:")
            logger.info("    - Increase semantic weights further")
            logger.info("    - Train with more data")
            logger.info("    - Try simple scaling factor (--simple_scale_factor 0.1)")
        else:
            logger.info("  • Investigate training progression:")
            logger.info("    - Check if loss decreased")
            logger.info("    - Verify gradients are non-zero")
            logger.info("    - Try different scaling factor")
        
        logger.info("=" * 80)
        logger.info("🔧 Advantages of No Normalization:")
        logger.info("  • No dependency on training data statistics")
        logger.info("  • Simpler training and evaluation pipeline")
        logger.info("  • Direct work with original CLIP space")
        logger.info("  • Easier to debug and understand")
        logger.info("  • No risk of normalization-related crashes")
        
        return 0 if success_level in ["excellent", "good", "fair"] else 1
        
    except Exception as e:
        logger.error(f"❌ Training failed with error: {e}")
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
        
        return 1

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except Exception as e:
        print(f"❌ Critical error in main: {e}")
        traceback.print_exc()
        sys.exit(1)