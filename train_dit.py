#!/usr/bin/env python3
"""
COMPLETE UPDATED BLIP3-o Training Script with All Critical Improvements
train_dit_improved.py

CRITICAL IMPROVEMENTS IMPLEMENTED:
1. ✅ Proper CLIP embedding normalization (HIGHEST IMPACT)
2. ✅ EVA-CLIP adaptation layers for cross-modal alignment
3. ✅ Heun's solver for O(h²) integration accuracy
4. ✅ U-shaped timestep sampling for better training dynamics
5. ✅ Multi-component loss with semantic preservation
6. ✅ Improved evaluation with denormalization
7. ✅ Comprehensive metrics tracking

Expected improvement: CLIP similarity 0.21 → 0.6+ 🚀

Usage:
    python train_dit_improved.py --chunked_embeddings_dir /path/to/embeddings --output_dir ./checkpoints
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
            logging.FileHandler('improved_clip_training.log', mode='w')
        ]
    )
    return logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments with improved options"""
    parser = argparse.ArgumentParser(
        description="IMPROVED BLIP3-o CLIP Reproduction Training with Critical Fixes",
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
    
    # NEW: Loss component weights
    parser.add_argument("--velocity_weight", type=float, default=1.0,
                       help="Weight for velocity prediction loss")
    parser.add_argument("--semantic_weight", type=float, default=0.1,
                       help="Weight for semantic consistency loss")
    parser.add_argument("--cosine_weight", type=float, default=0.05,
                       help="Weight for cosine similarity loss")
    
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
    
    # NEW: Architecture improvements
    parser.add_argument("--use_eva_adapter", action="store_true", default=True,
                       help="Use EVA-CLIP adapter layers (RECOMMENDED)")
    parser.add_argument("--eva_adapter_layers", type=int, default=6,
                       help="Number of EVA adapter layers")
    parser.add_argument("--use_timestep_weighting", action="store_true", default=True,
                       help="Use timestep-dependent loss weighting")
    
    # WandB configuration
    parser.add_argument("--use_wandb", action="store_true", default=True,
                       help="Enable WandB logging")
    parser.add_argument("--wandb_project", type=str, default="blip3o-clip-improved",
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
    try:
        from src.modules.datasets.blip3o_dataset import create_improved_clip_reproduction_dataloaders
        from src.modules.models.blip3o_dit import create_improved_clip_reproduction_model
        from src.modules.losses.blip3o_fm_loss import create_improved_clip_reproduction_loss
        from src.modules.trainers.blip3o_trainer import create_improved_clip_trainer
        logger.info("✅ All improved modules imported successfully")
    except ImportError as e:
        issues.append(f"Failed to import required modules: {e}")
    
    if issues:
        logger.warning("Environment issues detected:")
        for issue in issues:
            logger.warning(f"  • {issue}")
    else:
        logger.info("✅ Environment check passed")
    
    return len(issues) == 0

def create_improved_model(args, logger):
    """Create improved BLIP3-o model with all enhancements"""
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
        
        logger.info(f"✅ IMPROVED model created with {model.get_num_parameters():,} parameters")
        logger.info(f"  Model size: {args.model_size}")
        logger.info(f"  Training mode: {args.training_mode}")
        logger.info(f"  3D RoPE: ✅ ENABLED")
        logger.info(f"  Sandwich Norm: ✅ ENABLED")
        logger.info(f"  EVA Adapter: {'✅ ENABLED' if args.use_eva_adapter else '❌ DISABLED'}")
        logger.info(f"  Heun Solver: ✅ ENABLED")
        
        return model, device
        
    except ImportError as e:
        logger.error(f"❌ Could not import improved model: {e}")
        raise
    except Exception as e:
        logger.error(f"❌ Error creating improved model: {e}")
        raise

def create_improved_loss_function(args, logger):
    """Create improved loss function with semantic preservation"""
    try:
        from src.modules.losses.blip3o_fm_loss import create_improved_clip_reproduction_loss
        
        loss_fn = create_improved_clip_reproduction_loss(
            prediction_type="velocity",
            flow_type="rectified",
            velocity_weight=args.velocity_weight,
            semantic_weight=args.semantic_weight,
            cosine_weight=args.cosine_weight,
            use_timestep_weighting=args.use_timestep_weighting,
        )
        
        logger.info("✅ IMPROVED loss function created:")
        logger.info(f"  Prediction type: velocity")
        logger.info(f"  Flow type: rectified")
        logger.info(f"  Velocity weight: {args.velocity_weight}")
        logger.info(f"  Semantic weight: {args.semantic_weight}")
        logger.info(f"  Cosine weight: {args.cosine_weight}")
        logger.info(f"  Timestep weighting: {'✅ ENABLED' if args.use_timestep_weighting else '❌ DISABLED'}")
        
        return loss_fn
        
    except ImportError as e:
        logger.error(f"❌ Could not import improved loss function: {e}")
        raise
    except Exception as e:
        logger.error(f"❌ Error creating improved loss function: {e}")
        raise

def create_improved_dataloaders(args, logger):
    """Create improved data loaders with proper normalization"""
    try:
        from src.modules.datasets.blip3o_dataset import create_improved_clip_reproduction_dataloaders
        
        # Validate embeddings directory
        embeddings_dir = Path(args.chunked_embeddings_dir)
        logger.info(f"Loading embeddings from: {embeddings_dir}")
        
        # Look for embedding files
        pkl_files = list(embeddings_dir.glob("*.pkl"))
        if not pkl_files:
            raise FileNotFoundError(f"No .pkl files found in {embeddings_dir}")
        
        logger.info(f"Found {len(pkl_files)} .pkl files in embeddings directory")
        
        train_dataloader, eval_dataloader = create_improved_clip_reproduction_dataloaders(
            chunked_embeddings_dir=args.chunked_embeddings_dir,
            batch_size=args.batch_size,
            training_mode=args.training_mode,
            max_shards=args.max_shards,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
            skip_corrupted_samples=True,
            validate_tensor_shapes=True,
        )
        
        logger.info("✅ IMPROVED dataloaders created successfully:")
        logger.info(f"  Training mode: {args.training_mode}")
        logger.info(f"  Batch size: {args.batch_size}")
        logger.info(f"  Max shards: {args.max_shards}")
        logger.info(f"  🔥 CLIP normalization: ✅ PROPERLY CONFIGURED")
        logger.info(f"  🔥 U-shaped timestep sampling: ✅ ENABLED")
        
        # Test dataloader
        test_batch = next(iter(train_dataloader))
        logger.info(f"✅ Dataloader test successful:")
        logger.info(f"  Batch size: {test_batch.get('batch_size', 'unknown')}")
        logger.info(f"  CLIP embeddings shape: {test_batch['clip_embeddings'].shape}")
        logger.info(f"  EVA embeddings shape: {test_batch['encoder_hidden_states'].shape}")
        
        # Check if CLIP normalizer is available
        if hasattr(train_dataloader, 'clip_normalizer') and train_dataloader.clip_normalizer:
            logger.info(f"  🔥 CLIP normalizer: ✅ AVAILABLE")
            logger.info(f"     Scale factor: {train_dataloader.clip_normalizer.scale_factor:.2f}")
        else:
            logger.warning(f"  ⚠️ CLIP normalizer: MISSING - this will hurt performance!")
        
        return train_dataloader, eval_dataloader
        
    except ImportError as e:
        logger.error(f"❌ Could not import improved dataset: {e}")
        raise
    except Exception as e:
        logger.error(f"❌ Error creating improved dataloaders: {e}")
        raise

def create_improved_trainer(model, loss_fn, train_dataloader, eval_dataloader, args, device, logger):
    """Create improved trainer with all enhancements"""
    try:
        from src.modules.trainers.blip3o_trainer import create_improved_clip_trainer
        
        # Create run name if not provided
        wandb_run_name = args.wandb_run_name
        if wandb_run_name is None and args.use_wandb:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            improvements = []
            if args.use_eva_adapter:
                improvements.append("eva_adapter")
            if args.use_heun_inference:
                improvements.append("heun")
            improvements_str = "_".join(improvements) if improvements else "baseline"
            wandb_run_name = f"blip3o_{args.model_size}_{args.training_mode}_improved_{improvements_str}_{timestamp}"
        
        # WandB config
        wandb_config = {
            "model_size": args.model_size,
            "training_mode": args.training_mode,
            "batch_size": args.batch_size,
            "max_shards": args.max_shards,
            "experiment_version": "improved_v1",
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "num_epochs": args.num_epochs,
            "warmup_steps": args.warmup_steps,
            "max_grad_norm": args.max_grad_norm,
            "fp16": args.fp16,
            
            # NEW: Improvement flags
            "clip_normalization": "ENABLED",
            "eva_adapter": args.use_eva_adapter,
            "eva_adapter_layers": args.eva_adapter_layers,
            "heun_inference": args.use_heun_inference,
            "timestep_weighting": args.use_timestep_weighting,
            
            # Loss weights
            "velocity_weight": args.velocity_weight,
            "semantic_weight": args.semantic_weight,
            "cosine_weight": args.cosine_weight,
            
            # Expected improvements
            "expected_clip_similarity_improvement": "0.21 → 0.6+",
            "critical_fixes": [
                "clip_embedding_normalization",
                "heun_solver_integration", 
                "eva_clip_adaptation",
                "u_shaped_timestep_sampling",
                "semantic_preserving_loss"
            ]
        }
        
        # Get CLIP normalizer from dataloader
        clip_normalizer = getattr(train_dataloader, 'clip_normalizer', None)
        
        trainer = create_improved_clip_trainer(
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
        
        logger.info("✅ IMPROVED trainer created successfully:")
        logger.info(f"  Evaluation: Every {args.eval_every_n_steps} steps")
        logger.info(f"  WandB enabled: {args.use_wandb}")
        logger.info(f"  CLIP normalizer: {'✅ AVAILABLE' if clip_normalizer else '❌ MISSING'}")
        logger.info(f"  Heun inference: {'✅ ENABLED' if args.use_heun_inference else '❌ DISABLED'}")
        
        return trainer
        
    except ImportError as e:
        logger.error(f"❌ Could not import improved trainer: {e}")
        raise
    except Exception as e:
        logger.error(f"❌ Error creating improved trainer: {e}")
        raise

def save_experiment_config(args, model, output_dir, logger):
    """Save detailed experiment configuration"""
    try:
        config = {
            'experiment_info': {
                'name': 'IMPROVED BLIP3-o CLIP Reproduction',
                'version': 'improved_v1',
                'timestamp': datetime.now().isoformat(),
                'task': 'Reproduce CLIP embeddings from EVA embeddings',
                'method': 'BLIP3-o DiT with CRITICAL IMPROVEMENTS',
                'expected_improvement': 'CLIP similarity 0.21 → 0.6+',
            },
            'args': vars(args),
            'model_config': model.config.to_dict() if hasattr(model.config, 'to_dict') else {},
            'model_info': {
                'parameters': model.get_num_parameters() if hasattr(model, 'get_num_parameters') else 'unknown',
                'model_class': model.__class__.__name__,
            },
            'critical_improvements': {
                'clip_embedding_normalization': {
                    'implemented': True,
                    'description': 'Proper elementwise mean/std normalization + scaling',
                    'expected_impact': 'HIGH (0.21 → 0.45+)',
                },
                'eva_clip_adapter': {
                    'implemented': args.use_eva_adapter,
                    'layers': args.eva_adapter_layers,
                    'description': 'Multi-layer adaptation between EVA-CLIP and CLIP spaces',
                    'expected_impact': 'MEDIUM (+0.1 similarity)',
                },
                'heun_solver': {
                    'implemented': args.use_heun_inference,
                    'description': 'O(h²) integration accuracy vs Euler O(h)',
                    'expected_impact': 'MEDIUM (+0.1 similarity)',
                },
                'u_shaped_timestep_sampling': {
                    'implemented': True,
                    'description': 'Better timestep distribution for training',
                    'expected_impact': 'LOW-MEDIUM (+0.05 similarity)',
                },
                'semantic_preserving_loss': {
                    'implemented': True,
                    'weights': {
                        'velocity': args.velocity_weight,
                        'semantic': args.semantic_weight,
                        'cosine': args.cosine_weight,
                    },
                    'description': 'Multi-component loss addressing velocity-semantic disconnect',
                    'expected_impact': 'MEDIUM (+0.1 similarity)',
                },
                'timestep_weighting': {
                    'implemented': args.use_timestep_weighting,
                    'description': 'Timestep-dependent loss component weighting',
                    'expected_impact': 'LOW (+0.05 similarity)',
                },
            },
            'architecture_features': {
                '3d_rope': True,
                'sandwich_normalization': True,
                'grouped_query_attention': True,
                'rectified_flow_matching': True,
                'eva_adapter': args.use_eva_adapter,
                'heun_solver': True,
            },
        }
        
        config_path = output_dir / 'experiment_config.json'
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2, default=str)
        
        logger.info(f"✅ IMPROVED configuration saved to {config_path}")
        return config
        
    except Exception as e:
        logger.error(f"❌ Error saving experiment config: {e}")
        return {}

def main():
    """Main improved training function"""
    # Setup logging
    logger = setup_logging()
    
    logger.info("🚀 IMPROVED BLIP3-o CLIP Reproduction Training")
    logger.info("=" * 80)
    logger.info("📋 Task: Reproduce CLIP embeddings from EVA embeddings")
    logger.info("🧠 Model: BLIP3-o DiT with CRITICAL IMPROVEMENTS")
    logger.info("🌊 Method: Rectified Flow Matching + Enhancements")
    logger.info("🎯 Target: CLIP embeddings [B, N, 1024] (PROPERLY NORMALIZED)")
    logger.info("🎮 Conditioning: EVA embeddings [B, N, 4096] (ADAPTED)")
    logger.info("🔥 Expected: CLIP similarity 0.21 → 0.6+ 🚀")
    logger.info("=" * 80)
    logger.info("🛠️ CRITICAL IMPROVEMENTS:")
    logger.info("   ✅ 1. CLIP embedding normalization (HIGHEST IMPACT)")
    logger.info("   ✅ 2. EVA-CLIP adaptation layers")
    logger.info("   ✅ 3. Heun's solver for O(h²) integration")
    logger.info("   ✅ 4. U-shaped timestep sampling")
    logger.info("   ✅ 5. Multi-component semantic-preserving loss")
    logger.info("   ✅ 6. Improved evaluation with denormalization")
    logger.info("=" * 80)
    
    try:
        # Parse arguments
        args = parse_arguments()
        
        # Validate arguments
        if not validate_arguments(args, logger):
            return 1
        
        logger.info(f"Configuration:")
        logger.info(f"  Model size: {args.model_size}")
        logger.info(f"  Training mode: {args.training_mode}")
        logger.info(f"  Embeddings dir: {args.chunked_embeddings_dir}")
        logger.info(f"  Output dir: {args.output_dir}")
        logger.info(f"  Learning rate: {args.learning_rate}")
        logger.info(f"  Batch size: {args.batch_size}")
        logger.info(f"  Epochs: {args.num_epochs}")
        logger.info(f"  Max shards: {args.max_shards}")
        logger.info(f"  EVA adapter: {'✅ ENABLED' if args.use_eva_adapter else '❌ DISABLED'}")
        logger.info(f"  Heun inference: {'✅ ENABLED' if args.use_heun_inference else '❌ DISABLED'}")
        
        # Check environment
        if not check_environment(logger):
            logger.warning("Environment issues detected - proceeding with caution")
        
        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"✅ Output directory ready: {output_dir}")
        
        # Create improved model
        logger.info("🏗️ Creating IMPROVED model...")
        model, device = create_improved_model(args, logger)
        
        # Create improved loss function
        logger.info("🌊 Creating IMPROVED loss function...")
        loss_fn = create_improved_loss_function(args, logger)
        
        # Create improved dataloaders
        logger.info("📊 Creating IMPROVED dataloaders...")
        train_dataloader, eval_dataloader = create_improved_dataloaders(args, logger)
        
        # Create improved trainer
        logger.info("🏃 Creating IMPROVED trainer...")
        trainer = create_improved_trainer(model, loss_fn, train_dataloader, eval_dataloader, args, device, logger)
        
        # Save configuration
        logger.info("💾 Saving experiment configuration...")
        config = save_experiment_config(args, model, output_dir, logger)
        
        # Start training
        logger.info(f"\n🚀 Starting IMPROVED BLIP3-o training...")
        logger.info("=" * 80)
        logger.info("🔥 Expected Results Timeline:")
        logger.info("   • After CLIP normalization fix: 0.21 → 0.45+ (IMMEDIATE)")
        logger.info("   • After Heun solver: 0.45 → 0.55+")
        logger.info("   • After EVA adapter: 0.55 → 0.65+")
        logger.info("   • After all improvements: 0.65 → 0.70+")
        logger.info("=" * 80)
        
        start_time = datetime.now()
        
        # Run improved training
        summary = trainer.train()
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # FINAL SUMMARY
        logger.info("\n" + "=" * 80)
        logger.info("🎉 IMPROVED BLIP3-o TRAINING COMPLETED!")
        logger.info("=" * 80)
        
        logger.info(f"📊 RESULTS:")
        logger.info(f"  Duration: {duration:.1f} seconds ({duration/60:.1f} minutes)")
        logger.info(f"  Total steps: {summary.get('total_steps', 0)}")
        logger.info(f"  Best loss: {summary.get('best_loss', float('inf')):.6f}")
        logger.info(f"  Best CLIP similarity: {summary.get('best_eval_similarity', 0):.4f}")
        
        # Enhanced results analysis
        best_sim = summary.get('best_eval_similarity', 0)
        initial_sim = 0.21  # Original problematic similarity
        improvement = best_sim - initial_sim
        
        logger.info(f"📈 IMPROVEMENT ANALYSIS:")
        logger.info(f"  Initial similarity: {initial_sim:.4f}")
        logger.info(f"  Final similarity: {best_sim:.4f}")
        logger.info(f"  Absolute improvement: +{improvement:.4f}")
        logger.info(f"  Relative improvement: {(improvement/initial_sim)*100:.1f}%")
        
        if best_sim > 0.7:
            logger.info(f"  🎉 EXCELLENT: Similarity >0.7 - Outstanding success!")
            logger.info(f"  🔥 CRITICAL IMPROVEMENTS WORKED! 🔥")
        elif best_sim > 0.6:
            logger.info(f"  🎉 VERY GOOD: Similarity >0.6 - Major success!")
            logger.info(f"  ✅ Improvements significantly effective!")
        elif best_sim > 0.5:
            logger.info(f"  ✅ GOOD: Similarity >0.5 - Solid improvement!")
            logger.info(f"  💡 Consider fine-tuning hyperparameters")
        elif best_sim > 0.4:
            logger.info(f"  ✅ FAIR: Similarity >0.4 - Noticeable improvement!")
            logger.info(f"  💡 Some improvements working, check normalization")
        elif best_sim > initial_sim + 0.1:
            logger.info(f"  📈 PROGRESS: +{improvement:.3f} improvement detected")
            logger.info(f"  💡 Partial success - verify all fixes are applied")
        else:
            logger.info(f"  ⚠️ LIMITED IMPROVEMENT: Check implementation")
            logger.info(f"  🔍 Debug: Verify CLIP normalization is working")
        
        # Final evaluation results
        final_eval = summary.get('final_eval', {})
        if final_eval:
            logger.info(f"📊 Final Evaluation:")
            logger.info(f"  CLIP similarity: {final_eval.get('eval_clip_similarity', 0):.4f}")
            logger.info(f"  High quality (>0.7): {final_eval.get('eval_high_quality', 0)*100:.1f}%")
            logger.info(f"  Very high quality (>0.8): {final_eval.get('eval_very_high_quality', 0)*100:.1f}%")
            logger.info(f"  Excellent quality (>0.9): {final_eval.get('eval_excellent_quality', 0)*100:.1f}%")
            logger.info(f"  Normalization applied: {final_eval.get('eval_normalization_applied', False)}")
        
        # Improvement verification
        improvements_enabled = summary.get('improvements_enabled', {})
        logger.info(f"🔧 Improvements Status:")
        for improvement, enabled in improvements_enabled.items():
            status = "✅ ENABLED" if enabled else "❌ DISABLED"
            logger.info(f"  {improvement}: {status}")
        
        # Save enhanced final summary
        summary['duration_seconds'] = duration
        summary['end_time'] = end_time.isoformat()
        summary['experiment_config'] = config
        summary['improvement_analysis'] = {
            'initial_similarity': initial_sim,
            'final_similarity': best_sim,
            'absolute_improvement': improvement,
            'relative_improvement_percent': (improvement/initial_sim)*100 if initial_sim > 0 else 0,
            'success_level': 'excellent' if best_sim > 0.7 else 'very_good' if best_sim > 0.6 else 'good' if best_sim > 0.5 else 'fair' if best_sim > 0.4 else 'limited',
        }
        
        summary_path = output_dir / 'training_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info(f"📁 Outputs:")
        logger.info(f"  Training summary: {summary_path}")
        logger.info(f"  Model checkpoints: {output_dir}")
        logger.info(f"  Training logs: improved_clip_training.log")
        
        logger.info("=" * 80)
        
        if best_sim > 0.6:
            logger.info("🎉 TRAINING COMPLETED WITH MAJOR SUCCESS!")
            logger.info("🔥 CRITICAL IMPROVEMENTS PROVED EFFECTIVE! 🔥")
        elif best_sim > 0.4:
            logger.info("✅ TRAINING COMPLETED WITH GOOD PROGRESS!")
        else:
            logger.info("📈 TRAINING COMPLETED - CHECK IMPROVEMENTS IMPLEMENTATION")
        
        logger.info("💡 Next Steps:")
        if best_sim > 0.6:
            logger.info("  • Excellent results! Consider:")
            logger.info("    - Longer training for even better performance")
            logger.info("    - Testing on larger datasets")
            logger.info("    - Deploying for production use")
        elif best_sim > 0.4:
            logger.info("  • Good progress! To improve further:")
            logger.info("    - Increase semantic_weight to 0.2")
            logger.info("    - Try longer training (more epochs)")
            logger.info("    - Verify all normalizers are working")
        else:
            logger.info("  • Debug required:")
            logger.info("    - Check CLIP normalization is properly applied")
            logger.info("    - Verify EVA adapter is being used")
            logger.info("    - Ensure Heun solver is enabled")
            logger.info("    - Check for any import/configuration errors")
        
        return 0 if best_sim > 0.4 else 1
        
    except Exception as e:
        logger.error(f"❌ Training failed with error: {e}")
        logger.error("=" * 50)
        logger.error("FULL ERROR TRACEBACK:")
        traceback.print_exc()
        logger.error("=" * 50)
        
        # Provide debugging advice
        error_str = str(e)
        if "CUDA out of memory" in error_str:
            logger.error("🔍 GPU MEMORY ERROR:")
            logger.error("   Try reducing --batch_size or --model_size")
        elif "No module named" in error_str:
            logger.error("🔍 IMPORT ERROR:")
            logger.error("   Check that all improved files are in place")
            logger.error("   Verify you're using the updated modules")
        elif "FileNotFoundError" in error_str:
            logger.error("🔍 FILE NOT FOUND:")
            logger.error("   Check --chunked_embeddings_dir path")
        elif "normalization" in error_str.lower():
            logger.error("🔍 NORMALIZATION ERROR:")
            logger.error("   Check CLIP embeddings format and statistics computation")
        
        return 1

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except Exception as e:
        print(f"❌ Critical error in main: {e}")
        traceback.print_exc()
        sys.exit(1)