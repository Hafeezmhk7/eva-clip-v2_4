#!/usr/bin/env python3
"""
Clean BLIP3-o CLIP Reproduction Training Script
Simple implementation aligned with BLIP3-o paper

Usage:
    python train_dit_clean.py --chunked_embeddings_dir /path/to/embeddings --output_dir ./checkpoints
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

# Setup paths
sys.path.insert(0, str(Path(__file__).parent))

def setup_logging():
    """Setup simple logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('clean_clip_training.log', mode='w')
        ]
    )
    return logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Clean BLIP3-o CLIP Reproduction Training",
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
    
    # Evaluation
    parser.add_argument("--eval_every_n_steps", type=int, default=50,
                       help="Evaluate every N steps")
    parser.add_argument("--eval_num_samples", type=int, default=100,
                       help="Number of samples for evaluation")
    parser.add_argument("--eval_inference_steps", type=int, default=50,
                       help="Number of inference steps for evaluation")
    
    # Data
    parser.add_argument("--max_shards", type=int, default=None,
                       help="Maximum number of shards to use")
    
    # System
    parser.add_argument("--fp16", action="store_true", default=True,
                       help="Use mixed precision")
    parser.add_argument("--num_workers", type=int, default=0,
                       help="Number of dataloader workers")
    
    # WandB configuration
    parser.add_argument("--use_wandb", action="store_true", default=False,
                       help="Enable WandB logging")
    parser.add_argument("--wandb_project", type=str, default="blip3o-clip-clean",
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
    
    if issues:
        logger.warning("Environment issues detected:")
        for issue in issues:
            logger.warning(f"  • {issue}")
    else:
        logger.info("✅ Environment check passed")
    
    return len(issues) == 0

def create_model(args, logger):
    """Create BLIP3-o model"""
    try:
        from src.modules.models.blip3o_dit import create_clip_reproduction_model
        
        model = create_clip_reproduction_model(
            model_size=args.model_size,
            training_mode=args.training_mode,
            use_3d_rope=True,
            use_sandwich_norm=True,
        )
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        
        logger.info(f"Model created with {model.get_num_parameters():,} parameters")
        logger.info(f"  Model size: {args.model_size}")
        logger.info(f"  Training mode: {args.training_mode}")
        logger.info(f"  3D RoPE: Enabled")
        logger.info(f"  Sandwich Norm: Enabled")
        
        return model, device
        
    except ImportError as e:
        logger.error(f"❌ Could not import model: {e}")
        raise
    except Exception as e:
        logger.error(f"❌ Error creating model: {e}")
        raise

def create_loss_function(logger):
    """Create loss function"""
    try:
        from src.modules.losses.blip3o_fm_loss import create_clip_reproduction_loss
        
        loss_fn = create_clip_reproduction_loss(
            prediction_type="velocity",
            flow_type="rectified",
            loss_weight=1.0
        )
        
        logger.info("Loss function created:")
        logger.info("  Prediction type: velocity")
        logger.info("  Flow type: rectified")
        
        return loss_fn
        
    except ImportError as e:
        logger.error(f"❌ Could not import loss function: {e}")
        raise
    except Exception as e:
        logger.error(f"❌ Error creating loss function: {e}")
        raise

def create_dataloaders(args, logger):
    """Create data loaders"""
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
        
        train_dataloader, eval_dataloader = create_clip_reproduction_dataloaders(
            chunked_embeddings_dir=args.chunked_embeddings_dir,
            batch_size=args.batch_size,
            training_mode=args.training_mode,
            max_shards=args.max_shards,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
            skip_corrupted_samples=True,
            validate_tensor_shapes=True,
        )
        
        logger.info("Dataloaders created successfully:")
        logger.info(f"  Training mode: {args.training_mode}")
        logger.info(f"  Batch size: {args.batch_size}")
        logger.info(f"  Max shards: {args.max_shards}")
        
        # Test dataloader
        test_batch = next(iter(train_dataloader))
        logger.info(f"✅ Dataloader test successful:")
        logger.info(f"  Batch size: {test_batch.get('batch_size', 'unknown')}")
        logger.info(f"  CLIP embeddings shape: {test_batch['clip_embeddings'].shape}")
        logger.info(f"  EVA embeddings shape: {test_batch['encoder_hidden_states'].shape}")
        
        return train_dataloader, eval_dataloader
        
    except ImportError as e:
        logger.error(f"❌ Could not import dataset: {e}")
        raise
    except Exception as e:
        logger.error(f"❌ Error creating dataloaders: {e}")
        raise

def create_trainer(model, loss_fn, train_dataloader, eval_dataloader, args, device, logger):
    """Create trainer"""
    try:
        from src.modules.trainers.blip3o_trainer import create_clip_trainer
        
        # Create run name if not provided
        wandb_run_name = args.wandb_run_name
        if wandb_run_name is None and args.use_wandb:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            wandb_run_name = f"blip3o_{args.model_size}_{args.training_mode}_clean_{timestamp}"
        
        # WandB config
        wandb_config = {
            "model_size": args.model_size,
            "training_mode": args.training_mode,
            "batch_size": args.batch_size,
            "max_shards": args.max_shards,
            "experiment_version": "clean_v1",
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "num_epochs": args.num_epochs,
            "warmup_steps": args.warmup_steps,
            "max_grad_norm": args.max_grad_norm,
            "fp16": args.fp16,
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
            output_dir=args.output_dir,
            device=device,
            use_wandb=args.use_wandb,
            wandb_project=args.wandb_project,
            wandb_run_name=wandb_run_name,
            wandb_config=wandb_config,
        )
        
        logger.info("Trainer created successfully:")
        logger.info(f"  Evaluation: Every {args.eval_every_n_steps} steps")
        logger.info(f"  WandB enabled: {args.use_wandb}")
        
        return trainer
        
    except ImportError as e:
        logger.error(f"❌ Could not import trainer: {e}")
        raise
    except Exception as e:
        logger.error(f"❌ Error creating trainer: {e}")
        raise

def save_experiment_config(args, model, output_dir, logger):
    """Save experiment configuration"""
    try:
        config = {
            'experiment_info': {
                'name': 'Clean BLIP3-o CLIP Reproduction',
                'version': 'clean_v1',
                'timestamp': datetime.now().isoformat(),
                'task': 'Reproduce CLIP embeddings from EVA embeddings',
                'method': 'BLIP3-o DiT with Rectified Flow Matching',
            },
            'args': vars(args),
            'model_config': model.config.to_dict() if hasattr(model.config, 'to_dict') else {},
            'model_info': {
                'parameters': model.get_num_parameters() if hasattr(model, 'get_num_parameters') else 'unknown',
                'model_class': model.__class__.__name__,
            },
            'architecture_features': {
                '3d_rope': True,
                'sandwich_normalization': True,
                'grouped_query_attention': True,
                'rectified_flow_matching': True,
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
    """Main training function"""
    # Setup logging
    logger = setup_logging()
    
    logger.info("🚀 Clean BLIP3-o CLIP Reproduction Training")
    logger.info("=" * 60)
    logger.info("📋 Task: Reproduce CLIP embeddings from EVA embeddings")
    logger.info("🧠 Model: BLIP3-o DiT with 3D RoPE and Sandwich Normalization")
    logger.info("🌊 Method: Rectified Flow Matching")
    logger.info("🎯 Target: CLIP embeddings [B, N, 1024]")
    logger.info("🎮 Conditioning: EVA embeddings [B, N, 4096]")
    logger.info("=" * 60)
    
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
        
        # Check environment
        if not check_environment(logger):
            logger.warning("Environment issues detected - proceeding with caution")
        
        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"✅ Output directory ready: {output_dir}")
        
        # Create model
        logger.info("🏗️ Creating model...")
        model, device = create_model(args, logger)
        
        # Create loss function
        logger.info("🌊 Creating loss function...")
        loss_fn = create_loss_function(logger)
        
        # Create dataloaders
        logger.info("📊 Creating dataloaders...")
        train_dataloader, eval_dataloader = create_dataloaders(args, logger)
        
        # Create trainer
        logger.info("🏃 Creating trainer...")
        trainer = create_trainer(model, loss_fn, train_dataloader, eval_dataloader, args, device, logger)
        
        # Save configuration
        logger.info("💾 Saving experiment configuration...")
        config = save_experiment_config(args, model, output_dir, logger)
        
        # Start training
        logger.info(f"\n🚀 Starting clean BLIP3-o training...")
        logger.info("=" * 60)
        
        start_time = datetime.now()
        
        # Run training
        summary = trainer.train()
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # FINAL SUMMARY
        logger.info("\n" + "=" * 60)
        logger.info("🎉 CLEAN BLIP3-o TRAINING COMPLETED!")
        logger.info("=" * 60)
        
        logger.info(f"📊 RESULTS:")
        logger.info(f"  Duration: {duration:.1f} seconds ({duration/60:.1f} minutes)")
        logger.info(f"  Total steps: {summary.get('total_steps', 0)}")
        logger.info(f"  Best loss: {summary.get('best_loss', float('inf')):.6f}")
        logger.info(f"  Best CLIP similarity: {summary.get('best_eval_similarity', 0):.4f}")
        
        # Enhanced results analysis
        best_sim = summary.get('best_eval_similarity', 0)
        if best_sim > 0.8:
            logger.info(f"  🎉 EXCELLENT: Similarity >0.8 - Great results!")
        elif best_sim > 0.6:
            logger.info(f"  ✅ VERY GOOD: Similarity >0.6 - Solid performance!")
        elif best_sim > 0.4:
            logger.info(f"  ✅ GOOD: Similarity >0.4 - Promising results!")
        elif best_sim > 0.2:
            logger.info(f"  📈 FAIR: Similarity >0.2 - Learning observed!")
        else:
            logger.info(f"  ⚠️ NEEDS WORK: Similarity <0.2 - Check configuration")
        
        # Final evaluation results
        final_eval = summary.get('final_eval', {})
        if final_eval:
            logger.info(f"📊 Final Evaluation:")
            logger.info(f"  CLIP similarity: {final_eval.get('eval_clip_similarity', 0):.4f}")
            logger.info(f"  High quality (>0.7): {final_eval.get('eval_high_quality', 0)*100:.1f}%")
            logger.info(f"  Very high quality (>0.8): {final_eval.get('eval_very_high_quality', 0)*100:.1f}%")
            logger.info(f"  Excellent quality (>0.9): {final_eval.get('eval_excellent_quality', 0)*100:.1f}%")
        
        # Save enhanced final summary
        summary['duration_seconds'] = duration
        summary['end_time'] = end_time.isoformat()
        summary['experiment_config'] = config
        
        summary_path = output_dir / 'training_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info(f"📁 Outputs:")
        logger.info(f"  Training summary: {summary_path}")
        logger.info(f"  Model checkpoints: {output_dir}")
        logger.info(f"  Training logs: clean_clip_training.log")
        
        logger.info("=" * 60)
        logger.info("🎉 TRAINING COMPLETED SUCCESSFULLY!")
        
        return 0
        
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
            logger.error("   Check that all required files are in place")
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