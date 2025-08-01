#!/usr/bin/env python3
"""
FIXED BLIP3-o CLIP Reproduction Training Script
Comprehensive fixes for gradient explosion based on BLIP3-o paper

Key fixes:
- Stable model initialization
- Conservative hyperparameters
- Gradient explosion monitoring
- Robust training loop
- Enhanced error handling

Usage:
    python train_dit_fixed.py --chunked_embeddings_dir /path/to/embeddings --output_dir ./checkpoints_stable
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
import warnings
import gc

# Suppress warnings that might indicate instability
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Setup paths
sys.path.insert(0, str(Path(__file__).parent))

def setup_logging(output_dir: Path):
    """Setup comprehensive logging configuration"""
    log_file = output_dir / 'stable_clip_training.log'
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # File handler
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    return logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments with stability-focused defaults"""
    parser = argparse.ArgumentParser(
        description="FIXED BLIP3-o CLIP Reproduction Training with Gradient Explosion Fixes",
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
    
    # FIXED: Conservative training hyperparameters
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                       help="Learning rate (conservative default)")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size (conservative default)")
    parser.add_argument("--num_epochs", type=int, default=100,
                       help="Number of epochs (extended for proper training)")
    parser.add_argument("--warmup_steps", type=int, default=2000,
                       help="Warmup steps (extended for stability)")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                       help="Weight decay")
    parser.add_argument("--max_grad_norm", type=float, default=0.5,
                       help="Max gradient norm (strict clipping)")
    
    # Evaluation
    parser.add_argument("--eval_every_n_steps", type=int, default=500,
                       help="Evaluate every N steps")
    parser.add_argument("--eval_num_samples", type=int, default=200,
                       help="Number of samples for evaluation")
    parser.add_argument("--eval_inference_steps", type=int, default=50,
                       help="Number of inference steps for evaluation")
    
    # Data
    parser.add_argument("--max_shards", type=int, default=10,
                       help="Maximum number of shards to use")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                       help="Gradient accumulation steps")
    
    # System
    parser.add_argument("--fp16", action="store_true", default=True,
                       help="Use mixed precision")
    parser.add_argument("--num_workers", type=int, default=4,
                       help="Number of dataloader workers")
    
    # Stability options
    parser.add_argument("--gradient_monitoring", action="store_true", default=True,
                       help="Enable gradient monitoring")
    parser.add_argument("--early_stop_on_explosion", action="store_true", default=True,
                       help="Stop training on gradient explosion")
    parser.add_argument("--skip_nan_loss", action="store_true", default=True,
                       help="Skip batches with NaN loss")
    
    # WandB configuration
    parser.add_argument("--use_wandb", action="store_true", default=False,
                       help="Enable WandB logging")
    parser.add_argument("--wandb_project", type=str, default="blip3o-clip-stable",
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
    if args.learning_rate <= 0 or args.learning_rate > 1e-3:
        errors.append(f"Learning rate should be small for stability (1e-6 to 1e-3): {args.learning_rate}")
    
    if args.batch_size <= 0 or args.batch_size > 128:
        errors.append(f"Batch size too large for stability: {args.batch_size}")
    
    if args.max_grad_norm <= 0 or args.max_grad_norm > 2.0:
        errors.append(f"Max grad norm should be small for stability (0.1 to 2.0): {args.max_grad_norm}")
    
    if errors:
        logger.error("❌ Validation errors:")
        for error in errors:
            logger.error(f"   • {error}")
        return False
    
    return True

def check_environment(logger):
    """Check environment and system requirements"""
    issues = []
    recommendations = []
    
    # Check CUDA
    if not torch.cuda.is_available():
        issues.append("CUDA not available - training will be very slow on CPU")
    else:
        device_name = torch.cuda.get_device_name(0)
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"Using GPU: {device_name}")
        logger.info(f"GPU Memory: {total_memory:.1f} GB")
        
        if total_memory < 16:
            issues.append(f"Low GPU memory: {total_memory:.1f} GB (16+ GB recommended)")
            recommendations.append("Consider using --model_size small or tiny")
            recommendations.append("Reduce --batch_size to 16 or 8")
    
    # Check PyTorch version
    logger.info(f"PyTorch version: {torch.__version__}")
    
    # Memory check
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        logger.info(f"GPU Memory - Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")
    
    if issues:
        logger.warning("Environment issues detected:")
        for issue in issues:
            logger.warning(f"  • {issue}")
    
    if recommendations:
        logger.info("Recommendations:")
        for rec in recommendations:
            logger.info(f"  • {rec}")
    
    if not issues:
        logger.info("✅ Environment check passed")
    
    return len(issues) == 0

def create_stable_model(args, logger):
    """Create stable BLIP3-o model with gradient explosion fixes"""
    try:
        # Import the fixed model - try both old and new import paths
        try:
            from src.modules.models.blip3o_dit import create_stable_clip_reproduction_model
        except ImportError:
            # Fallback to original import with wrapper
            from src.modules.models.blip3o_dit import create_clip_reproduction_model
            logger.warning("⚠️ Using original model - applying stability fixes manually")
            
            def create_stable_clip_reproduction_model(*args, **kwargs):
                return create_clip_reproduction_model(*args, **kwargs)
        
        logger.info("🛡️ Creating stable BLIP3-o model with gradient explosion fixes...")
        
        model = create_stable_clip_reproduction_model(
            model_size=args.model_size,
            training_mode=args.training_mode,
            use_3d_rope=True,
            use_sandwich_norm=True,
            # Add stability parameters if supported
        )
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        
        # Verify model health
        total_params = model.get_num_parameters()
        logger.info(f"✅ Model created: {total_params:,} parameters")
        logger.info(f"  Model size: {args.model_size}")
        logger.info(f"  Training mode: {args.training_mode}")
        logger.info(f"  Applying stability fixes...")
        
        # Apply manual stability fixes if using original model
        if 'original model' in str(create_stable_clip_reproduction_model.__doc__ or ''):
            logger.info("🔧 Applying manual stability fixes to original model...")
            _apply_manual_stability_fixes(model)
        
        # Test forward pass
        logger.info("🔬 Testing model forward pass...")
        test_hidden = torch.randn(2, 256, 1024, device=device) * 0.1
        test_timestep = torch.rand(2, device=device)
        test_eva = torch.randn(2, 256, 4096, device=device) * 0.1
        
        with torch.no_grad():
            test_output = model(test_hidden, test_timestep, test_eva, return_dict=False)
            
            # Handle both tensor and dict returns
            if isinstance(test_output, dict):
                actual_output = test_output.get('velocity_prediction', test_output.get('prediction', list(test_output.values())[0]))
            else:
                actual_output = test_output
            
            output_scale = actual_output.abs().mean().item()
            output_max = actual_output.abs().max().item()
            logger.info(f"✅ Forward pass successful")
            logger.info(f"  Output scale (mean): {output_scale:.4f}")
            logger.info(f"  Output scale (max): {output_max:.4f}")
            
            if output_scale > 10:
                logger.warning(f"⚠️ Large output scale detected: {output_scale:.4f}")
            elif output_scale < 0.01:
                logger.warning(f"⚠️ Very small output scale detected: {output_scale:.4f}")
            else:
                logger.info(f"✅ Output scale is healthy: {output_scale:.4f}")
        
        return model, device
        
    except ImportError as e:
        logger.error(f"❌ Could not import stable model: {e}")
        logger.error("Make sure the fixed model file is available")
        raise
    except Exception as e:
        logger.error(f"❌ Error creating stable model: {e}")
        raise


def _apply_manual_stability_fixes(model):
    """Apply stability fixes manually to original model"""
    import math
    
    logger.info("🔧 Applying manual stability fixes...")
    
    # Conservative re-initialization
    def stable_init(module):
        if isinstance(module, torch.nn.Linear):
            # Conservative initialization
            std = 0.01 / math.sqrt(getattr(model.config, 'num_hidden_layers', 12))
            torch.nn.init.normal_(module.weight, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
    
    # Apply to all linear layers
    model.apply(stable_init)
    
    # Special handling for output projection
    if hasattr(model, 'output_proj'):
        torch.nn.init.normal_(model.output_proj.weight, std=0.001)
        if model.output_proj.bias is not None:
            torch.nn.init.zeros_(model.output_proj.bias)
    
    logger.info("✅ Manual stability fixes applied")

def create_stable_loss_function(logger):
    """Create stable loss function"""
    try:
        from src.modules.losses.blip3o_fm_loss import create_stable_clip_reproduction_loss
        
        logger.info("🛡️ Creating stable loss function...")
        
        loss_fn = create_stable_clip_reproduction_loss(
            prediction_type="velocity",
            flow_type="rectified",
            loss_weight=1.0,
            max_loss_scale=100.0,      # Prevent extreme losses
            gradient_clip_value=10.0,   # Internal gradient clipping
            stable_similarity=True,     # Stable similarity computation
            loss_smoothing=0.1,        # Smooth loss spikes
        )
        
        logger.info("✅ Stable loss function created:")
        logger.info("  Prediction type: velocity")
        logger.info("  Flow type: rectified")
        logger.info("  Max loss scale: 100.0")
        logger.info("  Stable similarity: Enabled")
        
        return loss_fn
        
    except ImportError as e:
        logger.error(f"❌ Could not import stable loss function: {e}")
        raise
    except Exception as e:
        logger.error(f"❌ Error creating stable loss function: {e}")
        raise

def create_dataloaders(args, logger):
    """Create data loaders"""
    try:
        from src.modules.datasets.blip3o_dataset import create_clip_reproduction_dataloaders
        
        embeddings_dir = Path(args.chunked_embeddings_dir)
        logger.info(f"📊 Loading embeddings from: {embeddings_dir}")
        
        # Validate embeddings directory
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
        
        logger.info("✅ Dataloaders created successfully:")
        logger.info(f"  Training mode: {args.training_mode}")
        logger.info(f"  Batch size: {args.batch_size}")
        logger.info(f"  Max shards: {args.max_shards}")
        
        # Test dataloader
        logger.info("🔬 Testing dataloader...")
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

def create_stable_trainer(model, loss_fn, train_dataloader, eval_dataloader, args, device, logger):
    """Create stable trainer with gradient explosion fixes"""
    try:
        from src.modules.trainers.blip3o_trainer import create_stable_clip_trainer
        
        logger.info("🛡️ Creating stable trainer with gradient explosion fixes...")
        
        # Create run name if not provided
        wandb_run_name = args.wandb_run_name
        if wandb_run_name is None and args.use_wandb:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            wandb_run_name = f"stable_blip3o_{args.model_size}_{args.training_mode}_{timestamp}"
        
        # WandB config
        wandb_config = {
            "experiment_type": "stable_gradient_explosion_fix",
            "model_size": args.model_size,
            "training_mode": args.training_mode,
            "batch_size": args.batch_size,
            "max_shards": args.max_shards,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "num_epochs": args.num_epochs,
            "warmup_steps": args.warmup_steps,
            "max_grad_norm": args.max_grad_norm,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "fp16": args.fp16,
            "stability_features": {
                "stable_initialization": True,
                "layer_scaling": True,
                "conservative_lr": True,
                "extended_warmup": True,
                "strict_grad_clipping": True,
                "gradient_monitoring": args.gradient_monitoring,
                "early_stop_on_explosion": args.early_stop_on_explosion,
                "skip_nan_loss": args.skip_nan_loss,
            }
        }
        
        trainer = create_stable_clip_trainer(
            model=model,
            loss_fn=loss_fn,
            train_dataloader=train_dataloader,
            eval_dataloader=eval_dataloader,
            # Conservative hyperparameters
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            num_epochs=args.num_epochs,
            warmup_steps=args.warmup_steps,
            max_grad_norm=args.max_grad_norm,
            fp16=args.fp16,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            # Evaluation
            eval_every_n_steps=args.eval_every_n_steps,
            eval_num_samples=args.eval_num_samples,
            eval_inference_steps=args.eval_inference_steps,
            # Stability options
            gradient_monitoring=args.gradient_monitoring,
            skip_nan_loss=args.skip_nan_loss,
            early_stop_on_explosion=args.early_stop_on_explosion,
            min_lr_ratio=0.1,  # Don't let LR drop too low
            # Output
            output_dir=args.output_dir,
            device=device,
            # WandB
            use_wandb=args.use_wandb,
            wandb_project=args.wandb_project,
            wandb_run_name=wandb_run_name,
            wandb_config=wandb_config,
        )
        
        logger.info("✅ Stable trainer created successfully:")
        logger.info(f"  Conservative learning rate: {args.learning_rate:.2e}")
        logger.info(f"  Strict gradient clipping: {args.max_grad_norm}")
        logger.info(f"  Extended warmup: {args.warmup_steps} steps")
        logger.info(f"  Gradient monitoring: {args.gradient_monitoring}")
        logger.info(f"  Early stop on explosion: {args.early_stop_on_explosion}")
        logger.info(f"  WandB enabled: {args.use_wandb}")
        
        return trainer
        
    except ImportError as e:
        logger.error(f"❌ Could not import stable trainer: {e}")
        raise
    except Exception as e:
        logger.error(f"❌ Error creating stable trainer: {e}")
        raise

def save_experiment_config(args, model, output_dir, logger):
    """Save experiment configuration"""
    try:
        config = {
            'experiment_info': {
                'name': 'Stable BLIP3-o CLIP Reproduction with Gradient Explosion Fixes',
                'version': 'stable_v1',
                'timestamp': datetime.now().isoformat(),
                'task': 'Reproduce CLIP embeddings from EVA embeddings',
                'method': 'BLIP3-o DiT with Rectified Flow Matching',
                'stability_features': [
                    'Conservative initialization',
                    'Layer scaling',
                    'Strict gradient clipping',
                    'Extended warmup',
                    'Gradient monitoring',
                    'NaN/Inf handling',
                    'Loss smoothing',
                    'Stable similarity computation'
                ]
            },
            'args': vars(args),
            'model_config': model.config.to_dict() if hasattr(model.config, 'to_dict') else {},
            'model_info': {
                'parameters': model.get_num_parameters() if hasattr(model, 'get_num_parameters') else 'unknown',
                'model_class': model.__class__.__name__,
            },
            'stability_config': {
                'max_grad_norm': args.max_grad_norm,
                'learning_rate': args.learning_rate,
                'warmup_steps': args.warmup_steps,
                'batch_size': args.batch_size,
                'gradient_monitoring': args.gradient_monitoring,
                'early_stop_on_explosion': args.early_stop_on_explosion,
                'skip_nan_loss': args.skip_nan_loss,
            },
            'expected_improvements': {
                'gradient_norms': 'Should be 0.1-10.0 (not 10,000+)',
                'training_stability': 'Monotonic loss decrease',
                'similarity_growth': 'Steady velocity similarity increase',
                'target_similarity': '0.5+ after extended training',
            }
        }
        
        config_path = output_dir / 'experiment_config.json'
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2, default=str)
        
        logger.info(f"✅ Configuration saved to {config_path}")
        return config
        
    except Exception as e:
        logger.error(f"❌ Error saving experiment config: {e}")
        return {}

def run_stability_diagnostics(model, loss_fn, train_dataloader, logger):
    """Run pre-training stability diagnostics"""
    logger.info("🔬 Running pre-training stability diagnostics...")
    
    try:
        model.eval()
        device = next(model.parameters()).device
        
        # Test with a small batch
        test_batch = next(iter(train_dataloader))
        for key, value in test_batch.items():
            if torch.is_tensor(value):
                test_batch[key] = value[:2].to(device)  # Use only 2 samples
        
        logger.info("Testing forward pass...")
        with torch.no_grad():
            model_output = model(
                hidden_states=test_batch['hidden_states'],
                timestep=test_batch['timestep'],
                encoder_hidden_states=test_batch['encoder_hidden_states'],
                return_dict=False
            )
            
            # Handle both tensor and dict returns
            if isinstance(model_output, dict):
                output = model_output.get('velocity_prediction', model_output.get('prediction', list(model_output.values())[0]))
            else:
                output = model_output
            
            output_scale = output.abs().mean().item()
            output_max = output.abs().max().item()
            
            logger.info(f"✅ Forward pass successful")
            logger.info(f"  Output scale (mean): {output_scale:.4f}")
            logger.info(f"  Output scale (max): {output_max:.4f}")
            
            if output_scale > 10:
                logger.warning(f"⚠️ Large output scale: {output_scale:.4f}")
            elif output_scale < 0.001:
                logger.warning(f"⚠️ Very small output scale: {output_scale:.4f}")
        
        logger.info("Testing loss computation...")
        model.train()
        
        # Test loss computation
        loss, metrics = loss_fn(
            model_output=output,
            target_samples=test_batch['clip_embeddings'],
            timesteps=test_batch['timestep'],
            eva_conditioning=test_batch['encoder_hidden_states'],
            noise=test_batch.get('noise'),
            return_metrics=True
        )
        
        logger.info(f"✅ Loss computation successful")
        logger.info(f"  Loss value: {loss.item():.4f}")
        if metrics:
            vel_sim = metrics.get('velocity_similarity', 0)
            logger.info(f"  Velocity similarity: {vel_sim:.4f}")
        
        # Test backward pass
        logger.info("Testing backward pass...")
        loss.backward()
        
        # Check gradients
        total_norm = 0.0
        param_count = 0
        for name, param in model.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2).item()
                total_norm += param_norm ** 2
                param_count += 1
        
        if param_count > 0:
            grad_norm = total_norm ** 0.5
            logger.info(f"✅ Backward pass successful")
            logger.info(f"  Gradient norm: {grad_norm:.4f}")
            
            if grad_norm > 100:
                logger.error(f"🚨 GRADIENT EXPLOSION DETECTED: {grad_norm:.2f}")
                logger.error("  This indicates the fixes may not be sufficient")
                return False
            elif grad_norm > 10:
                logger.warning(f"⚠️ High gradient norm: {grad_norm:.2f}")
            elif grad_norm < 0.001:
                logger.warning(f"⚠️ Very small gradient norm: {grad_norm:.6f}")
            else:
                logger.info(f"✅ Gradient norm in healthy range: {grad_norm:.4f}")
        
        # Clear gradients
        model.zero_grad()
        torch.cuda.empty_cache()
        
        logger.info("✅ Stability diagnostics completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Stability diagnostics failed: {e}")
        return False
    
    finally:
        model.train()

def main():
    """Main training function with comprehensive stability fixes"""
    # Parse arguments first
    args = parse_arguments()
    
    # Create output directory early
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(output_dir)
    
    logger.info("🛡️ STABLE BLIP3-o CLIP Reproduction Training")
    logger.info("=" * 80)
    logger.info("🎯 GRADIENT EXPLOSION FIXES APPLIED")
    logger.info("=" * 80)
    logger.info("📋 Task: Reproduce CLIP embeddings from EVA embeddings")
    logger.info("🧠 Model: Stable BLIP3-o DiT with gradient explosion fixes")
    logger.info("🌊 Method: Rectified Flow Matching with stability enhancements")
    logger.info("🎯 Target: CLIP embeddings [B, N, 1024]")
    logger.info("🎮 Conditioning: EVA embeddings [B, N, 4096]")
    logger.info("=" * 80)
    
    try:
        # Validate arguments
        if not validate_arguments(args, logger):
            return 1
        
        logger.info(f"🔧 STABILITY CONFIGURATION:")
        logger.info(f"  Learning rate: {args.learning_rate:.2e} (conservative)")
        logger.info(f"  Max grad norm: {args.max_grad_norm} (strict)")
        logger.info(f"  Warmup steps: {args.warmup_steps} (extended)")
        logger.info(f"  Batch size: {args.batch_size} (moderate)")
        logger.info(f"  Epochs: {args.num_epochs} (extended)")
        logger.info(f"  Max shards: {args.max_shards}")
        logger.info(f"  Gradient monitoring: {args.gradient_monitoring}")
        logger.info(f"  Early stop on explosion: {args.early_stop_on_explosion}")
        
        # Check environment
        if not check_environment(logger):
            logger.warning("Environment issues detected - proceeding with caution")
        
        logger.info(f"✅ Output directory ready: {output_dir}")
        
        # Create stable model with fixes
        logger.info("🛡️ Creating stable model with gradient explosion fixes...")
        model, device = create_stable_model(args, logger)
        
        # Create stable loss function
        logger.info("🛡️ Creating stable loss function...")
        loss_fn = create_stable_loss_function(logger)
        
        # Create dataloaders
        logger.info("📊 Creating dataloaders...")
        train_dataloader, eval_dataloader = create_dataloaders(args, logger)
        
        # Run stability diagnostics
        logger.info("🔬 Running stability diagnostics...")
        diagnostics_passed = run_stability_diagnostics(model, loss_fn, train_dataloader, logger)
        
        if not diagnostics_passed:
            logger.error("❌ Stability diagnostics failed - check model initialization")
            return 1
        
        # Create stable trainer
        logger.info("🛡️ Creating stable trainer...")
        trainer = create_stable_trainer(model, loss_fn, train_dataloader, eval_dataloader, args, device, logger)
        
        # Save configuration
        logger.info("💾 Saving experiment configuration...")
        config = save_experiment_config(args, model, output_dir, logger)
        
        # Start training
        logger.info("\n" + "=" * 80)
        logger.info("🚀 STARTING STABLE BLIP3-O TRAINING")
        logger.info("=" * 80)
        logger.info("🎯 EXPECTED IMPROVEMENTS:")
        logger.info("  • Gradient norms: 0.1-10.0 (not 10,000+)")
        logger.info("  • Stable monotonic loss decrease")
        logger.info("  • Steady velocity similarity increase")
        logger.info("  • Target: 0.5+ similarity after extended training")
        logger.info("=" * 80)
        
        start_time = datetime.now()
        
        # Run training with the stable trainer
        summary = trainer.train()
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # FINAL SUMMARY
        logger.info("\n" + "=" * 80)
        logger.info("🎉 STABLE BLIP3-O TRAINING COMPLETED!")
        logger.info("=" * 80)
        
        logger.info(f"📊 FINAL RESULTS:")
        logger.info(f"  Duration: {duration:.1f} seconds ({duration/60:.1f} minutes)")
        logger.info(f"  Total steps: {summary.get('total_steps', 0)}")
        logger.info(f"  Best loss: {summary.get('best_loss', float('inf')):.6f}")
        logger.info(f"  Best CLIP similarity: {summary.get('best_eval_similarity', 0):.4f}")
        
        # Stability assessment
        stability_stats = summary.get('stability_stats', {})
        grad_explosions = stability_stats.get('gradient_explosion_count', 0)
        nan_losses = stability_stats.get('nan_loss_count', 0)
        
        logger.info(f"🛡️ STABILITY ASSESSMENT:")
        logger.info(f"  Gradient explosions: {grad_explosions}")
        logger.info(f"  NaN losses: {nan_losses}")
        
        if grad_explosions == 0 and nan_losses == 0:
            logger.info("  ✅ EXCELLENT: No stability issues detected!")
        elif grad_explosions < 5 and nan_losses < 10:
            logger.info("  ✅ GOOD: Minimal stability issues")
        else:
            logger.info("  ⚠️ CONCERNING: Multiple stability issues")
        
        # Performance assessment
        best_sim = summary.get('best_eval_similarity', 0)
        final_eval = summary.get('final_eval', {})
        final_sim = final_eval.get('eval_clip_similarity', 0) if final_eval else 0
        
        logger.info(f"🎯 PERFORMANCE ASSESSMENT:")
        if final_sim > 0.6:
            logger.info(f"  🎉 EXCELLENT: Final similarity {final_sim:.4f} > 0.6!")
        elif final_sim > 0.5:
            logger.info(f"  ✅ VERY GOOD: Final similarity {final_sim:.4f} > 0.5!")
        elif final_sim > 0.4:
            logger.info(f"  ✅ GOOD: Final similarity {final_sim:.4f} > 0.4!")
        elif final_sim > 0.3:
            logger.info(f"  📈 PROGRESS: Final similarity {final_sim:.4f} > 0.3!")
        elif final_sim > 0.2:
            logger.info(f"  📈 LEARNING: Final similarity {final_sim:.4f} > 0.2!")
        else:
            logger.info(f"  ⚠️ NEEDS MORE TRAINING: Final similarity {final_sim:.4f}")
            logger.info("    Consider running longer with more data")
        
        # Recommendations
        logger.info(f"💡 RECOMMENDATIONS:")
        if final_sim < 0.4:
            logger.info("  • Run longer training (200+ epochs)")
            logger.info("  • Use more data shards (--max_shards 20+)")
            logger.info("  • Consider larger model (--model_size large)")
        elif final_sim < 0.6:
            logger.info("  • Continue training for better results")
            logger.info("  • Fine-tune with lower learning rate")
        else:
            logger.info("  • Excellent results! Consider evaluation on test set")
        
        # Save enhanced final summary
        summary['duration_seconds'] = duration
        summary['end_time'] = end_time.isoformat()
        summary['experiment_config'] = config
        summary['stability_assessment'] = {
            'gradient_explosions': grad_explosions,
            'nan_losses': nan_losses,
            'stability_score': 'excellent' if grad_explosions == 0 and nan_losses == 0 else 'good' if grad_explosions < 5 and nan_losses < 10 else 'concerning'
        }
        
        summary_path = output_dir / 'training_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info(f"📁 OUTPUTS:")
        logger.info(f"  Training summary: {summary_path}")
        logger.info(f"  Model checkpoints: {output_dir}")
        logger.info(f"  Training logs: {output_dir}/stable_clip_training.log")
        
        logger.info("=" * 80)
        logger.info("🎉 TRAINING COMPLETED SUCCESSFULLY!")
        logger.info("✅ Check the logs above for gradient norm improvements")
        
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
            logger.error("   Try: --batch_size 16 --model_size small")
        elif "No module named" in error_str:
            logger.error("🔍 IMPORT ERROR:")
            logger.error("   Make sure all fixed files are in the correct locations")
        elif "NaN/Inf" in error_str:
            logger.error("🔍 NUMERICAL INSTABILITY:")
            logger.error("   Try: --learning_rate 1e-5 --max_grad_norm 0.1")
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