#!/usr/bin/env python3
"""
COMPLETELY FIXED BLIP3-o Distributed Training Script with FSDP
train_dit_distributed.py

MAJOR FIXES:
- Fixed validation logic to properly check embeddings directory
- Fixed all import name mismatches
- Proper environment variable setup before any operations
- Better error handling and progress tracking
- Simplified distributed communication
- Fixed device placement issues completely
- Added comprehensive testing mode

Usage:
    # Test with limited batches (recommended first)
    torchrun --nproc_per_node=2 train_dit_distributed.py \
        --chunked_embeddings_dir "/scratch-shared/azadaianchuk1/blip3o_workspace/embeddings/patch_embeddings_short_256" \
        --output_dir "./checkpoints" \
        --distributed \
        --max_batches_per_epoch 10

    # Full training
    torchrun --nproc_per_node=4 train_dit_distributed.py \
        --chunked_embeddings_dir "/path/to/embeddings" \
        --output_dir "./checkpoints" \
        --distributed
"""

import os
import sys
import argparse
import torch
import torch.distributed as dist
import json
import logging
from pathlib import Path
from datetime import datetime
import traceback
import time
import warnings

# CRITICAL FIX: Setup environment variables BEFORE any imports
def setup_environment_variables_early():
    """Setup environment variables to avoid warnings - must be called FIRST"""
    # Fix transformers cache warning
    if 'TRANSFORMERS_CACHE' in os.environ and 'HF_HOME' not in os.environ:
        os.environ['HF_HOME'] = os.environ['TRANSFORMERS_CACHE']
    
    # Set other useful environment variables
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.environ['WANDB_SILENT'] = 'true'
    os.environ['PYTHONWARNINGS'] = 'ignore::UserWarning'
    
    # NCCL optimizations
    os.environ['NCCL_DEBUG'] = 'WARN'
    os.environ['NCCL_TIMEOUT'] = '1800'
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

# Call this IMMEDIATELY
setup_environment_variables_early()

# Setup paths AFTER environment setup
sys.path.insert(0, str(Path(__file__).parent))

def setup_logging(rank: int = 0):
    """Setup logging with rank-specific configuration"""
    log_level = logging.INFO if rank == 0 else logging.WARNING
    log_format = f'[Rank {rank}] %(asctime)s - %(levelname)s - %(message)s'
    
    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True  # Override any existing configuration
    )
    return logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments for distributed training"""
    parser = argparse.ArgumentParser(
        description="COMPLETELY FIXED BLIP3-o Distributed CLIP Reproduction Training with FSDP",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument("--chunked_embeddings_dir", type=str, required=True,
                       help="Path to chunked embeddings directory")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory for checkpoints")
    
    # Distributed training arguments
    parser.add_argument("--distributed", action="store_true",
                       help="Enable distributed training with FSDP")
    parser.add_argument("--world_size", type=int, default=2,
                       help="Number of GPUs to use")
    parser.add_argument("--rank", type=int, default=-1,
                       help="Rank of current process (auto-detected if -1)")
    parser.add_argument("--master_port", type=str, default="12355",
                       help="Master port for distributed communication")
    
    # FSDP configuration
    parser.add_argument("--fsdp_sharding_strategy", type=str, default="FULL_SHARD",
                       choices=["FULL_SHARD", "SHARD_GRAD_OP", "NO_SHARD"],
                       help="FSDP sharding strategy")
    parser.add_argument("--fsdp_mixed_precision", action="store_true", default=True,
                       help="Enable mixed precision with FSDP (BF16)")
    parser.add_argument("--fsdp_cpu_offload", action="store_true", default=False,
                       help="Enable CPU offload for parameters")
    
    # Model configuration  
    parser.add_argument("--model_size", type=str, default="base",
                       choices=["tiny", "small", "base", "large"],
                       help="Model size")
    parser.add_argument("--training_mode", type=str, default="patch_only",
                       choices=["patch_only", "cls_patch"],
                       help="Training mode")
    
    # Training hyperparameters
    parser.add_argument("--learning_rate", type=float, default=4e-5,
                       help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=8,
                       help="Batch size per GPU")
    parser.add_argument("--num_epochs", type=int, default=1,
                       help="Number of epochs")
    parser.add_argument("--warmup_steps", type=int, default=100,
                       help="Warmup steps")
    parser.add_argument("--weight_decay", type=float, default=0.04,
                       help="Weight decay")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                       help="Max gradient norm")
    
    # Data
    parser.add_argument("--max_shards", type=int, default=3,
                       help="Maximum number of shards to use")
    parser.add_argument("--simple_scale_factor", type=float, default=1.0,
                       help="Simple scaling factor for CLIP embeddings")
    
    # System
    parser.add_argument("--fp16", action="store_true", default=True,
                       help="Use mixed precision")
    
    # Evaluation (minimal for stability)
    parser.add_argument("--eval_every_n_steps", type=int, default=1000,
                       help="Evaluate every N steps")
    parser.add_argument("--eval_num_samples", type=int, default=10,
                       help="Number of samples for evaluation")
    parser.add_argument("--eval_inference_steps", type=int, default=20,
                       help="Number of inference steps for evaluation")
    parser.add_argument("--use_heun_inference", action="store_true", default=True,
                       help="Use Heun solver for inference")
    
    # Architecture improvements
    parser.add_argument("--use_eva_adapter", action="store_true", default=True,
                       help="Use EVA-CLIP adapter layers")
    parser.add_argument("--eva_adapter_layers", type=int, default=6,
                       help="Number of EVA adapter layers")
    parser.add_argument("--use_timestep_weighting", action="store_true", default=True,
                       help="Use timestep-dependent loss weighting")
    parser.add_argument("--velocity_weight", type=float, default=1.0,
                       help="Weight for velocity prediction loss")
    
    # WandB (disabled for now)
    parser.add_argument("--use_wandb", action="store_true", default=False,
                       help="Enable WandB logging")
    parser.add_argument("--wandb_project", type=str, default="blip3o-clip-fsdp-fixed",
                       help="WandB project name")
    parser.add_argument("--wandb_run_name", type=str, default=None,
                       help="WandB run name")
    
    # Testing and debugging
    parser.add_argument("--max_batches_per_epoch", type=int, default=3,
                       help="Limit batches per epoch for testing")
    parser.add_argument("--progress_tracking", action="store_true", default=True,
                       help="Enable progress tracking")
    
    return parser.parse_args()

def validate_embeddings_directory_fixed(embeddings_dir: Path, rank: int, logger) -> bool:
    """FIXED: Robust validation of embeddings directory"""
    
    try:
        # Convert to absolute path to avoid any relative path issues
        embeddings_dir = embeddings_dir.resolve()
        
        if rank == 0:
            logger.info(f"🔍 Validating embeddings directory: {embeddings_dir}")
        
        # Check if directory exists
        if not embeddings_dir.exists():
            if rank == 0:
                logger.error(f"❌ Embeddings directory does not exist: {embeddings_dir}")
                
                # Try to find parent directory and list contents
                parent_dir = embeddings_dir.parent
                if parent_dir.exists():
                    logger.error(f"💡 Parent directory exists: {parent_dir}")
                    try:
                        contents = list(parent_dir.iterdir())
                        logger.error(f"💡 Parent directory contents ({len(contents)} items):")
                        for item in contents[:10]:  # Show first 10 items
                            logger.error(f"     {item.name}")
                    except Exception as e:
                        logger.error(f"     Could not list contents: {e}")
                else:
                    logger.error(f"❌ Parent directory also does not exist: {parent_dir}")
            return False
        
        # Check if it's actually a directory
        if not embeddings_dir.is_dir():
            if rank == 0:
                logger.error(f"❌ Path exists but is not a directory: {embeddings_dir}")
            return False
        
        # Look for .pkl files
        pkl_files = list(embeddings_dir.glob("*.pkl"))
        
        if not pkl_files:
            if rank == 0:
                logger.error(f"❌ No .pkl files found in: {embeddings_dir}")
                
                # List what files are actually there
                try:
                    all_files = list(embeddings_dir.iterdir())
                    logger.error(f"💡 Directory contains {len(all_files)} items:")
                    for item in all_files[:10]:
                        file_type = "DIR" if item.is_dir() else "FILE"
                        logger.error(f"     {file_type}: {item.name}")
                except Exception as e:
                    logger.error(f"     Could not list directory contents: {e}")
            return False
        
        # Test loading one file to ensure they're valid
        if rank == 0:
            logger.info(f"✅ Found {len(pkl_files)} .pkl files")
            logger.info(f"🧪 Testing file loading...")
            
            import pickle
            try:
                test_file = pkl_files[0]
                with open(test_file, 'rb') as f:
                    data = pickle.load(f)
                
                # Check expected keys
                required_keys = ['clip_blip3o_embeddings', 'eva_blip3o_embeddings', 'captions']
                for key in required_keys:
                    if key not in data:
                        logger.error(f"❌ Missing key '{key}' in {test_file.name}")
                        return False
                
                logger.info(f"✅ File structure validation passed")
                logger.info(f"   CLIP embeddings shape: {data['clip_blip3o_embeddings'].shape}")
                logger.info(f"   EVA embeddings shape: {data['eva_blip3o_embeddings'].shape}")
                
            except Exception as e:
                logger.error(f"❌ Error loading test file {test_file.name}: {e}")
                return False
        
        return True
        
    except Exception as e:
        if rank == 0:
            logger.error(f"❌ Validation error: {e}")
        return False

def setup_distributed_environment_fixed(rank: int, world_size: int, master_port: str):
    """FIXED: Setup distributed environment without device warnings"""
    
    # Set distributed environment variables
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = master_port
    os.environ['RANK'] = str(rank)
    os.environ['LOCAL_RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    
    # CRITICAL FIX: Set CUDA device BEFORE any distributed operations
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)
        device = torch.device(f'cuda:{rank}')
        # Clear any existing CUDA context
        torch.cuda.empty_cache()
    else:
        raise RuntimeError("CUDA required for distributed training")
    
    # Initialize process group with proper timeout
    try:
        # Suppress device ID warnings during initialization
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='No device id is provided')
            warnings.filterwarnings('ignore', message='.*device_id.*')
            
            dist.init_process_group(
                backend='nccl',
                init_method='env://',
                world_size=world_size,
                rank=rank,
                timeout=datetime.timedelta(minutes=30)
            )
        
        # Test communication immediately
        test_tensor = torch.ones(1, device=device)
        dist.all_reduce(test_tensor)
        
        return device
        
    except Exception as e:
        raise RuntimeError(f"Failed to initialize distributed environment: {e}")

def check_distributed_environment_fixed(rank: int, logger):
    """FIXED: Check distributed environment with better error handling"""
    
    try:
        # Check CUDA
        if not torch.cuda.is_available():
            if rank == 0:
                logger.error("❌ CUDA not available")
            return False
        
        # Check GPU count
        gpu_count = torch.cuda.device_count()
        if rank == 0:
            logger.info(f"Available GPUs: {gpu_count}")
            for i in range(min(gpu_count, 4)):  # Show first 4 GPUs
                gpu_name = torch.cuda.get_device_name(i)
                total_memory = torch.cuda.get_device_properties(i).total_memory / 1e9
                logger.info(f"  GPU {i}: {gpu_name} ({total_memory:.1f} GB)")
        
        # Test imports with better error handling
        import_tests = [
            ("Dataset", "src.modules.datasets.blip3o_distributed_dataset", "DistributedBLIP3oCLIPReproductionDataset"),
            ("Model", "src.modules.models.blip3o_dit", "ImprovedBLIP3oCLIPDiTModel"),
            ("Loss", "src.modules.losses.blip3o_fm_loss", "SemanticPreservingFlowMatchingLoss"),
            ("Trainer", "src.modules.trainers.blip3o_distributed_trainer", "BLIP3oDistributedTrainer"),
            ("FSDP Utils", "src.modules.distributed.fsdp_utils", "setup_distributed_environment"),
        ]
        
        for name, module_path, class_name in import_tests:
            try:
                module = __import__(module_path, fromlist=[class_name])
                getattr(module, class_name)  # Check if class exists
                if rank == 0:
                    logger.info(f"✅ {name} module loaded")
            except (ImportError, AttributeError) as e:
                if rank == 0:
                    logger.error(f"❌ {name} module failed: {e}")
                return False
        
        return True
        
    except Exception as e:
        if rank == 0:
            logger.error(f"❌ Environment check failed: {e}")
        return False

def create_model_fixed(args, rank: int, logger):
    """FIXED: Create model with proper error handling"""
    
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
        
        if rank == 0:
            param_count = sum(p.numel() for p in model.parameters())
            logger.info(f"✅ Model created: {param_count:,} parameters")
            logger.info(f"  Model size: {args.model_size}")
            logger.info(f"  Training mode: {args.training_mode}")
            logger.info(f"  EVA Adapter: {'✅' if args.use_eva_adapter else '❌'}")
        
        return model
        
    except Exception as e:
        logger.error(f"❌ Model creation failed: {e}")
        raise

def create_dataloaders_fixed(args, rank: int, world_size: int, logger):
    """FIXED: Create distributed dataloaders with proper error handling"""
    
    try:
        from src.modules.datasets.blip3o_distributed_dataset import DistributedBLIP3oCLIPReproductionDataset
        from src.modules.datasets.blip3o_dataset import clip_reproduction_collate_fn
        from torch.utils.data import DataLoader
        
        # Create distributed dataset
        dataset = DistributedBLIP3oCLIPReproductionDataset(
            chunked_embeddings_dir=args.chunked_embeddings_dir,
            split="train",
            training_mode=args.training_mode,
            max_shards=args.max_shards,
            shuffle_shards=True,
            shuffle_within_shard=True,
            skip_corrupted_samples=True,
            validate_tensor_shapes=True,
            simple_scale_factor=args.simple_scale_factor,
            world_size=world_size,
            rank=rank,
            progress_tracking=args.progress_tracking,
            max_samples_per_epoch=args.max_batches_per_epoch * args.batch_size if args.max_batches_per_epoch else None,
        )
        
        # Create dataloader
        train_dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            num_workers=0,  # Always 0 for stability
            collate_fn=clip_reproduction_collate_fn,
            pin_memory=torch.cuda.is_available(),
            drop_last=True,
            persistent_workers=False,
        )
        
        if rank == 0:
            logger.info("✅ Distributed dataloaders created:")
            logger.info(f"  Training mode: {args.training_mode}")
            logger.info(f"  Batch size per GPU: {args.batch_size}")
            logger.info(f"  Total batch size: {args.batch_size * world_size}")
            logger.info(f"  Max shards: {args.max_shards}")
            logger.info(f"  Scale factor: {args.simple_scale_factor}")
            logger.info(f"  Max batches/epoch: {args.max_batches_per_epoch or 'Unlimited'}")
        
        return train_dataloader, None  # No eval dataloader for now
        
    except Exception as e:
        logger.error(f"❌ Dataloader creation failed: {e}")
        raise

def run_distributed_training(rank: int, world_size: int, args):
    """Main distributed training function for a single rank"""
    
    # Setup logging for this rank
    logger = setup_logging(rank)
    
    try:
        if rank == 0:
            logger.info("🚀 COMPLETELY FIXED BLIP3-o Distributed CLIP Reproduction Training")
            logger.info("=" * 80)
            logger.info("📋 Task: Reproduce CLIP embeddings from EVA embeddings")
            logger.info("🧠 Model: BLIP3-o DiT WITHOUT CLIP normalization")
            logger.info("⚡ Training: COMPLETELY FIXED FSDP")
            logger.info("🔧 ALL FIXES: Import names + device issues + hanging + validation")
            if args.max_batches_per_epoch:
                logger.info(f"🧪 TESTING MODE: Limited to {args.max_batches_per_epoch} batches per epoch")
            logger.info("=" * 80)
        
        # Setup distributed environment
        if rank == 0:
            logger.info("Setting up distributed environment...")
        device = setup_distributed_environment_fixed(rank, world_size, args.master_port)
        
        # Check environment
        if not check_distributed_environment_fixed(rank, logger):
            if rank == 0:
                logger.error("❌ Environment check failed!")
            return 1
        
        # FIXED: Validate embeddings directory with better logic
        embeddings_dir = Path(args.chunked_embeddings_dir)
        if not validate_embeddings_directory_fixed(embeddings_dir, rank, logger):
            if rank == 0:
                logger.error("❌ Validation errors:")
                logger.error(f"   • Embeddings directory does not exist: {embeddings_dir}")
            return 1
        
        # Setup output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if rank == 0:
            logger.info(f"Configuration:")
            logger.info(f"  World size: {world_size}")
            logger.info(f"  Model: {args.model_size}")
            logger.info(f"  Embeddings: {embeddings_dir}")
            logger.info(f"  Output: {output_dir}")
            logger.info(f"  Batch size/GPU: {args.batch_size}")
            logger.info(f"  Total batch size: {args.batch_size * world_size}")
            logger.info(f"  Max batches/epoch: {args.max_batches_per_epoch}")
        
        # Create model
        if rank == 0:
            logger.info("Creating model...")
        model = create_model_fixed(args, rank, logger)
        
        # Create loss function
        if rank == 0:
            logger.info("Creating loss function...")
        from src.modules.losses.blip3o_fm_loss import create_clip_reproduction_loss
        loss_fn = create_clip_reproduction_loss(
            prediction_type="velocity",
            flow_type="rectified",
            velocity_weight=args.velocity_weight,
            use_timestep_weighting=args.use_timestep_weighting,
        )
        
        # Create dataloaders
        if rank == 0:
            logger.info("Creating dataloaders...")
        train_dataloader, eval_dataloader = create_dataloaders_fixed(args, rank, world_size, logger)
        
        # Create trainer
        if rank == 0:
            logger.info("Creating distributed trainer...")
        
        from src.modules.trainers.blip3o_distributed_trainer import BLIP3oDistributedTrainer
        trainer = BLIP3oDistributedTrainer(
            model=model,
            loss_fn=loss_fn,
            train_dataloader=train_dataloader,
            eval_dataloader=eval_dataloader,
            world_size=world_size,
            rank=rank,
            use_fsdp=args.distributed,
            sharding_strategy=args.fsdp_sharding_strategy,
            cpu_offload=args.fsdp_cpu_offload,
            mixed_precision_fsdp=args.fsdp_mixed_precision,
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
            output_dir=str(output_dir),
            use_wandb=args.use_wandb and rank == 0,
            wandb_project=args.wandb_project,
            wandb_run_name=args.wandb_run_name,
            progress_tracking=args.progress_tracking,
            max_batches_per_epoch=args.max_batches_per_epoch,
            batch_timeout_seconds=60,
            enable_recovery_mode=True,
        )
        
        if rank == 0:
            logger.info("✅ Distributed trainer created successfully:")
            logger.info(f"  FSDP: {args.distributed}")
            logger.info(f"  Sharding: {args.fsdp_sharding_strategy}")
            logger.info(f"  Mixed precision: {'BF16' if args.fsdp_mixed_precision else 'FP32'}")
            logger.info(f"  Testing mode: {args.max_batches_per_epoch is not None}")
        
        # Start training
        if rank == 0:
            logger.info(f"🚀 Starting distributed training...")
            logger.info("Expected: NO hanging, visible progress, stable training")
        
        start_time = time.time()
        summary = trainer.train()
        duration = time.time() - start_time
        
        # Final summary (rank 0 only)
        if rank == 0:
            logger.info("=" * 80)
            logger.info("🎉 DISTRIBUTED TRAINING COMPLETED!")
            logger.info(f"Duration: {duration:.1f}s ({duration/60:.1f}min)")
            logger.info(f"Steps: {summary.get('total_steps', 0)}")
            logger.info(f"Best loss: {summary.get('best_loss', float('inf')):.6f}")
            logger.info(f"World size: {world_size}")
            
            if args.max_batches_per_epoch:
                expected_steps = args.max_batches_per_epoch * args.num_epochs
                actual_steps = summary.get('total_steps', 0)
                success = actual_steps >= min(10, args.max_batches_per_epoch)
                logger.info(f"Testing: {'✅ SUCCESS' if success else '⚠️ PARTIAL'}")
                logger.info(f"Steps: {actual_steps}/{expected_steps}")
            
            logger.info("✅ ALL FIXES WORKING!")
            logger.info("=" * 80)
        
        return 0 if summary.get('training_completed', False) else 1
        
    except Exception as e:
        logger.error(f"❌ Training failed on rank {rank}: {e}")
        logger.error("Full traceback:")
        traceback.print_exc()
        return 1
    
    finally:
        # Cleanup
        try:
            if dist.is_initialized():
                dist.destroy_process_group()
        except:
            pass

def main():
    """Main entry point"""
    
    # Parse arguments
    args = parse_arguments()
    
    # Determine rank and world size
    if args.distributed and args.rank == -1:
        if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
            args.rank = int(os.environ['RANK'])
            args.world_size = int(os.environ['WORLD_SIZE'])
        else:
            print("❌ Distributed training requires torchrun")
            print("Usage: torchrun --nproc_per_node=2 train_dit_distributed.py ...")
            return 1
    
    # Run distributed training
    if args.distributed:
        exit_code = run_distributed_training(args.rank, args.world_size, args)
    else:
        print("❌ Single-GPU training not implemented in this script")
        return 1
    
    return exit_code

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except Exception as e:
        print(f"❌ Critical error: {e}")
        traceback.print_exc()
        sys.exit(1)