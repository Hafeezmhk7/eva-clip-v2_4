#!/usr/bin/env python3
"""
Multi-GPU Embedding Extraction for BLIP3-o
src/modules/extract_embeddings_distributed.py

Distributes TAR file processing across multiple GPUs for faster preprocessing.
Each GPU processes assigned TAR files independently with coordinated saving.
"""

import os
import sys
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import argparse
from pathlib import Path
import json
import time
import logging
from typing import List, Optional, Dict, Any
import pickle
from tqdm import tqdm

# Setup paths
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import single-GPU extraction functions
from src.modules.extract_embeddings_g import (
    load_models, 
    extract_clip_features_with_cls,
    extract_eva_features_with_cls,
    setup_temp_manager,
    find_data_files,
    cleanup_memory
)

logger = logging.getLogger(__name__)


def setup_distributed(rank: int, world_size: int, master_port: str = "12355"):
    """Initialize distributed training"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = master_port
    os.environ['RANK'] = str(rank)
    os.environ['LOCAL_RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    
    # Initialize the process group
    dist.init_process_group(
        backend='nccl',
        init_method=f'env://',
        world_size=world_size,
        rank=rank
    )
    
    # Set CUDA device
    torch.cuda.set_device(rank)
    device = torch.device(f'cuda:{rank}')
    
    if rank == 0:
        logger.info(f"✅ Distributed initialized: {world_size} GPUs")
    
    return device


def cleanup_distributed():
    """Clean up distributed training"""
    if dist.is_initialized():
        dist.destroy_process_group()


def distribute_tar_files(tar_files: List[str], world_size: int, rank: int) -> List[str]:
    """Distribute TAR files across GPUs"""
    assigned_files = []
    for i, tar_file in enumerate(tar_files):
        if i % world_size == rank:
            assigned_files.append(tar_file)
    
    logger.info(f"Rank {rank}: Assigned {len(assigned_files)} TAR files")
    return assigned_files


def get_gpu_specific_output_path(base_path: Path, rank: int, shard_idx: int, mode_suffix: str) -> Path:
    """Generate GPU-specific output paths to prevent conflicts"""
    return base_path / f"embeddings_shard_{shard_idx:05d}_{mode_suffix}_gpu{rank}.pkl"


def process_tar_files_on_gpu(
    rank: int,
    world_size: int,
    tar_files: List[str],
    output_dir: Path,
    working_dir: Path,
    batch_size: int = 32,
    include_cls: bool = True,
    target_tokens: int = 257,
    master_port: str = "12355"
):
    """Process assigned TAR files on a specific GPU"""
    
    # Setup distributed
    device = setup_distributed(rank, world_size, master_port)
    
    # Setup logging for this rank
    log_format = f'[GPU {rank}] %(asctime)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_format)
    rank_logger = logging.getLogger(f'rank_{rank}')
    
    try:
        rank_logger.info(f"Starting extraction on GPU {rank}")
        
        # Load models on this GPU
        clip_processor, clip_model, eva_processor, eva_model = load_models(device)
        rank_logger.info(f"Models loaded on GPU {rank}")
        
        # Get assigned TAR files
        assigned_files = distribute_tar_files(tar_files, world_size, rank)
        
        if not assigned_files:
            rank_logger.info(f"No files assigned to GPU {rank}")
            return
        
        # Process each assigned TAR file
        mode_suffix = "cls_patch" if include_cls else "patch_only"
        total_samples = 0
        processed_files = 0
        
        for shard_idx, tar_file in enumerate(assigned_files):
            rank_logger.info(f"Processing TAR file {shard_idx + 1}/{len(assigned_files)}: {Path(tar_file).name}")
            
            # Generate unique output path for this GPU
            actual_shard_idx = tar_files.index(tar_file)  # Global shard index
            output_path = get_gpu_specific_output_path(
                output_dir, rank, actual_shard_idx, mode_suffix
            )
            
            # Check if already processed
            if output_path.exists():
                rank_logger.info(f"Shard already exists, skipping: {output_path.name}")
                continue
            
            # Import single-GPU processing function
            from src.modules.extract_embeddings_g import process_single_tar
            
            # Process this TAR file
            result = process_single_tar(
                tar_file_path=tar_file,
                shard_idx=actual_shard_idx,
                clip_processor=clip_processor,
                clip_model=clip_model,
                eva_processor=eva_processor,
                eva_model=eva_model,
                device=device,
                output_dir=output_dir,
                working_dir=working_dir / f"gpu_{rank}",
                batch_size=batch_size,
                include_cls=include_cls,
                target_tokens=target_tokens
            )
            
            if result and result['success']:
                total_samples += result['total_samples']
                processed_files += 1
                rank_logger.info(f"✅ Completed shard {actual_shard_idx}: {result['total_samples']} samples")
            else:
                rank_logger.error(f"❌ Failed to process shard {actual_shard_idx}")
            
            # Cleanup memory after each file
            cleanup_memory()
        
        rank_logger.info(f"GPU {rank} completed: {processed_files} files, {total_samples} samples")
        
        # Synchronize all GPUs before consolidation
        if dist.is_initialized():
            dist.barrier()
            rank_logger.info(f"GPU {rank} reached synchronization barrier")
        
    except Exception as e:
        rank_logger.error(f"Error on GPU {rank}: {e}")
        raise
    
    finally:
        cleanup_distributed()


def consolidate_gpu_outputs(
    output_dir: Path,
    world_size: int,
    mode_suffix: str,
    total_shards: int
) -> Dict[str, Any]:
    """Consolidate outputs from all GPUs into final shard files"""
    
    logger.info("🔄 Consolidating GPU outputs...")
    
    consolidation_results = {
        'consolidated_shards': 0,
        'total_samples': 0,
        'consolidation_errors': 0,
        'final_files': []
    }
    
    for shard_idx in range(total_shards):
        # Look for GPU-specific files for this shard
        gpu_files = []
        shard_data_parts = []
        
        for rank in range(world_size):
            gpu_output_path = get_gpu_specific_output_path(
                output_dir, rank, shard_idx, mode_suffix
            )
            
            if gpu_output_path.exists():
                try:
                    with open(gpu_output_path, 'rb') as f:
                        shard_data = pickle.load(f)
                    shard_data_parts.append(shard_data)
                    gpu_files.append(gpu_output_path)
                except Exception as e:
                    logger.warning(f"⚠️ Error loading {gpu_output_path}: {e}")
                    consolidation_results['consolidation_errors'] += 1
        
        # Consolidate if we have data from any GPU
        if shard_data_parts:
            try:
                # Use the first part as base
                consolidated_data = shard_data_parts[0].copy()
                
                # If multiple GPUs processed the same shard (shouldn't happen), merge
                if len(shard_data_parts) > 1:
                    logger.warning(f"Multiple GPU outputs for shard {shard_idx}, merging...")
                    
                    all_clip = [part['clip_blip3o_embeddings'] for part in shard_data_parts]
                    all_eva = [part['eva_blip3o_embeddings'] for part in shard_data_parts]
                    all_captions = []
                    all_keys = []
                    
                    for part in shard_data_parts:
                        all_captions.extend(part['captions'])
                        all_keys.extend(part['keys'])
                    
                    consolidated_data.update({
                        'clip_blip3o_embeddings': torch.cat(all_clip, dim=0),
                        'eva_blip3o_embeddings': torch.cat(all_eva, dim=0),
                        'captions': all_captions,
                        'keys': all_keys,
                        'total_samples': sum(part['total_samples'] for part in shard_data_parts)
                    })
                
                # Save consolidated shard
                final_output_path = output_dir / f"embeddings_shard_{shard_idx:05d}_{mode_suffix}.pkl"
                with open(final_output_path, 'wb') as f:
                    pickle.dump(consolidated_data, f, protocol=pickle.HIGHEST_PROTOCOL)
                
                consolidation_results['consolidated_shards'] += 1
                consolidation_results['total_samples'] += consolidated_data['total_samples']
                consolidation_results['final_files'].append(str(final_output_path))
                
                logger.info(f"✅ Consolidated shard {shard_idx}: {consolidated_data['total_samples']} samples")
                
                # Clean up GPU-specific files
                for gpu_file in gpu_files:
                    try:
                        gpu_file.unlink()
                        logger.debug(f"Cleaned up: {gpu_file.name}")
                    except Exception as e:
                        logger.warning(f"⚠️ Could not clean up {gpu_file}: {e}")
                
            except Exception as e:
                logger.error(f"❌ Error consolidating shard {shard_idx}: {e}")
                consolidation_results['consolidation_errors'] += 1
    
    logger.info(f"✅ Consolidation completed:")
    logger.info(f"   Consolidated shards: {consolidation_results['consolidated_shards']}")
    logger.info(f"   Total samples: {consolidation_results['total_samples']:,}")
    logger.info(f"   Errors: {consolidation_results['consolidation_errors']}")
    
    return consolidation_results


def create_distributed_manifest(
    output_dir: Path,
    consolidation_results: Dict[str, Any],
    world_size: int,
    include_cls: bool,
    target_tokens: int,
    processing_time: float
):
    """Create manifest for distributed extraction"""
    
    manifest_data = {
        'extraction_info': {
            'method': 'distributed_multi_gpu',
            'world_size': world_size,
            'extraction_time_seconds': processing_time,
            'timestamp': time.time()
        },
        'consolidation_results': consolidation_results,
        'token_info': {
            'tokens_per_sample': target_tokens,
            'include_cls': include_cls,
            'cls_token_position': 0 if include_cls else None,
            'patch_tokens_range': [1, 257] if include_cls else [0, 256],
        },
        'format_version': f'blip3o_{target_tokens}_tokens_{"cls_" if include_cls else ""}patch_distributed_v1',
        'total_shards': consolidation_results['consolidated_shards'],
        'total_samples': consolidation_results['total_samples'],
        'usage': {
            'training_command': f'python train_dit_distributed.py --chunked_embeddings_dir {output_dir} --distributed',
        }
    }
    
    manifest_path = output_dir / "embeddings_manifest.json"
    with open(manifest_path, 'w') as f:
        json.dump(manifest_data, f, indent=2)
    
    logger.info(f"✅ Distributed manifest saved: {manifest_path}")
    return manifest_path


def main():
    """Main distributed embedding extraction"""
    
    parser = argparse.ArgumentParser(description="Multi-GPU Embedding Extraction for BLIP3-o")
    parser.add_argument("--world_size", type=int, default=4,
                       help="Number of GPUs to use")
    parser.add_argument("--master_port", type=str, default="12355",
                       help="Master port for distributed communication")
    parser.add_argument("--include_cls", action="store_true", default=False,
                       help="Include CLS token (257 tokens) or patches only (256 tokens)")
    parser.add_argument("--max_shards", type=int, default=None,
                       help="Maximum number of shards to process")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size per GPU")
    
    args = parser.parse_args()
    
    # Setup
    target_tokens = 257 if args.include_cls else 256
    mode_name = "CLS+Patches" if args.include_cls else "Patches only"
    
    print("🚀 Multi-GPU BLIP3-o Embedding Extraction")
    print("=" * 70)
    print(f"GPUs: {args.world_size}")
    print(f"Mode: {mode_name} ({target_tokens} tokens)")
    print(f"Batch size per GPU: {args.batch_size}")
    print("=" * 70)
    
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for multi-GPU extraction")
    
    if torch.cuda.device_count() < args.world_size:
        raise RuntimeError(f"Requested {args.world_size} GPUs, but only {torch.cuda.device_count()} available")
    
    # Setup temp manager
    temp_manager = setup_temp_manager()
    
    if temp_manager:
        mode_suffix = "cls_patch" if args.include_cls else "patch_only"
        embeddings_dir = temp_manager.create_embeddings_subdirectory(f"{mode_suffix}_{target_tokens}_tokens")
        working_dir = temp_manager.get_working_dir()
        temp_manager.setup_model_cache()
        
        print(f"✅ Using structured temp management")
        print(f"📁 Embeddings dir: {embeddings_dir}")
        print(f"📁 Working dir: {working_dir}")
    else:
        # Fallback
        mode_suffix = "cls_patch" if args.include_cls else "patch_only"
        embeddings_dir = Path(f"./embeddings_{mode_suffix}")
        working_dir = Path("./working")
        
        embeddings_dir.mkdir(parents=True, exist_ok=True)
        working_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"⚠️ Using fallback temp management")
        print(f"📁 Embeddings dir: {embeddings_dir}")
    
    # Find TAR files
    try:
        tar_files = find_data_files(temp_manager, max_shards=args.max_shards)
    except Exception as e:
        print(f"❌ {e}")
        return 1
    
    print(f"📤 Output directory: {embeddings_dir}")
    print(f"🔄 Processing {len(tar_files)} TAR files across {args.world_size} GPUs...")
    
    start_time = time.time()
    
    # Launch distributed processing
    try:
        # Use torch.multiprocessing.spawn for multi-GPU processing
        mp.spawn(
            process_tar_files_on_gpu,
            args=(
                args.world_size,
                tar_files,
                embeddings_dir,
                working_dir,
                args.batch_size,
                args.include_cls,
                target_tokens,
                args.master_port
            ),
            nprocs=args.world_size,
            join=True
        )
        
        print("✅ All GPU processes completed")
        
    except Exception as e:
        print(f"❌ Distributed processing failed: {e}")
        return 1
    
    # Consolidate results (single-threaded)
    print("\n🔄 Consolidating results from all GPUs...")
    
    try:
        consolidation_results = consolidate_gpu_outputs(
            embeddings_dir,
            args.world_size,
            mode_suffix,
            len(tar_files)
        )
        
        # Create distributed manifest
        processing_time = time.time() - start_time
        manifest_path = create_distributed_manifest(
            embeddings_dir,
            consolidation_results,
            args.world_size,
            args.include_cls,
            target_tokens,
            processing_time
        )
        
        # Final results
        print("\n" + "=" * 80)
        print("🎉 MULTI-GPU EXTRACTION COMPLETED!")
        print("=" * 80)
        print(f"📊 SUMMARY:")
        print(f"   GPUs used: {args.world_size}")
        print(f"   Mode: {mode_name} ({target_tokens} tokens)")
        print(f"   TAR files processed: {len(tar_files)}")
        print(f"   Consolidated shards: {consolidation_results['consolidated_shards']}")
        print(f"   Total samples: {consolidation_results['total_samples']:,}")
        print(f"   Processing time: {processing_time:.1f}s")
        print(f"   Speedup: ~{args.world_size:.1f}x (theoretical)")
        print(f"   Embeddings location: {embeddings_dir}")
        print(f"   Manifest: {manifest_path}")
        
        if consolidation_results['consolidated_shards'] == len(tar_files):
            print(f"\n🎉 SUCCESS! All shards processed successfully!")
            print("Ready for distributed BLIP3-o training!")
            print(f"\nNext step:")
            print(f"python train_dit_distributed.py --chunked_embeddings_dir {embeddings_dir} --distributed --world_size {args.world_size}")
        else:
            print(f"\n⚠️ Some shards may have failed processing")
            print(f"Check logs for details")
        
        print("=" * 80)
        
        return 0
        
    except Exception as e:
        print(f"❌ Consolidation failed: {e}")
        return 1


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except Exception as e:
        print(f"❌ Critical error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)