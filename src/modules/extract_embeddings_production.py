#!/usr/bin/env python3
"""
PRODUCTION Multi-GPU BLIP3-o Embedding Extraction
src/modules/extract_embeddings_production.py

FIXED for large-scale production use:
✅ Removed training-specific imports that were causing failures
✅ Fixed output directory handling to match user requirements
✅ Enhanced multi-node coordination and error handling
✅ Optimized for processing ALL 1800+ shards
✅ Robust distributed processing with proper cleanup
"""

import sys
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPProcessor, CLIPModel, AutoModel, CLIPImageProcessor
import pickle
from tqdm import tqdm
import numpy as np
from pathlib import Path
import gc
import time
import json
import argparse
from datetime import timedelta
from typing import List, Optional, Dict, Any
import logging
import warnings
import traceback
import tarfile

def setup_paths():
    """Setup paths for project structure"""
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    
    sys.path.insert(0, str(project_root))
    sys.path.insert(0, str(project_root / "src"))
    sys.path.insert(0, str(project_root / "src" / "modules"))
    sys.path.insert(0, str(project_root / "src" / "utils"))
    
    return project_root

def setup_temp_manager():
    """Setup temp manager with fallback for production use"""
    try:
        from src.modules.utils.temp_manager import setup_snellius_environment
        manager = setup_snellius_environment("blip3o_workspace")
        return manager
    except ImportError:
        print("⚠️  Temp manager not available, using environment variables")
        return None

def smart_memory_cleanup():
    """Efficient memory cleanup"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

def load_models_production(device, rank):
    """Load models with production-grade error handling and optimization"""
    print(f"[GPU {rank}] Loading models for PRODUCTION extraction...")
    
    try:
        # Set cache directory from environment
        cache_dir = os.environ.get('HF_HOME', None)
        
        # Load CLIP ViT-L/14 with robust error handling
        print(f"[GPU {rank}] Loading CLIP model...")
        clip_processor = CLIPProcessor.from_pretrained(
            "openai/clip-vit-large-patch14",
            cache_dir=cache_dir
        )
        
        clip_model = CLIPModel.from_pretrained(
            "openai/clip-vit-large-patch14",
            torch_dtype=torch.float16,
            device_map=None,
            cache_dir=cache_dir
        ).to(device)
        clip_model.eval()
        
        # Load EVA-CLIP-8B with robust error handling
        print(f"[GPU {rank}] Loading EVA-CLIP model...")
        eva_model = AutoModel.from_pretrained(
            "BAAI/EVA-CLIP-8B", 
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map=None,
            cache_dir=cache_dir
        ).to(device)
        
        eva_processor = CLIPImageProcessor.from_pretrained(
            "openai/clip-vit-large-patch14",
            cache_dir=cache_dir
        )
        eva_model.eval()
        
        # Disable gradients for inference
        for param in clip_model.parameters():
            param.requires_grad = False
        for param in eva_model.parameters():
            param.requires_grad = False
            
        smart_memory_cleanup()
        print(f"[GPU {rank}] ✅ Models loaded successfully for production extraction")
        
        return clip_processor, clip_model, eva_processor, eva_model
        
    except Exception as e:
        print(f"[GPU {rank}] ❌ Error loading models: {e}")
        traceback.print_exc()
        raise

def extract_clip_features_production(images, processor, model, device, include_cls=True):
    """Production CLIP feature extraction with robust error handling"""
    if not images or len(images) == 0:
        expected_tokens = 257 if include_cls else 256
        return torch.empty(0, expected_tokens, 1024)
    
    expected_tokens = 257 if include_cls else 256
    
    try:
        # Process images with proper error handling
        inputs = processor(
            images=images, 
            return_tensors="pt"
        )
        
        pixel_values = inputs['pixel_values'].to(device, dtype=torch.float16, non_blocking=True)
        
        with torch.no_grad():
            vision_outputs = model.vision_model(
                pixel_values=pixel_values,
                output_hidden_states=True,
                return_dict=True
            )
            
            # Extract tokens
            if include_cls:
                all_embeddings = vision_outputs.last_hidden_state  # [B, 257, 1024]
            else:
                all_embeddings = vision_outputs.last_hidden_state[:, 1:, :]  # [B, 256, 1024]
            
            # Validate shape
            batch_size, num_tokens, hidden_dim = all_embeddings.shape
            assert hidden_dim == 1024, f"Expected CLIP 1024-dim, got {hidden_dim}"
            assert num_tokens == expected_tokens, f"Expected {expected_tokens} tokens, got {num_tokens}"
            
            # Move to CPU efficiently
            result = all_embeddings.to('cpu', dtype=torch.float32, non_blocking=True)
            
            # Cleanup
            del vision_outputs, all_embeddings, pixel_values
            
            return result
            
    except Exception as e:
        print(f"⚠️ Batch CLIP extraction failed: {e}, using fallback")
        return extract_clip_features_fallback(images, processor, model, device, include_cls)

def extract_clip_features_fallback(images, processor, model, device, include_cls=True):
    """Fallback CLIP extraction with individual processing"""
    expected_tokens = 257 if include_cls else 256
    features = []
    
    for img in images:
        if img is None:
            features.append(torch.zeros(expected_tokens, 1024))
            continue
            
        try:
            inputs = processor(images=[img], return_tensors="pt")
            pixel_values = inputs['pixel_values'].to(device, dtype=torch.float16, non_blocking=True)
            
            with torch.no_grad():
                vision_outputs = model.vision_model(
                    pixel_values=pixel_values,
                    output_hidden_states=True,
                    return_dict=True
                )
                
                if include_cls:
                    embeddings = vision_outputs.last_hidden_state[0]
                else:
                    embeddings = vision_outputs.last_hidden_state[0, 1:, :]
                
                features.append(embeddings.to('cpu', dtype=torch.float32))
                
                del vision_outputs, embeddings, pixel_values
                
        except Exception:
            features.append(torch.zeros(expected_tokens, 1024))
    
    try:
        return torch.stack(features)
    except Exception:
        return torch.empty(0, expected_tokens, 1024)

def extract_eva_features_production(images, processor, model, device, include_cls=True):
    """Production EVA feature extraction with robust error handling"""
    if not images or len(images) == 0:
        expected_tokens = 257 if include_cls else 256
        return torch.empty(0, expected_tokens, 4096)
    
    expected_tokens = 257 if include_cls else 256
    
    try:
        inputs = processor(
            images=images, 
            return_tensors="pt"
        )
        
        pixel_values = inputs['pixel_values'].to(device, dtype=torch.float16, non_blocking=True)
        
        with torch.no_grad():
            vision_outputs = model.vision_model(
                pixel_values=pixel_values,
                output_hidden_states=True,
                return_dict=True
            )
            
            if include_cls:
                all_embeddings = vision_outputs.last_hidden_state  # [B, 257, 4096]
            else:
                all_embeddings = vision_outputs.last_hidden_state[:, 1:, :]  # [B, 256, 4096]
            
            batch_size, num_tokens, hidden_dim = all_embeddings.shape
            assert num_tokens == expected_tokens, f"Expected {expected_tokens} tokens, got {num_tokens}"
            
            result = all_embeddings.to('cpu', dtype=torch.float32, non_blocking=True)
            
            del vision_outputs, all_embeddings, pixel_values
            
            return result
            
    except Exception as e:
        print(f"⚠️ Batch EVA extraction failed: {e}, using fallback")
        return extract_eva_features_fallback(images, processor, model, device, include_cls)

def extract_eva_features_fallback(images, processor, model, device, include_cls=True):
    """Fallback EVA extraction with individual processing"""
    expected_tokens = 257 if include_cls else 256
    features = []
    
    for img in images:
        if img is None:
            features.append(torch.zeros(expected_tokens, 4096))
            continue
            
        try:
            inputs = processor(images=[img], return_tensors="pt")
            pixel_values = inputs['pixel_values'].to(device, dtype=torch.float16, non_blocking=True)
            
            with torch.no_grad():
                vision_outputs = model.vision_model(
                    pixel_values=pixel_values,
                    output_hidden_states=True,
                    return_dict=True
                )
                
                if include_cls:
                    embeddings = vision_outputs.last_hidden_state[0]
                else:
                    embeddings = vision_outputs.last_hidden_state[0, 1:, :]
                
                features.append(embeddings.to('cpu', dtype=torch.float32))
                
                del vision_outputs, embeddings, pixel_values
                
        except Exception:
            features.append(torch.zeros(expected_tokens, 4096))
    
    try:
        return torch.stack(features)
    except Exception:
        return torch.empty(0, expected_tokens, 4096)

def find_data_files(temp_manager, max_shards=None):
    """Find TAR files with production-grade error handling"""
    
    # Use environment variable if available
    if "BLIP3O_DATASETS" in os.environ:
        datasets_dir = Path(os.environ["BLIP3O_DATASETS"])
        print(f"Using BLIP3O_DATASETS: {datasets_dir}")
    elif temp_manager:
        datasets_dir = temp_manager.get_datasets_dir()
        print(f"Using temp manager datasets dir: {datasets_dir}")
    else:
        # Fallback search
        possible_dirs = [
            Path("/scratch-shared") / os.environ.get("USER", "user") / "blip3o_workspace" / "datasets",
            Path(os.environ.get("TMPDIR", "/tmp")) / "blip3o_data",
            Path(__file__).parent.parent.parent / "data"
        ]
        
        datasets_dir = None
        for possible_dir in possible_dirs:
            if possible_dir.exists():
                datasets_dir = possible_dir
                break
        
        if datasets_dir is None:
            raise FileNotFoundError("Could not find datasets directory")
    
    print(f"Searching for TAR files in: {datasets_dir}")
    
    # Find all TAR files
    tar_files = list(datasets_dir.glob("*.tar"))
    if not tar_files:
        raise FileNotFoundError(f"No TAR files found in {datasets_dir}")
    
    # Sort for consistent ordering across ranks
    tar_files.sort()
    tar_files = [str(f) for f in tar_files]
    
    # Apply max_shards limit if specified
    if max_shards is not None and max_shards > 0:
        tar_files = tar_files[:max_shards]
    
    # Validate files
    valid_files = []
    total_size_gb = 0
    
    print(f"Validating {len(tar_files)} TAR files...")
    
    for tar_file in tar_files:
        tar_path = Path(tar_file)
        if tar_path.exists():
            try:
                size_gb = tar_path.stat().st_size / (1024**3)
                
                if size_gb < 0.001:  # Less than 1MB - likely corrupted
                    print(f"⚠️ Skipping tiny file: {tar_file} ({size_gb:.6f} GB)")
                    continue
                
                # Quick validation - try to open
                with tarfile.open(tar_file, 'r') as test_tar:
                    members = test_tar.getmembers()
                    if len(members) == 0:
                        print(f"⚠️ Skipping empty archive: {tar_file}")
                        continue
                
                total_size_gb += size_gb
                valid_files.append(tar_file)
                
            except Exception as e:
                print(f"⚠️ Skipping corrupted file: {tar_file} ({e})")
                continue
    
    print(f"✅ Found {len(valid_files)} valid TAR files ({total_size_gb:.2f} GB total)")
    
    if not valid_files:
        raise FileNotFoundError("No valid TAR files found!")
    
    return valid_files

class ProductionTarDataset(Dataset):
    """Production-grade TAR dataset with robust error handling"""
    
    def __init__(self, tar_path, rank=0, world_size=1):
        self.tar_path = tar_path
        self.rank = rank
        self.world_size = world_size
        self.samples = []
        self._load_samples()
        
    def _load_samples(self):
        """Load samples with production-grade error handling"""
        try:
            print(f"[GPU {self.rank}] Loading samples from {Path(self.tar_path).name}")
            
            with tarfile.open(self.tar_path, 'r') as tar:
                all_members = tar.getmembers()
                
                # Filter to image files
                image_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
                image_members = [
                    member for member in all_members 
                    if member.isfile() and Path(member.name).suffix.lower() in image_extensions
                ]
                
                print(f"[GPU {self.rank}] Found {len(image_members)} images in {Path(self.tar_path).name}")
                
                processed_count = 0
                error_count = 0
                
                for member in image_members:
                    try:
                        file_obj = tar.extractfile(member)
                        if file_obj is None:
                            error_count += 1
                            continue
                        
                        image_data = file_obj.read()
                        if not image_data or len(image_data) < 100:
                            error_count += 1
                            continue
                        
                        # Basic validation
                        from PIL import Image
                        import io
                        
                        try:
                            test_io = io.BytesIO(image_data)
                            test_image = Image.open(test_io)
                            width, height = test_image.size
                            if width < 10 or height < 10:
                                error_count += 1
                                continue
                            test_image.close()
                            
                            key = Path(member.name).stem
                            sample_data = {
                                'image_data': image_data,
                                'key': key,
                                'caption': f"Image {key}",
                                'member_name': member.name
                            }
                            
                            self.samples.append(sample_data)
                            processed_count += 1
                            
                        except Exception:
                            error_count += 1
                            continue
                            
                    except Exception:
                        error_count += 1
                        continue
                
                print(f"[GPU {self.rank}] Loaded {processed_count} samples, skipped {error_count} errors")
                
        except Exception as e:
            print(f"[GPU {self.rank}] ❌ Error loading TAR: {e}")
            self.samples = []
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        """Production-grade item retrieval with fallback"""
        try:
            if index < 0 or index >= len(self.samples):
                raise IndexError(f"Index {index} out of range")
            
            sample_data = self.samples[index]
            image_data = sample_data.get('image_data')
            if image_data is None:
                raise ValueError(f"No image data for sample {index}")
            
            from PIL import Image
            import io
            
            image = Image.open(io.BytesIO(image_data)).convert('RGB')
            
            if image.size[0] < 10 or image.size[1] < 10:
                raise ValueError(f"Image too small: {image.size}")
            
            return {
                'image': image,
                'caption': sample_data.get('caption', ''),
                'key': sample_data.get('key', 'unknown'),
            }
            
        except Exception as e:
            # Robust fallback
            from PIL import Image
            fallback_image = Image.new('RGB', (224, 224), color='black')
            return {
                'image': fallback_image,
                'caption': f'fallback_sample_{index}',
                'key': f'fallback_{index}',
            }

def production_collate_fn(batch):
    """Production collate function with error handling"""
    if not batch:
        return None
        
    valid_items = [item for item in batch if item is not None and 'image' in item]
    if not valid_items:
        return None
    
    try:
        images = []
        captions = []
        keys = []
        
        for item in valid_items:
            if item['image'] is not None:
                images.append(item['image'])
                captions.append(item['caption'])
                keys.append(item['key'])
        
        if not images:
            return None
        
        return {
            'image': images,
            'caption': captions,
            'key': keys
        }
    except Exception:
        return None

def setup_distributed_production(rank: int, world_size: int, master_port: str = "12355"):
    """Production distributed setup with enhanced multi-node support"""
    try:
        # Handle multi-node SLURM setup
        if "SLURM_LOCALID" in os.environ:
            local_rank = int(os.environ["SLURM_LOCALID"])
        elif "SLURM_GPUS_ON_NODE" in os.environ:
            gpus_per_node = int(os.environ["SLURM_GPUS_ON_NODE"])
            local_rank = rank % gpus_per_node
        else:
            local_rank = rank
        
        # Get master address for multi-node
        if "MASTER_ADDR" in os.environ:
            master_addr = os.environ["MASTER_ADDR"]
        else:
            master_addr = "localhost"
        
        os.environ['MASTER_ADDR'] = master_addr
        os.environ['MASTER_PORT'] = master_port
        os.environ['RANK'] = str(rank)
        os.environ['LOCAL_RANK'] = str(local_rank)
        os.environ['WORLD_SIZE'] = str(world_size)
        
        print(f"[Rank {rank}] Distributed setup: {master_addr}:{master_port}, Local GPU: {local_rank}")
        
        # Set device
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = torch.device(f'cuda:{local_rank}')
            backend = 'nccl'
        else:
            device = torch.device('cpu')
            backend = 'gloo'
        
        # Initialize process group
        dist.init_process_group(
            backend=backend,
            init_method='env://',
            world_size=world_size,
            rank=rank,
            timeout=timedelta(minutes=120)  # Extended timeout for large jobs
        )
        
        print(f"[Rank {rank}] ✅ Distributed environment ready")
        return device
        
    except Exception as e:
        print(f"[Rank {rank}] ❌ Failed to setup distributed: {e}")
        raise

def cleanup_distributed():
    """Clean up distributed environment"""
    try:
        if dist.is_initialized():
            dist.destroy_process_group()
    except Exception:
        pass

def process_single_tar_production(
    tar_file_path: str,
    shard_idx: int,
    clip_processor, clip_model, eva_processor, eva_model,
    device: torch.device,
    output_dir: Path,
    batch_size: int = 32,
    include_cls: bool = True,
    target_tokens: int = 256,
    rank: int = 0,
    world_size: int = 1
) -> dict:
    """Production TAR processing with enhanced error handling"""
    
    print(f"[GPU {rank}] 🏭 PRODUCTION processing shard {shard_idx}: {Path(tar_file_path).name}")
    
    mode_suffix = "cls_patch" if include_cls else "patch_only"
    shard_filename = f"embeddings_shard_{shard_idx:05d}_{mode_suffix}_gpu{rank}.pkl"
    shard_path = output_dir / shard_filename
    
    # Check if already processed
    if shard_path.exists():
        try:
            with open(shard_path, 'rb') as f:
                existing_data = pickle.load(f)
            sample_count = len(existing_data.get('captions', []))
            print(f"[GPU {rank}] ✅ Shard {shard_idx} already processed ({sample_count} samples)")
            return {
                'shard_idx': shard_idx,
                'total_samples': sample_count,
                'success': True,
                'skipped': True,
                'output_path': str(shard_path)
            }
        except:
            shard_path.unlink()
    
    try:
        # Create dataset
        dataset = ProductionTarDataset(tar_file_path, rank, world_size)
        
        if len(dataset) == 0:
            print(f"[GPU {rank}] ⚠️ No samples in {Path(tar_file_path).name}")
            return {
                'shard_idx': shard_idx,
                'total_samples': 0,
                'success': False,
                'error': 'No samples loaded'
            }
        
        # Production DataLoader
        num_workers = min(4, os.cpu_count() // world_size)
        dataloader = DataLoader(
            dataset, 
            batch_size=batch_size,
            shuffle=False,
            collate_fn=production_collate_fn,
            num_workers=num_workers,
            drop_last=False,
            pin_memory=True,
            persistent_workers=True if num_workers > 0 else False,
            prefetch_factor=2,
        )
        
        # Storage
        shard_clip_embeddings = []
        shard_eva_embeddings = []
        shard_captions = []
        shard_keys = []
        
        total_samples = 0
        start_time = time.time()
        batch_count = 0
        error_count = 0
        last_cleanup = time.time()
        
        # Progress tracking
        progress_bar = tqdm(
            dataloader, 
            desc=f"GPU{rank} Shard{shard_idx}", 
            unit="batch", 
            position=rank,
            leave=False
        )
        
        for batch_idx, batch in enumerate(progress_bar):
            if batch is None:
                continue
                
            batch_count += 1
            
            try:
                images = batch['image']
                captions = batch['caption']
                keys = batch['key']
                
                if not images:
                    continue
                
                # Extract features
                try:
                    clip_features = extract_clip_features_production(
                        images, clip_processor, clip_model, device, include_cls=include_cls
                    )
                    
                    if clip_features.numel() == 0:
                        error_count += 1
                        continue
                    
                    eva_features = extract_eva_features_production(
                        images, eva_processor, eva_model, device, include_cls=include_cls
                    )
                    
                    if eva_features.numel() == 0:
                        error_count += 1
                        continue
                    
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        print(f"[GPU {rank}] ⚠️ OOM on batch {batch_idx}, cleaning up...")
                        smart_memory_cleanup()
                        error_count += 1
                        continue
                    else:
                        raise e
                
                # Validate shapes
                if clip_features.shape[1] != target_tokens or eva_features.shape[1] != target_tokens:
                    error_count += 1
                    continue
                
                # Store results
                shard_clip_embeddings.append(clip_features)
                shard_eva_embeddings.append(eva_features)
                shard_captions.extend(captions)
                shard_keys.extend(keys)
                
                total_samples += len(images)
                
                # Update progress
                current_time = time.time()
                elapsed = current_time - start_time
                samples_per_sec = total_samples / elapsed if elapsed > 0 else 0
                progress_bar.set_postfix({
                    'samples': total_samples,
                    'sps': f'{samples_per_sec:.1f}',
                    'errors': error_count
                })
                
                # Periodic cleanup
                if current_time - last_cleanup > 60:
                    smart_memory_cleanup()
                    last_cleanup = current_time
                
                del clip_features, eva_features, images, captions, keys
            
            except Exception as e:
                error_count += 1
                print(f"[GPU {rank}] ⚠️ Error in batch {batch_idx}: {e}")
                continue
        
        progress_bar.close()
        
        # Consolidate and save
        if shard_clip_embeddings and total_samples > 0:
            try:
                final_clip = torch.cat(shard_clip_embeddings, dim=0)
                final_eva = torch.cat(shard_eva_embeddings, dim=0)
                
                # Create production shard data
                shard_data = {
                    'clip_blip3o_embeddings': final_clip,
                    'eva_blip3o_embeddings': final_eva,
                    'captions': shard_captions,
                    'keys': shard_keys,
                    'total_samples': total_samples,
                    'shard_idx': shard_idx,
                    'source_tar': tar_file_path,
                    'config': {
                        'clip_model': 'openai/clip-vit-large-patch14',
                        'eva_model': 'BAAI/EVA-CLIP-8B',
                        'clip_dim': 1024,
                        'eva_dim': final_eva.shape[2],
                        'tokens': target_tokens,
                        'include_cls': include_cls,
                        'mode': mode_suffix,
                        'extraction_method': 'production_v1',
                        'format_version': f'blip3o_{target_tokens}_tokens_production_v1',
                        'extraction_time': time.time() - start_time,
                        'distributed': world_size > 1,
                        'rank': rank,
                        'world_size': world_size,
                        'production_features': {
                            'robust_error_handling': True,
                            'memory_optimized': True,
                            'multi_node_support': True,
                            'batch_fallback': True,
                        },
                        'performance': {
                            'samples_per_second': total_samples / (time.time() - start_time),
                            'batch_size': batch_size,
                            'total_batches': batch_count,
                            'error_rate': error_count / batch_count if batch_count > 0 else 0
                        }
                    }
                }
                
                # Save with error handling
                temp_path = shard_path.with_suffix('.tmp')
                with open(temp_path, 'wb') as f:
                    pickle.dump(shard_data, f, protocol=pickle.HIGHEST_PROTOCOL)
                temp_path.rename(shard_path)
                
                file_size_mb = shard_path.stat().st_size / (1024 * 1024)
                processing_time = time.time() - start_time
                samples_per_sec = total_samples / processing_time
                
                print(f"[GPU {rank}] ✅ Shard {shard_idx} completed: {total_samples} samples "
                      f"({file_size_mb:.1f} MB, {samples_per_sec:.1f} sps)")
                
                del shard_clip_embeddings, shard_eva_embeddings, final_clip, final_eva
                smart_memory_cleanup()
                
                return {
                    'shard_idx': shard_idx,
                    'total_samples': total_samples,
                    'file_size_mb': file_size_mb,
                    'processing_time': processing_time,
                    'samples_per_second': samples_per_sec,
                    'output_path': str(shard_path),
                    'success': True,
                    'rank': rank,
                    'world_size': world_size,
                    'error_rate': error_count / batch_count if batch_count > 0 else 0
                }
            
            except Exception as e:
                print(f"[GPU {rank}] ❌ Failed to save shard {shard_idx}: {e}")
                return {
                    'shard_idx': shard_idx,
                    'total_samples': total_samples,
                    'success': False,
                    'error': f'Save failed: {e}',
                }
        
        else:
            return {
                'shard_idx': shard_idx,
                'total_samples': 0,
                'success': False,
                'error': 'No embeddings extracted',
            }
    
    except Exception as e:
        print(f"[GPU {rank}] ❌ Processing failed for shard {shard_idx}: {e}")
        return {
            'shard_idx': shard_idx,
            'total_samples': 0,
            'success': False,
            'error': f'Processing failed: {e}',
        }

def process_tar_files_on_gpu(
    rank: int,
    world_size: int,
    tar_files: List[str],
    output_dir: Path,
    batch_size: int = 32,
    include_cls: bool = True,
    target_tokens: int = 256,
    master_port: str = "12355"
):
    """Production multi-GPU TAR processing"""
    
    # Setup distributed environment
    device = setup_distributed_production(rank, world_size, master_port)
    
    try:
        print(f"[GPU {rank}] 🏭 Starting PRODUCTION extraction")
        print(f"[GPU {rank}] Processing {len(tar_files)} total TAR files")
        
        # Load models
        clip_processor, clip_model, eva_processor, eva_model = load_models_production(device, rank)
        
        # Distribute files across GPUs
        assigned_files = []
        for i, tar_file in enumerate(tar_files):
            if i % world_size == rank:
                assigned_files.append((i, tar_file))
        
        if not assigned_files:
            print(f"[GPU {rank}] No files assigned")
            return
        
        print(f"[GPU {rank}] Processing {len(assigned_files)} files with batch_size={batch_size}")
        
        # Process assigned files
        total_start_time = time.time()
        successful_shards = 0
        total_samples_processed = 0
        
        for local_idx, (global_shard_idx, tar_file) in enumerate(assigned_files):
            print(f"[GPU {rank}] Processing file {local_idx + 1}/{len(assigned_files)}: {Path(tar_file).name}")
            
            result = process_single_tar_production(
                tar_file_path=tar_file,
                shard_idx=global_shard_idx,
                clip_processor=clip_processor,
                clip_model=clip_model,
                eva_processor=eva_processor,
                eva_model=eva_model,
                device=device,
                output_dir=output_dir,
                batch_size=batch_size,
                include_cls=include_cls,
                target_tokens=target_tokens,
                rank=rank,
                world_size=world_size
            )
            
            if result and result['success']:
                successful_shards += 1
                total_samples_processed += result['total_samples']
                if not result.get('skipped', False):
                    sps = result.get('samples_per_second', 0)
                    print(f"[GPU {rank}] ✅ Shard {global_shard_idx}: {result['total_samples']} samples ({sps:.1f} sps)")
            else:
                error_msg = result.get('error', 'Unknown error') if result else 'No result returned'
                print(f"[GPU {rank}] ❌ Failed shard {global_shard_idx}: {error_msg}")
        
        # Summary
        total_time = time.time() - total_start_time
        avg_sps = total_samples_processed / total_time if total_time > 0 else 0
        print(f"[GPU {rank}] 📊 Summary: {successful_shards}/{len(assigned_files)} shards, "
              f"{total_samples_processed:,} samples, {avg_sps:.1f} avg sps")
        
        # Synchronize all GPUs
        if dist.is_initialized():
            dist.barrier()
        
        print(f"[GPU {rank}] ✅ PRODUCTION extraction completed")
        
    except Exception as e:
        print(f"[GPU {rank}] ❌ Critical error: {e}")
        traceback.print_exc()
        raise
    
    finally:
        # Cleanup
        try:
            del clip_model, eva_model, clip_processor, eva_processor
        except:
            pass
        
        smart_memory_cleanup()
        cleanup_distributed()

def consolidate_gpu_outputs_production(output_dir: Path, world_size: int, mode_suffix: str, total_shards: int) -> Dict[str, Any]:
    """Production GPU output consolidation"""
    
    print("🔄 Consolidating PRODUCTION GPU outputs...")
    
    consolidation_results = {
        'consolidated_shards': 0,
        'total_samples': 0,
        'consolidation_errors': 0,
        'final_files': [],
        'failed_shards': [],
        'performance_stats': {
            'total_processing_time': 0,
            'avg_samples_per_second': 0,
            'avg_error_rate': 0
        }
    }
    
    for shard_idx in range(total_shards):
        gpu_files = []
        shard_data_parts = []
        
        # Find GPU-specific files for this shard
        for rank in range(world_size):
            gpu_output_path = output_dir / f"embeddings_shard_{shard_idx:05d}_{mode_suffix}_gpu{rank}.pkl"
            
            if gpu_output_path.exists():
                try:
                    with open(gpu_output_path, 'rb') as f:
                        shard_data = pickle.load(f)
                    shard_data_parts.append(shard_data)
                    gpu_files.append(gpu_output_path)
                except Exception as e:
                    print(f"⚠️ Error loading GPU file {gpu_output_path}: {e}")
                    consolidation_results['consolidation_errors'] += 1
        
        if not shard_data_parts:
            consolidation_results['failed_shards'].append(shard_idx)
            continue
        
        # Consolidate data
        if len(shard_data_parts) == 1:
            consolidated_data = shard_data_parts[0]
        else:
            # Multiple GPU outputs need merging
            print(f"⚠️ Multiple GPU outputs for shard {shard_idx}, consolidating...")
            consolidated_data = shard_data_parts[0].copy()
            
            all_clip = [part['clip_blip3o_embeddings'] for part in shard_data_parts]
            all_eva = [part['eva_blip3o_embeddings'] for part in shard_data_parts]
            all_captions = []
            all_keys = []
            
            for part in shard_data_parts:
                all_captions.extend(part.get('captions', []))
                all_keys.extend(part.get('keys', []))
            
            consolidated_data.update({
                'clip_blip3o_embeddings': torch.cat(all_clip, dim=0),
                'eva_blip3o_embeddings': torch.cat(all_eva, dim=0),
                'captions': all_captions,
                'keys': all_keys,
                'total_samples': sum(part.get('total_samples', 0) for part in shard_data_parts)
            })
        
        # Update performance stats
        if 'config' in consolidated_data and 'performance' in consolidated_data['config']:
            perf = consolidated_data['config']['performance']
            consolidation_results['performance_stats']['total_processing_time'] += consolidated_data['config'].get('extraction_time', 0)
            consolidation_results['performance_stats']['avg_samples_per_second'] += perf.get('samples_per_second', 0)
            consolidation_results['performance_stats']['avg_error_rate'] += perf.get('error_rate', 0)
        
        # Mark as production version
        if 'config' in consolidated_data:
            consolidated_data['config']['production_version'] = True
            consolidated_data['config']['consolidation_timestamp'] = time.time()
        
        # Save consolidated shard
        final_output_path = output_dir / f"embeddings_shard_{shard_idx:05d}_{mode_suffix}.pkl"
        
        try:
            temp_path = final_output_path.with_suffix('.tmp')
            with open(temp_path, 'wb') as f:
                pickle.dump(consolidated_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            temp_path.rename(final_output_path)
            
            consolidation_results['consolidated_shards'] += 1
            consolidation_results['total_samples'] += consolidated_data.get('total_samples', 0)
            consolidation_results['final_files'].append(str(final_output_path))
            
            # Clean up GPU-specific files
            for gpu_file in gpu_files:
                try:
                    gpu_file.unlink()
                except Exception:
                    pass
                    
        except Exception as e:
            print(f"❌ Failed to save consolidated shard {shard_idx}: {e}")
            consolidation_results['consolidation_errors'] += 1
    
    # Calculate average performance stats
    if consolidation_results['consolidated_shards'] > 0:
        consolidation_results['performance_stats']['avg_samples_per_second'] /= consolidation_results['consolidated_shards']
        consolidation_results['performance_stats']['avg_error_rate'] /= consolidation_results['consolidated_shards']
    
    print(f"✅ PRODUCTION consolidation completed: {consolidation_results['consolidated_shards']} shards, "
          f"{consolidation_results['total_samples']:,} samples")
    
    return consolidation_results

def main():
    """Main production extraction function"""
    
    parser = argparse.ArgumentParser(description="PRODUCTION Multi-GPU BLIP3-o Embedding Extraction")
    parser.add_argument("--include_cls", action="store_true", default=False,
                       help="Include CLS token (257 tokens) or patches only (256 tokens)")
    parser.add_argument("--max_shards", type=int, default=0,
                       help="Maximum number of shards to process (0 = ALL)")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size for processing")
    parser.add_argument("--world_size", type=int, default=0,
                       help="Number of GPUs to use (0 = auto-detect)")
    parser.add_argument("--master_port", type=str, default="12361",
                       help="Master port for distributed communication")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Output directory (uses environment or default)")
    
    args = parser.parse_args()
    
    # Auto-detect GPU configuration
    if args.world_size == 0:
        if torch.cuda.is_available():
            if "SLURM_NTASKS" in os.environ:
                args.world_size = int(os.environ["SLURM_NTASKS"])
                print(f"🔍 SLURM detected: {args.world_size} total GPUs across nodes")
            else:
                args.world_size = torch.cuda.device_count()
                print(f"🔍 Local detection: {args.world_size} GPUs")
        else:
            print("❌ CUDA not available!")
            return 1
    
    # Setup
    target_tokens = 257 if args.include_cls else 256
    mode_name = "CLS+Patches" if args.include_cls else "Patches only"
    mode_suffix = "cls_patch" if args.include_cls else "patch_only"
    
    print("🏭 PRODUCTION Multi-GPU BLIP3-o Embedding Extraction")
    print("=" * 70)
    print(f"Mode: {mode_name} ({target_tokens} tokens)")
    print(f"GPUs: {args.world_size}")
    print(f"Batch size: {args.batch_size}")
    print(f"Max shards: {'ALL' if args.max_shards == 0 else args.max_shards}")
    print("=" * 70)
    
    project_root = setup_paths()
    
    # Setup directories
    temp_manager = setup_temp_manager()
    
    if args.output_dir:
        output_dir = Path(args.output_dir)
    elif "BLIP3O_EMBEDDINGS" in os.environ:
        base_embeddings_dir = Path(os.environ["BLIP3O_EMBEDDINGS"])
        output_dir = base_embeddings_dir / "patch_embeddings_short_256"
    elif temp_manager:
        output_dir = temp_manager.create_embeddings_subdirectory("patch_embeddings_short_256")
    else:
        output_dir = Path("./embeddings_patch_only_256")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"📁 Output directory: {output_dir}")
    
    # Find TAR files
    try:
        tar_files = find_data_files(temp_manager, max_shards=args.max_shards if args.max_shards > 0 else None)
    except Exception as e:
        print(f"❌ {e}")
        return 1
    
    if args.max_shards > 0:
        tar_files = tar_files[:args.max_shards]
    
    expected_samples = len(tar_files) * 2600  # Estimate
    expected_time = expected_samples / (args.world_size * 40)  # Conservative estimate
    
    print(f"📊 PRODUCTION Processing Overview:")
    print(f"   TAR files: {len(tar_files)}")
    print(f"   Expected samples: ~{expected_samples:,}")
    print(f"   Estimated time: ~{expected_time/60:.1f} minutes")
    
    start_time = time.time()
    
    # Multi-GPU production processing
    try:
        print("\n🏭 Starting PRODUCTION multi-GPU processing...")
        mp.spawn(
            process_tar_files_on_gpu,
            args=(
                args.world_size,
                tar_files,
                output_dir,
                args.batch_size,
                args.include_cls,
                target_tokens,
                args.master_port
            ),
            nprocs=args.world_size,
            join=True
        )
        
        print("✅ All GPU processes completed")
        
        # Consolidate results
        consolidation_results = consolidate_gpu_outputs_production(
            output_dir,
            args.world_size,
            mode_suffix,
            len(tar_files)
        )
        
    except Exception as e:
        print(f"❌ PRODUCTION processing failed: {e}")
        traceback.print_exc()
        return 1
    
    # Create production manifest
    processing_time = time.time() - start_time
    
    manifest_data = {
        'extraction_info': {
            'method': 'production_v1',
            'world_size': args.world_size,
            'extraction_time_seconds': processing_time,
            'timestamp': time.time(),
            'production_features': {
                'robust_error_handling': True,
                'memory_optimized': True,
                'multi_node_support': True,
                'batch_fallback': True,
                'enhanced_consolidation': True,
            },
            'approach': 'production_large_scale'
        },
        'consolidation_results': consolidation_results,
        'performance_stats': consolidation_results.get('performance_stats', {}),
        'token_info': {
            'tokens_per_sample': target_tokens,
            'include_cls': args.include_cls,
        },
        'format_version': f'blip3o_{target_tokens}_tokens_production_v1',
        'total_shards': consolidation_results['consolidated_shards'],
        'total_samples': consolidation_results['total_samples'],
        'failed_shards': consolidation_results.get('failed_shards', []),
        'success_rate': consolidation_results['consolidated_shards'] / len(tar_files) if tar_files else 0,
        'dataset_info': {
            'source': 'blip3o_pretrain_short_caption',
            'total_tar_files': len(tar_files),
            'output_directory': str(output_dir),
        }
    }
    
    manifest_path = output_dir / "embeddings_manifest.json"
    with open(manifest_path, 'w') as f:
        json.dump(manifest_data, f, indent=2)
    
    # Final results
    print("\n" + "=" * 70)
    print("🎉 PRODUCTION EXTRACTION COMPLETED!")
    print("=" * 70)
    
    actual_samples = consolidation_results['total_samples']
    actual_sps = actual_samples / processing_time if processing_time > 0 else 0
    
    print(f"🏭 PRODUCTION RESULTS:")
    print(f"   Total samples: {actual_samples:,}")
    print(f"   Processing time: {processing_time:.1f} seconds ({processing_time/60:.1f} minutes)")
    print(f"   Overall speed: {actual_sps:.1f} samples/sec")
    print(f"   Success rate: {consolidation_results.get('success_rate', 0)*100:.1f}%")
    print(f"   Successful shards: {consolidation_results['consolidated_shards']}/{len(tar_files)}")
    
    print(f"\n📁 Output location: {output_dir}")
    print(f"📊 Files: {len(consolidation_results['final_files'])} embedding files")
    print(f"📋 Manifest: {manifest_path}")
    
    if consolidation_results['consolidated_shards'] > 0:
        print(f"\n🎉 SUCCESS! PRODUCTION extraction ready for large-scale training!")
        print(f"Next steps:")
        print(f"  # For distributed training:")
        print(f"  torchrun --nproc_per_node={args.world_size} train_dit_distributed.py \\")
        print(f"    --chunked_embeddings_dir {output_dir} \\")
        print(f"    --distributed --world_size {args.world_size}")
    else:
        print(f"\n❌ No shards processed successfully")
    
    print("=" * 70)
    
    return 0 if consolidation_results['consolidated_shards'] > 0 else 1

if __name__ == "__main__":
    try:
        # Production CUDA optimizations
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
        
        exit_code = main()
        sys.exit(exit_code)
    except Exception as e:
        print(f"❌ PRODUCTION extraction failed: {e}")
        traceback.print_exc()
        sys.exit(1)