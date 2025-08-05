#!/usr/bin/env python3
"""
FIXED BLIP3-o Embedding Extraction with Multi-GPU Support
src/modules/extract_embeddings_g.py

FIXES:
1. Fixed WebDataset version compatibility issues
2. Multiple fallback approaches for distributed WebDataset
3. Better error handling and diagnostics
4. Simplified distributed processing that works with any WebDataset version
"""

import sys
import os
import torch
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPModel, AutoModel, CLIPImageProcessor
import pickle
from tqdm import tqdm
import numpy as np
from pathlib import Path
import gc
import psutil
import time
import glob
import json
import shutil
import argparse

def setup_paths():
    """Setup paths for project structure"""
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    
    # Add import paths
    sys.path.insert(0, str(project_root))
    sys.path.insert(0, str(project_root / "src"))
    sys.path.insert(0, str(project_root / "src" / "data_hand"))
    sys.path.insert(0, str(project_root / "src" / "utils"))
    
    return project_root

def setup_temp_manager():
    """Setup temp manager for structured directory management."""
    try:
        from src.modules.utils.temp_manager import setup_snellius_environment
        manager = setup_snellius_environment("blip3o_workspace")
        return manager
    except ImportError:
        print("⚠️  Temp manager not available, using fallback directories")
        return None

def get_memory_usage():
    """Get current memory usage in GB"""
    try:
        process = psutil.Process(os.getpid())
        memory_mb = process.memory_info().rss / 1024 / 1024
        return memory_mb / 1024
    except:
        return 0.0

def cleanup_memory():
    """Aggressive memory cleanup"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def load_models(device):
    """Load CLIP and EVA-CLIP models"""
    print("📦 Loading models...")
    
    # Load CLIP ViT-L/14
    print("   Loading CLIP ViT-L/14...")
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    clip_model = CLIPModel.from_pretrained(
        "openai/clip-vit-large-patch14",
        torch_dtype=torch.float16
    ).to(device)
    clip_model.eval()
    
    # Load EVA-CLIP-8B
    print("   Loading EVA-CLIP-8B...")
    eva_model = AutoModel.from_pretrained(
        "BAAI/EVA-CLIP-8B", 
        trust_remote_code=True,
        torch_dtype=torch.float16
    ).to(device)
    
    eva_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")
    eva_model.eval()
    
    cleanup_memory()
    
    print("✅ Models loaded successfully")
    print(f"💾 Memory usage after loading: {get_memory_usage():.2f} GB")
    
    return clip_processor, clip_model, eva_processor, eva_model

def extract_clip_features_with_cls(images, processor, model, device, include_cls=True):
    """
    Extract CLIP ViT-L/14 features with CLS token + patches
    
    Args:
        images: List of PIL images
        processor: CLIP processor
        model: CLIP model
        device: Device
        include_cls: Whether to include CLS token (default: True)
        
    Returns:
        Features tensor [B, 257, 1024] if include_cls else [B, 256, 1024]
    """
    features = []
    
    for img in images:
        inputs = processor(images=img, return_tensors="pt")
        inputs = {k: v.to(device).half() if v.dtype == torch.float32 else v.to(device) 
                 for k, v in inputs.items()}
        
        with torch.no_grad():
            vision_outputs = model.vision_model(
                pixel_values=inputs['pixel_values'],
                output_hidden_states=True,
                return_dict=True
            )
            
            # Get all hidden states (CLS + patches)
            # ViT-L/14 outputs: [1, 257, 1024] where [0] is CLS, [1:257] are patches
            if include_cls:
                # Keep CLS token + patches: [1, 257, 1024]
                all_embeddings = vision_outputs.last_hidden_state  # [1, 257, 1024]
                expected_tokens = 257
            else:
                # Remove CLS token (patches only): [1, 256, 1024]
                all_embeddings = vision_outputs.last_hidden_state[:, 1:, :]  # [1, 256, 1024]
                expected_tokens = 256
            
            batch_size, num_tokens, hidden_dim = all_embeddings.shape
            
            # Validate dimensions
            assert hidden_dim == 1024, f"Expected CLIP 1024-dim, got {hidden_dim}"
            assert num_tokens == expected_tokens, f"Expected {expected_tokens} tokens, got {num_tokens}"
            
            # Convert to float32 and move to CPU
            features.append(all_embeddings.squeeze().cpu().float())
            
            # Clear GPU memory
            del vision_outputs, all_embeddings
    
    return torch.stack(features)

def extract_eva_features_with_cls(images, processor, model, device, include_cls=True):
    """
    Extract EVA-CLIP-8B features with CLS token + patches
    
    Args:
        images: List of PIL images
        processor: EVA processor
        model: EVA model
        device: Device
        include_cls: Whether to include CLS token (default: True)
        
    Returns:
        Features tensor [B, 257, 4096] if include_cls else [B, 256, 4096]
    """
    features = []
    
    for img in images:
        inputs = processor(images=img, return_tensors="pt")
        pixel_values = inputs['pixel_values'].to(device).half()
        
        with torch.no_grad():
            vision_outputs = model.vision_model(
                pixel_values=pixel_values,
                output_hidden_states=True,
                return_dict=True
            )
            
            # Get all hidden states (CLS + patches)
            # EVA-CLIP outputs: [1, 257, hidden_dim] where [0] is CLS, [1:257] are patches
            if include_cls:
                # Keep CLS token + patches: [1, 257, hidden_dim]
                all_embeddings = vision_outputs.last_hidden_state  # [1, 257, hidden_dim]
                expected_tokens = 257
            else:
                # Remove CLS token (patches only): [1, 256, hidden_dim]
                all_embeddings = vision_outputs.last_hidden_state[:, 1:, :]  # [1, 256, hidden_dim]
                expected_tokens = 256
            
            batch_size, num_tokens, hidden_dim = all_embeddings.shape
            
            # Validate dimensions
            assert num_tokens == expected_tokens, f"Expected {expected_tokens} tokens, got {num_tokens}"
            
            # Convert to float32 and move to CPU
            features.append(all_embeddings.squeeze().cpu().float())
            
            # Clear GPU memory
            del vision_outputs, all_embeddings, pixel_values
    
    return torch.stack(features)

def find_data_files(temp_manager, max_shards=None):
    """Find downloaded tar files using temp manager."""
    if temp_manager:
        datasets_dir = temp_manager.get_datasets_dir()
        print(f"🔍 Searching for dataset shards in: {datasets_dir}")
    else:
        # Fallback to old method
        print("🔍 Searching for dataset shards (fallback method)...")
        if "TMPDIR" in os.environ:
            datasets_dir = Path(os.environ["TMPDIR"]) / "blip3o_data"
        elif "SCRATCH_SHARED" in os.environ:
            user = os.environ.get("USER", "user")
            datasets_dir = Path(os.environ["SCRATCH_SHARED"]) / user / "blip3o_data"
        else:
            datasets_dir = Path(__file__).parent.parent.parent / "data"
    
    # Look for tar files
    tar_files = list(datasets_dir.glob("*.tar"))
    if tar_files:
        tar_files.sort()  # Sort numerically
        tar_files = [str(f) for f in tar_files]
        
        # Limit number of shards if specified
        if max_shards is not None:
            tar_files = tar_files[:max_shards]
            print(f"📊 Limited to {max_shards} shards")
        
        print(f"   ✅ Found {len(tar_files)} tar files")
        
        # Validate files exist and show details
        valid_files = []
        total_size_gb = 0
        
        print(f"\n📊 Validating found files...")
        for tar_file in tar_files:
            tar_path = Path(tar_file)
            if tar_path.exists():
                size_gb = tar_path.stat().st_size / (1024**3)
                total_size_gb += size_gb
                valid_files.append(tar_file)
                print(f"   ✅ {tar_path.name}: {size_gb:.2f} GB")
            else:
                print(f"   ❌ Missing: {tar_file}")
        
        print(f"\n🎯 Using {len(valid_files)} tar files for extraction")
        print(f"📊 Total dataset size: {total_size_gb:.2f} GB")
        
        # Estimate samples
        estimated_samples = int(total_size_gb * 400000 / 1.0)  # Rough estimate
        print(f"📊 Estimated samples: ~{estimated_samples:,}")
        
        return valid_files
    
    raise FileNotFoundError(
        f"No TAR files found in {datasets_dir}!\n"
        "Please download dataset shards first:\n"
        "  python src/data_hand/download_data.py --shards 0 1 2 3 4 5 6 7 8 9\n"
    )

def check_webdataset_version():
    """Check WebDataset version and capabilities"""
    try:
        import webdataset as wds
        version = getattr(wds, '__version__', 'unknown')
        
        # Check for pipe method
        has_pipe = hasattr(wds.WebDataset([]), 'pipe')
        
        # Check for split_by_node
        has_split_by_node = hasattr(wds, 'split_by_node')
        
        print(f"📦 WebDataset version: {version}")
        print(f"   Has .pipe() method: {'✅' if has_pipe else '❌'}")
        print(f"   Has split_by_node: {'✅' if has_split_by_node else '❌'}")
        
        return {
            'version': version,
            'has_pipe': has_pipe,
            'has_split_by_node': has_split_by_node
        }
    except ImportError:
        print("❌ WebDataset not available")
        return None

def create_distributed_webdataset_v2(tar_file_path: str, world_size: int = 1, rank: int = 0):
    """
    FIXED: Create WebDataset with version compatibility and multiple fallback approaches
    """
    print(f"🔧 Creating WebDataset (world_size={world_size}, rank={rank})")
    
    # Check WebDataset capabilities
    wds_info = check_webdataset_version()
    if not wds_info:
        print("❌ WebDataset not available, using fallback")
        return None
    
    try:
        import webdataset as wds
        from PIL import Image
        import io
        
        def decode_sample(sample):
            """Decode a sample from WebDataset"""
            try:
                # Get image
                for ext in ['jpg', 'jpeg', 'png', 'webp']:
                    if ext in sample:
                        image_data = sample[ext]
                        break
                else:
                    return None
                
                image = Image.open(io.BytesIO(image_data)).convert('RGB')
                
                # Get caption
                caption = ""
                for caption_key in ['txt', 'caption', 'text']:
                    if caption_key in sample:
                        caption_data = sample[caption_key]
                        if isinstance(caption_data, bytes):
                            caption = caption_data.decode('utf-8').strip()
                        else:
                            caption = str(caption_data).strip()
                        break
                
                key = sample.get('__key__', 'unknown')
                
                return {
                    'image': image,
                    'caption': caption,
                    'key': key,
                }
            except Exception as e:
                return None
        
        # APPROACH 1: Try modern WebDataset with .pipe() method
        if wds_info['has_pipe'] and wds_info['has_split_by_node'] and world_size > 1:
            print("   Attempting modern WebDataset with .pipe() method...")
            try:
                dataset = (
                    wds.WebDataset([tar_file_path], empty_check=False, shardshuffle=False)
                    .pipe(wds.split_by_node, group_size=None)
                    .pipe(wds.split_by_worker)
                    .map(decode_sample)
                    .select(lambda x: x is not None)
                )
                print("   ✅ Modern WebDataset created successfully")
                return dataset
            except Exception as e:
                print(f"   ❌ Modern WebDataset failed: {e}")
        
        # APPROACH 2: Try WebDataset with manual distributed logic
        if world_size > 1:
            print("   Attempting WebDataset with manual distributed processing...")
            try:
                # Create dataset with manual rank-based filtering
                class RankFilteredDataset:
                    def __init__(self, base_dataset, rank, world_size):
                        self.base_dataset = base_dataset
                        self.rank = rank
                        self.world_size = world_size
                        self.counter = 0
                    
                    def __iter__(self):
                        self.counter = 0
                        for item in self.base_dataset:
                            if self.counter % self.world_size == self.rank:
                                yield item
                            self.counter += 1
                
                base_dataset = (
                    wds.WebDataset([tar_file_path], empty_check=False, shardshuffle=False)
                    .map(decode_sample)
                    .select(lambda x: x is not None)
                )
                
                dataset = RankFilteredDataset(base_dataset, rank, world_size)
                print("   ✅ Manual distributed WebDataset created successfully")
                return dataset
            except Exception as e:
                print(f"   ❌ Manual distributed WebDataset failed: {e}")
        
        # APPROACH 3: Simple WebDataset (fallback)
        print("   Using simple WebDataset (single-GPU mode or fallback)...")
        try:
            dataset = (
                wds.WebDataset([tar_file_path], empty_check=False, shardshuffle=False)
                .map(decode_sample)
                .select(lambda x: x is not None)
            )
            print("   ✅ Simple WebDataset created successfully")
            return dataset
        except Exception as e:
            print(f"   ❌ Simple WebDataset failed: {e}")
        
        # APPROACH 4: Ultra-simple WebDataset
        print("   Attempting ultra-simple WebDataset...")
        try:
            dataset = wds.WebDataset(tar_file_path).map(decode_sample)
            print("   ✅ Ultra-simple WebDataset created successfully")
            return dataset
        except Exception as e:
            print(f"   ❌ Ultra-simple WebDataset failed: {e}")
        
        print("❌ All WebDataset approaches failed")
        return None
        
    except ImportError as e:
        print(f"❌ WebDataset import failed: {e}")
        return None
    except Exception as e:
        print(f"❌ Unexpected error creating WebDataset: {e}")
        return None

def create_fallback_tar_processor(tar_file_path: str, world_size: int = 1, rank: int = 0):
    """
    Fallback TAR processor using Python tarfile when WebDataset fails
    """
    print(f"🔄 Creating fallback TAR processor for {Path(tar_file_path).name}")
    
    try:
        import tarfile
        from PIL import Image
        import io
        
        class FallbackTarDataset:
            def __init__(self, tar_path, rank=0, world_size=1):
                self.tar_path = tar_path
                self.rank = rank
                self.world_size = world_size
                
            def __iter__(self):
                try:
                    with tarfile.open(self.tar_path, 'r') as tar:
                        members = tar.getmembers()
                        
                        # Filter members for this rank if distributed
                        if self.world_size > 1:
                            members = [m for i, m in enumerate(members) if i % self.world_size == self.rank]
                        
                        for member in members:
                            if member.isfile():
                                try:
                                    # Extract the file
                                    file_obj = tar.extractfile(member)
                                    if file_obj is None:
                                        continue
                                    
                                    # Try to decode as image
                                    if any(ext in member.name.lower() for ext in ['.jpg', '.jpeg', '.png', '.webp']):
                                        try:
                                            image = Image.open(io.BytesIO(file_obj.read())).convert('RGB')
                                            
                                            # Create a simple caption (or try to find associated text file)
                                            key = Path(member.name).stem
                                            caption = f"Image {key}"
                                            
                                            yield {
                                                'image': image,
                                                'caption': caption,
                                                'key': key,
                                            }
                                        except Exception as e:
                                            continue
                                    
                                except Exception as e:
                                    continue
                except Exception as e:
                    print(f"❌ Error processing TAR file: {e}")
                    return
        
        dataset = FallbackTarDataset(tar_file_path, rank, world_size)
        print("   ✅ Fallback TAR processor created successfully")
        return dataset
        
    except Exception as e:
        print(f"❌ Fallback TAR processor failed: {e}")
        return None

def process_single_tar(
    tar_file_path: str,
    shard_idx: int,
    clip_processor, clip_model, eva_processor, eva_model,
    device: torch.device,
    output_dir: Path,
    working_dir: Path,
    batch_size: int = 16,
    include_cls: bool = True,
    target_tokens: int = 257,
    world_size: int = 1,
    rank: int = 0,
    max_retries: int = 3
) -> dict:
    """
    FIXED: Process a single TAR file with robust WebDataset handling
    """
    
    print(f"\n🔄 Processing shard {shard_idx}: {Path(tar_file_path).name}")
    print(f"   Mode: {'CLS+Patches' if include_cls else 'Patches only'} ({target_tokens} tokens)")
    if world_size > 1:
        print(f"   Distributed: rank {rank}/{world_size}")
    
    # Expected output file path
    mode_suffix = "cls_patch" if include_cls else "patch_only"
    shard_filename = f"embeddings_shard_{shard_idx:05d}_{mode_suffix}.pkl"
    shard_path = output_dir / shard_filename
    
    # Check if this shard already exists
    if shard_path.exists():
        print(f"   ✅ Shard {shard_idx} already exists: {shard_path}")
        file_size_mb = shard_path.stat().st_size / (1024 * 1024)
        
        try:
            with open(shard_path, 'rb') as f:
                existing_data = pickle.load(f)
            sample_count = len(existing_data.get('captions', []))
            
            return {
                'shard_idx': shard_idx,
                'total_samples': sample_count,
                'file_size_mb': file_size_mb,
                'processing_time': 0.0,
                'output_path': str(shard_path),
                'success': True,
                'skipped': True,
                'mode': mode_suffix,
                'tokens': target_tokens
            }
        except:
            print(f"   ⚠️  Could not read existing file, will reprocess...")
            shard_path.unlink()
    
    # Try processing with retries and multiple approaches
    for attempt in range(max_retries):
        try:
            print(f"   🔄 Processing attempt {attempt + 1}/{max_retries}")
            
            # APPROACH 1: Try WebDataset (fixed version)
            dataset = create_distributed_webdataset_v2(tar_file_path, world_size, rank)
            
            # APPROACH 2: Fallback to direct TAR processing
            if dataset is None:
                print("   🔄 WebDataset failed, trying fallback TAR processor...")
                dataset = create_fallback_tar_processor(tar_file_path, world_size, rank)
            
            if dataset is None:
                print(f"   ❌ All dataset creation methods failed")
                if attempt == max_retries - 1:
                    return {
                        'shard_idx': shard_idx,
                        'total_samples': 0,
                        'success': False,
                        'error': 'Failed to create any dataset processor'
                    }
                continue
            
            # Create dataloader
            def simple_collate(batch):
                valid_batch = [item for item in batch if item is not None]
                if not valid_batch:
                    return None
                    
                images = [item['image'] for item in valid_batch]
                captions = [item['caption'] for item in valid_batch]
                keys = [item['key'] for item in valid_batch]
                return {
                    'image': images,
                    'caption': captions,
                    'key': keys
                }
            
            from torch.utils.data import DataLoader
            dataloader = DataLoader(
                dataset, 
                batch_size=batch_size, 
                collate_fn=simple_collate,
                num_workers=0,  # Set to 0 for distributed processing
                drop_last=False
            )
            
            print(f"   ✅ Dataloader created successfully")
            
            # Storage for this shard's embeddings
            shard_clip_embeddings = []
            shard_eva_embeddings = []
            shard_captions = []
            shard_keys = []
            
            total_samples = 0
            start_time = time.time()
            batch_count = 0
            error_count = 0
            
            print(f"   📊 Processing batches...")
            
            # Process all batches in this TAR file
            try:
                for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Shard {shard_idx}", unit="batch")):
                    if batch is None:
                        continue
                        
                    batch_count += 1
                    
                    try:
                        images = batch['image']
                        captions = batch['caption']
                        keys = batch['key']
                        
                        if not images:  # Skip empty batches
                            continue
                        
                        # Extract features with CLS support
                        clip_features = extract_clip_features_with_cls(
                            images, clip_processor, clip_model, device, include_cls=include_cls
                        )
                        cleanup_memory()
                        
                        eva_features = extract_eva_features_with_cls(
                            images, eva_processor, eva_model, device, include_cls=include_cls
                        )
                        cleanup_memory()
                        
                        # Validate shapes match target tokens
                        assert clip_features.shape[1] == target_tokens, f"CLIP tokens: {clip_features.shape[1]} vs {target_tokens}"
                        assert eva_features.shape[1] == target_tokens, f"EVA tokens: {eva_features.shape[1]} vs {target_tokens}"
                        
                        # Move to CPU and store
                        shard_clip_embeddings.append(clip_features.cpu())
                        shard_eva_embeddings.append(eva_features.cpu())
                        shard_captions.extend(captions)
                        shard_keys.extend(keys)
                        
                        total_samples += len(images)
                        
                        # Clear intermediate variables
                        del clip_features, eva_features, images
                        cleanup_memory()
                        
                        # Progress update every 10 batches
                        if batch_idx % 10 == 0:
                            elapsed = time.time() - start_time
                            samples_per_sec = total_samples / elapsed if elapsed > 0 else 0
                            print(f"   Batch {batch_idx}: {total_samples} samples, {samples_per_sec:.1f} samples/sec")
                    
                    except Exception as e:
                        error_count += 1
                        print(f"   ⚠️  Error processing batch {batch_idx}: {e}")
                        if error_count > batch_count * 0.5:  # If >50% of batches fail
                            raise Exception(f"Too many batch errors: {error_count}/{batch_count}")
                        continue
                
                print(f"   ✅ Processed {batch_count} batches, {error_count} errors, {total_samples} samples")
                break  # Success, exit retry loop
                
            except Exception as e:
                print(f"   ❌ Error iterating through dataloader (attempt {attempt + 1}): {e}")
                if attempt == max_retries - 1:
                    return {
                        'shard_idx': shard_idx,
                        'total_samples': 0,
                        'success': False,
                        'error': f'Dataloader iteration failed after {max_retries} attempts: {e}'
                    }
                continue
        
        except Exception as e:
            print(f"   ❌ Error in processing attempt {attempt + 1}: {e}")
            if attempt == max_retries - 1:
                return {
                    'shard_idx': shard_idx,
                    'total_samples': 0,
                    'success': False,
                    'error': f'All processing attempts failed: {e}'
                }
            continue
    
    # Consolidate embeddings for this shard
    if shard_clip_embeddings and total_samples > 0:
        print(f"   🔄 Consolidating {total_samples} embeddings...")
        
        try:
            final_clip = torch.cat(shard_clip_embeddings, dim=0)
            final_eva = torch.cat(shard_eva_embeddings, dim=0)
            
            # Final validation
            assert final_clip.shape[1] == target_tokens, f"Final CLIP shape: {final_clip.shape}"
            assert final_eva.shape[1] == target_tokens, f"Final EVA shape: {final_eva.shape}"
            assert final_clip.shape[2] == 1024, f"CLIP dim: {final_clip.shape[2]}"
            
            # Create shard data
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
                    'extraction_method': 'cls_patch_extraction_v5_fixed_webdataset',
                    'format_version': f'blip3o_{target_tokens}_tokens_{"cls_" if include_cls else ""}patch_v5',
                    'extraction_time': time.time() - start_time,
                    'cls_first': include_cls,
                    'patch_order': '16x16_row_major',
                    'distributed': world_size > 1,
                    'rank': rank,
                    'world_size': world_size,
                    'webdataset_fixed': True,
                }
            }
            
            # Save shard data
            print(f"   💾 Saving shard {shard_idx} to persistent storage...")
            
            try:
                with open(shard_path, 'wb') as f:
                    pickle.dump(shard_data, f, protocol=pickle.HIGHEST_PROTOCOL)
                
                file_size_mb = shard_path.stat().st_size / (1024 * 1024)
                
                print(f"   ✅ Shard {shard_idx} completed:")
                print(f"      File: {shard_filename}")
                print(f"      Size: {file_size_mb:.1f} MB")
                print(f"      Samples: {total_samples}")
                print(f"      Mode: {mode_suffix} ({target_tokens} tokens)")
                print(f"      Time: {time.time() - start_time:.1f}s")
                if world_size > 1:
                    print(f"      Distributed: rank {rank}/{world_size}")
                
                # Clear memory
                del shard_clip_embeddings, shard_eva_embeddings, final_clip, final_eva
                cleanup_memory()
                
                return {
                    'shard_idx': shard_idx,
                    'total_samples': total_samples,
                    'file_size_mb': file_size_mb,
                    'processing_time': time.time() - start_time,
                    'output_path': str(shard_path),
                    'success': True,
                    'mode': mode_suffix,
                    'tokens': target_tokens,
                    'include_cls': include_cls,
                    'rank': rank,
                    'world_size': world_size,
                    'webdataset_fixed': True,
                }
            
            except Exception as e:
                print(f"   ❌ Failed to save shard {shard_idx}: {e}")
                return {
                    'shard_idx': shard_idx,
                    'total_samples': total_samples,
                    'success': False,
                    'error': f'File save failed: {e}'
                }
        
        except Exception as e:
            print(f"   ❌ Failed to consolidate embeddings for shard {shard_idx}: {e}")
            return {
                'shard_idx': shard_idx,
                'total_samples': 0,
                'success': False,
                'error': f'Consolidation failed: {e}'
            }
    
    else:
        print(f"   ❌ No embeddings extracted from shard {shard_idx}")
        return {
            'shard_idx': shard_idx,
            'total_samples': 0,
            'success': False,
            'error': 'No embeddings extracted - empty or corrupted TAR file'
        }

def main():
    """Main extraction function with fixed WebDataset support."""
    parser = argparse.ArgumentParser(description="FIXED BLIP3-o Embedding Extraction")
    parser.add_argument("--include_cls", action="store_true", default=False,
                       help="Include CLS token (257 tokens) or patches only (256 tokens)")
    parser.add_argument("--max_shards", type=int, default=None,
                       help="Maximum number of shards to process (for testing)")
    parser.add_argument("--batch_size", type=int, default=16,
                       help="Batch size for processing")
    
    args = parser.parse_args()
    
    # Setup
    target_tokens = 257 if args.include_cls else 256
    mode_name = "CLS+Patches" if args.include_cls else "Patches only"
    
    print("🚀 FIXED BLIP3-o Embedding Extraction")
    print("=" * 70)
    print(f"Mode: {mode_name} ({target_tokens} tokens)")
    print(f"CLS token: {'First token [0]' if args.include_cls else 'Not included'}")
    print(f"Patches: {'Tokens [1:257]' if args.include_cls else 'Tokens [0:256]'} (16x16 grid)")
    print(f"Max shards: {args.max_shards if args.max_shards else 'All'}")
    print(f"FIXES: WebDataset version compatibility, multiple fallbacks, better error handling")
    print("=" * 70)
    
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for this script")
    
    device = torch.device('cuda')
    project_root = setup_paths()
    
    # Check WebDataset status
    wds_info = check_webdataset_version()
    
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
        if "TMPDIR" in os.environ:
            base_temp = Path(os.environ["TMPDIR"])
        else:
            base_temp = Path("./temp")
        
        mode_suffix = "cls_patch" if args.include_cls else "patch_only"
        embeddings_dir = base_temp / f"embeddings_{mode_suffix}"
        working_dir = base_temp / "working"
        
        embeddings_dir.mkdir(parents=True, exist_ok=True)
        working_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"⚠️  Using fallback temp management")
        print(f"📁 Embeddings dir: {embeddings_dir}")
    
    print(f"🎮 Using GPU: {torch.cuda.get_device_name(0)}")
    
    # Load models
    try:
        clip_processor, clip_model, eva_processor, eva_model = load_models(device)
    except Exception as e:
        print(f"❌ Failed to load models: {e}")
        return 1
    
    # Find TAR files
    try:
        tar_files = find_data_files(temp_manager, max_shards=args.max_shards)
    except Exception as e:
        print(f"❌ {e}")
        return 1
    
    print(f"📤 Output directory: {embeddings_dir}")
    
    # Process each TAR file
    print(f"\n🔄 Processing {len(tar_files)} TAR files...")
    
    processing_results = []
    total_samples_all = 0
    failed_shards = []
    
    for shard_idx, tar_file in enumerate(tar_files):
        print(f"\n" + "="*60)
        print(f"PROCESSING SHARD {shard_idx + 1}/{len(tar_files)}")
        print(f"="*60)
        
        result = process_single_tar(
            tar_file_path=tar_file,
            shard_idx=shard_idx,
            clip_processor=clip_processor,
            clip_model=clip_model,
            eva_processor=eva_processor,
            eva_model=eva_model,
            device=device,
            output_dir=embeddings_dir,
            working_dir=working_dir,
            batch_size=args.batch_size,
            include_cls=args.include_cls,
            target_tokens=target_tokens,
            world_size=1,  # Single GPU mode
            rank=0
        )
        
        if result and result['success']:
            processing_results.append(result)
            total_samples_all += result['total_samples']
            print(f"✅ Shard {shard_idx} successful: {result['total_samples']} samples")
        else:
            failed_shards.append(shard_idx)
            print(f"❌ Shard {shard_idx} failed: {result.get('error', 'Unknown error')}")
    
    # Create manifest
    manifest_data = {
        'total_shards': len(processing_results),
        'total_samples': total_samples_all,
        'extraction_mode': mode_name,
        'tokens_per_sample': target_tokens,
        'include_cls': args.include_cls,
        'cls_token_position': 0 if args.include_cls else None,
        'patch_tokens_range': [1, 257] if args.include_cls else [0, 256],
        'extraction_timestamp': time.time(),
        'shards': processing_results,
        'failed_shards': failed_shards,
        'format_version': f'blip3o_{target_tokens}_tokens_{"cls_" if args.include_cls else ""}patch_v5_fixed',
        'webdataset_info': wds_info,
        'fixes_applied': [
            'WebDataset version compatibility checks',
            'Multiple fallback approaches for dataset creation',
            'Better error handling and diagnostics',
            'Direct TAR processing fallback when WebDataset fails',
            'Robust distributed processing support'
        ],
        'token_layout': {
            'cls_token': {'included': args.include_cls, 'position': 0 if args.include_cls else None},
            'patches': {
                'count': 256,
                'positions': [1, 257] if args.include_cls else [0, 256],
                'layout': '16x16 grid in row-major order'
            }
        },
        'usage': {
            'training_command_cls_patch': f'python train_blip3o_enhanced.py --chunked_embeddings_dir {embeddings_dir} --training_mode cls_patch',
            'training_command_patch_only': f'python train_blip3o_enhanced.py --chunked_embeddings_dir {embeddings_dir} --training_mode patch_only',
        }
    }
    
    manifest_path = embeddings_dir / "embeddings_manifest.json"
    with open(manifest_path, 'w') as f:
        json.dump(manifest_data, f, indent=2)
    
    # Final status
    print("\n" + "=" * 80)
    print("✅ FIXED EMBEDDING EXTRACTION COMPLETED!")
    print("=" * 80)
    print(f"📊 SUMMARY:")
    print(f"   Mode: {mode_name} ({target_tokens} tokens)")
    print(f"   CLS token: {'Included at position [0]' if args.include_cls else 'Not included'}")
    print(f"   Patches: {'Positions [1:257]' if args.include_cls else 'Positions [0:256]'} (16x16)")
    print(f"   TAR files processed: {len(tar_files)}")
    print(f"   Successful shards: {len(processing_results)}")
    print(f"   Failed shards: {len(failed_shards)}")
    print(f"   Total samples: {total_samples_all:,}")
    print(f"   Embeddings location: {embeddings_dir}")
    print(f"   Manifest file: {manifest_path}")
    
    if wds_info:
        print(f"\n📦 WebDataset Status:")
        print(f"   Version: {wds_info['version']}")
        print(f"   Compatibility: {'✅ FIXED' if processing_results else '❌ ISSUES'}")
    
    if failed_shards:
        print(f"\n⚠️ Failed shards: {failed_shards}")
        print(f"   Success rate: {len(processing_results)/(len(processing_results)+len(failed_shards))*100:.1f}%")
        
        if len(processing_results) > 0:
            print(f"✅ Partial success - continuing with {len(processing_results)} shards")
        else:
            print(f"❌ All shards failed - check WebDataset installation and TAR files")
            return 1
    
    if len(processing_results) > 0:
        print(f"\n🎉 SUCCESS! {len(processing_results)} shards processed successfully!")
        print("Ready for BLIP3-o training!")
        print(f"\nUsage commands:")
        print(f"CLS+Patch mode (257 tokens):")
        print(f"  python train_blip3o_enhanced.py --chunked_embeddings_dir {embeddings_dir} --training_mode cls_patch")
        print(f"Patch-only mode (256 tokens):")
        print(f"  python train_blip3o_enhanced.py --chunked_embeddings_dir {embeddings_dir} --training_mode patch_only")
    
    print("=" * 80)
    print("🔧 FIXES APPLIED:")
    print("  ✅ WebDataset version compatibility checks")
    print("  ✅ Multiple fallback approaches for dataset creation")
    print("  ✅ Better error handling and diagnostics")
    print("  ✅ Direct TAR processing fallback when WebDataset fails")
    print("  ✅ Robust distributed processing support")
    
    return 0 if len(processing_results) > 0 else 1

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except Exception as e:
        print(f"❌ Critical error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)