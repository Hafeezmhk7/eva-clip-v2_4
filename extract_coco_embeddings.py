#!/usr/bin/env python3
"""
MS-COCO Embedding Extraction for BLIP3-o Evaluation
extract_coco_embeddings.py

Extracts CLIP and EVA-CLIP embeddings from MS-COCO validation dataset
and saves them for memory-efficient evaluation.

Usage:
    python extract_coco_embeddings.py --coco_root ./data/coco --output_dir /path/to/output
"""

import os
import sys
import argparse
import torch
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPModel, AutoModel, CLIPImageProcessor
import pickle
from tqdm import tqdm
import numpy as np
from pathlib import Path
import gc
import time
import json
import logging
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Any, Optional

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Setup paths
sys.path.insert(0, str(Path(__file__).parent))

def setup_temp_manager():
    """Setup temp manager for structured directory management."""
    try:
        from src.modules.utils.temp_manager import setup_snellius_environment
        manager = setup_snellius_environment("blip3o_workspace")
        return manager
    except ImportError:
        logger.warning("⚠️  Temp manager not available, using fallback directories")
        return None

def get_memory_usage():
    """Get current GPU memory usage in GB"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        return allocated, reserved
    return 0.0, 0.0

def cleanup_memory():
    """Aggressive memory cleanup"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

class COCODataset(Dataset):
    """MS-COCO validation dataset for embedding extraction"""
    
    def __init__(self, coco_root: str, max_samples: int = None):
        self.coco_root = Path(coco_root)
        self.max_samples = max_samples
        
        # Load annotations
        annotations_file = self.coco_root / "annotations" / "captions_val2017.json"
        images_dir = self.coco_root / "val2017"
        
        # Check different possible paths
        if not images_dir.exists():
            images_dir = self.coco_root / "images" / "val2017"
        
        if not annotations_file.exists():
            logger.error(f"❌ Annotations file not found: {annotations_file}")
            raise FileNotFoundError(f"Annotations file not found: {annotations_file}")
        
        if not images_dir.exists():
            logger.error(f"❌ Images directory not found: {images_dir}")
            raise FileNotFoundError(f"Images directory not found: {images_dir}")
        
        # Load COCO annotations
        with open(annotations_file, 'r') as f:
            coco_data = json.load(f)
        
        # Get valid images with captions
        images_info = {img['id']: img for img in coco_data['images']}
        
        # Group captions by image
        captions_by_image = {}
        for ann in coco_data['annotations']:
            img_id = ann['image_id']
            if img_id not in captions_by_image:
                captions_by_image[img_id] = []
            captions_by_image[img_id].append(ann['caption'])
        
        # Prepare valid samples
        valid_samples = []
        for img_id, img_info in images_info.items():
            if img_id in captions_by_image:
                img_path = images_dir / img_info['file_name']
                if img_path.exists():
                    # Use the first caption for each image
                    caption = captions_by_image[img_id][0]
                    valid_samples.append({
                        'image_id': img_id,
                        'file_name': img_info['file_name'],
                        'image_path': img_path,
                        'caption': caption,
                        'all_captions': captions_by_image[img_id]
                    })
                    
                    if max_samples and len(valid_samples) >= max_samples:
                        break
        
        self.samples = valid_samples
        logger.info(f"✅ Loaded {len(self.samples)} MS-COCO validation samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        try:
            image = Image.open(sample['image_path']).convert('RGB')
            return {
                'image': image,
                'image_id': sample['image_id'],
                'file_name': sample['file_name'],
                'caption': sample['caption'],
                'all_captions': sample['all_captions']
            }
        except Exception as e:
            logger.warning(f"⚠️ Error loading image {sample['file_name']}: {e}")
            # Return a dummy black image
            dummy_image = Image.new('RGB', (224, 224), color='black')
            return {
                'image': dummy_image,
                'image_id': sample['image_id'],
                'file_name': sample['file_name'],
                'caption': sample['caption'],
                'all_captions': sample['all_captions']
            }

def load_models(device, use_half_precision=True):
    """Load CLIP and EVA-CLIP models"""
    logger.info("📦 Loading CLIP and EVA-CLIP models...")
    
    dtype = torch.float16 if use_half_precision else torch.float32
    
    # Load CLIP ViT-L/14
    logger.info("   Loading CLIP ViT-L/14...")
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    clip_model = CLIPModel.from_pretrained(
        "openai/clip-vit-large-patch14",
        torch_dtype=dtype
    ).to(device)
    clip_model.eval()
    
    # Load EVA-CLIP-8B
    logger.info("   Loading EVA-CLIP-8B...")
    eva_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")
    eva_model = AutoModel.from_pretrained(
        "BAAI/EVA-CLIP-8B", 
        trust_remote_code=True,
        torch_dtype=dtype
    ).to(device)
    eva_model.eval()
    
    cleanup_memory()
    
    allocated, reserved = get_memory_usage()
    logger.info(f"✅ Models loaded successfully")
    logger.info(f"💾 GPU memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
    
    return clip_processor, clip_model, eva_processor, eva_model

def extract_clip_features(images, processor, model, device, include_cls=True):
    """Extract CLIP features with CLS token + patches"""
    features = []
    
    # Get model dtype
    model_dtype = next(model.parameters()).dtype
    
    for img in images:
        inputs = processor(images=img, return_tensors="pt")
        # Ensure inputs match model dtype
        inputs = {
            k: v.to(device, dtype=model_dtype) if v.dtype.is_floating_point else v.to(device)
            for k, v in inputs.items()
        }
        
        with torch.no_grad():
            vision_outputs = model.vision_model(
                pixel_values=inputs['pixel_values'],
                output_hidden_states=True,
                return_dict=True
            )
            
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
    
    cleanup_memory()
    return torch.stack(features)

def extract_eva_features(images, processor, model, device, include_cls=True):
    """Extract EVA-CLIP features with CLS token + patches"""
    features = []
    
    # Get model dtype
    model_dtype = next(model.parameters()).dtype
    
    for img in images:
        inputs = processor(images=img, return_tensors="pt")
        # Ensure inputs match model dtype
        pixel_values = inputs['pixel_values'].to(device, dtype=model_dtype)
        
        with torch.no_grad():
            vision_outputs = model.vision_model(
                pixel_values=pixel_values,
                output_hidden_states=True,
                return_dict=True
            )
            
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
    
    cleanup_memory()
    return torch.stack(features)

def process_coco_embeddings(
    coco_root: str,
    output_dir: Path,
    device: torch.device,
    batch_size: int = 8,
    max_samples: int = None,
    include_cls: bool = True,
    use_half_precision: bool = True,
    save_every_n_batches: int = 50
):
    """Process MS-COCO validation set and extract embeddings"""
    
    logger.info(f"🔄 Processing MS-COCO validation set...")
    logger.info(f"   COCO root: {coco_root}")
    logger.info(f"   Output directory: {output_dir}")
    logger.info(f"   Batch size: {batch_size}")
    logger.info(f"   Max samples: {max_samples if max_samples else 'All'}")
    logger.info(f"   Include CLS: {include_cls}")
    logger.info(f"   Half precision: {use_half_precision}")
    
    # Load models
    clip_processor, clip_model, eva_processor, eva_model = load_models(device, use_half_precision)
    
    # Create dataset
    dataset = COCODataset(coco_root=coco_root, max_samples=max_samples)
    
    def collate_fn(batch):
        """Custom collate function to handle batch processing"""
        images = [item['image'] for item in batch if item['image'] is not None]
        image_ids = [item['image_id'] for item in batch]
        file_names = [item['file_name'] for item in batch]
        captions = [item['caption'] for item in batch]
        all_captions = [item['all_captions'] for item in batch]
        
        return {
            'images': images,
            'image_ids': image_ids,
            'file_names': file_names,
            'captions': captions,
            'all_captions': all_captions
        }
    
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=2,
        collate_fn=collate_fn
    )
    
    # Storage for embeddings
    all_clip_embeddings = []
    all_eva_embeddings = []
    all_metadata = []
    
    total_batches = len(dataloader)
    logger.info(f"📊 Processing {total_batches} batches...")
    
    batch_data = []
    
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Extracting embeddings")):
        try:
            images = batch['images']
            
            if not images:
                logger.warning(f"⚠️ Empty batch {batch_idx}, skipping...")
                continue
            
            # Extract CLIP features
            clip_features = extract_clip_features(
                images, clip_processor, clip_model, device, include_cls=include_cls
            )
            
            # Extract EVA features
            eva_features = extract_eva_features(
                images, eva_processor, eva_model, device, include_cls=include_cls
            )
            
            # Store embeddings and metadata
            all_clip_embeddings.append(clip_features)
            all_eva_embeddings.append(eva_features)
            
            # Store metadata for this batch
            batch_metadata = []
            for i in range(len(images)):
                metadata = {
                    'image_id': batch['image_ids'][i],
                    'file_name': batch['file_names'][i],
                    'caption': batch['captions'][i],
                    'all_captions': batch['all_captions'][i],
                    'batch_idx': batch_idx,
                    'sample_idx': i
                }
                batch_metadata.append(metadata)
                all_metadata.extend(batch_metadata)
            
            # Save intermediate results every N batches to avoid memory issues
            if (batch_idx + 1) % save_every_n_batches == 0:
                logger.info(f"💾 Saving intermediate results after batch {batch_idx + 1}/{total_batches}")
                
                # Concatenate current batch data
                if all_clip_embeddings:
                    clip_embeddings_tensor = torch.cat(all_clip_embeddings, dim=0)
                    eva_embeddings_tensor = torch.cat(all_eva_embeddings, dim=0)
                    
                    # Save intermediate file
                    intermediate_file = output_dir / f"coco_embeddings_batch_{batch_idx + 1}.pkl"
                    intermediate_data = {
                        'clip_embeddings': clip_embeddings_tensor,
                        'eva_embeddings': eva_embeddings_tensor,
                        'metadata': all_metadata.copy(),
                        'config': {
                            'include_cls': include_cls,
                            'tokens': 257 if include_cls else 256,
                            'clip_dim': 1024,
                            'eva_dim': eva_embeddings_tensor.shape[-1],
                            'batch_size': batch_size,
                            'end_batch': batch_idx + 1,
                            'samples_count': len(all_metadata)
                        }
                    }
                    
                    with open(intermediate_file, 'wb') as f:
                        pickle.dump(intermediate_data, f, protocol=pickle.HIGHEST_PROTOCOL)
                    
                    logger.info(f"   Saved {len(all_metadata)} samples to {intermediate_file}")
                    
                    # Clear memory
                    all_clip_embeddings.clear()
                    all_eva_embeddings.clear()
                    all_metadata.clear()
                    
                    del clip_embeddings_tensor, eva_embeddings_tensor, intermediate_data
                    cleanup_memory()
            
            # Memory monitoring
            allocated, reserved = get_memory_usage()
            if allocated > 80:  # Warning if using > 80GB
                logger.warning(f"⚠️ High GPU memory usage: {allocated:.1f}GB")
                cleanup_memory()
            
        except Exception as e:
            logger.error(f"❌ Error processing batch {batch_idx}: {e}")
            continue
    
    # Save final batch if any remaining
    if all_clip_embeddings:
        logger.info(f"💾 Saving final batch data...")
        
        clip_embeddings_tensor = torch.cat(all_clip_embeddings, dim=0)
        eva_embeddings_tensor = torch.cat(all_eva_embeddings, dim=0)
        
        final_file = output_dir / f"coco_embeddings_final.pkl"
        final_data = {
            'clip_embeddings': clip_embeddings_tensor,
            'eva_embeddings': eva_embeddings_tensor,
            'metadata': all_metadata,
            'config': {
                'include_cls': include_cls,
                'tokens': 257 if include_cls else 256,
                'clip_dim': 1024,
                'eva_dim': eva_embeddings_tensor.shape[-1],
                'batch_size': batch_size,
                'is_final': True,
                'samples_count': len(all_metadata)
            }
        }
        
        with open(final_file, 'wb') as f:
            pickle.dump(final_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        logger.info(f"   Saved final {len(all_metadata)} samples to {final_file}")
    
    logger.info("✅ MS-COCO embedding extraction completed!")

def consolidate_embeddings(output_dir: Path, include_cls: bool = True):
    """Consolidate all intermediate embedding files into a single file"""
    
    logger.info(f"🔄 Consolidating embedding files in {output_dir}...")
    
    # Find all embedding files
    embedding_files = sorted(output_dir.glob("coco_embeddings_*.pkl"))
    
    if not embedding_files:
        logger.error("❌ No embedding files found to consolidate!")
        return None
    
    logger.info(f"Found {len(embedding_files)} embedding files to consolidate")
    
    all_clip_embeddings = []
    all_eva_embeddings = []
    all_metadata = []
    total_samples = 0
    
    for file_path in embedding_files:
        logger.info(f"   Loading {file_path.name}...")
        
        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            
            all_clip_embeddings.append(data['clip_embeddings'])
            all_eva_embeddings.append(data['eva_embeddings'])
            all_metadata.extend(data['metadata'])
            total_samples += data['config']['samples_count']
            
            logger.info(f"     Added {data['config']['samples_count']} samples")
            
        except Exception as e:
            logger.error(f"   ❌ Error loading {file_path}: {e}")
            continue
    
    if not all_clip_embeddings:
        logger.error("❌ No valid embedding data found!")
        return None
    
    # Concatenate all embeddings
    logger.info(f"🔗 Concatenating {len(all_clip_embeddings)} embedding tensors...")
    
    final_clip_embeddings = torch.cat(all_clip_embeddings, dim=0)
    final_eva_embeddings = torch.cat(all_eva_embeddings, dim=0)
    
    logger.info(f"✅ Consolidated embeddings:")
    logger.info(f"   CLIP embeddings: {final_clip_embeddings.shape}")
    logger.info(f"   EVA embeddings: {final_eva_embeddings.shape}")
    logger.info(f"   Total samples: {total_samples}")
    
    # Create final consolidated file
    consolidated_data = {
        'clip_embeddings': final_clip_embeddings,
        'eva_embeddings': final_eva_embeddings,
        'metadata': all_metadata,
        'config': {
            'include_cls': include_cls,
            'tokens': 257 if include_cls else 256,
            'clip_dim': 1024,
            'eva_dim': final_eva_embeddings.shape[-1],
            'total_samples': total_samples,
            'consolidated': True,
            'source_files': [f.name for f in embedding_files],
            'extraction_timestamp': time.time(),
            'format_version': 'coco_embeddings_v1'
        }
    }
    
    # Save consolidated file
    consolidated_file = output_dir / "coco_embeddings_consolidated.pkl"
    logger.info(f"💾 Saving consolidated embeddings to {consolidated_file}...")
    
    with open(consolidated_file, 'wb') as f:
        pickle.dump(consolidated_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    file_size_gb = consolidated_file.stat().st_size / (1024**3)
    logger.info(f"✅ Consolidated file saved: {file_size_gb:.2f} GB")
    
    # Create manifest file
    manifest = {
        'consolidated_file': str(consolidated_file),
        'total_samples': total_samples,
        'clip_shape': list(final_clip_embeddings.shape),
        'eva_shape': list(final_eva_embeddings.shape),
        'file_size_gb': file_size_gb,
        'include_cls': include_cls,
        'tokens': 257 if include_cls else 256,
        'created': time.strftime('%Y-%m-%d %H:%M:%S'),
        'usage': {
            'evaluation_script': 'eval_blip3o_coco.py --use_precomputed_embeddings',
            'embeddings_file': str(consolidated_file)
        }
    }
    
    manifest_file = output_dir / "coco_embeddings_manifest.json"
    with open(manifest_file, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    logger.info(f"📋 Manifest saved to: {manifest_file}")
    
    return consolidated_file

def main():
    """Main function for MS-COCO embedding extraction"""
    
    parser = argparse.ArgumentParser(description="Extract MS-COCO embeddings for BLIP3-o evaluation")
    parser.add_argument("--coco_root", type=str, required=True,
                       help="Path to MS-COCO dataset root directory")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Output directory for embeddings (default: auto-detect from temp manager)")
    parser.add_argument("--batch_size", type=int, default=8,
                       help="Batch size for processing")
    parser.add_argument("--max_samples", type=int, default=None,
                       help="Maximum number of samples to process (None for all)")
    parser.add_argument("--include_cls", action="store_true", default=True,
                       help="Include CLS token (257 tokens) or patches only (256 tokens)")
    parser.add_argument("--disable_half_precision", action="store_true",
                       help="Disable half precision (use float32)")
    parser.add_argument("--save_every_n_batches", type=int, default=50,
                       help="Save intermediate results every N batches")
    parser.add_argument("--consolidate_only", action="store_true",
                       help="Only consolidate existing embedding files")
    
    args = parser.parse_args()
    
    logger.info("🚀 MS-COCO Embedding Extraction for BLIP3-o Evaluation")
    logger.info("=" * 70)
    logger.info(f"COCO root: {args.coco_root}")
    logger.info(f"Include CLS: {args.include_cls} ({'257' if args.include_cls else '256'} tokens)")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Max samples: {args.max_samples if args.max_samples else 'All'}")
    logger.info(f"Half precision: {not args.disable_half_precision}")
    
    # Setup output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        # Try to use temp manager
        temp_manager = setup_temp_manager()
        if temp_manager:
            output_dir = temp_manager.get_working_dir() / "coco_embeddings"
            logger.info(f"✅ Using temp manager working directory: {output_dir}")
        else:
            output_dir = Path("./coco_embeddings")
            logger.warning(f"⚠️ Using fallback directory: {output_dir}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"📁 Output directory: {output_dir}")
    
    # Check CUDA
    if not torch.cuda.is_available():
        logger.error("❌ CUDA not available - this script requires GPU")
        return 1
    
    device = torch.device("cuda")
    logger.info(f"🎮 Using GPU: {torch.cuda.get_device_name(0)}")
    
    try:
        if args.consolidate_only:
            # Only consolidate existing files
            logger.info("🔗 Consolidating existing embedding files...")
            consolidated_file = consolidate_embeddings(output_dir, args.include_cls)
            if consolidated_file:
                logger.info(f"✅ Consolidation completed: {consolidated_file}")
                return 0
            else:
                logger.error("❌ Consolidation failed")
                return 1
        else:
            # Extract embeddings
            process_coco_embeddings(
                coco_root=args.coco_root,
                output_dir=output_dir,
                device=device,
                batch_size=args.batch_size,
                max_samples=args.max_samples,
                include_cls=args.include_cls,
                use_half_precision=not args.disable_half_precision,
                save_every_n_batches=args.save_every_n_batches
            )
            
            # Consolidate all files
            logger.info("🔗 Consolidating embedding files...")
            consolidated_file = consolidate_embeddings(output_dir, args.include_cls)
            
            if consolidated_file:
                logger.info(f"🎉 MS-COCO embedding extraction completed successfully!")
                logger.info(f"📁 Consolidated embeddings: {consolidated_file}")
                logger.info(f"")
                logger.info(f"💡 Usage in evaluation:")
                logger.info(f"   python eval_blip3o_coco.py \\")
                logger.info(f"     --model_path /path/to/model \\")
                logger.info(f"     --coco_embeddings_file {consolidated_file}")
                return 0
            else:
                logger.error("❌ Consolidation failed")
                return 1
        
    except Exception as e:
        logger.error(f"❌ Extraction failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)