#!/usr/bin/env python3
"""
COCO Evaluation WITHOUT CLIP Normalization
eval_blip3o_coco.py

CHANGES:
1. Removed all normalization/denormalization logic
2. Works directly with raw CLIP embeddings
3. Simplified evaluation to work in original CLIP space
4. Removed clip_normalizer dependencies
"""

import os
import sys
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import json
import numpy as np
from pathlib import Path
import logging
from tqdm import tqdm
import time
import pickle
from typing import Dict, Any, Optional, Union, Tuple, List

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Setup paths
sys.path.insert(0, str(Path(__file__).parent))

# Import BLIP3-o modules
try:
    from src.modules.models.blip3o_dit import ImprovedBLIP3oCLIPDiTModel, BLIP3oCLIPDiTConfig
    logger.info("✅ BLIP3-o modules imported successfully")
except ImportError as e:
    logger.error(f"❌ Failed to import BLIP3-o modules: {e}")
    sys.exit(1)


class COCODatasetAsTrainingFormat(Dataset):
    """
    COCO dataset that returns items in same format as training dataset
    Works with raw CLIP embeddings (no normalization)
    """
    
    def __init__(self, embeddings_file: str, max_samples: int = None, training_mode: str = "patch_only"):
        self.embeddings_file = Path(embeddings_file)
        self.training_mode = training_mode
        self.expected_tokens = 257 if training_mode == "cls_patch" else 256
        
        logger.info(f"📂 Loading COCO embeddings: {self.embeddings_file}")
        logger.info(f"🎯 Training mode: {training_mode} ({self.expected_tokens} tokens)")
        logger.info(f"🔑 Working with RAW CLIP embeddings (NO NORMALIZATION)")
        
        # Load embeddings
        self._load_embeddings()
        
        # Apply max_samples limit
        if max_samples is not None and max_samples < len(self.metadata):
            self.clip_embeddings = self.clip_embeddings[:max_samples]
            self.eva_embeddings = self.eva_embeddings[:max_samples]  
            self.metadata = self.metadata[:max_samples]
            logger.info(f"🔢 Limited to {max_samples} samples")
        
        self.num_samples = len(self.metadata)
        
        logger.info(f"✅ COCO dataset loaded:")
        logger.info(f"   Samples: {self.num_samples:,}")
        logger.info(f"   CLIP shape: {self.clip_embeddings.shape}")
        logger.info(f"   EVA shape: {self.eva_embeddings.shape}")
    
    def _load_embeddings(self):
        """Load and process COCO embeddings"""
        with open(self.embeddings_file, 'rb') as f:
            data = pickle.load(f)
        
        # Extract data
        clip_emb = data['clip_embeddings']
        eva_emb = data['eva_embeddings']
        metadata = data['metadata']
        
        # Convert to tensors
        if not torch.is_tensor(clip_emb):
            clip_emb = torch.tensor(clip_emb, dtype=torch.float32)
        if not torch.is_tensor(eva_emb):
            eva_emb = torch.tensor(eva_emb, dtype=torch.float32)
        
        logger.info(f"📊 Raw tensor shapes:")
        logger.info(f"   CLIP: {clip_emb.shape}")
        logger.info(f"   EVA: {eva_emb.shape}")
        logger.info(f"   Metadata: {len(metadata)}")
        
        # Handle different tensor formats to get [num_samples, seq_len, embed_dim]
        if clip_emb.dim() == 2:
            # Format: [total_tokens, embed_dim] -> [num_samples, tokens_per_sample, embed_dim]
            num_samples = len(metadata)
            tokens_per_sample = clip_emb.shape[0] // num_samples
            
            clip_emb = clip_emb.view(num_samples, tokens_per_sample, -1)
            eva_emb = eva_emb.view(num_samples, tokens_per_sample, -1)
            
            logger.info(f"🔧 Reshaped from 2D to 3D:")
            logger.info(f"   CLIP: {clip_emb.shape}")
            logger.info(f"   EVA: {eva_emb.shape}")
        
        # Adapt token count if needed
        current_tokens = clip_emb.shape[1]
        if current_tokens != self.expected_tokens:
            logger.info(f"🔧 Adapting tokens: {current_tokens} -> {self.expected_tokens}")
            
            if current_tokens == 256 and self.expected_tokens == 257:
                # Add CLS token
                clip_cls = clip_emb.mean(dim=1, keepdim=True)
                eva_cls = eva_emb.mean(dim=1, keepdim=True)
                clip_emb = torch.cat([clip_cls, clip_emb], dim=1)
                eva_emb = torch.cat([eva_cls, eva_emb], dim=1)
                
            elif current_tokens == 257 and self.expected_tokens == 256:
                # Remove CLS token
                clip_emb = clip_emb[:, 1:, :]
                eva_emb = eva_emb[:, 1:, :]
        
        # Final validation
        assert clip_emb.shape[1] == self.expected_tokens, f"CLIP tokens: {clip_emb.shape[1]} vs {self.expected_tokens}"
        assert eva_emb.shape[1] == self.expected_tokens, f"EVA tokens: {eva_emb.shape[1]} vs {self.expected_tokens}"
        assert clip_emb.shape[2] == 1024, f"CLIP dim: {clip_emb.shape[2]}"
        assert eva_emb.shape[2] == 4096, f"EVA dim: {eva_emb.shape[2]}"
        
        self.clip_embeddings = clip_emb
        self.eva_embeddings = eva_emb
        self.metadata = metadata
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        """Return item in same format as training dataset"""
        # Get embeddings (raw CLIP embeddings - no normalization)
        eva_emb = self.eva_embeddings[idx]  # [seq_len, 4096]
        clip_emb = self.clip_embeddings[idx]  # [seq_len, 1024]
        
        # Get metadata
        metadata_item = self.metadata[idx]
        if isinstance(metadata_item, dict):
            caption = metadata_item.get('caption', '')
            image_id = metadata_item.get('image_id', idx)
        else:
            caption = str(metadata_item)
            image_id = idx
        
        # Return in same format as training dataset
        return {
            'eva_embeddings': eva_emb,
            'clip_embeddings': clip_emb,  # Raw CLIP embeddings
            'caption': caption,
            'key': f"coco_{image_id}",
            'sample_idx': idx,
            'training_mode': self.training_mode,
            'num_tokens': self.expected_tokens,
        }


class SimpleModelLoader:
    """Simple model loader"""
    
    def __init__(self, model_path: str, device: torch.device):
        self.model_path = Path(model_path)
        self.device = device
    
    def load_model(self):
        """Load model with automatic config detection"""
        checkpoint_files = list(self.model_path.glob("checkpoint_step_*.pt"))
        if not checkpoint_files:
            raise FileNotFoundError(f"No checkpoint files found in {self.model_path}")
        
        checkpoint_file = max(checkpoint_files, key=lambda x: int(x.stem.split('_')[-1]))
        logger.info(f"Loading checkpoint: {checkpoint_file}")
        
        checkpoint = torch.load(checkpoint_file, map_location=self.device)
        state_dict = checkpoint['model_state_dict']
        
        # Auto-detect config from state dict
        hidden_size = state_dict['blocks.0.mlp.gate_proj.weight'].shape[1]
        intermediate_size = state_dict['blocks.0.mlp.gate_proj.weight'].shape[0]
        num_layers = max([int(key.split('.')[1]) for key in state_dict.keys() 
                         if key.startswith('blocks.') and '.mlp.gate_proj.weight' in key]) + 1
        
        # Try common configurations
        for num_heads, num_kv_heads in [(12, 4), (16, 4), (8, 4)]:
            if hidden_size % num_heads == 0:
                try:
                    config = BLIP3oCLIPDiTConfig(
                        hidden_size=hidden_size,
                        num_hidden_layers=num_layers,
                        num_attention_heads=num_heads,
                        num_key_value_heads=num_kv_heads,
                        intermediate_size=intermediate_size,
                        eva_embedding_size=4096,
                        clip_embedding_size=1024,
                        num_tokens=256,
                        use_3d_rope=True,
                        use_sandwich_norm=True,
                        use_eva_adapter=True,
                    )
                    
                    model = ImprovedBLIP3oCLIPDiTModel(config).to(self.device)
                    model.load_state_dict(state_dict)
                    model.eval()
                    
                    logger.info(f"✅ Model loaded successfully")
                    return model, config, checkpoint
                    
                except Exception as e:
                    continue
        
        raise RuntimeError("Could not load model with any valid configuration")


class COCOEvaluator:
    """
    COCO Evaluator without CLIP normalization
    Works directly with raw CLIP embeddings
    """
    
    def __init__(self, model_path: str, device: torch.device, 
                 num_inference_steps: int = 50, use_heun: bool = True):
        self.device = device
        self.num_inference_steps = num_inference_steps
        self.use_heun = use_heun
        
        logger.info(f"🔬 COCO Evaluator (NO NORMALIZATION)")
        logger.info(f"🔑 Working with raw CLIP embeddings")
        
        # Load model
        loader = SimpleModelLoader(model_path, device)
        self.model, self.config, self.checkpoint = loader.load_model()
        self.model = self.model.half()  # Use half precision
        
        logger.info("✅ Setup complete (NO NORMALIZATION)")

    def _generate_with_heun(self, eva_features: torch.Tensor) -> torch.Tensor:
        """Generate using Heun's method"""
        batch_size, seq_len, _ = eva_features.shape
        
        eva_features = eva_features.to(self.device, dtype=torch.float16)
        
        # Start from noise
        x = torch.randn(batch_size, seq_len, 1024, device=self.device, dtype=torch.float16)
        
        # Linear schedule
        timesteps = torch.linspace(1.0, 0.0, self.num_inference_steps + 1, device=self.device)[:-1]
        
        for i, t in enumerate(timesteps):
            t_batch = torch.full((batch_size,), t.item(), device=self.device, dtype=torch.float16)
            dt = timesteps[i] - timesteps[i + 1] if i < len(timesteps) - 1 else timesteps[i]
            
            if self.use_heun:
                # Heun's method
                with torch.amp.autocast('cuda'):
                    v1 = self.model(
                        hidden_states=x,
                        timestep=t_batch,
                        encoder_hidden_states=eva_features,
                        return_dict=False
                    )
                
                if isinstance(v1, dict):
                    v1 = v1.get('velocity_prediction', list(v1.values())[0])
                
                x_mid = x + dt * v1
                t_mid = torch.full((batch_size,), max(0.0, t.item() - dt), device=self.device, dtype=torch.float16)
                
                with torch.amp.autocast('cuda'):
                    v2 = self.model(
                        hidden_states=x_mid,
                        timestep=t_mid,
                        encoder_hidden_states=eva_features,
                        return_dict=False
                    )
                
                if isinstance(v2, dict):
                    v2 = v2.get('velocity_prediction', list(v2.values())[0])
                
                v_avg = (v1 + v2) / 2.0
                x = x + dt * v_avg
            else:
                # Euler method
                with torch.amp.autocast('cuda'):
                    velocity = self.model(
                        hidden_states=x,
                        timestep=t_batch,
                        encoder_hidden_states=eva_features,
                        return_dict=False
                    )
                
                if isinstance(velocity, dict):
                    velocity = velocity.get('velocity_prediction', list(velocity.values())[0])
                
                x = x + dt * velocity
            
            # Clamp for stability
            x = torch.clamp(x, min=-10.0, max=10.0)
        
        return x

    def evaluate(self, embeddings_file: str, max_samples: int = None, 
                batch_size: int = 4, training_mode: str = "patch_only") -> Dict[str, Any]:
        """
        Evaluate without normalization - all in raw CLIP space
        """
        start_time = time.time()
        
        logger.info(f"🔬 Starting COCO evaluation (NO NORMALIZATION)")
        logger.info(f"📂 File: {embeddings_file}")
        logger.info(f"🔑 Working with raw CLIP embeddings")
        
        # Create dataset working with raw embeddings
        dataset = COCODatasetAsTrainingFormat(
            embeddings_file=embeddings_file,
            max_samples=max_samples,
            training_mode=training_mode
        )
        
        # Simple collate function (no normalization)
        def eval_collate_fn(batch):
            """Simple collate function without normalization"""
            eva_embeddings = torch.stack([item['eva_embeddings'] for item in batch])
            clip_embeddings = torch.stack([item['clip_embeddings'] for item in batch])  # Raw
            
            return {
                'encoder_hidden_states': eva_embeddings,
                'clip_embeddings': clip_embeddings,  # Raw CLIP embeddings
                'batch_size': len(batch),
                'captions': [item['caption'] for item in batch],
                'keys': [item['key'] for item in batch],
            }
        
        # Create dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=eval_collate_fn,
            num_workers=0,
            drop_last=False
        )
        
        logger.info(f"✅ Created dataloader (NO NORMALIZATION)")
        logger.info(f"   Dataset size: {len(dataset)}")
        logger.info(f"   Batch size: {batch_size}")
        
        # Process batches
        all_generated = []
        all_targets = []
        samples_processed = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
                eva_features = batch['encoder_hidden_states'].to(self.device)
                target_clip = batch['clip_embeddings'].to(self.device)  # Raw CLIP
                
                # Generate (returns raw CLIP embeddings)
                generated = self._generate_with_heun(eva_features)
                
                # Store results (both in raw CLIP space)
                all_generated.append(generated.cpu().float())
                all_targets.append(target_clip.cpu().float())
                
                samples_processed += eva_features.shape[0]
                
                # Memory cleanup
                del eva_features, generated
                torch.cuda.empty_cache()
        
        # Compute metrics (all in raw CLIP space)
        all_generated = torch.cat(all_generated, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        # Per-image similarity (raw CLIP space)
        gen_per_image = all_generated.mean(dim=1)
        tgt_per_image = all_targets.mean(dim=1)
        
        gen_norm = F.normalize(gen_per_image, p=2, dim=-1)
        tgt_norm = F.normalize(tgt_per_image, p=2, dim=-1)
        similarity = F.cosine_similarity(gen_norm, tgt_norm, dim=-1)
        
        # Quality metrics
        high_quality = (similarity > 0.7).float().mean().item()
        very_high_quality = (similarity > 0.8).float().mean().item()
        excellent_quality = (similarity > 0.9).float().mean().item()
        
        # MSE loss
        mse_loss = F.mse_loss(all_generated, all_targets).item()
        
        eval_metrics = {
            'eval_clip_similarity': similarity.mean().item(),
            'eval_mse_loss': mse_loss,
            'eval_high_quality': high_quality,
            'eval_very_high_quality': very_high_quality,
            'eval_excellent_quality': excellent_quality,
            'eval_similarity_std': similarity.std().item(),
            'eval_samples': samples_processed,
            'evaluation_time_seconds': time.time() - start_time,
            'inference_steps': self.num_inference_steps,
            'use_heun_solver': self.use_heun,
            'normalization': 'DISABLED',
            'raw_clip_space': True,
        }
        
        # Print results
        self.print_results(eval_metrics)
        
        return eval_metrics

    def print_results(self, metrics):
        """Print results"""
        logger.info("\n" + "="*80)
        logger.info("📊 COCO EVALUATION RESULTS (NO NORMALIZATION)")
        logger.info("="*80)
        
        similarity = metrics['eval_clip_similarity']
        
        logger.info(f"📊 Results:")
        logger.info(f"   CLIP Similarity: {similarity:.4f} (RAW CLIP space)")
        
        logger.info(f"🏆 Quality:")
        logger.info(f"   High (>0.7): {metrics['eval_high_quality']*100:.1f}%")
        logger.info(f"   Very High (>0.8): {metrics['eval_very_high_quality']*100:.1f}%")
        logger.info(f"   Excellent (>0.9): {metrics['eval_excellent_quality']*100:.1f}%")
        
        logger.info(f"⚙️ Details:")
        logger.info(f"   Samples: {metrics['eval_samples']:,}")
        logger.info(f"   Time: {metrics['evaluation_time_seconds']:.1f}s")
        logger.info(f"   Normalization: DISABLED")
        logger.info(f"   Space: Raw CLIP embeddings")
        
        logger.info("="*80)


def main():
    parser = argparse.ArgumentParser(description="COCO Evaluation without CLIP normalization")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to model directory")
    parser.add_argument("--coco_embeddings_file", type=str, required=True,
                       help="Path to COCO embeddings file")
    parser.add_argument("--max_samples", type=int, default=None,
                       help="Max samples to evaluate")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Batch size")
    parser.add_argument("--num_inference_steps", type=int, default=50,
                       help="Number of inference steps")
    parser.add_argument("--training_mode", type=str, default="patch_only",
                       choices=["patch_only", "cls_patch"],
                       help="Training mode")
    parser.add_argument("--use_heun", action="store_true", default=True,
                       help="Use Heun solver")
    
    args = parser.parse_args()
    
    logger.info("🔬 COCO Evaluation (NO NORMALIZATION)")
    logger.info("="*70)
    logger.info("🔑 Working with raw CLIP embeddings")
    logger.info("📋 No normalization/denormalization applied")
    logger.info("="*70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        evaluator = COCOEvaluator(
            args.model_path,
            device,
            num_inference_steps=args.num_inference_steps,
            use_heun=args.use_heun
        )
        
        results = evaluator.evaluate(
            embeddings_file=args.coco_embeddings_file,
            max_samples=args.max_samples,
            batch_size=args.batch_size,
            training_mode=args.training_mode
        )
        
        if results:
            logger.info("🎉 COCO evaluation completed successfully!")
            similarity = results['eval_clip_similarity']
            logger.info(f"📊 Final CLIP similarity: {similarity:.4f} (RAW CLIP space)")
            logger.info(f"🔑 Evaluation performed without normalization")
            
            return 0
        else:
            logger.error("❌ Evaluation failed")
            return 1
            
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)