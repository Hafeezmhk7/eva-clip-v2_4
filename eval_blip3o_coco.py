#!/usr/bin/env python3
"""
COCO Evaluation Using EXACT Same Pipeline as Training
eval_coco_same_pipeline.py

KEY INSIGHT: Use the exact same dataloader and collate function as training!
This fixes the tensor shape issues by replicating the working training evaluation approach.
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
    from src.modules.datasets.blip3o_dataset import (
        UltraConservativeCLIPNormalizer, 
        ultra_conservative_clip_reproduction_collate_fn,
        BLIP3oCLIPReproductionDataset  # We'll adapt this
    )
    logger.info("✅ BLIP3-o modules imported successfully")
except ImportError as e:
    logger.error(f"❌ Failed to import BLIP3-o modules: {e}")
    sys.exit(1)


class COCOAsTrainingDataset(Dataset):
    """
    CRITICAL FIX: Adapt COCO embeddings to look exactly like training data
    This makes COCO evaluation use the SAME pipeline as training evaluation
    """
    
    def __init__(self, embeddings_file: str, max_samples: int = None):
        self.embeddings_file = Path(embeddings_file)
        self.max_samples = max_samples
        
        logger.info(f"📂 Loading COCO embeddings as training-format dataset: {self.embeddings_file}")
        
        # Load embeddings data
        with open(self.embeddings_file, 'rb') as f:
            data = pickle.load(f)
        
        self.clip_embeddings = data['clip_embeddings']
        self.eva_embeddings = data['eva_embeddings']
        self.metadata = data['metadata']
        
        # Convert to tensors and ensure proper shape
        if not torch.is_tensor(self.clip_embeddings):
            self.clip_embeddings = torch.tensor(self.clip_embeddings, dtype=torch.float32)
        if not torch.is_tensor(self.eva_embeddings):
            self.eva_embeddings = torch.tensor(self.eva_embeddings, dtype=torch.float32)
        
        # Ensure 3D format [batch, seq_len, embed_dim]
        if self.clip_embeddings.dim() == 2:
            num_samples = len(self.metadata)
            tokens_per_sample = self.clip_embeddings.shape[0] // num_samples
            self.clip_embeddings = self.clip_embeddings.view(num_samples, tokens_per_sample, -1)
            
        if self.eva_embeddings.dim() == 2:
            num_samples = len(self.metadata)
            tokens_per_sample = self.eva_embeddings.shape[0] // num_samples
            self.eva_embeddings = self.eva_embeddings.view(num_samples, tokens_per_sample, -1)
        
        # Ensure sequence length consistency
        clip_seq_len = self.clip_embeddings.shape[1]
        eva_seq_len = self.eva_embeddings.shape[1]
        if clip_seq_len != eva_seq_len:
            min_seq_len = min(clip_seq_len, eva_seq_len)
            self.clip_embeddings = self.clip_embeddings[:, :min_seq_len, :]
            self.eva_embeddings = self.eva_embeddings[:, :min_seq_len, :]
            logger.info(f"🔧 Adjusted sequence length to {min_seq_len}")
        
        # Apply max_samples limit
        if max_samples is not None and max_samples < len(self.metadata):
            self.clip_embeddings = self.clip_embeddings[:max_samples]
            self.eva_embeddings = self.eva_embeddings[:max_samples]
            self.metadata = self.metadata[:max_samples]
        
        self.num_samples = len(self.metadata)
        
        logger.info(f"✅ COCO dataset adapted for training pipeline:")
        logger.info(f"   Samples: {self.num_samples:,}")
        logger.info(f"   CLIP shape: {self.clip_embeddings.shape}")
        logger.info(f"   EVA shape: {self.eva_embeddings.shape}")
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        """
        CRITICAL: Return data in EXACT same format as training dataset
        This ensures the collate function works identically to training
        """
        return {
            'eva_embeddings': self.eva_embeddings[idx],      # [seq_len, 4096]
            'clip_embeddings': self.clip_embeddings[idx],    # [seq_len, 1024] 
            'caption': self.metadata[idx]['caption'],
            'key': f"coco_{self.metadata[idx]['image_id']}",
            'sample_idx': idx,
            'training_mode': 'patch_only',  # Assumes patch_only mode
            'num_tokens': self.clip_embeddings.shape[1],
        }


class ModelLoader:
    """Model loader with automatic configuration detection"""
    
    def __init__(self, model_path: str, device: torch.device):
        self.model_path = Path(model_path)
        self.device = device
    
    def load_model_with_manual_config(self):
        """Load model by trying different common configurations"""
        
        # Find checkpoint
        checkpoint_files = list(self.model_path.glob("checkpoint_step_*.pt"))
        if not checkpoint_files:
            raise FileNotFoundError(f"No checkpoint files found in {self.model_path}")
        
        checkpoint_file = max(checkpoint_files, key=lambda x: int(x.stem.split('_')[-1]))
        logger.info(f"Loading checkpoint: {checkpoint_file}")
        
        checkpoint = torch.load(checkpoint_file, map_location=self.device)
        state_dict = checkpoint['model_state_dict']
        
        # Get actual dimensions from the checkpoint
        intermediate_size = state_dict['blocks.0.mlp.gate_proj.weight'].shape[0]
        hidden_size = state_dict['blocks.0.mlp.gate_proj.weight'].shape[1]
        
        # Count layers
        num_layers = max([int(key.split('.')[1]) for key in state_dict.keys() 
                         if key.startswith('blocks.') and '.mlp.gate_proj.weight' in key]) + 1
        
        logger.info(f"Detected: hidden_size={hidden_size}, layers={num_layers}, intermediate_size={intermediate_size}")
        
        # Try common attention head configurations
        configs_to_try = [
            (12, 4), (16, 4), (8, 4), (12, 12),
            (hidden_size // 64, 4) if hidden_size % 64 == 0 else (12, 4),
        ]
        
        for num_heads, num_kv_heads in configs_to_try:
            if (hidden_size % num_heads == 0 and 
                num_heads % num_kv_heads == 0 and
                num_kv_heads <= num_heads):
                
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
                        training_mode="patch_only",
                        use_3d_rope=True,
                        use_sandwich_norm=True,
                        use_eva_adapter=True,
                    )
                    
                    model = ImprovedBLIP3oCLIPDiTModel(config).to(self.device)
                    model.load_state_dict(state_dict)
                    model.eval()
                    
                    logger.info(f"✅ Successfully loaded with config: heads={num_heads}, kv_heads={num_kv_heads}")
                    return model, config, checkpoint
                    
                except Exception as e:
                    logger.warning(f"⚠️ Config {num_heads}/{num_kv_heads} failed: {e}")
                    continue
        
        raise RuntimeError("Could not load model with any valid configuration")


class SamePipelineCOCOEvaluator:
    """
    COCO Evaluator using EXACT same pipeline as training evaluation
    This is the key fix - we replicate the training evaluation approach exactly
    """
    
    def __init__(self, model_path: str, device: torch.device, 
                 num_inference_steps: int = 50, use_heun: bool = True, 
                 use_half_precision: bool = True,
                 training_embeddings_dir: str = None):
        self.device = device
        self.num_inference_steps = num_inference_steps
        self.use_heun = use_heun
        self.use_half_precision = use_half_precision
        
        logger.info(f"🔬 Same Pipeline COCO Evaluator")
        logger.info(f"🎯 Key insight: Use EXACT same data pipeline as training!")
        logger.info(f"Inference steps: {num_inference_steps}")
        logger.info(f"Using Heun solver: {use_heun}")
        logger.info(f"Half precision: {use_half_precision}")
        
        # Load BLIP3-o model
        loader = ModelLoader(model_path, device)
        self.blip3o_model, self.config, self.checkpoint = loader.load_model_with_manual_config()
        
        # Set model dtype and ensure consistency
        if use_half_precision:
            self.blip3o_model = self.blip3o_model.half()
            self.model_dtype = torch.float16
        else:
            self.blip3o_model = self.blip3o_model.float()
            self.model_dtype = torch.float32
        
        logger.info(f"🔧 Model dtype: {self.model_dtype}")
        
        # Load CLIP normalizer from checkpoint or training data
        self.clip_normalizer = self._load_clip_normalizer_from_checkpoint()
        
        logger.info("✅ Model loaded successfully")

    def _load_clip_normalizer_from_checkpoint(self):
        """Load CLIP normalizer from checkpoint (exactly like training)"""
        normalizer_state = self.checkpoint.get('clip_normalizer_state')
        if normalizer_state and normalizer_state.get('stats_computed', False):
            logger.info("🔍 Loading normalizer from checkpoint...")
            try:
                normalizer = UltraConservativeCLIPNormalizer(embedding_dim=1024)
                normalizer.scale_factor = normalizer_state.get('scale_factor', 1.5)
                normalizer.stats_computed = True
                
                if ('clip_mean' in normalizer_state and 'clip_std' in normalizer_state and
                    normalizer_state['clip_mean'] is not None and normalizer_state['clip_std'] is not None):
                    normalizer.clip_mean = torch.tensor(normalizer_state['clip_mean'])
                    normalizer.clip_std = torch.tensor(normalizer_state['clip_std'])
                
                logger.info("✅ Normalizer loaded from checkpoint!")
                return normalizer
            except Exception as e:
                logger.warning(f"⚠️ Failed to load normalizer from checkpoint: {e}")
        
        # Fallback: create approximate normalizer
        logger.info("🔄 Using approximate normalizer...")
        normalizer = UltraConservativeCLIPNormalizer(embedding_dim=1024)
        normalizer.scale_factor = 1.5
        normalizer.stats_computed = True
        normalizer.clip_mean = torch.full((1, 1, 1024), -0.697)
        normalizer.clip_std = torch.full((1, 1, 1024), 2.897)
        return normalizer

    def _safe_generate_exactly_like_training(self, eva_features: torch.Tensor, num_steps: int = 50) -> torch.Tensor:
        """
        Generate using EXACT same approach as training evaluation in _safe_generate_with_heun
        This replicates the trainer's _safe_generate_with_heun method exactly
        """
        try:
            batch_size, seq_len, _ = eva_features.shape
            
            # CRITICAL: Ensure all tensors match model dtype (exactly like training)
            eva_features = eva_features.to(self.device, dtype=self.model_dtype)
            
            # Start from standard Gaussian noise with correct dtype
            x = torch.randn(
                batch_size, seq_len, 1024,
                device=self.device, dtype=self.model_dtype
            )
            
            # Linear timestep schedule with correct dtype
            timesteps = torch.linspace(1.0, 0.0, num_steps + 1, device=self.device, dtype=self.model_dtype)[:-1]
            
            for i, t in enumerate(timesteps):
                t_batch = torch.full((batch_size,), t.item(), device=self.device, dtype=self.model_dtype)
                
                # Compute step size
                if i < len(timesteps) - 1:
                    dt = timesteps[i] - timesteps[i + 1]
                else:
                    dt = timesteps[i]
                dt = dt.item()
                
                dt_tensor = torch.tensor(dt, device=self.device, dtype=self.model_dtype)
                
                if self.use_heun:
                    # Heun's method (exactly like training)
                    try:
                        # First velocity prediction
                        with torch.amp.autocast('cuda', enabled=self.use_half_precision):
                            v1 = self.blip3o_model(
                                hidden_states=x.to(self.model_dtype),
                                timestep=t_batch.to(self.model_dtype),
                                encoder_hidden_states=eva_features.to(self.model_dtype),
                                return_dict=False
                            )
                        
                        if isinstance(v1, dict):
                            v1 = v1.get('velocity_prediction', v1.get('prediction', list(v1.values())[0]))
                        
                        v1 = v1.to(self.model_dtype)
                        
                        # Check for issues
                        if torch.isnan(v1).any() or torch.isinf(v1).any():
                            logger.warning(f"⚠️ Unhealthy v1 at step {i}, falling back to Euler")
                            x = x.to(self.model_dtype) + dt_tensor * v1.to(self.model_dtype)
                            continue
                        
                        # Predict intermediate point
                        x_mid = x.to(self.model_dtype) + dt_tensor * v1.to(self.model_dtype)
                        t_mid = torch.full((batch_size,), max(0.0, t.item() - dt), device=self.device, dtype=self.model_dtype)
                        
                        # Second velocity prediction
                        with torch.amp.autocast('cuda', enabled=self.use_half_precision):
                            v2 = self.blip3o_model(
                                hidden_states=x_mid.to(self.model_dtype),
                                timestep=t_mid.to(self.model_dtype),
                                encoder_hidden_states=eva_features.to(self.model_dtype),
                                return_dict=False
                            )
                        
                        if isinstance(v2, dict):
                            v2 = v2.get('velocity_prediction', v2.get('prediction', list(v2.values())[0]))
                        
                        v2 = v2.to(self.model_dtype)
                        
                        if torch.isnan(v2).any() or torch.isinf(v2).any():
                            logger.warning(f"⚠️ Unhealthy v2 at step {i}, using v1 only")
                            x = x.to(self.model_dtype) + dt_tensor * v1.to(self.model_dtype)
                            continue
                        
                        # Heun's corrector
                        v_avg = (v1.to(self.model_dtype) + v2.to(self.model_dtype)) / 2.0
                        x = x.to(self.model_dtype) + dt_tensor * v_avg.to(self.model_dtype)
                        
                    except Exception as e:
                        logger.warning(f"⚠️ Heun step failed at {i}: {e}, using Euler fallback")
                        # Fallback to Euler
                        try:
                            with torch.amp.autocast('cuda', enabled=self.use_half_precision):
                                velocity = self.blip3o_model(
                                    hidden_states=x.to(self.model_dtype),
                                    timestep=t_batch.to(self.model_dtype),
                                    encoder_hidden_states=eva_features.to(self.model_dtype),
                                    return_dict=False
                                )
                            if isinstance(velocity, dict):
                                velocity = velocity.get('velocity_prediction', velocity.get('prediction', list(velocity.values())[0]))
                            x = x.to(self.model_dtype) + dt_tensor * velocity.to(self.model_dtype)
                        except Exception as e2:
                            logger.error(f"❌ Euler fallback failed at step {i}: {e2}")
                            return None
                else:
                    # Euler method
                    try:
                        with torch.amp.autocast('cuda', enabled=self.use_half_precision):
                            velocity = self.blip3o_model(
                                hidden_states=x.to(self.model_dtype),
                                timestep=t_batch.to(self.model_dtype),
                                encoder_hidden_states=eva_features.to(self.model_dtype),
                                return_dict=False
                            )
                        
                        if isinstance(velocity, dict):
                            velocity = velocity.get('velocity_prediction', velocity.get('prediction', list(velocity.values())[0]))
                        
                        x = x.to(self.model_dtype) + dt_tensor * velocity.to(self.model_dtype)
                    except Exception as e:
                        logger.error(f"❌ Euler method failed at step {i}: {e}")
                        return None
                
                # Conservative clamping (exactly like training)
                x = torch.clamp(x.to(self.model_dtype), min=-10.0, max=10.0)
                
                # Check for unhealthy outputs
                if torch.isnan(x).any() or torch.isinf(x).any():
                    logger.error(f"❌ Unhealthy generation at step {i}")
                    return None
            
            return x.to(self.model_dtype)
            
        except Exception as e:
            logger.error(f"❌ Generation failed: {e}")
            return None

    def evaluate_using_training_pipeline(
        self, 
        embeddings_file: str, 
        max_samples: int = None, 
        batch_size: int = 4,
        output_dir: str = None, 
        save_results: bool = False
    ) -> Dict[str, Any]:
        """
        CRITICAL FIX: Evaluate using EXACT same pipeline as training
        This creates a dataloader and uses the same collate function as training
        """
        start_time = time.time()
        
        logger.info(f"🔬 Starting COCO evaluation using EXACT same pipeline as training")
        logger.info(f"📂 Embeddings file: {embeddings_file}")
        logger.info(f"🎯 Key: Using same dataloader + collate function as training!")
        
        # Create output directory
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
        
        # CRITICAL: Create dataset that mimics training format
        dataset = COCOAsTrainingDataset(embeddings_file=embeddings_file, max_samples=max_samples)
        
        # CRITICAL: Use EXACT same collate function as training
        def evaluation_collate_fn(batch):
            """Use the EXACT same collate function as training evaluation"""
            result = ultra_conservative_clip_reproduction_collate_fn(batch)
            
            # Apply CLIP normalization (exactly like training)
            if self.clip_normalizer and self.clip_normalizer.stats_computed:
                result['clip_embeddings_original'] = result['clip_embeddings'].clone()
                result['clip_embeddings'] = self.clip_normalizer.normalize(result['clip_embeddings'])
                # Update targets accordingly
                result['velocity_target'] = result['clip_embeddings'] - result['noise']
                # Update noisy input
                t_expanded = result['timestep'].view(-1, 1, 1)
                result['hidden_states'] = (1 - t_expanded) * result['noise'] + t_expanded * result['clip_embeddings']
            
            return result
        
        # CRITICAL: Create dataloader exactly like training
        dataloader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=False,
            collate_fn=evaluation_collate_fn,
            num_workers=0,  # Keep simple for evaluation
            drop_last=False
        )
        
        logger.info(f"✅ Created dataloader using EXACT same pipeline as training")
        logger.info(f"   Using ultra_conservative_clip_reproduction_collate_fn")
        logger.info(f"   Same normalization as training")
        
        # Storage for results
        all_generated_normalized = []
        all_targets_original = []
        samples_processed = 0
        evaluation_errors = 0
        
        logger.info(f"📊 Processing batches with {self.model_dtype} dtype...")
        
        # CRITICAL: Process batches exactly like training evaluation
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating", unit="batch")):
                try:
                    # Get data from batch (exactly like training evaluation)
                    eva_features = batch['encoder_hidden_states'].to(self.device)
                    target_clip_normalized = batch['clip_embeddings'].to(self.device)
                    
                    # Get original targets if available
                    if 'clip_embeddings_original' in batch:
                        target_clip_original = batch['clip_embeddings_original'].to(self.device)
                    else:
                        target_clip_original = None
                    
                    # Show memory usage periodically
                    if batch_idx % 10 == 0:
                        if torch.cuda.is_available():
                            allocated = torch.cuda.memory_allocated() / 1024**3
                            total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                            logger.info(f"💾 Batch {batch_idx}: GPU {allocated:.1f}GB/{total:.1f}GB ({allocated/total*100:.1f}%)")
                    
                    # Log debug info for first batch
                    if batch_idx == 0:
                        logger.info(f"🔧 Debug Info (Same as Training):")
                        logger.info(f"   Model dtype: {self.model_dtype}")
                        logger.info(f"   EVA features dtype: {eva_features.dtype}")
                        logger.info(f"   Target CLIP dtype: {target_clip_normalized.dtype}")
                        logger.info(f"   EVA features shape: {eva_features.shape}")
                        logger.info(f"   Target CLIP shape: {target_clip_normalized.shape}")
                        logger.info(f"   Batch keys: {list(batch.keys())}")
                    
                    # Generate using EXACT same method as training evaluation
                    generated_clip_normalized = self._safe_generate_exactly_like_training(
                        eva_features=eva_features,
                        num_steps=self.num_inference_steps,
                    )
                    
                    if generated_clip_normalized is None:
                        evaluation_errors += 1
                        logger.warning(f"⚠️ Generation failed for batch {batch_idx}")
                        continue
                    
                    # Store results (exactly like training evaluation)
                    all_generated_normalized.append(generated_clip_normalized.cpu().float())
                    if target_clip_original is not None:
                        all_targets_original.append(target_clip_original.cpu().float())
                    else:
                        # Denormalize if we don't have original
                        if self.clip_normalizer and self.clip_normalizer.stats_computed:
                            denorm_target = self.clip_normalizer.denormalize(target_clip_normalized)
                            all_targets_original.append(denorm_target.cpu().float())
                        else:
                            all_targets_original.append(target_clip_normalized.cpu().float())
                    
                    samples_processed += eva_features.shape[0]
                    
                    # Clear GPU memory
                    del eva_features, generated_clip_normalized
                    torch.cuda.empty_cache()
                    
                except Exception as e:
                    evaluation_errors += 1
                    logger.warning(f"⚠️ Error processing batch {batch_idx}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
        
        if not all_generated_normalized:
            logger.error("❌ No evaluation samples processed successfully")
            return None
        
        # Process results (exactly like training evaluation)
        logger.info(f"🔄 Processing results...")
        try:
            all_generated_normalized = torch.cat(all_generated_normalized, dim=0)
            all_targets_original = torch.cat(all_targets_original, dim=0)
            
            # Compute metrics using the same approach as training
            eval_metrics = self._compute_evaluation_metrics_like_training(
                all_generated_normalized, all_targets_original
            )
            
            eval_metrics.update({
                'eval_samples': samples_processed,
                'eval_errors': evaluation_errors,
                'eval_success_rate': (samples_processed - evaluation_errors) / max(samples_processed, 1),
                'evaluation_time_seconds': time.time() - start_time,
                'samples_per_second': samples_processed / (time.time() - start_time),
                'inference_steps': self.num_inference_steps,
                'use_heun_solver': self.use_heun,
                'same_pipeline_as_training': True,  # This is the key fix
                'collate_function_used': 'ultra_conservative_clip_reproduction_collate_fn',
                'normalization_applied': self.clip_normalizer is not None and self.clip_normalizer.stats_computed,
            })
            
        except Exception as e:
            logger.error(f"❌ Error processing evaluation results: {e}")
            return None
        
        # Print results
        self.print_results(eval_metrics)
        
        # Save results
        if save_results and output_dir:
            results_file = output_path / "coco_evaluation_same_pipeline.json"
            with open(results_file, 'w') as f:
                json.dump(eval_metrics, f, indent=2)
            
            logger.info(f"💾 Results saved to: {results_file}")
        
        return eval_metrics

    def _compute_evaluation_metrics_like_training(self, generated: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
        """Compute metrics exactly like training evaluation"""
        try:
            # Per-image metrics (exactly like training)
            gen_per_image = generated.mean(dim=1)
            tgt_per_image = target.mean(dim=1)
            
            # Robust cosine similarity (exactly like training)
            gen_norm = F.normalize(gen_per_image, p=2, dim=-1)
            tgt_norm = F.normalize(tgt_per_image, p=2, dim=-1)
            similarity = F.cosine_similarity(gen_norm, tgt_norm, dim=-1)
            
            # Quality metrics (same thresholds as training)
            high_quality = (similarity > 0.7).float().mean().item()
            very_high_quality = (similarity > 0.8).float().mean().item()
            excellent_quality = (similarity > 0.9).float().mean().item()
            
            # MSE loss
            mse_loss = F.mse_loss(generated, target).item()
            
            return {
                'eval_clip_similarity': similarity.mean().item(),
                'eval_mse_loss': mse_loss,
                'eval_high_quality': high_quality,
                'eval_very_high_quality': very_high_quality,
                'eval_excellent_quality': excellent_quality,
                'eval_similarity_std': similarity.std().item(),
            }
            
        except Exception as e:
            logger.warning(f"⚠️ Error computing metrics: {e}")
            return {
                'eval_clip_similarity': 0.0,
                'eval_mse_loss': float('inf'),
                'eval_error': str(e),
            }

    def print_results(self, metrics):
        """Print comprehensive results"""
        logger.info("\n" + "="*80)
        logger.info("📊 BLIP3-o COCO EVALUATION RESULTS (SAME PIPELINE AS TRAINING)")
        logger.info("="*80)
        
        current_similarity = metrics['eval_clip_similarity']
        training_reference = 0.912  # From training logs
        
        logger.info(f"📊 Main Results:")
        logger.info(f"   Current CLIP Similarity: {current_similarity:.4f}")
        logger.info(f"   Training Reference: {training_reference:.4f}")
        
        diff_pct = ((current_similarity - training_reference) / training_reference) * 100
        if abs(diff_pct) < 5:
            status_icon = "✅"
            status_text = "Very close to training"
        elif abs(diff_pct) < 15:
            status_icon = "📊"
            status_text = "Reasonably close to training"
        else:
            status_icon = "⚠️"
            status_text = "Significant difference from training"
        
        logger.info(f"   Difference: {diff_pct:+.1f}% {status_icon} {status_text}")
        
        # Quality distribution
        logger.info(f"🏆 Quality Distribution:")
        logger.info(f"   High (>0.7): {metrics['eval_high_quality']*100:.1f}%")
        logger.info(f"   Very High (>0.8): {metrics['eval_very_high_quality']*100:.1f}%")
        logger.info(f"   Excellent (>0.9): {metrics['eval_excellent_quality']*100:.1f}%")
        
        # Evaluation details
        logger.info(f"⚙️ Evaluation Details:")
        logger.info(f"   Samples: {metrics['eval_samples']:,}")
        logger.info(f"   Success Rate: {metrics['eval_success_rate']*100:.1f}%")
        logger.info(f"   Same Pipeline as Training: {'✅' if metrics.get('same_pipeline_as_training', False) else '❌'}")
        logger.info(f"   Collate Function: {metrics.get('collate_function_used', 'unknown')}")
        logger.info(f"   Time: {metrics['evaluation_time_seconds']:.1f}s")
        
        logger.info("="*80)


def main():
    parser = argparse.ArgumentParser(description="COCO Evaluation Using Same Pipeline as Training")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to the trained model directory")
    parser.add_argument("--coco_embeddings_file", type=str, required=True,
                       help="Path to pre-computed COCO embeddings file")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Output directory for results")
    parser.add_argument("--max_samples", type=int, default=None,
                       help="Maximum number of samples to evaluate")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Batch size for evaluation")
    parser.add_argument("--num_inference_steps", type=int, default=50,
                       help="Number of inference steps for generation")
    parser.add_argument("--use_heun", action="store_true", default=True,
                       help="Use Heun solver instead of Euler")
    parser.add_argument("--save_results", action="store_true", default=True,
                       help="Save detailed results to files")
    parser.add_argument("--disable_half_precision", action="store_true",
                       help="Disable half precision (use float32)")
    
    args = parser.parse_args()
    
    logger.info("🔬 COCO Evaluation Using EXACT Same Pipeline as Training")
    logger.info("="*70)
    logger.info("🎯 KEY FIX: Use same dataloader + collate function as training!")
    logger.info("📋 This should eliminate the tensor shape mismatch issue")
    logger.info("="*70)
    
    # Check paths
    if not Path(args.model_path).exists():
        logger.error(f"❌ Model path not found: {args.model_path}")
        return 1
    
    if not Path(args.coco_embeddings_file).exists():
        logger.error(f"❌ COCO embeddings file not found: {args.coco_embeddings_file}")
        return 1
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Create output directory if not specified
    if args.output_dir is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        args.output_dir = f"./coco_eval_same_pipeline_{timestamp}"
    
    try:
        evaluator = SamePipelineCOCOEvaluator(
            args.model_path, 
            device,
            num_inference_steps=args.num_inference_steps,
            use_heun=args.use_heun,
            use_half_precision=not args.disable_half_precision,
        )
        
        results = evaluator.evaluate_using_training_pipeline(
            embeddings_file=args.coco_embeddings_file,
            max_samples=args.max_samples,
            batch_size=args.batch_size,
            output_dir=args.output_dir,
            save_results=args.save_results
        )
        
        if results:
            logger.info("🎉 Same Pipeline Evaluation completed successfully!")
            similarity = results['eval_clip_similarity']
            logger.info(f"📊 Final CLIP similarity: {similarity:.4f}")
            
            if results.get('same_pipeline_as_training', False):
                logger.info("✅ Used EXACT same pipeline as training - this should fix the issue!")
            
            if similarity > 0.8:
                logger.info("🎉 EXCELLENT performance!")
            elif similarity > 0.6:
                logger.info("✅ GOOD performance!")
            elif similarity > 0.4:
                logger.info("📈 FAIR performance!")
            else:
                logger.info("⚠️ Performance needs investigation")
            
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