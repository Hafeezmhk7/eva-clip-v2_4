#!/usr/bin/env python3
"""
BLIP3-o COCO Evaluation Script (Compatible with SLURM job)
eval_blip3o_coco.py

Updated to support all arguments expected by the SLURM job script.

Usage:
    python eval_blip3o_coco.py --model_path ./checkpoints/model --coco_root ./data/coco
"""

import os
import sys
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import json
import numpy as np
from PIL import Image
from pathlib import Path
import logging
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import seaborn as sns

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Setup paths
sys.path.insert(0, str(Path(__file__).parent))

# Import with fallbacks
try:
    from transformers import CLIPProcessor, CLIPModel, AutoModel, CLIPImageProcessor
    logger.info("✅ Transformers imported successfully")
except ImportError as e:
    logger.error(f"❌ Failed to import transformers: {e}")
    sys.exit(1)

# Try to import BLIP3-o modules
try:
    from src.modules.models.blip3o_dit import ImprovedBLIP3oCLIPDiTModel, BLIP3oCLIPDiTConfig
    logger.info("✅ BLIP3-o model modules imported")
except ImportError as e:
    logger.error(f"❌ Failed to import BLIP3-o model: {e}")
    sys.exit(1)

class COCOEvalDataset(Dataset):
    """COCO dataset for evaluation"""
    
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
        
        with open(annotations_file, 'r') as f:
            coco_data = json.load(f)
        
        # Get valid images
        images_info = {img['id']: img for img in coco_data['images']}
        valid_images = []
        
        for img_id, img_info in images_info.items():
            img_path = images_dir / img_info['file_name']
            if img_path.exists():
                valid_images.append((img_id, img_info))
                if max_samples and len(valid_images) >= max_samples:
                    break
        
        self.images = valid_images
        self.images_dir = images_dir
        logger.info(f"✅ Loaded {len(self.images)} images for evaluation")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_id, img_info = self.images[idx]
        img_path = self.images_dir / img_info['file_name']
        image = Image.open(img_path).convert('RGB')
        return {
            'image': image, 
            'image_id': img_id,
            'file_name': img_info['file_name']
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
            # (num_attention_heads, num_key_value_heads)
            (12, 4),
            (16, 4), 
            (8, 4),
            (12, 12),
            (hidden_size // 64, 4) if hidden_size % 64 == 0 else (12, 4),
        ]
        
        for num_heads, num_kv_heads in configs_to_try:
            # Ensure valid configuration
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
                    return model, config
                    
                except Exception as e:
                    logger.warning(f"⚠️ Config {num_heads}/{num_kv_heads} failed: {e}")
                    continue
        
        raise RuntimeError("Could not load model with any valid configuration")

class COCOEvaluator:
    """COCO evaluator with comprehensive metrics and visualization"""
    
    def __init__(self, model_path: str, coco_root: str, device: torch.device, 
                 num_inference_steps: int = 50, use_heun: bool = False, 
                 use_half_precision: bool = True):
        self.device = device
        self.num_inference_steps = num_inference_steps
        self.use_heun = use_heun
        self.use_half_precision = use_half_precision
        
        logger.info(f"🔬 BLIP3-o COCO Evaluator")
        logger.info(f"Inference steps: {num_inference_steps}")
        logger.info(f"Using Heun solver: {use_heun}")
        logger.info(f"Half precision: {use_half_precision}")
        
        # Load BLIP3-o model
        loader = ModelLoader(model_path, device)
        self.blip3o_model, self.config = loader.load_model_with_manual_config()
        
        # Convert to appropriate precision
        if use_half_precision:
            self.blip3o_model = self.blip3o_model.half()
            self.model_dtype = torch.float16
            self._ensure_model_dtype_consistency()
        else:
            self.blip3o_model = self.blip3o_model.float()
            self.model_dtype = torch.float32
        
        # Load feature extraction models with matching precision
        model_precision = torch.float16 if use_half_precision else torch.float32
        
        logger.info("📦 Loading CLIP...")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=model_precision).to(device)
        self.clip_model.eval()
        
        logger.info("📦 Loading EVA-CLIP...")
        self.eva_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")
        self.eva_model = AutoModel.from_pretrained("BAAI/EVA-CLIP-8B", trust_remote_code=True, torch_dtype=model_precision).to(device)
        self.eva_model.eval()
        
        logger.info("✅ All models loaded")
        logger.info(f"🔧 Model dtype: {self.model_dtype}")
    
    def _ensure_model_dtype_consistency(self):
        """Ensure all model parameters and buffers have consistent dtype (only for half precision)"""
        if not self.use_half_precision:
            return
            
        logger.info("🔧 Ensuring model dtype consistency...")
        
        # Check for any parameters that might still be in float32
        float32_params = []
        for name, param in self.blip3o_model.named_parameters():
            if param.dtype == torch.float32:
                float32_params.append(name)
        
        if float32_params:
            logger.warning(f"⚠️ Found {len(float32_params)} float32 parameters, converting to {self.model_dtype}")
            for name in float32_params[:5]:  # Show first 5
                logger.warning(f"   - {name}")
            if len(float32_params) > 5:
                logger.warning(f"   ... and {len(float32_params) - 5} more")
        
        # Check buffers too
        float32_buffers = []
        for name, buffer in self.blip3o_model.named_buffers():
            if buffer.dtype == torch.float32:
                float32_buffers.append(name)
        
        if float32_buffers:
            logger.warning(f"⚠️ Found {len(float32_buffers)} float32 buffers, converting to {self.model_dtype}")
            for name in float32_buffers[:5]:  # Show first 5
                logger.warning(f"   - {name}")
        
        # Force conversion of all parameters and buffers
        self.blip3o_model = self.blip3o_model.to(dtype=self.model_dtype)
        
        # Double-check
        remaining_float32 = []
        for name, param in self.blip3o_model.named_parameters():
            if param.dtype == torch.float32:
                remaining_float32.append(name)
        for name, buffer in self.blip3o_model.named_buffers():
            if buffer.dtype == torch.float32:
                remaining_float32.append(name)
        
        if remaining_float32:
            logger.error(f"❌ Still have {len(remaining_float32)} float32 tensors after conversion!")
            for name in remaining_float32:
                logger.error(f"   - {name}")
        else:
            logger.info("✅ All model tensors converted to consistent dtype")
    
    def extract_features(self, images):
        """Extract CLIP and EVA features"""
        clip_features = []
        eva_features = []
        
        for img in images:
            # CLIP features (remove CLS token for patch_only mode)
            clip_inputs = self.clip_processor(images=img, return_tensors="pt")
            clip_inputs = {k: v.to(self.device).half() if v.dtype == torch.float32 else v.to(self.device) 
                          for k, v in clip_inputs.items()}
            
            with torch.no_grad():
                clip_outputs = self.clip_model.vision_model(**clip_inputs, output_hidden_states=True)
                clip_emb = clip_outputs.last_hidden_state[:, 1:, :]  # Remove CLS, get patches [1, 256, 1024]
                clip_features.append(clip_emb.squeeze().cpu().float())
            
            # EVA features (remove CLS token for patch_only mode)
            eva_inputs = self.eva_processor(images=img, return_tensors="pt")
            eva_inputs = {k: v.to(self.device).half() if v.dtype == torch.float32 else v.to(self.device) 
                         for k, v in eva_inputs.items()}
            
            with torch.no_grad():
                eva_outputs = self.eva_model.vision_model(**eva_inputs, output_hidden_states=True)
                eva_emb = eva_outputs.last_hidden_state[:, 1:, :]  # Remove CLS, get patches
                eva_features.append(eva_emb.squeeze().cpu().float())
        
        return torch.stack(clip_features), torch.stack(eva_features)
    
    def generate_embeddings(self, eva_features):
        """Generate CLIP embeddings using the specified solver with proper dtype handling"""
        # Ensure consistent dtype throughout
        model_dtype = next(self.blip3o_model.parameters()).dtype
        eva_features = eva_features.to(self.device, dtype=model_dtype)
        batch_size, seq_len, _ = eva_features.shape
        
        logger.debug(f"🔧 Generation setup: model_dtype={model_dtype}, eva_shape={eva_features.shape}")
        
        # Start from noise with correct dtype
        x = torch.randn(batch_size, seq_len, 1024, device=self.device, dtype=model_dtype)
        
        # Linear timesteps from 1.0 to 0.0 with correct dtype
        timesteps = torch.linspace(1.0, 0.0, self.num_inference_steps + 1, device=self.device, dtype=model_dtype)[:-1]
        
        with torch.no_grad():
            for i, t in enumerate(timesteps):
                t_batch = torch.full((batch_size,), t.item(), device=self.device, dtype=model_dtype)
                
                # Get velocity
                try:
                    # Ensure all inputs have consistent dtype
                    x = x.to(dtype=model_dtype)
                    t_batch = t_batch.to(dtype=model_dtype)
                    eva_features = eva_features.to(dtype=model_dtype)
                    
                    # Debug dtype information
                    logger.debug(f"Step {i}: x.dtype={x.dtype}, t_batch.dtype={t_batch.dtype}, eva.dtype={eva_features.dtype}")
                    
                    velocity = self.blip3o_model(
                        hidden_states=x,
                        timestep=t_batch,
                        encoder_hidden_states=eva_features,
                        return_dict=False
                    )
                    
                    if isinstance(velocity, dict):
                        velocity = velocity.get('velocity_prediction', list(velocity.values())[0])
                    
                    # Ensure velocity has correct dtype
                    velocity = velocity.to(dtype=model_dtype)
                    
                    # Calculate step size
                    dt = timesteps[i] - timesteps[i + 1] if i < len(timesteps) - 1 else timesteps[i]
                    
                    if self.use_heun:
                        # Heun's method (second-order)
                        x_temp = x + dt * velocity
                        x_temp = x_temp.to(dtype=model_dtype)
                        
                        # Get second velocity prediction
                        if i < len(timesteps) - 1:
                            t_next = timesteps[i + 1]
                            t_next_batch = torch.full((batch_size,), t_next.item(), device=self.device, dtype=model_dtype)
                            
                            velocity_next = self.blip3o_model(
                                hidden_states=x_temp,
                                timestep=t_next_batch,
                                encoder_hidden_states=eva_features,
                                return_dict=False
                            )
                            
                            if isinstance(velocity_next, dict):
                                velocity_next = velocity_next.get('velocity_prediction', list(velocity_next.values())[0])
                            
                            velocity_next = velocity_next.to(dtype=model_dtype)
                            
                            # Heun update
                            x = x + dt * 0.5 * (velocity + velocity_next)
                        else:
                            x = x_temp
                    else:
                        # Euler step
                        x = x + dt * velocity
                    
                    # Ensure x maintains correct dtype
                    x = x.to(dtype=model_dtype)
                    x = torch.clamp(x, -10, 10)
                    
                except Exception as e:
                    logger.error(f"❌ Generation step {i} failed: {e}")
                    logger.error(f"   x.dtype: {x.dtype}, eva_features.dtype: {eva_features.dtype}")
                    logger.error(f"   t_batch.dtype: {t_batch.dtype}, model_dtype: {model_dtype}")
                    
                    # Check model parameter dtypes
                    param_dtypes = set()
                    for name, param in self.blip3o_model.named_parameters():
                        param_dtypes.add(param.dtype)
                        if param.dtype != model_dtype:
                            logger.error(f"   Mismatched param {name}: {param.dtype}")
                            break
                    logger.error(f"   Model param dtypes: {param_dtypes}")
                    
                    # If first step fails, it's likely a fundamental dtype issue
                    if i == 0:
                        logger.error("❌ First step failed - likely fundamental dtype mismatch")
                        # Try to continue with fallback approach
                        try:
                            logger.info("🔄 Attempting fallback with explicit float32 conversion...")
                            
                            # Convert everything to float32 as fallback
                            x_f32 = x.float()
                            t_batch_f32 = t_batch.float()
                            eva_features_f32 = eva_features.float()
                            
                            # Temporarily convert model to float32
                            self.blip3o_model = self.blip3o_model.float()
                            
                            velocity = self.blip3o_model(
                                hidden_states=x_f32,
                                timestep=t_batch_f32,
                                encoder_hidden_states=eva_features_f32,
                                return_dict=False
                            )
                            
                            if isinstance(velocity, dict):
                                velocity = velocity.get('velocity_prediction', list(velocity.values())[0])
                            
                            # Update with float32
                            dt = timesteps[i] - timesteps[i + 1] if i < len(timesteps) - 1 else timesteps[i]
                            x = x_f32 + dt.float() * velocity
                            x = torch.clamp(x, -10, 10)
                            
                            # Convert back to target dtype for next iteration
                            x = x.to(dtype=model_dtype)
                            self.blip3o_model = self.blip3o_model.to(dtype=model_dtype)
                            
                            logger.info("✅ Fallback successful, continuing with mixed precision")
                            
                        except Exception as e2:
                            logger.error(f"❌ Fallback also failed: {e2}")
                            break
                    else:
                        break
        
        return x.cpu().float()
    
    def compute_metrics(self, generated_clip, target_clip):
        """Compute comprehensive evaluation metrics"""
        
        # Per-image similarity (global features)
        pred_global = generated_clip.mean(dim=1)  # [B, 1024]
        target_global = target_clip.mean(dim=1)  # [B, 1024]
        
        # Normalize for cosine similarity
        pred_global_norm = F.normalize(pred_global, p=2, dim=-1)
        target_global_norm = F.normalize(target_global, p=2, dim=-1)
        
        # Cosine similarities
        cosine_similarities = F.cosine_similarity(pred_global_norm, target_global_norm, dim=-1)
        
        # Per-token similarity
        pred_tokens_norm = F.normalize(generated_clip, p=2, dim=-1)  # [B, 256, 1024]
        target_tokens_norm = F.normalize(target_clip, p=2, dim=-1)  # [B, 256, 1024]
        token_similarities = F.cosine_similarity(pred_tokens_norm, target_tokens_norm, dim=-1)  # [B, 256]
        token_similarities_mean = token_similarities.mean(dim=-1)  # [B]
        
        # MSE and L1 losses
        mse_loss = F.mse_loss(generated_clip, target_clip, reduction='none').mean(dim=[1, 2])  # [B]
        l1_loss = F.l1_loss(generated_clip, target_clip, reduction='none').mean(dim=[1, 2])  # [B]
        
        return {
            'cosine_similarities': cosine_similarities.numpy(),
            'token_similarities': token_similarities_mean.numpy(),
            'mse_losses': mse_loss.numpy(),
            'l1_losses': l1_loss.numpy(),
        }
    
    def create_visualizations(self, all_metrics, output_dir):
        """Create comprehensive visualizations"""
        plt.style.use('default')
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('BLIP3-o COCO Evaluation Results', fontsize=16, fontweight='bold')
        
        cosine_sims = all_metrics['cosine_similarities']
        token_sims = all_metrics['token_similarities']
        mse_losses = all_metrics['mse_losses']
        l1_losses = all_metrics['l1_losses']
        
        # 1. Cosine similarity distribution
        axes[0, 0].hist(cosine_sims, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].axvline(cosine_sims.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {cosine_sims.mean():.3f}')
        axes[0, 0].axvline(np.median(cosine_sims), color='orange', linestyle='--', linewidth=2, label=f'Median: {np.median(cosine_sims):.3f}')
        axes[0, 0].set_xlabel('Cosine Similarity')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Global Cosine Similarity Distribution')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Token similarity distribution
        axes[0, 1].hist(token_sims, bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
        axes[0, 1].axvline(token_sims.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {token_sims.mean():.3f}')
        axes[0, 1].set_xlabel('Token Similarity')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Per-Token Cosine Similarity Distribution')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Quality distribution pie chart
        high_quality = (cosine_sims > 0.7).sum()
        very_high_quality = (cosine_sims > 0.8).sum()
        excellent_quality = (cosine_sims > 0.9).sum()
        
        quality_labels = ['Poor (<0.7)', 'High (0.7-0.8)', 'Very High (0.8-0.9)', 'Excellent (>0.9)']
        quality_counts = [
            len(cosine_sims) - high_quality,
            high_quality - very_high_quality,
            very_high_quality - excellent_quality,
            excellent_quality
        ]
        quality_colors = ['lightcoral', 'gold', 'lightgreen', 'darkgreen']
        
        axes[0, 2].pie(quality_counts, labels=quality_labels, colors=quality_colors, autopct='%1.1f%%', startangle=90)
        axes[0, 2].set_title('Quality Distribution')
        
        # 4. MSE Loss distribution
        axes[1, 0].hist(mse_losses, bins=50, alpha=0.7, color='salmon', edgecolor='black')
        axes[1, 0].axvline(mse_losses.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {mse_losses.mean():.4f}')
        axes[1, 0].set_xlabel('MSE Loss')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('MSE Loss Distribution')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Correlation plot
        axes[1, 1].scatter(cosine_sims, token_sims, alpha=0.6, c='purple', s=20)
        axes[1, 1].set_xlabel('Global Cosine Similarity')
        axes[1, 1].set_ylabel('Token Cosine Similarity')
        axes[1, 1].set_title('Global vs Token Similarity Correlation')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Add correlation coefficient
        correlation = np.corrcoef(cosine_sims, token_sims)[0, 1]
        axes[1, 1].text(0.05, 0.95, f'r = {correlation:.3f}', transform=axes[1, 1].transAxes, 
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        # 6. Performance summary
        axes[1, 2].axis('off')
        summary_text = f"""
Performance Summary:

Mean Cosine Similarity: {cosine_sims.mean():.4f}
Median Cosine Similarity: {np.median(cosine_sims):.4f}
Std Deviation: {cosine_sims.std():.4f}

Quality Distribution:
• High (>0.7): {(cosine_sims > 0.7).mean()*100:.1f}%
• Very High (>0.8): {(cosine_sims > 0.8).mean()*100:.1f}%
• Excellent (>0.9): {(cosine_sims > 0.9).mean()*100:.1f}%

Samples: {len(cosine_sims):,}
Inference Steps: {self.num_inference_steps}
Solver: {'Heun' if self.use_heun else 'Euler'}
"""
        axes[1, 2].text(0.1, 0.9, summary_text, transform=axes[1, 2].transAxes, fontsize=12,
                        verticalalignment='top', bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
        
        plt.tight_layout()
        
        # Save plots
        plot_file = output_dir / "evaluation_plots.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        logger.info(f"📊 Plots saved to: {plot_file}")
        plt.close()
    
    def evaluate(self, coco_root: str, max_samples: int = None, batch_size: int = 4, 
                 output_dir: str = None, save_results: bool = False):
        """Run comprehensive evaluation"""
        start_time = time.time()
        
        logger.info(f"🔬 Starting COCO evaluation")
        logger.info(f"Max samples: {max_samples if max_samples else 'All'}")
        logger.info(f"Batch size: {batch_size}")
        
        # Create output directory
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
        
        # Create dataset
        dataset = COCOEvalDataset(coco_root=coco_root, max_samples=max_samples)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, 
                               collate_fn=lambda x: {
                                   'images': [item['image'] for item in x],
                                   'image_ids': [item['image_id'] for item in x],
                                   'file_names': [item['file_name'] for item in x]
                               })
        
        all_metrics = {
            'cosine_similarities': [],
            'token_similarities': [],
            'mse_losses': [],
            'l1_losses': [],
            'image_ids': [],
            'file_names': []
        }
        
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
            try:
                images = batch['images']
                image_ids = batch['image_ids']
                file_names = batch['file_names']
                
                # Extract features
                clip_features, eva_features = self.extract_features(images)
                
                # Generate CLIP embeddings
                generated_clip = self.generate_embeddings(eva_features)
                
                # Compute metrics
                batch_metrics = self.compute_metrics(generated_clip, clip_features)
                
                # Store results
                all_metrics['cosine_similarities'].extend(batch_metrics['cosine_similarities'])
                all_metrics['token_similarities'].extend(batch_metrics['token_similarities'])
                all_metrics['mse_losses'].extend(batch_metrics['mse_losses'])
                all_metrics['l1_losses'].extend(batch_metrics['l1_losses'])
                all_metrics['image_ids'].extend(image_ids)
                all_metrics['file_names'].extend(file_names)
                
            except Exception as e:
                logger.warning(f"⚠️ Batch {batch_idx} failed: {e}")
                continue
        
        # Convert to numpy arrays
        for key in ['cosine_similarities', 'token_similarities', 'mse_losses', 'l1_losses']:
            all_metrics[key] = np.array(all_metrics[key])
        
        evaluation_time = time.time() - start_time
        
        # Compute summary metrics
        if len(all_metrics['cosine_similarities']) > 0:
            cosine_sims = all_metrics['cosine_similarities']
            
            summary_metrics = {
                'cosine_similarity_mean': float(cosine_sims.mean()),
                'cosine_similarity_median': float(np.median(cosine_sims)),
                'cosine_similarity_std': float(cosine_sims.std()),
                'cosine_similarity_min': float(cosine_sims.min()),
                'cosine_similarity_max': float(cosine_sims.max()),
                'token_similarity_mean': float(all_metrics['token_similarities'].mean()),
                'mse_loss': float(all_metrics['mse_losses'].mean()),
                'l1_loss': float(all_metrics['l1_losses'].mean()),
                'high_quality_percentage': float((cosine_sims > 0.7).mean() * 100),
                'very_high_quality_percentage': float((cosine_sims > 0.8).mean() * 100),
                'excellent_quality_percentage': float((cosine_sims > 0.9).mean() * 100),
                'num_samples': len(cosine_sims),
                'evaluation_time_seconds': evaluation_time,
                'samples_per_second': len(cosine_sims) / evaluation_time,
                'inference_steps': self.num_inference_steps,
                'use_heun_solver': self.use_heun,
                'denormalized': False  # This would be True if CLIP denormalization was applied
            }
            
            # Print results
            self.print_results(summary_metrics)
            
            # Save results if requested
            if save_results and output_dir:
                # Save detailed results
                detailed_results = {
                    'summary_metrics': summary_metrics,
                    'per_sample_results': {
                        'cosine_similarities': all_metrics['cosine_similarities'].tolist(),
                        'token_similarities': all_metrics['token_similarities'].tolist(),
                        'mse_losses': all_metrics['mse_losses'].tolist(),
                        'l1_losses': all_metrics['l1_losses'].tolist(),
                        'image_ids': all_metrics['image_ids'],
                        'file_names': all_metrics['file_names']
                    },
                    'evaluation_config': {
                        'model_path': str(self.blip3o_model),
                        'coco_root': coco_root,
                        'max_samples': max_samples,
                        'batch_size': batch_size,
                        'num_inference_steps': self.num_inference_steps,
                        'use_heun': self.use_heun
                    }
                }
                
                # Save files
                results_file = output_path / "evaluation_results.json"
                with open(results_file, 'w') as f:
                    json.dump(detailed_results, f, indent=2)
                
                metrics_file = output_path / "metrics_summary.json"
                with open(metrics_file, 'w') as f:
                    json.dump(summary_metrics, f, indent=2)
                
                logger.info(f"💾 Results saved to: {output_path}")
                
                # Create visualizations
                self.create_visualizations(all_metrics, output_path)
            
            return summary_metrics
        else:
            logger.error("❌ No samples processed successfully")
            return None
    
    def print_results(self, metrics):
        """Print formatted results"""
        logger.info("\n" + "="*60)
        logger.info("📊 BLIP3-o COCO EVALUATION RESULTS")
        logger.info("="*60)
        logger.info(f"🎯 Mean Cosine Similarity: {metrics['cosine_similarity_mean']:.4f}")
        logger.info(f"📊 Median: {metrics['cosine_similarity_median']:.4f}")
        logger.info(f"📊 Std: {metrics['cosine_similarity_std']:.4f}")
        logger.info(f"📊 Range: [{metrics['cosine_similarity_min']:.4f}, {metrics['cosine_similarity_max']:.4f}]")
        logger.info(f"")
        logger.info(f"🏆 Quality Distribution:")
        logger.info(f"   High (>0.7): {metrics['high_quality_percentage']:.1f}%")
        logger.info(f"   Very High (>0.8): {metrics['very_high_quality_percentage']:.1f}%")
        logger.info(f"   Excellent (>0.9): {metrics['excellent_quality_percentage']:.1f}%")
        logger.info(f"")
        logger.info(f"📊 Additional Metrics:")
        logger.info(f"   Token Similarity: {metrics['token_similarity_mean']:.4f}")
        logger.info(f"   MSE Loss: {metrics['mse_loss']:.6f}")
        logger.info(f"   L1 Loss: {metrics['l1_loss']:.6f}")
        logger.info(f"")
        logger.info(f"⚙️ Configuration:")
        logger.info(f"   Samples: {metrics['num_samples']:,}")
        logger.info(f"   Inference Steps: {metrics['inference_steps']}")
        logger.info(f"   Solver: {'Heun' if metrics['use_heun_solver'] else 'Euler'}")
        logger.info(f"   Evaluation Time: {metrics['evaluation_time_seconds']:.1f}s")
        logger.info(f"   Speed: {metrics['samples_per_second']:.1f} samples/sec")
        
        # Assessment
        mean_sim = metrics['cosine_similarity_mean']
        if mean_sim > 0.8:
            assessment = "🎉 EXCELLENT"
        elif mean_sim > 0.6:
            assessment = "✅ VERY GOOD"
        elif mean_sim > 0.4:
            assessment = "👍 GOOD"
        elif mean_sim > 0.2:
            assessment = "📈 FAIR"
        else:
            assessment = "⚠️ NEEDS IMPROVEMENT"
        
        logger.info(f"")
        logger.info(f"🏆 Assessment: {assessment}")
        logger.info("="*60)

def main():
    parser = argparse.ArgumentParser(description="BLIP3-o COCO Evaluation")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to the trained model directory")
    parser.add_argument("--coco_root", type=str, default="./data/coco",
                       help="Path to COCO dataset root directory")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Output directory for results")
    parser.add_argument("--max_samples", type=int, default=None,
                       help="Maximum number of samples to evaluate (None for all)")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Batch size for evaluation")
    parser.add_argument("--num_inference_steps", type=int, default=50,
                       help="Number of inference steps for generation")
    parser.add_argument("--use_heun", action="store_true",
                       help="Use Heun solver instead of Euler")
    parser.add_argument("--save_results", action="store_true",
                       help="Save detailed results to files")
    parser.add_argument("--disable_half_precision", action="store_true",
                       help="Disable half precision (use float32) to avoid dtype issues")
    
    args = parser.parse_args()
    
    logger.info("🔬 BLIP3-o COCO Evaluation")
    logger.info("="*50)
    
    # Check paths
    if not Path(args.model_path).exists():
        logger.error(f"❌ Model path not found: {args.model_path}")
        return 1
    
    if not Path(args.coco_root).exists():
        logger.error(f"❌ COCO root not found: {args.coco_root}")
        return 1
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Create output directory if not specified
    if args.output_dir is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        args.output_dir = f"./coco_eval_results_{timestamp}"
    
    try:
        evaluator = COCOEvaluator(
            args.model_path, 
            args.coco_root, 
            device,
            num_inference_steps=args.num_inference_steps,
            use_heun=args.use_heun,
            use_half_precision=not args.disable_half_precision
        )
        
        results = evaluator.evaluate(
            coco_root=args.coco_root,
            max_samples=args.max_samples,
            batch_size=args.batch_size,
            output_dir=args.output_dir,
            save_results=args.save_results
        )
        
        if results:
            logger.info("🎉 Evaluation completed successfully!")
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