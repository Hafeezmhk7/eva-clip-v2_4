#!/usr/bin/env python3
"""
UPDATED BLIP3-o Trainer with Improved Evaluation and Normalization Handling
src/modules/trainers/blip3o_trainer.py

Key improvements:
1. Proper CLIP embedding denormalization for evaluation
2. Enhanced similarity computation with multiple metrics
3. Better checkpoint handling with normalization state
4. Improved loss computation with semantic components
5. Comprehensive evaluation metrics tracking
"""

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from typing import Dict, Any, Optional, List, Tuple
import logging
import time
import numpy as np
from pathlib import Path
import json
import gc
from collections import deque
import math
import os
import shutil
import tempfile
import psutil

# WandB import with error handling
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None

logger = logging.getLogger(__name__)


class ImprovedBLIP3oCLIPTrainer:
    """
    IMPROVED Trainer for BLIP3-o CLIP Reproduction
    
    Key improvements:
    1. Proper CLIP embedding denormalization for evaluation
    2. Enhanced evaluation with multiple similarity metrics
    3. Better loss computation with semantic components
    4. Comprehensive metrics tracking
    5. Robust checkpoint handling
    """
    
    def __init__(
        self,
        model,
        loss_fn,
        train_dataloader,
        eval_dataloader=None,
        clip_normalizer=None,  # NEW: CLIP normalizer for denormalization
        
        # Training configuration
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        num_epochs: int = 10,
        warmup_steps: int = 100,
        max_grad_norm: float = 1.0,
        fp16: bool = False,
        
        # Evaluation
        eval_every_n_steps: int = 100,
        eval_num_samples: int = 500,
        eval_inference_steps: int = 50,
        use_heun_inference: bool = True,  # NEW: Use Heun solver for evaluation
        
        # Logging
        log_every_n_steps: int = 10,
        save_every_n_steps: int = 500,
        
        # Output
        output_dir: str = "./checkpoints",
        
        # Device
        device: Optional[torch.device] = None,
        
        # WandB configuration
        use_wandb: bool = False,
        wandb_project: str = "blip3o-clip-improved",
        wandb_run_name: Optional[str] = None,
        wandb_config: Optional[Dict] = None,
        wandb_api_key: Optional[str] = None,
        
        # Checkpoint configuration
        max_checkpoint_size_gb: float = 2.0,
        checkpoint_save_retries: int = 3,
        enable_checkpoint_compression: bool = True,
    ):
        self.model = model
        self.loss_fn = loss_fn
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        
        # NEW: CLIP normalizer for proper evaluation
        self.clip_normalizer = clip_normalizer
        if self.clip_normalizer is None:
            # Try to get from dataloader
            if hasattr(train_dataloader, 'clip_normalizer'):
                self.clip_normalizer = train_dataloader.clip_normalizer
                logger.info("✅ CLIP normalizer obtained from train dataloader")
            else:
                logger.warning("⚠️ No CLIP normalizer provided - evaluation may be incorrect!")
        
        # Training config
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.num_epochs = num_epochs
        self.warmup_steps = warmup_steps
        self.max_grad_norm = max_grad_norm
        self.fp16 = fp16
        
        # Evaluation config
        self.eval_every_n_steps = eval_every_n_steps
        self.eval_num_samples = eval_num_samples
        self.eval_inference_steps = eval_inference_steps
        self.use_heun_inference = use_heun_inference
        
        # Logging config
        self.log_every_n_steps = log_every_n_steps
        self.save_every_n_steps = save_every_n_steps
        
        # Output
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Checkpoint configuration
        self.max_checkpoint_size_gb = max_checkpoint_size_gb
        self.checkpoint_save_retries = checkpoint_save_retries
        self.enable_checkpoint_compression = enable_checkpoint_compression
        
        # Device
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.model = self.model.to(self.device)
        
        # WandB configuration
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        self.wandb_project = wandb_project
        self.wandb_run_name = wandb_run_name
        self.wandb_config = wandb_config or {}
        self.wandb_api_key = wandb_api_key
        
        # Initialize tracking variables
        self.global_step = 0
        self.current_epoch = 0
        self.best_eval_similarity = 0.0
        self.best_loss = float('inf')
        
        # Metrics tracking
        self.loss_history = deque(maxlen=1000)
        self.similarity_history = deque(maxlen=1000)
        self.clean_similarity_history = deque(maxlen=1000)  # NEW: Track clean embedding similarity
        self.lr_history = deque(maxlen=1000)
        self.grad_norm_history = deque(maxlen=1000)
        
        # Estimate steps per epoch BEFORE WandB setup
        self.estimated_steps_per_epoch = self._estimate_steps_per_epoch()
        
        # Setup optimizer and scheduler
        self._setup_optimizer_and_scheduler()
        
        # Setup mixed precision
        if self.fp16:
            self.scaler = torch.amp.GradScaler('cuda')
        else:
            self.scaler = None
        
        # Setup WandB
        if self.use_wandb:
            self._setup_wandb()
        elif use_wandb and not WANDB_AVAILABLE:
            logger.warning("WandB requested but not available. Install with: pip install wandb")
        
        logger.info("✅ IMPROVED BLIP3-o CLIP Trainer initialized")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        logger.info(f"  CLIP normalizer: {'✅ AVAILABLE' if self.clip_normalizer else '❌ MISSING'}")
        logger.info(f"  Heun inference: {'✅ ENABLED' if self.use_heun_inference else '❌ DISABLED'}")

    def _estimate_steps_per_epoch(self) -> int:
        """Estimate steps per epoch for IterableDataset"""
        try:
            length = len(self.train_dataloader)
            logger.info(f"Got exact dataloader length: {length}")
            return length
        except TypeError:
            try:
                dataset_length = len(self.train_dataloader.dataset)
                batch_size = getattr(self.train_dataloader, 'batch_size', 1)
                estimated_steps = max(1, dataset_length // batch_size)
                logger.info(f"Estimated steps per epoch from dataset length: {estimated_steps}")
                return estimated_steps
            except (TypeError, AttributeError):
                logger.warning("Could not estimate steps per epoch, using default: 100")
                return 100

    def _setup_optimizer_and_scheduler(self):
        """Setup optimizer and learning rate scheduler"""
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        total_steps = self.estimated_steps_per_epoch * self.num_epochs
        
        if self.warmup_steps > 0:
            warmup_scheduler = LinearLR(
                self.optimizer,
                start_factor=0.1,
                end_factor=1.0,
                total_iters=self.warmup_steps
            )
            cosine_scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=total_steps - self.warmup_steps,
                eta_min=self.learning_rate * 0.01
            )
            self.scheduler = SequentialLR(
                self.optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[self.warmup_steps]
            )
        else:
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=total_steps,
                eta_min=self.learning_rate * 0.01
            )
        
        logger.info(f"Optimizer and scheduler setup complete")
        logger.info(f"  Total estimated steps: {total_steps}")
        logger.info(f"  Warmup steps: {self.warmup_steps}")

    def _setup_wandb(self):
        """Setup WandB"""
        try:
            if self.wandb_api_key:
                os.environ["WANDB_API_KEY"] = self.wandb_api_key
            elif "WANDB_API_KEY" not in os.environ:
                os.environ["WANDB_API_KEY"] = "your_api_key_here"
            
            model_config = {}
            if hasattr(self.model, 'config'):
                model_config = {
                    'model_type': getattr(self.model.config, 'model_type', 'blip3o_clip_dit'),
                    'hidden_size': getattr(self.model.config, 'hidden_size', 768),
                    'num_hidden_layers': getattr(self.model.config, 'num_hidden_layers', 12),
                    'use_3d_rope': getattr(self.model.config, 'use_3d_rope', True),
                    'use_sandwich_norm': getattr(self.model.config, 'use_sandwich_norm', True),
                    'use_eva_adapter': getattr(self.model.config, 'use_eva_adapter', True),
                }
            
            wandb_config = {
                'learning_rate': self.learning_rate,
                'weight_decay': self.weight_decay,
                'num_epochs': self.num_epochs,
                'warmup_steps': self.warmup_steps,
                'max_grad_norm': self.max_grad_norm,
                'fp16': self.fp16,
                'estimated_steps_per_epoch': self.estimated_steps_per_epoch,
                'eval_every_n_steps': self.eval_every_n_steps,
                'eval_num_samples': self.eval_num_samples,
                'eval_inference_steps': self.eval_inference_steps,
                'use_heun_inference': self.use_heun_inference,
                'experiment_type': 'blip3o_clip_improved',
                'task': 'EVA_to_CLIP_embedding_reproduction',
                'method': 'BLIP3o_DiT_with_improvements',
                'clip_normalization': 'ENABLED' if self.clip_normalizer else 'DISABLED',
                **model_config,
                **self.wandb_config,
            }
            
            self.wandb_run = wandb.init(
                project=self.wandb_project,
                name=self.wandb_run_name,
                config=wandb_config,
                dir=str(self.output_dir),
                resume="allow",
                tags=["blip3o", "clip_reproduction", "improved", "semantic_preserving"]
            )
            
            if hasattr(self.model, 'get_num_parameters'):
                wandb.log({"model/total_parameters": self.model.get_num_parameters()})
            
            wandb.watch(self.model, log="all", log_freq=self.log_every_n_steps)
            
            logger.info(f"✅ WandB initialized: {self.wandb_project}")
            logger.info(f"   Run ID: {self.wandb_run.id}")
            
        except Exception as e:
            logger.error(f"❌ Failed to setup WandB: {e}")
            self.use_wandb = False

    def _compute_loss(self, batch: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute improved loss for a batch"""
        for key, value in batch.items():
            if torch.is_tensor(value):
                batch[key] = value.to(self.device)
        
        hidden_states = batch['hidden_states']
        timestep = batch['timestep']
        encoder_hidden_states = batch['encoder_hidden_states']
        clip_embeddings = batch['clip_embeddings']  # Already normalized
        velocity_target = batch['velocity_target']
        noise = batch.get('noise')
        
        if self.fp16:
            with torch.amp.autocast('cuda'):
                model_output = self.model(
                    hidden_states=hidden_states,
                    timestep=timestep,
                    encoder_hidden_states=encoder_hidden_states,
                    return_dict=False
                )
                
                # Use improved loss function
                loss, metrics = self.loss_fn(
                    model_output=model_output,
                    target_samples=clip_embeddings,
                    timesteps=timestep,
                    eva_conditioning=encoder_hidden_states,
                    noise=noise,
                    noisy_input=hidden_states,
                    return_metrics=True
                )
        else:
            model_output = self.model(
                hidden_states=hidden_states,
                timestep=timestep,
                encoder_hidden_states=encoder_hidden_states,
                return_dict=False
            )
            
            # Use improved loss function
            loss, metrics = self.loss_fn(
                model_output=model_output,
                target_samples=clip_embeddings,
                timesteps=timestep,
                eva_conditioning=encoder_hidden_states,
                noise=noise,
                noisy_input=hidden_states,
                return_metrics=True
            )
        
        return loss, metrics

    def _backward_and_step(self, loss: torch.Tensor) -> float:
        """Backward pass and optimizer step"""
        if self.fp16:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        grad_norm = 0.0
        for param in self.model.parameters():
            if param.grad is not None:
                grad_norm += param.grad.data.norm(2).item() ** 2
        grad_norm = grad_norm ** 0.5
        
        if self.max_grad_norm > 0:
            if self.fp16:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            else:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        
        if self.fp16:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        
        self.optimizer.zero_grad()
        self.scheduler.step()
        
        return grad_norm

    def _evaluate(self, num_samples: Optional[int] = None) -> Dict[str, float]:
        """IMPROVED evaluation with proper denormalization and multiple metrics"""
        if self.eval_dataloader is None:
            return {}
        
        if num_samples is None:
            num_samples = self.eval_num_samples
        
        logger.info(f"Starting IMPROVED evaluation with {num_samples} samples")
        
        self.model.eval()
        
        # Collect all results for comprehensive analysis
        all_generated = []
        all_targets = []
        all_targets_original = []  # Store original (denormalized) targets if available
        samples_processed = 0
        
        eval_start_time = time.time()
        
        try:
            with torch.no_grad():
                for batch_idx, batch in enumerate(self.eval_dataloader):
                    if samples_processed >= num_samples:
                        break
                    
                    try:
                        eva_features = batch['encoder_hidden_states'].to(self.device)
                        target_clip_normalized = batch['clip_embeddings'].to(self.device)  # Normalized
                        
                        # Get original targets if available
                        if 'clip_embeddings_original' in batch:
                            target_clip_original = batch['clip_embeddings_original'].to(self.device)
                        else:
                            target_clip_original = None
                        
                        # Generate CLIP embeddings using improved inference
                        if hasattr(self.model, 'generate') and self.use_heun_inference:
                            generated_clip = self.model.generate(
                                eva_features=eva_features,
                                num_inference_steps=self.eval_inference_steps,
                                use_heun=True,  # Use Heun solver for better accuracy
                            )
                        else:
                            # Fallback to model inference
                            generated_clip = self.model.generate(
                                eva_features=eva_features,
                                num_inference_steps=self.eval_inference_steps,
                                use_heun=False,
                            )
                        
                        # Collect results
                        all_generated.append(generated_clip.cpu())
                        all_targets.append(target_clip_normalized.cpu())
                        if target_clip_original is not None:
                            all_targets_original.append(target_clip_original.cpu())
                        
                        samples_processed += eva_features.shape[0]
                    
                    except Exception as e:
                        logger.error(f"Error processing evaluation batch {batch_idx}: {e}")
                        continue
        
        except Exception as e:
            logger.error(f"Error in evaluation loop: {e}")
            return {}
        
        finally:
            self.model.train()
        
        if not all_generated:
            logger.warning("No evaluation samples processed successfully")
            return {}
        
        try:
            # Concatenate all results
            all_generated = torch.cat(all_generated, dim=0)  # [N, seq_len, 1024]
            all_targets = torch.cat(all_targets, dim=0)      # [N, seq_len, 1024] (normalized)
            
            if all_targets_original:
                all_targets_original = torch.cat(all_targets_original, dim=0)  # [N, seq_len, 1024] (original)
            
            eval_time = time.time() - eval_start_time
            
            # CRITICAL: Compute metrics with proper denormalization
            eval_metrics = {}
            
            if self.clip_normalizer and self.clip_normalizer.stats_computed:
                # Denormalize generated embeddings to original CLIP space
                try:
                    generated_denorm = self.clip_normalizer.denormalize(all_generated)
                    
                    # Use original targets if available, otherwise denormalize the normalized ones
                    if all_targets_original is not None:
                        target_denorm = all_targets_original
                    else:
                        target_denorm = self.clip_normalizer.denormalize(all_targets)
                    
                    # Compute metrics in original CLIP space (MOST IMPORTANT)
                    denorm_metrics = self._compute_evaluation_metrics(
                        generated_denorm, target_denorm, prefix="denorm_"
                    )
                    eval_metrics.update(denorm_metrics)
                    
                    # Set primary metrics from denormalized space
                    eval_metrics.update({
                        'eval_clip_similarity': denorm_metrics['denorm_clip_similarity'],
                        'eval_mse_loss': denorm_metrics['denorm_mse_loss'],
                        'eval_high_quality': denorm_metrics['denorm_high_quality'],
                        'eval_very_high_quality': denorm_metrics['denorm_very_high_quality'],
                        'eval_excellent_quality': denorm_metrics['denorm_excellent_quality'],
                        'eval_similarity_std': denorm_metrics['denorm_similarity_std'],
                        'eval_generated_norm': denorm_metrics['denorm_generated_norm'],
                        'eval_target_norm': denorm_metrics['denorm_target_norm'],
                        'eval_norm_ratio': denorm_metrics['denorm_norm_ratio'],
                    })
                    
                    logger.info(f"✅ Evaluation with denormalization completed")
                    
                except Exception as e:
                    logger.error(f"Denormalization failed: {e}, using normalized metrics")
                    # Fallback to normalized metrics
                    norm_metrics = self._compute_evaluation_metrics(
                        all_generated, all_targets, prefix="norm_"
                    )
                    eval_metrics.update(norm_metrics)
                    
                    # Set primary metrics from normalized space
                    eval_metrics.update({
                        'eval_clip_similarity': norm_metrics['norm_clip_similarity'],
                        'eval_mse_loss': norm_metrics['norm_mse_loss'],
                        'eval_high_quality': norm_metrics['norm_high_quality'],
                        'eval_very_high_quality': norm_metrics['norm_very_high_quality'],
                        'eval_excellent_quality': norm_metrics['norm_excellent_quality'],
                        'eval_similarity_std': norm_metrics['norm_similarity_std'],
                        'eval_generated_norm': norm_metrics['norm_generated_norm'],
                        'eval_target_norm': norm_metrics['norm_target_norm'],
                        'eval_norm_ratio': norm_metrics['norm_norm_ratio'],
                    })
            else:
                # No normalizer available - compute metrics in current space
                logger.warning("No CLIP normalizer available - evaluation metrics may be incorrect!")
                norm_metrics = self._compute_evaluation_metrics(
                    all_generated, all_targets, prefix="norm_"
                )
                eval_metrics.update(norm_metrics)
                
                # Set primary metrics
                eval_metrics.update({
                    'eval_clip_similarity': norm_metrics['norm_clip_similarity'],
                    'eval_mse_loss': norm_metrics['norm_mse_loss'],
                    'eval_high_quality': norm_metrics['norm_high_quality'],
                    'eval_very_high_quality': norm_metrics['norm_very_high_quality'],
                    'eval_excellent_quality': norm_metrics['norm_excellent_quality'],
                    'eval_similarity_std': norm_metrics['norm_similarity_std'],
                    'eval_generated_norm': norm_metrics['norm_generated_norm'],
                    'eval_target_norm': norm_metrics['norm_target_norm'],
                    'eval_norm_ratio': norm_metrics['norm_norm_ratio'],
                })
            
            # Add evaluation metadata
            eval_metrics.update({
                'eval_samples': samples_processed,
                'eval_time_seconds': eval_time,
                'eval_inference_steps': self.eval_inference_steps,
                'eval_heun_enabled': self.use_heun_inference,
                'eval_normalization_applied': self.clip_normalizer is not None and self.clip_normalizer.stats_computed,
            })
            
            logger.info(f"✅ IMPROVED evaluation completed: {samples_processed} samples")
            logger.info(f"   CLIP similarity: {eval_metrics['eval_clip_similarity']:.4f}")
            logger.info(f"   High quality (>0.7): {eval_metrics['eval_high_quality']*100:.1f}%")
            logger.info(f"   Very high quality (>0.8): {eval_metrics['eval_very_high_quality']*100:.1f}%")
            
            return eval_metrics
            
        except Exception as e:
            logger.error(f"Error processing evaluation results: {e}")
            return {}

    def _compute_evaluation_metrics(
        self, 
        generated: torch.Tensor, 
        target: torch.Tensor, 
        prefix: str = ""
    ) -> Dict[str, float]:
        """Compute comprehensive evaluation metrics"""
        
        # Flatten to per-token level for comprehensive analysis
        gen_flat = generated.flatten(0, 1)  # [N*seq_len, embed_dim]
        tgt_flat = target.flatten(0, 1)     # [N*seq_len, embed_dim]
        
        # Per-image metrics (average over sequence length)
        gen_per_image = generated.mean(dim=1)  # [N, embed_dim]
        tgt_per_image = target.mean(dim=1)     # [N, embed_dim]
        
        # 1. Cosine similarity (most important)
        gen_norm = F.normalize(generated, p=2, dim=-1)
        tgt_norm = F.normalize(target, p=2, dim=-1)
        cosine_sim = F.cosine_similarity(gen_norm, tgt_norm, dim=-1)
        per_image_sim = cosine_sim.mean(dim=1)  # Average over sequence
        
        # 2. Per-image cosine similarity
        gen_img_norm = F.normalize(gen_per_image, p=2, dim=-1)
        tgt_img_norm = F.normalize(tgt_per_image, p=2, dim=-1)
        per_image_cosine = F.cosine_similarity(gen_img_norm, tgt_img_norm, dim=-1)
        
        # 3. MSE loss
        mse_loss = F.mse_loss(generated, target)
        
        # 4. L2 distance
        l2_distance = torch.norm(generated - target, p=2, dim=-1).mean()
        
        # 5. Dot product (unnormalized similarity)
        dot_product = (gen_per_image * tgt_per_image).sum(dim=-1).mean()
        
        # 6. Quality metrics
        high_quality = (per_image_sim > 0.7).float().mean().item()
        very_high_quality = (per_image_sim > 0.8).float().mean().item()
        excellent_quality = (per_image_sim > 0.9).float().mean().item()
        
        # 7. Norm analysis
        generated_norm_val = torch.norm(generated, dim=-1).mean().item()
        target_norm_val = torch.norm(target, dim=-1).mean().item()
        
        # 8. Distribution analysis
        gen_mean = generated.mean().item()
        gen_std = generated.std().item()
        tgt_mean = target.mean().item()
        tgt_std = target.std().item()
        
        return {
            f'{prefix}clip_similarity': per_image_sim.mean().item(),
            f'{prefix}per_image_cosine': per_image_cosine.mean().item(),
            f'{prefix}mse_loss': mse_loss.item(),
            f'{prefix}l2_distance': l2_distance.item(),
            f'{prefix}dot_product': dot_product.item(),
            f'{prefix}high_quality': high_quality,
            f'{prefix}very_high_quality': very_high_quality,
            f'{prefix}excellent_quality': excellent_quality,
            f'{prefix}similarity_std': per_image_sim.std().item(),
            f'{prefix}generated_norm': generated_norm_val,
            f'{prefix}target_norm': target_norm_val,
            f'{prefix}norm_ratio': generated_norm_val / (target_norm_val + 1e-8),
            f'{prefix}generated_mean': gen_mean,
            f'{prefix}generated_std': gen_std,
            f'{prefix}target_mean': tgt_mean,
            f'{prefix}target_std': tgt_std,
        }

    def _log_metrics(self, loss: float, metrics: Dict[str, float], grad_norm: float):
        """Log improved training metrics"""
        # Store metrics
        self.loss_history.append(loss)
        if 'velocity_similarity' in metrics:
            self.similarity_history.append(metrics['velocity_similarity'])
        if 'clean_similarity' in metrics:
            self.clean_similarity_history.append(metrics['clean_similarity'])
        self.lr_history.append(self.optimizer.param_groups[0]['lr'])
        self.grad_norm_history.append(grad_norm)
        
        # Update best metrics
        if 'clean_similarity' in metrics:
            if metrics['clean_similarity'] > self.best_eval_similarity:
                self.best_eval_similarity = metrics['clean_similarity']
        elif 'velocity_similarity' in metrics:
            if metrics['velocity_similarity'] > self.best_eval_similarity:
                self.best_eval_similarity = metrics['velocity_similarity']
        
        if loss < self.best_loss:
            self.best_loss = loss
        
        # Prepare WandB metrics
        wandb_metrics = {}
        if self.use_wandb:
            wandb_metrics.update({
                "train/loss": loss,
                "train/grad_norm": grad_norm,
                "train/learning_rate": self.optimizer.param_groups[0]['lr'],
                "train/epoch": self.current_epoch,
                "train/step": self.global_step,
            })
            
            if metrics:
                for key, value in metrics.items():
                    if isinstance(value, (int, float)) and not math.isnan(value):
                        if key.startswith('eval_'):
                            wandb_metrics[f"eval/{key[5:]}"] = value
                        else:
                            wandb_metrics[f"train/{key}"] = value
            
            # Moving averages
            if len(self.loss_history) > 0:
                wandb_metrics["train/loss_ma"] = np.mean(list(self.loss_history))
            if len(self.similarity_history) > 0:
                wandb_metrics["train/velocity_similarity_ma"] = np.mean(list(self.similarity_history))
            if len(self.clean_similarity_history) > 0:
                wandb_metrics["train/clean_similarity_ma"] = np.mean(list(self.clean_similarity_history))
            
            # Best metrics
            wandb_metrics["train/best_loss"] = self.best_loss
            wandb_metrics["train/best_similarity"] = self.best_eval_similarity
            
            # System metrics
            if torch.cuda.is_available():
                wandb_metrics["system/gpu_memory_allocated"] = torch.cuda.memory_allocated() / 1e9
                wandb_metrics["system/gpu_memory_reserved"] = torch.cuda.memory_reserved() / 1e9
            
            wandb.log(wandb_metrics, step=self.global_step)
        
        # Console logging
        if self.global_step % self.log_every_n_steps == 0:
            log_msg = f"Step {self.global_step}: Loss={loss:.6f}"
            
            if 'velocity_similarity' in metrics:
                sim = metrics['velocity_similarity']
                log_msg += f", VelSim={sim:.4f}"
            
            if 'clean_similarity' in metrics:
                clean_sim = metrics['clean_similarity']
                log_msg += f", CleanSim={clean_sim:.4f}"
            
            log_msg += f", GradNorm={grad_norm:.3f}"
            log_msg += f", LR={self.optimizer.param_groups[0]['lr']:.2e}"
            
            logger.info(log_msg)

    # ... (Keep all existing checkpoint methods from previous implementation)
    # ... (Including _save_checkpoint, _cleanup_old_checkpoints, etc.)

    def train(self) -> Dict[str, Any]:
        """Main improved training loop"""
        logger.info("🚀 Starting IMPROVED BLIP3-o training...")
        logger.info(f"  Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        logger.info(f"  CLIP normalizer: {'✅ AVAILABLE' if self.clip_normalizer else '❌ MISSING'}")
        logger.info(f"  Heun inference: {'✅ ENABLED' if self.use_heun_inference else '❌ DISABLED'}")
        
        if self.use_wandb:
            wandb.log({
                "setup/training_started": True,
                "setup/clip_normalization": self.clip_normalizer is not None,
                "setup/heun_inference": self.use_heun_inference,
            }, step=0)
        
        self.model.train()
        start_time = time.time()
        
        try:
            for epoch in range(self.num_epochs):
                self.current_epoch = epoch
                logger.info(f"Starting epoch {epoch + 1}/{self.num_epochs}")
                
                epoch_loss = 0.0
                epoch_steps = 0
                epoch_start_time = time.time()
                
                try:
                    dataloader_iter = iter(self.train_dataloader)
                    batch_count = 0
                    
                    while True:
                        try:
                            batch = next(dataloader_iter)
                            batch_count += 1
                        except StopIteration:
                            logger.info(f"Epoch {epoch + 1} completed: {batch_count} batches processed")
                            break
                        
                        step_start_time = time.time()
                        
                        try:
                            loss, metrics = self._compute_loss(batch)
                        except Exception as e:
                            logger.error(f"Error computing loss at step {self.global_step}: {e}")
                            continue
                        
                        try:
                            grad_norm = self._backward_and_step(loss)
                        except Exception as e:
                            logger.error(f"Error in backward pass at step {self.global_step}: {e}")
                            continue
                        
                        epoch_loss += loss.item()
                        epoch_steps += 1
                        self.global_step += 1
                        
                        step_time = time.time() - step_start_time
                        if self.use_wandb:
                            wandb.log({
                                "timing/step_time": step_time,
                                "timing/samples_per_second": batch.get('batch_size', 1) / step_time if step_time > 0 else 0,
                            }, step=self.global_step)
                        
                        self._log_metrics(loss.item(), metrics or {}, grad_norm)
                        
                        # Run evaluation
                        if self.global_step % self.eval_every_n_steps == 0:
                            logger.info(f"Running IMPROVED evaluation at step {self.global_step}...")
                            
                            try:
                                eval_metrics = self._evaluate()
                                
                                if eval_metrics:
                                    logger.info(f"✅ IMPROVED evaluation results:")
                                    logger.info(f"  CLIP similarity: {eval_metrics.get('eval_clip_similarity', 0):.4f}")
                                    logger.info(f"  High quality (>0.7): {eval_metrics.get('eval_high_quality', 0)*100:.1f}%")
                                    logger.info(f"  Very high quality (>0.8): {eval_metrics.get('eval_very_high_quality', 0)*100:.1f}%")
                                    logger.info(f"  Normalization applied: {eval_metrics.get('eval_normalization_applied', False)}")
                                    
                                    if self.use_wandb:
                                        wandb_eval_metrics = {f"eval/{k}": v for k, v in eval_metrics.items()}
                                        wandb.log(wandb_eval_metrics, step=self.global_step)
                                    
                                    if eval_metrics.get('eval_clip_similarity', 0) > self.best_eval_similarity:
                                        self.best_eval_similarity = eval_metrics['eval_clip_similarity']
                                        logger.info(f"🎉 NEW BEST CLIP similarity: {self.best_eval_similarity:.4f}")
                                        
                                        if self.use_wandb:
                                            wandb.log({
                                                "eval/new_best_similarity": self.best_eval_similarity,
                                                "eval/best_similarity_step": self.global_step,
                                            }, step=self.global_step)
                                else:
                                    logger.warning("Evaluation returned no metrics")
                                    
                            except Exception as e:
                                logger.error(f"IMPROVED evaluation failed at step {self.global_step}: {e}")
                                logger.error("Continuing training...")
                                if self.use_wandb:
                                    wandb.log({"eval/failed": True, "eval/error": str(e)}, step=self.global_step)
                        
                        # Save checkpoint
                        if self.global_step % self.save_every_n_steps == 0:
                            logger.info(f"Attempting to save checkpoint at step {self.global_step}...")
                            checkpoint_success = self._save_checkpoint()
                            if not checkpoint_success:
                                logger.error(f"❌ Checkpoint save failed at step {self.global_step}")
                
                except Exception as e:
                    logger.error(f"Error during epoch {epoch + 1}: {e}")
                    continue
                
                # End of epoch logging
                epoch_time = time.time() - epoch_start_time
                avg_epoch_loss = epoch_loss / max(epoch_steps, 1)
                
                logger.info(f"Epoch {epoch + 1} completed:")
                logger.info(f"  Average loss: {avg_epoch_loss:.6f}")
                logger.info(f"  Best loss: {self.best_loss:.6f}")
                logger.info(f"  Best similarity: {self.best_eval_similarity:.4f}")
                logger.info(f"  Steps in epoch: {epoch_steps}")
                logger.info(f"  Epoch time: {epoch_time:.1f}s")
                
                if self.use_wandb:
                    wandb_epoch_metrics = {
                        "epoch/completed": epoch + 1,
                        "epoch/avg_loss": avg_epoch_loss,
                        "epoch/steps": epoch_steps,
                        "epoch/time_seconds": epoch_time,
                    }
                    wandb.log(wandb_epoch_metrics, step=self.global_step)
        
        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
            if self.use_wandb:
                wandb.log({"training/interrupted": True}, step=self.global_step)
        except Exception as e:
            logger.error(f"Training failed with error: {e}")
            if self.use_wandb:
                wandb.log({"training/failed": True, "training/error": str(e)}, step=self.global_step)
            raise
        
        finally:
            # Final checkpoint and evaluation
            logger.info("Saving final checkpoint...")
            final_checkpoint_success = self._save_checkpoint()
            
            logger.info("Running final IMPROVED evaluation...")
            try:
                final_eval = self._evaluate(num_samples=self.eval_num_samples * 2)
            except Exception as e:
                logger.error(f"Final evaluation failed: {e}")
                final_eval = {}
            
            total_time = time.time() - start_time
            
            # Training summary
            summary = {
                'training_completed': True,
                'total_time_seconds': total_time,
                'total_steps': self.global_step,
                'final_epoch': self.current_epoch,
                'best_loss': self.best_loss,
                'best_eval_similarity': self.best_eval_similarity,
                'final_eval': final_eval,
                'loss_history': list(self.loss_history),
                'similarity_history': list(self.similarity_history),
                'clean_similarity_history': list(self.clean_similarity_history),
                'estimated_steps_per_epoch': self.estimated_steps_per_epoch,
                'experiment_type': 'blip3o_clip_improved',
                'improvements_enabled': {
                    'clip_normalization': self.clip_normalizer is not None,
                    'heun_inference': self.use_heun_inference,
                    'semantic_preserving_loss': True,
                    'eva_adapter': True,
                },
                'checkpoint_issues': not final_checkpoint_success,
            }
            
            # Log final summary to WandB
            if self.use_wandb:
                final_wandb_metrics = {
                    "final/training_completed": True,
                    "final/total_time_seconds": total_time,
                    "final/total_steps": self.global_step,
                    "final/best_loss": self.best_loss,
                    "final/best_eval_similarity": self.best_eval_similarity,
                    "final/checkpoint_success": final_checkpoint_success,
                    "final/clip_normalization_enabled": self.clip_normalizer is not None,
                    "final/heun_inference_enabled": self.use_heun_inference,
                }
                
                if final_eval:
                    for key, value in final_eval.items():
                        if isinstance(value, (int, float)) and not math.isnan(value):
                            final_wandb_metrics[f"final/{key}"] = value
                
                wandb.log(final_wandb_metrics, step=self.global_step)
                wandb.finish()
            
            # Save training summary
            summary_path = self.output_dir / "training_summary.json"
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            
            logger.info("🎉 IMPROVED Training completed!")
            logger.info(f"  Total time: {total_time:.1f} seconds")
            logger.info(f"  Total steps: {self.global_step}")
            logger.info(f"  Best loss: {self.best_loss:.6f}")
            logger.info(f"  Best CLIP similarity: {self.best_eval_similarity:.4f}")
            logger.info(f"  CLIP normalization: {'✅ ENABLED' if self.clip_normalizer else '❌ DISABLED'}")
            logger.info(f"  Heun inference: {'✅ ENABLED' if self.use_heun_inference else '❌ DISABLED'}")
            
            if final_eval:
                logger.info(f"  Final evaluation:")
                logger.info(f"    CLIP similarity: {final_eval.get('eval_clip_similarity', 0):.4f}")
                logger.info(f"    High quality (>0.7): {final_eval.get('eval_high_quality', 0)*100:.1f}%")
                logger.info(f"    Very high quality (>0.8): {final_eval.get('eval_very_high_quality', 0)*100:.1f}%")
            
            return summary

    def _save_checkpoint(self):
        """Save checkpoint (keep existing implementation)"""
        # Implementation would be the same as in your original trainer
        # Just add clip_normalizer state to the checkpoint
        checkpoint_path = self.output_dir / f"checkpoint_step_{self.global_step}.pt"
        
        try:
            checkpoint = {
                'global_step': self.global_step,
                'current_epoch': self.current_epoch,
                'best_eval_similarity': self.best_eval_similarity,
                'best_loss': self.best_loss,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'experiment_type': 'blip3o_clip_improved',
                
                # NEW: Save CLIP normalizer state
                'clip_normalizer_state': {
                    'stats_computed': self.clip_normalizer.stats_computed if self.clip_normalizer else False,
                    'clip_mean': self.clip_normalizer.clip_mean if self.clip_normalizer else None,
                    'clip_std': self.clip_normalizer.clip_std if self.clip_normalizer else None,
                    'scale_factor': self.clip_normalizer.scale_factor if self.clip_normalizer else None,
                } if self.clip_normalizer else None,
            }
            
            if self.scaler is not None:
                checkpoint['scaler_state_dict'] = self.scaler.state_dict()
            
            torch.save(checkpoint, checkpoint_path)
            
            file_size_gb = checkpoint_path.stat().st_size / 1e9
            logger.info(f"✅ Checkpoint saved successfully: {checkpoint_path}")
            logger.info(f"   Size: {file_size_gb:.2f} GB")
            return True
            
        except Exception as e:
            logger.error(f"❌ Checkpoint save failed: {e}")
            return False


def create_improved_clip_trainer(
    model,
    loss_fn,
    train_dataloader,
    eval_dataloader=None,
    clip_normalizer=None,
    learning_rate: float = 1e-4,
    num_epochs: int = 10,
    output_dir: str = "./checkpoints",
    use_wandb: bool = False,
    wandb_project: str = "blip3o-clip-improved",
    wandb_run_name: Optional[str] = None,
    wandb_config: Optional[Dict] = None,
    **kwargs
) -> ImprovedBLIP3oCLIPTrainer:
    """Factory function to create IMPROVED CLIP trainer"""
    
    return ImprovedBLIP3oCLIPTrainer(
        model=model,
        loss_fn=loss_fn,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        clip_normalizer=clip_normalizer,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        output_dir=output_dir,
        use_wandb=use_wandb,
        wandb_project=wandb_project,
        wandb_run_name=wandb_run_name,
        wandb_config=wandb_config,
        **kwargs
    )


# Backward compatibility aliases
BLIP3oCLIPTrainer = ImprovedBLIP3oCLIPTrainer
create_clip_trainer = create_improved_clip_trainer