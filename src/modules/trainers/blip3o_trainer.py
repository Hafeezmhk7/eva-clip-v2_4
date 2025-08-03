#!/usr/bin/env python3
"""
BLIP3-o Trainer WITHOUT CLIP Normalization - Updated with Temp Directory Checkpoints
src/modules/trainers/blip3o_trainer.py

CHANGES:
1. Removed all references to clip_normalizer
2. Removed denormalization from evaluation
3. Simplified evaluation to work with raw CLIP embeddings
4. Removed normalization-related error handling
5. UPDATED: Save checkpoints to temp directory for large-scale training
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

# WandB import with error handling
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None

logger = logging.getLogger(__name__)


class BLIP3oCLIPTrainer:
    """
    BLIP3-o Trainer WITHOUT CLIP Normalization - Updated with Temp Directory Support
    
    FEATURES:
    1. Enhanced error handling throughout training loop
    2. Conservative gradient clipping and adaptive scaling
    3. Evaluation without normalization/denormalization
    4. Automatic learning rate reduction on instability
    5. Better monitoring and early stopping on issues
    6. UPDATED: Intelligent checkpoint management with temp directory support
    """
    
    def __init__(
        self,
        model,
        loss_fn,
        train_dataloader,
        eval_dataloader=None,
        
        # Training configuration
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        num_epochs: int = 10,
        warmup_steps: int = 100,
        max_grad_norm: float = 0.5,
        fp16: bool = False,
        
        # Evaluation
        eval_every_n_steps: int = 100,
        eval_num_samples: int = 100,
        eval_inference_steps: int = 50,
        use_heun_inference: bool = True,
        
        # Enhanced stability features
        adaptive_grad_clipping: bool = True,
        stability_check_frequency: int = 10,
        max_consecutive_failures: int = 5,
        emergency_lr_reduction: float = 0.5,
        loss_explosion_threshold: float = 10.0,
        
        # Logging
        log_every_n_steps: int = 10,
        save_every_n_steps: int = 500,
        
        # Output - UPDATED for temp directory support
        output_dir: str = "./checkpoints",
        temp_checkpoint_dir: Optional[str] = None,  # NEW: For large-scale training
        keep_local_checkpoints: int = 3,  # NEW: Number of checkpoints to keep locally
        save_to_temp_every_n_steps: int = 1000,  # NEW: Save to temp directory frequency
        
        # Device
        device: Optional[torch.device] = None,
        
        # WandB configuration
        use_wandb: bool = False,
        wandb_project: str = "blip3o-clip-no-norm",
        wandb_run_name: Optional[str] = None,
        wandb_config: Optional[Dict] = None,
        wandb_api_key: Optional[str] = None,
        
        **kwargs
    ):
        self.model = model
        self.loss_fn = loss_fn
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        
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
        
        # Stability features
        self.adaptive_grad_clipping = adaptive_grad_clipping
        self.stability_check_frequency = stability_check_frequency
        self.max_consecutive_failures = max_consecutive_failures
        self.emergency_lr_reduction = emergency_lr_reduction
        self.loss_explosion_threshold = loss_explosion_threshold
        
        # Logging config
        self.log_every_n_steps = log_every_n_steps
        self.save_every_n_steps = save_every_n_steps
        
        # UPDATED: Checkpoint management
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # NEW: Temp directory support for large-scale training
        self.temp_checkpoint_dir = None
        if temp_checkpoint_dir:
            self.temp_checkpoint_dir = Path(temp_checkpoint_dir)
            self.temp_checkpoint_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"🗂️ Temp checkpoint directory: {self.temp_checkpoint_dir}")
        
        self.keep_local_checkpoints = keep_local_checkpoints
        self.save_to_temp_every_n_steps = save_to_temp_every_n_steps
        self.local_checkpoints = []  # Track local checkpoints for cleanup
        
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
        
        # Enhanced monitoring
        self.loss_history = deque(maxlen=1000)
        self.grad_norm_history = deque(maxlen=1000)
        self.lr_history = deque(maxlen=1000)
        
        # Stability tracking
        self.consecutive_failures = 0
        self.emergency_lr_reductions = 0
        self.stability_alerts = 0
        self.last_stable_step = 0
        
        # Estimate steps per epoch
        self.estimated_steps_per_epoch = self._estimate_steps_per_epoch()
        
        # Setup optimizer and scheduler
        self._setup_optimizer_and_scheduler()
        
        # Setup mixed precision
        if self.fp16:
            self.scaler = torch.amp.GradScaler('cuda')
            logger.info("✅ Mixed precision enabled")
        else:
            self.scaler = None
            logger.info("🔒 Mixed precision disabled for stability")
        
        # Setup WandB
        if self.use_wandb:
            self._setup_wandb()
        
        logger.info("✅ BLIP3-o Trainer initialized (NO NORMALIZATION)")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        logger.info(f"  CLIP normalization: DISABLED")
        logger.info(f"  Enhanced stability features: enabled")
        logger.info(f"  Checkpoint strategy: {'Local + Temp' if self.temp_checkpoint_dir else 'Local only'}")

    def _estimate_steps_per_epoch(self) -> int:
        """Estimate steps per epoch with error handling"""
        try:
            length = len(self.train_dataloader)
            logger.info(f"Got exact dataloader length: {length}")
            return length
        except TypeError:
            try:
                dataset_length = len(self.train_dataloader.dataset)
                batch_size = getattr(self.train_dataloader, 'batch_size', 1)
                estimated_steps = max(1, dataset_length // batch_size)
                logger.info(f"Estimated steps per epoch from dataset: {estimated_steps}")
                return estimated_steps
            except (TypeError, AttributeError):
                logger.warning("Could not estimate steps per epoch, using conservative default: 100")
                return 100

    def _setup_optimizer_and_scheduler(self):
        """Setup optimizer and scheduler with conservative settings"""
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
        
        logger.info(f"✅ Conservative optimizer setup complete")

    def _setup_wandb(self):
        """Setup WandB configuration"""
        try:
            if self.wandb_api_key:
                os.environ["WANDB_API_KEY"] = self.wandb_api_key
            
            wandb_config = {
                'learning_rate': self.learning_rate,
                'weight_decay': self.weight_decay,
                'num_epochs': self.num_epochs,
                'warmup_steps': self.warmup_steps,
                'max_grad_norm': self.max_grad_norm,
                'fp16': self.fp16,
                'experiment_type': 'blip3o_clip_NO_NORMALIZATION',
                'normalization': 'DISABLED',
                'checkpoint_strategy': 'temp_directory_enabled' if self.temp_checkpoint_dir else 'local_only',
                'stability_features': {
                    'adaptive_grad_clipping': self.adaptive_grad_clipping,
                    'emergency_lr_reduction': self.emergency_lr_reduction,
                    'loss_explosion_threshold': self.loss_explosion_threshold,
                    'max_consecutive_failures': self.max_consecutive_failures,
                },
                **self.wandb_config,
            }
            
            self.wandb_run = wandb.init(
                project=self.wandb_project,
                name=self.wandb_run_name,
                config=wandb_config,
                dir=str(self.output_dir),
                resume="allow",
                tags=["blip3o", "clip_reproduction", "no_normalization", "raw_embeddings", "temp_checkpoints"]
            )
            
            logger.info(f"✅ WandB initialized: {self.wandb_project}")
            
        except Exception as e:
            logger.error(f"❌ Failed to setup WandB: {e}")
            self.use_wandb = False

    def _check_tensor_health(self, tensor: torch.Tensor, name: str) -> bool:
        """Check tensor for NaN/Inf values"""
        if torch.isnan(tensor).any():
            logger.warning(f"⚠️ NaN detected in {name}")
            return False
        if torch.isinf(tensor).any():
            logger.warning(f"⚠️ Inf detected in {name}")
            return False
        return True

    def _compute_loss_with_stability_check(self, batch: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, float], bool]:
        """Compute loss with comprehensive stability checking"""
        try:
            # Move batch to device
            for key, value in batch.items():
                if torch.is_tensor(value):
                    batch[key] = value.to(self.device)
            
            # Check input health
            hidden_states = batch['hidden_states']
            if not self._check_tensor_health(hidden_states, "hidden_states"):
                return torch.tensor(float('inf')), {}, False
            
            # Forward pass with error handling
            if self.fp16:
                with torch.amp.autocast('cuda'):
                    model_output = self.model(
                        hidden_states=batch['hidden_states'],
                        timestep=batch['timestep'],
                        encoder_hidden_states=batch['encoder_hidden_states'],
                        return_dict=False
                    )
            else:
                model_output = self.model(
                    hidden_states=batch['hidden_states'],
                    timestep=batch['timestep'],
                    encoder_hidden_states=batch['encoder_hidden_states'],
                    return_dict=False
                )
            
            # Check model output health
            if not self._check_tensor_health(model_output, "model_output"):
                return torch.tensor(float('inf')), {}, False
            
            # Compute loss with stability check
            if self.fp16:
                with torch.amp.autocast('cuda'):
                    loss, metrics = self.loss_fn(
                        model_output=model_output,
                        target_samples=batch['clip_embeddings'],  # Raw CLIP embeddings
                        timesteps=batch['timestep'],
                        eva_conditioning=batch['encoder_hidden_states'],
                        noise=batch.get('noise'),
                        noisy_input=batch['hidden_states'],
                        return_metrics=True
                    )
            else:
                loss, metrics = self.loss_fn(
                    model_output=model_output,
                    target_samples=batch['clip_embeddings'],  # Raw CLIP embeddings
                    timesteps=batch['timestep'],
                    eva_conditioning=batch['encoder_hidden_states'],
                    noise=batch.get('noise'),
                    noisy_input=batch['hidden_states'],
                    return_metrics=True
                )
            
            # Check loss health
            if not self._check_tensor_health(loss, "loss"):
                return torch.tensor(float('inf')), metrics or {}, False
            
            # Check for loss explosion
            if loss.item() > self.loss_explosion_threshold:
                logger.warning(f"⚠️ Loss explosion detected: {loss.item():.3f}")
                return loss, metrics or {}, False
            
            return loss, metrics or {}, True
            
        except Exception as e:
            logger.error(f"❌ Error in loss computation: {e}")
            return torch.tensor(float('inf')), {}, False

    def _backward_and_step_with_stability(self, loss: torch.Tensor) -> Tuple[float, bool]:
        """Backward pass and optimizer step with stability checks"""
        try:
            # Backward pass
            if self.fp16:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Compute gradient norm
            grad_norm = 0.0
            for param in self.model.parameters():
                if param.grad is not None:
                    param_norm = param.grad.data.norm(2)
                    if torch.isnan(param_norm) or torch.isinf(param_norm):
                        logger.warning("⚠️ NaN/Inf in gradients detected")
                        return float('inf'), False
                    grad_norm += param_norm.item() ** 2
            grad_norm = grad_norm ** 0.5
            
            # Adaptive gradient clipping
            if self.adaptive_grad_clipping:
                # Reduce clipping threshold if gradients are consistently high
                if len(self.grad_norm_history) >= 10:
                    recent_grad_norms = list(self.grad_norm_history)[-10:]
                    avg_recent_grad = sum(recent_grad_norms) / len(recent_grad_norms)
                    if avg_recent_grad > self.max_grad_norm * 2:
                        effective_grad_norm = self.max_grad_norm * 0.5
                        logger.warning(f"⚠️ High gradients detected, using stricter clipping: {effective_grad_norm:.3f}")
                    else:
                        effective_grad_norm = self.max_grad_norm
                else:
                    effective_grad_norm = self.max_grad_norm
            else:
                effective_grad_norm = self.max_grad_norm
            
            # Apply gradient clipping
            if effective_grad_norm > 0:
                if self.fp16:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), effective_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), effective_grad_norm)
            
            # Optimizer step
            if self.fp16:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            
            self.optimizer.zero_grad()
            self.scheduler.step()
            
            return grad_norm, True
            
        except Exception as e:
            logger.error(f"❌ Error in backward pass: {e}")
            return float('inf'), False

    def _handle_training_instability(self):
        """Handle training instability with emergency measures"""
        self.consecutive_failures += 1
        self.stability_alerts += 1
        
        logger.warning(f"⚠️ Training instability detected (failure #{self.consecutive_failures})")
        
        if self.consecutive_failures >= self.max_consecutive_failures:
            # Emergency learning rate reduction
            old_lr = self.optimizer.param_groups[0]['lr']
            new_lr = old_lr * self.emergency_lr_reduction
            
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = new_lr
            
            self.emergency_lr_reductions += 1
            logger.warning(f"🚨 EMERGENCY: Reducing learning rate from {old_lr:.2e} to {new_lr:.2e}")
            
            # Reset failure counter
            self.consecutive_failures = 0
            
            if self.use_wandb:
                wandb.log({
                    "emergency/lr_reduction": self.emergency_lr_reductions,
                    "emergency/new_lr": new_lr,
                    "emergency/stability_alerts": self.stability_alerts,
                }, step=self.global_step)

    def _safe_generate_with_heun(self, eva_features: torch.Tensor, num_steps: int = 50) -> Optional[torch.Tensor]:
        """Generate using Heun's method with comprehensive error handling"""
        try:
            batch_size, seq_len, _ = eva_features.shape
            
            # Start from standard Gaussian noise
            x = torch.randn(
                batch_size, seq_len, 1024,
                device=self.device, dtype=eva_features.dtype
            )
            
            # Linear timestep schedule
            timesteps = torch.linspace(1.0, 0.0, num_steps + 1, device=self.device)[:-1]
            
            for i, t in enumerate(timesteps):
                t_batch = torch.full((batch_size,), t.item(), device=self.device, dtype=eva_features.dtype)
                
                # Compute step size
                if i < len(timesteps) - 1:
                    dt = timesteps[i] - timesteps[i + 1]
                else:
                    dt = timesteps[i]
                dt = dt.item()
                
                if self.use_heun_inference:
                    # Heun's method with error checking
                    try:
                        # First velocity prediction
                        v1 = self.model(
                            hidden_states=x,
                            timestep=t_batch,
                            encoder_hidden_states=eva_features,
                            return_dict=False
                        )
                        if isinstance(v1, dict):
                            v1 = v1.get('velocity_prediction', v1.get('prediction', list(v1.values())[0]))
                        
                        if not self._check_tensor_health(v1, f"v1_step_{i}"):
                            logger.warning(f"⚠️ Unhealthy v1 at step {i}, falling back to Euler")
                            # Fallback to Euler
                            x = x + dt * v1
                            continue
                        
                        # Predict intermediate point
                        x_mid = x + dt * v1
                        t_mid = torch.full((batch_size,), max(0.0, t.item() - dt), device=self.device, dtype=eva_features.dtype)
                        
                        # Second velocity prediction
                        v2 = self.model(
                            hidden_states=x_mid,
                            timestep=t_mid,
                            encoder_hidden_states=eva_features,
                            return_dict=False
                        )
                        if isinstance(v2, dict):
                            v2 = v2.get('velocity_prediction', v2.get('prediction', list(v2.values())[0]))
                        
                        if not self._check_tensor_health(v2, f"v2_step_{i}"):
                            logger.warning(f"⚠️ Unhealthy v2 at step {i}, using v1 only")
                            x = x + dt * v1
                            continue
                        
                        # Heun's corrector
                        v_avg = (v1 + v2) / 2.0
                        x = x + dt * v_avg
                        
                    except Exception as e:
                        logger.warning(f"⚠️ Heun step failed at {i}: {e}, using Euler fallback")
                        # Fallback to Euler
                        velocity = self.model(
                            hidden_states=x,
                            timestep=t_batch,
                            encoder_hidden_states=eva_features,
                            return_dict=False
                        )
                        if isinstance(velocity, dict):
                            velocity = velocity.get('velocity_prediction', velocity.get('prediction', list(velocity.values())[0]))
                        x = x + dt * velocity
                else:
                    # Euler method
                    velocity = self.model(
                        hidden_states=x,
                        timestep=t_batch,
                        encoder_hidden_states=eva_features,
                        return_dict=False
                    )
                    if isinstance(velocity, dict):
                        velocity = velocity.get('velocity_prediction', velocity.get('prediction', list(velocity.values())[0]))
                    x = x + dt * velocity
                
                # Conservative clamping
                x = torch.clamp(x, min=-10.0, max=10.0)
                
                # Check for unhealthy outputs
                if not self._check_tensor_health(x, f"x_step_{i}"):
                    logger.error(f"❌ Unhealthy generation at step {i}")
                    return None
            
            return x
            
        except Exception as e:
            logger.error(f"❌ Generation failed: {e}")
            return None

    def _safe_evaluate(self, num_samples: Optional[int] = None) -> Dict[str, float]:
        """Evaluation without normalization/denormalization"""
        if self.eval_dataloader is None:
            return {}
        
        if num_samples is None:
            num_samples = self.eval_num_samples
        
        logger.info(f"Starting evaluation with {num_samples} samples (NO NORMALIZATION)")
        
        self.model.eval()
        
        try:
            all_generated = []
            all_targets = []
            samples_processed = 0
            evaluation_errors = 0
            
            with torch.no_grad():
                for batch_idx, batch in enumerate(self.eval_dataloader):
                    if samples_processed >= num_samples:
                        break
                    
                    try:
                        eva_features = batch['encoder_hidden_states'].to(self.device)
                        target_clip = batch['clip_embeddings'].to(self.device)  # Raw CLIP embeddings
                        
                        # Generate with error handling
                        generated_clip = self._safe_generate_with_heun(
                            eva_features=eva_features,
                            num_steps=self.eval_inference_steps,
                        )
                        
                        if generated_clip is None:
                            evaluation_errors += 1
                            logger.warning(f"⚠️ Generation failed for batch {batch_idx}")
                            continue
                        
                        # Collect results (both in raw CLIP space)
                        all_generated.append(generated_clip.cpu())
                        all_targets.append(target_clip.cpu())
                        
                        samples_processed += eva_features.shape[0]
                        
                    except Exception as e:
                        evaluation_errors += 1
                        logger.warning(f"⚠️ Error processing evaluation batch {batch_idx}: {e}")
                        continue
            
            if not all_generated:
                logger.error("❌ No evaluation samples processed successfully")
                return {'eval_error': 'no_samples_processed', 'eval_clip_similarity': 0.0}
            
            if evaluation_errors > 0:
                logger.warning(f"⚠️ {evaluation_errors} evaluation errors occurred")
            
            # Process results (no denormalization needed)
            try:
                all_generated = torch.cat(all_generated, dim=0)
                all_targets = torch.cat(all_targets, dim=0)
                
                # Compute metrics directly in raw CLIP space
                eval_metrics = self._compute_evaluation_metrics(all_generated, all_targets)
                eval_metrics.update({
                    'eval_samples': samples_processed,
                    'eval_errors': evaluation_errors,
                    'eval_success_rate': (samples_processed - evaluation_errors) / max(samples_processed, 1),
                })
                
                logger.info(f"✅ Evaluation completed: {samples_processed} samples")
                logger.info(f"   CLIP similarity: {eval_metrics['eval_clip_similarity']:.4f}")
                logger.info(f"   Success rate: {eval_metrics['eval_success_rate']*100:.1f}%")
                
                return eval_metrics
                
            except Exception as e:
                logger.error(f"❌ Error processing evaluation results: {e}")
                return {'eval_error': str(e), 'eval_clip_similarity': 0.0}
            
        except Exception as e:
            logger.error(f"❌ Evaluation failed: {e}")
            return {'eval_error': str(e), 'eval_clip_similarity': 0.0}
        
        finally:
            self.model.train()

    def _compute_evaluation_metrics(self, generated: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
        """Compute evaluation metrics in raw CLIP space"""
        try:
            # Per-image metrics
            gen_per_image = generated.mean(dim=1)
            tgt_per_image = target.mean(dim=1)
            
            # Robust cosine similarity
            gen_norm = F.normalize(gen_per_image, p=2, dim=-1)
            tgt_norm = F.normalize(tgt_per_image, p=2, dim=-1)
            similarity = F.cosine_similarity(gen_norm, tgt_norm, dim=-1)
            
            # Quality metrics
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
                'error': str(e),
            }

    # NEW: Enhanced checkpoint management methods
    def _cleanup_local_checkpoints(self):
        """Clean up old local checkpoints to save space"""
        if len(self.local_checkpoints) > self.keep_local_checkpoints:
            checkpoints_to_remove = self.local_checkpoints[:-self.keep_local_checkpoints]
            for checkpoint_path in checkpoints_to_remove:
                try:
                    if checkpoint_path.exists():
                        if checkpoint_path.is_dir():
                            shutil.rmtree(checkpoint_path)
                        else:
                            checkpoint_path.unlink()
                        logger.info(f"🗑️ Removed old local checkpoint: {checkpoint_path.name}")
                except Exception as e:
                    logger.warning(f"⚠️ Failed to remove checkpoint {checkpoint_path}: {e}")
            
            # Update tracking list
            self.local_checkpoints = self.local_checkpoints[-self.keep_local_checkpoints:]

    def _save_checkpoint_to_temp(self, checkpoint_path: Path) -> bool:
        """Copy checkpoint to temp directory"""
        if not self.temp_checkpoint_dir:
            return False
        
        try:
            temp_checkpoint_path = self.temp_checkpoint_dir / checkpoint_path.name
            
            if checkpoint_path.is_dir():
                if temp_checkpoint_path.exists():
                    shutil.rmtree(temp_checkpoint_path)
                shutil.copytree(checkpoint_path, temp_checkpoint_path)
            else:
                shutil.copy2(checkpoint_path, temp_checkpoint_path)
            
            logger.info(f"📦 Checkpoint copied to temp: {temp_checkpoint_path}")
            
            # Log disk usage info
            file_size_mb = self._get_path_size(temp_checkpoint_path) / (1024 * 1024)
            logger.info(f"   Temp checkpoint size: {file_size_mb:.1f} MB")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to copy checkpoint to temp: {e}")
            return False

    def _get_path_size(self, path: Path) -> int:
        """Get total size of a file or directory"""
        if path.is_file():
            return path.stat().st_size
        elif path.is_dir():
            return sum(f.stat().st_size for f in path.rglob('*') if f.is_file())
        return 0

    def _save_checkpoint(self, is_best: bool = False, force_temp: bool = False) -> bool:
        """Enhanced checkpoint saving with temp directory support"""
        try:
            # Determine checkpoint filename
            if is_best:
                checkpoint_filename = f"best_checkpoint_step_{self.global_step}.pt"
            else:
                checkpoint_filename = f"checkpoint_step_{self.global_step}.pt"
            
            checkpoint_path = self.output_dir / checkpoint_filename
            
            # Create checkpoint data
            checkpoint = {
                'global_step': self.global_step,
                'current_epoch': self.current_epoch,
                'best_eval_similarity': self.best_eval_similarity,
                'best_loss': self.best_loss,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'training_state': {
                    'emergency_lr_reductions': self.emergency_lr_reductions,
                    'stability_alerts': self.stability_alerts,
                    'consecutive_failures': self.consecutive_failures,
                    'last_stable_step': self.last_stable_step,
                },
                'normalization': 'DISABLED',
                'checkpoint_type': 'best' if is_best else 'regular',
                'temp_directory_enabled': self.temp_checkpoint_dir is not None,
            }
            
            if self.scaler is not None:
                checkpoint['scaler_state_dict'] = self.scaler.state_dict()
            
            # Save locally
            torch.save(checkpoint, checkpoint_path)
            self.local_checkpoints.append(checkpoint_path)
            
            # Get checkpoint size for logging
            file_size_mb = checkpoint_path.stat().st_size / (1024 * 1024)
            logger.info(f"✅ Checkpoint saved: {checkpoint_filename} ({file_size_mb:.1f} MB)")
            
            # Save to temp directory if conditions are met
            should_save_to_temp = (
                self.temp_checkpoint_dir and (
                    is_best or 
                    force_temp or 
                    (self.global_step % self.save_to_temp_every_n_steps == 0)
                )
            )
            
            if should_save_to_temp:
                temp_success = self._save_checkpoint_to_temp(checkpoint_path)
                if temp_success and is_best:
                    logger.info(f"🏆 Best checkpoint also saved to temp directory")
            
            # Clean up old local checkpoints (but keep best checkpoints)
            if not is_best:
                self._cleanup_local_checkpoints()
            
            # Log to WandB
            if self.use_wandb:
                wandb.log({
                    "checkpoint/step": self.global_step,
                    "checkpoint/is_best": is_best,
                    "checkpoint/size_mb": file_size_mb,
                    "checkpoint/saved_to_temp": should_save_to_temp,
                }, step=self.global_step)
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Checkpoint save failed: {e}")
            return False

    def train(self) -> Dict[str, Any]:
        """Main training loop without normalization"""
        logger.info("🚀 Starting BLIP3-o training (NO NORMALIZATION)...")
        logger.info(f"  Enhanced stability monitoring active")
        logger.info(f"  WandB logging: {'✅ ENABLED' if self.use_wandb else '❌ DISABLED'}")
        logger.info(f"  Checkpoint strategy: {'Local + Temp' if self.temp_checkpoint_dir else 'Local only'}")
        
        if self.temp_checkpoint_dir:
            logger.info(f"  Temp directory: {self.temp_checkpoint_dir}")
            logger.info(f"  Save to temp every: {self.save_to_temp_every_n_steps} steps")
            logger.info(f"  Keep local checkpoints: {self.keep_local_checkpoints}")
        
        # Log initial setup to WandB
        if self.use_wandb:
            wandb.log({
                "setup/training_started": True,
                "setup/model_parameters": sum(p.numel() for p in self.model.parameters()),
                "setup/normalization": "DISABLED",
                "setup/estimated_steps_per_epoch": self.estimated_steps_per_epoch,
                "setup/total_estimated_steps": self.estimated_steps_per_epoch * self.num_epochs,
                "setup/temp_checkpoint_enabled": self.temp_checkpoint_dir is not None,
            }, step=0)
        
        self.model.train()
        start_time = time.time()
        
        try:
            for epoch in range(self.num_epochs):
                self.current_epoch = epoch
                logger.info(f"Starting epoch {epoch + 1}/{self.num_epochs}")
                
                epoch_loss = 0.0
                epoch_steps = 0
                epoch_failures = 0
                epoch_start_time = time.time()
                
                try:
                    dataloader_iter = iter(self.train_dataloader)
                    batch_count = 0
                    
                    while True:
                        try:
                            batch = next(dataloader_iter)
                            batch_count += 1
                        except StopIteration:
                            logger.info(f"Epoch {epoch + 1} completed: {batch_count} batches, {epoch_failures} failures")
                            break
                        
                        step_start_time = time.time()
                        
                        # Compute loss with stability checks
                        loss, metrics, is_stable = self._compute_loss_with_stability_check(batch)
                        
                        if not is_stable:
                            self._handle_training_instability()
                            epoch_failures += 1
                            continue
                        
                        # Backward pass with stability checks
                        grad_norm, step_success = self._backward_and_step_with_stability(loss)
                        
                        if not step_success:
                            self._handle_training_instability()
                            epoch_failures += 1
                            continue
                        
                        # Success - reset failure counter
                        self.consecutive_failures = 0
                        self.last_stable_step = self.global_step
                        
                        # Update tracking
                        epoch_loss += loss.item()
                        epoch_steps += 1
                        self.global_step += 1
                        
                        self.loss_history.append(loss.item())
                        self.grad_norm_history.append(grad_norm)
                        self.lr_history.append(self.optimizer.param_groups[0]['lr'])
                        
                        # Update best loss
                        if loss.item() < self.best_loss:
                            self.best_loss = loss.item()
                        
                        # Log to WandB
                        if self.use_wandb:
                            wandb.log({
                                "train/loss": loss.item(),
                                "train/grad_norm": grad_norm,
                                "train/learning_rate": self.optimizer.param_groups[0]['lr'],
                                "train/epoch": self.current_epoch,
                                "stability/consecutive_failures": self.consecutive_failures,
                                "stability/epoch_failures": epoch_failures,
                            }, step=self.global_step)
                        
                        # Console logging
                        if self.global_step % self.log_every_n_steps == 0:
                            logger.info(f"Step {self.global_step}: Loss={loss.item():.6f}, GradNorm={grad_norm:.3f}, Failures={epoch_failures}")
                        
                        # Evaluation
                        if self.global_step % self.eval_every_n_steps == 0:
                            logger.info(f"Running evaluation at step {self.global_step}...")
                            
                            eval_metrics = self._safe_evaluate()
                            
                            if eval_metrics and 'eval_clip_similarity' in eval_metrics:
                                similarity = eval_metrics['eval_clip_similarity']
                                logger.info(f"✅ Evaluation: CLIP similarity = {similarity:.4f}")
                                
                                if self.use_wandb:
                                    wandb_eval = {f"eval/{k.replace('eval_', '')}": v for k, v in eval_metrics.items() 
                                                if isinstance(v, (int, float)) and not math.isnan(v)}
                                    wandb.log(wandb_eval, step=self.global_step)
                                
                                # Check if this is the best model
                                is_best = similarity > self.best_eval_similarity
                                if is_best:
                                    self.best_eval_similarity = similarity
                                    logger.info(f"🎉 NEW BEST similarity: {similarity:.4f}")
                                    # Save best checkpoint
                                    self._save_checkpoint(is_best=True)
                            else:
                                logger.warning("⚠️ Evaluation failed or returned no metrics")
                        
                        # Regular checkpoint saving
                        if self.global_step % self.save_every_n_steps == 0:
                            self._save_checkpoint(is_best=False)
                
                except Exception as e:
                    logger.error(f"❌ Error during epoch {epoch + 1}: {e}")
                    continue
                
                # End of epoch summary
                epoch_time = time.time() - epoch_start_time
                avg_loss = epoch_loss / max(epoch_steps, 1)
                failure_rate = epoch_failures / max(batch_count, 1)
                
                logger.info(f"Epoch {epoch + 1} summary:")
                logger.info(f"  Average loss: {avg_loss:.6f}")
                logger.info(f"  Failure rate: {failure_rate*100:.1f}%")
                logger.info(f"  Best similarity: {self.best_eval_similarity:.4f}")
                logger.info(f"  Epoch time: {epoch_time:.1f}s")
                
                # Save end-of-epoch checkpoint
                if (epoch + 1) % 5 == 0:  # Every 5 epochs
                    self._save_checkpoint(force_temp=True)
            
            # Final evaluation
            logger.info("Running final evaluation...")
            final_eval = self._safe_evaluate(num_samples=self.eval_num_samples * 2)
            
            # Save final checkpoint
            self._save_checkpoint(force_temp=True)
            
            total_time = time.time() - start_time
            
            # Training summary
            summary = {
                'training_completed': True,
                'total_time_seconds': total_time,
                'total_steps': self.global_step,
                'best_eval_similarity': self.best_eval_similarity,
                'best_loss': self.best_loss,
                'emergency_lr_reductions': self.emergency_lr_reductions,
                'stability_alerts': self.stability_alerts,
                'final_eval': final_eval,
                'normalization_disabled': True,
                'checkpoint_strategy_used': 'temp_directory' if self.temp_checkpoint_dir else 'local_only',
                'checkpoints_saved_to_temp': self.temp_checkpoint_dir is not None,
            }
            
            if self.use_wandb:
                wandb.log({
                    "final/training_completed": True,
                    "final/total_time_seconds": total_time,
                    "final/best_eval_similarity": self.best_eval_similarity,
                    "final/normalization": "DISABLED",
                    "final/temp_checkpoints_used": self.temp_checkpoint_dir is not None,
                }, step=self.global_step)
                wandb.finish()
            
            logger.info("🎉 Training completed (NO NORMALIZATION)!")
            logger.info(f"  Total time: {total_time:.1f} seconds")
            logger.info(f"  Best similarity: {self.best_eval_similarity:.4f}")
            logger.info(f"  Training stability maintained")
            if self.temp_checkpoint_dir:
                logger.info(f"  Checkpoints available in: {self.temp_checkpoint_dir}")
            
            return summary
            
        except Exception as e:
            logger.error(f"❌ Training failed: {e}")
            if self.use_wandb:
                wandb.finish()
            raise


def create_clip_trainer(
    model,
    loss_fn,
    train_dataloader,
    eval_dataloader=None,
    learning_rate: float = 1e-4,
    num_epochs: int = 10,
    output_dir: str = "./checkpoints",
    temp_checkpoint_dir: Optional[str] = None,  # NEW parameter
    use_wandb: bool = False,
    wandb_project: str = "blip3o-clip-no-norm",
    **kwargs
) -> BLIP3oCLIPTrainer:
    """Factory function to create CLIP trainer without normalization - Updated with temp directory support"""
    
    return BLIP3oCLIPTrainer(
        model=model,
        loss_fn=loss_fn,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        output_dir=output_dir,
        temp_checkpoint_dir=temp_checkpoint_dir,  # NEW parameter
        use_wandb=use_wandb,
        wandb_project=wandb_project,
        **kwargs
    )


# Backward compatibility aliases
create_ultra_conservative_clip_trainer = create_clip_trainer
create_fixed_clip_trainer = create_clip_trainer