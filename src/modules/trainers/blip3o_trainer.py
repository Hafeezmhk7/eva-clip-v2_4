#!/usr/bin/env python3
"""
FIXED BLIP3-o Trainer for CLIP Reproduction - Gradient Explosion Fixes
Based on BLIP3-o paper and flow matching literature best practices
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
import warnings

# WandB import with error handling
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None

logger = logging.getLogger(__name__)


class StableBLIP3oCLIPTrainer:
    """
    FIXED Trainer for BLIP3-o CLIP Reproduction with gradient explosion fixes
    
    Key fixes based on BLIP3-o paper and flow matching literature:
    - Conservative initialization and learning rates
    - Proper gradient clipping implementation
    - Stable mixed precision training
    - Progressive learning rate schedules
    - Training health monitoring
    """
    
    def __init__(
        self,
        model,
        loss_fn,
        train_dataloader,
        eval_dataloader=None,
        # FIXED: Conservative training configuration
        learning_rate: float = 5e-5,  # Much lower than original 1e-4
        weight_decay: float = 0.01,
        num_epochs: int = 100,  # Longer training
        warmup_steps: int = 2000,  # Much longer warmup
        max_grad_norm: float = 0.5,  # Stricter clipping
        fp16: bool = True,
        # FIXED: More conservative evaluation
        eval_every_n_steps: int = 500,
        eval_num_samples: int = 200,
        eval_inference_steps: int = 50,
        # Enhanced monitoring
        log_every_n_steps: int = 10,
        save_every_n_steps: int = 1000,
        gradient_monitoring: bool = True,
        # Output
        output_dir: str = "./checkpoints",
        # Device
        device: Optional[torch.device] = None,
        # WandB configuration
        use_wandb: bool = False,
        wandb_project: str = "blip3o-clip-stable",
        wandb_run_name: Optional[str] = None,
        wandb_config: Optional[Dict] = None,
        wandb_api_key: Optional[str] = None,
        # FIXED: Stability options
        gradient_accumulation_steps: int = 1,
        skip_nan_loss: bool = True,
        early_stop_on_explosion: bool = True,
        min_lr_ratio: float = 0.1,  # Don't let LR drop too low
    ):
        self.model = model
        self.loss_fn = loss_fn
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        
        # FIXED: Conservative training config
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.num_epochs = num_epochs
        self.warmup_steps = warmup_steps
        self.max_grad_norm = max_grad_norm
        self.fp16 = fp16
        self.gradient_accumulation_steps = gradient_accumulation_steps
        
        # Evaluation config
        self.eval_every_n_steps = eval_every_n_steps
        self.eval_num_samples = eval_num_samples
        self.eval_inference_steps = eval_inference_steps
        
        # Monitoring config
        self.log_every_n_steps = log_every_n_steps
        self.save_every_n_steps = save_every_n_steps
        self.gradient_monitoring = gradient_monitoring
        
        # Stability options
        self.skip_nan_loss = skip_nan_loss
        self.early_stop_on_explosion = early_stop_on_explosion
        self.min_lr_ratio = min_lr_ratio
        
        # Output
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Device
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.model = self.model.to(self.device)
        
        # FIXED: Apply stable initialization
        self._apply_stable_initialization()
        
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
        
        # FIXED: Enhanced metrics tracking for stability monitoring
        self.loss_history = deque(maxlen=1000)
        self.similarity_history = deque(maxlen=1000)
        self.lr_history = deque(maxlen=1000)
        self.grad_norm_history = deque(maxlen=1000)
        self.gradient_explosion_count = 0
        self.consecutive_explosions = 0
        self.nan_loss_count = 0
        
        # Estimate steps per epoch BEFORE setup
        self.estimated_steps_per_epoch = self._estimate_steps_per_epoch()
        
        # FIXED: Setup optimizer and scheduler with stability
        self._setup_stable_optimizer_and_scheduler()
        
        # FIXED: Setup stable mixed precision
        if self.fp16:
            self.scaler = torch.amp.GradScaler('cuda', 
                                               init_scale=1024,  # Conservative init scale
                                               growth_factor=1.5,  # Slower growth
                                               backoff_factor=0.8,  # More aggressive backoff
                                               growth_interval=1000)  # Less frequent growth
        else:
            self.scaler = None
        
        # Setup WandB
        if self.use_wandb:
            self._setup_wandb()
        elif use_wandb and not WANDB_AVAILABLE:
            logger.warning("WandB requested but not available. Install with: pip install wandb")
        
        logger.info("🛡️ Stable BLIP3-o CLIP Trainer initialized with gradient explosion fixes")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        logger.info(f"  Learning rate: {self.learning_rate:.2e} (conservative)")
        logger.info(f"  Max grad norm: {self.max_grad_norm} (strict)")
        logger.info(f"  Warmup steps: {self.warmup_steps} (extended)")

    def _apply_stable_initialization(self):
        """Apply stable initialization based on BLIP3-o paper recommendations"""
        logger.info("🔧 Applying stable initialization for gradient stability...")
        
        # Conservative scaling factor based on model depth
        num_layers = getattr(self.model.config, 'num_hidden_layers', 12)
        scale_factor = 1.0 / math.sqrt(num_layers)  # Scale down with depth
        
        def init_weights(module):
            if isinstance(module, torch.nn.Linear):
                # Xavier uniform with conservative scaling
                torch.nn.init.xavier_uniform_(module.weight, gain=scale_factor)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, torch.nn.Embedding):
                # Small normal initialization for embeddings
                torch.nn.init.normal_(module.weight, std=0.01)
            elif isinstance(module, torch.nn.LayerNorm) or isinstance(module, torch.nn.modules.normalization.RMSNorm):
                if hasattr(module, 'weight') and module.weight is not None:
                    torch.nn.init.ones_(module.weight)
                if hasattr(module, 'bias') and module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
        
        # Apply to entire model
        self.model.apply(init_weights)
        
        # Special handling for output projection (critical for flow matching)
        if hasattr(self.model, 'output_proj'):
            torch.nn.init.normal_(self.model.output_proj.weight, std=0.02)
            if self.model.output_proj.bias is not None:
                torch.nn.init.zeros_(self.model.output_proj.bias)
        
        # Special handling for position embeddings
        if hasattr(self.model, 'pos_embed'):
            torch.nn.init.normal_(self.model.pos_embed, std=0.01)
        
        logger.info(f"✅ Stable initialization applied with scale factor: {scale_factor:.4f}")

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
                estimated_steps = max(1, dataset_length // (batch_size * self.gradient_accumulation_steps))
                logger.info(f"Estimated steps per epoch from dataset length: {estimated_steps}")
                return estimated_steps
            except (TypeError, AttributeError):
                logger.warning("Could not estimate steps per epoch, using default: 100")
                return 100

    def _setup_stable_optimizer_and_scheduler(self):
        """Setup optimizer and learning rate scheduler with stability fixes"""
        
        # FIXED: Conservative AdamW settings based on flow matching literature
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.95),  # More conservative beta2
            eps=1e-8,
            amsgrad=False  # Can cause instability
        )
        
        total_steps = self.estimated_steps_per_epoch * self.num_epochs
        
        # FIXED: More stable learning rate schedule
        if self.warmup_steps > 0:
            # Very gradual warmup to prevent early explosion
            warmup_scheduler = LinearLR(
                self.optimizer,
                start_factor=0.01,  # Start at 1% of target LR
                end_factor=1.0,
                total_iters=self.warmup_steps
            )
            
            # Cosine with higher minimum to prevent learning shutdown
            min_lr = self.learning_rate * self.min_lr_ratio
            cosine_scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=total_steps - self.warmup_steps,
                eta_min=min_lr
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
                eta_min=self.learning_rate * self.min_lr_ratio
            )
        
        logger.info(f"✅ Stable optimizer and scheduler setup complete")
        logger.info(f"  Total estimated steps: {total_steps}")
        logger.info(f"  Warmup steps: {self.warmup_steps}")
        logger.info(f"  Min LR ratio: {self.min_lr_ratio}")

    def _setup_wandb(self):
        """Setup WandB with stability monitoring"""
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
                }
            
            wandb_config = {
                'learning_rate': self.learning_rate,
                'weight_decay': self.weight_decay,
                'num_epochs': self.num_epochs,
                'warmup_steps': self.warmup_steps,
                'max_grad_norm': self.max_grad_norm,
                'fp16': self.fp16,
                'gradient_accumulation_steps': self.gradient_accumulation_steps,
                'estimated_steps_per_epoch': self.estimated_steps_per_epoch,
                'eval_every_n_steps': self.eval_every_n_steps,
                'experiment_type': 'blip3o_clip_stable',
                'stability_features': {
                    'stable_initialization': True,
                    'conservative_lr': True,
                    'extended_warmup': True,
                    'strict_grad_clipping': True,
                    'gradient_monitoring': self.gradient_monitoring,
                },
                **model_config,
                **self.wandb_config,
            }
            
            self.wandb_run = wandb.init(
                project=self.wandb_project,
                name=self.wandb_run_name,
                config=wandb_config,
                dir=str(self.output_dir),
                resume="allow",
                tags=["blip3o", "clip_reproduction", "stable", "gradient_explosion_fix"]
            )
            
            if hasattr(self.model, 'get_num_parameters'):
                wandb.log({"model/total_parameters": self.model.get_num_parameters()})
            
            wandb.watch(self.model, log="all", log_freq=self.log_every_n_steps * 5)  # Less frequent
            
            logger.info(f"✅ WandB initialized: {self.wandb_project}")
            logger.info(f"   Run ID: {self.wandb_run.id}")
            
        except Exception as e:
            logger.error(f"❌ Failed to setup WandB: {e}")
            self.use_wandb = False

    def _check_tensor_health(self, tensor: torch.Tensor, name: str) -> bool:
        """Check tensor for NaN/Inf values"""
        if torch.isnan(tensor).any():
            logger.error(f"🚨 NaN detected in {name}")
            return False
        if torch.isinf(tensor).any():
            logger.error(f"🚨 Inf detected in {name}")
            return False
        return True

    def _compute_loss_stable(self, batch: Dict[str, Any]) -> Tuple[Optional[torch.Tensor], Optional[Dict[str, float]]]:
        """Compute loss with stability checks"""
        
        # Move tensors to device and check health
        for key, value in batch.items():
            if torch.is_tensor(value):
                batch[key] = value.to(self.device)
                if not self._check_tensor_health(value, f"input_{key}"):
                    return None, None
        
        hidden_states = batch['hidden_states']
        timestep = batch['timestep']
        encoder_hidden_states = batch['encoder_hidden_states']
        clip_embeddings = batch['clip_embeddings']
        noise = batch.get('noise')
        
        try:
            if self.fp16:
                with torch.amp.autocast('cuda', dtype=torch.float16):
                    model_output = self.model(
                        hidden_states=hidden_states,
                        timestep=timestep,
                        encoder_hidden_states=encoder_hidden_states,
                        return_dict=False
                    )
                    
                    # Check model output health
                    if not self._check_tensor_health(model_output, "model_output"):
                        return None, None
                    
                    loss, metrics = self.loss_fn(
                        model_output=model_output,
                        target_samples=clip_embeddings,
                        timesteps=timestep,
                        eva_conditioning=encoder_hidden_states,
                        noise=noise,
                        return_metrics=True
                    )
            else:
                model_output = self.model(
                    hidden_states=hidden_states,
                    timestep=timestep,
                    encoder_hidden_states=encoder_hidden_states,
                    return_dict=False
                )
                
                if not self._check_tensor_health(model_output, "model_output"):
                    return None, None
                
                loss, metrics = self.loss_fn(
                    model_output=model_output,
                    target_samples=clip_embeddings,
                    timesteps=timestep,
                    eva_conditioning=encoder_hidden_states,
                    noise=noise,
                    return_metrics=True
                )
            
            # Check loss health
            if not self._check_tensor_health(loss, "loss"):
                return None, None
            
            # Additional stability checks
            if loss.item() > 100.0:  # Suspiciously high loss
                logger.warning(f"⚠️ Very high loss detected: {loss.item():.2f}")
                if loss.item() > 1000.0:
                    logger.error("🚨 Extremely high loss - skipping step")
                    return None, None
            
            return loss, metrics
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                logger.error("🚨 GPU out of memory during forward pass")
                torch.cuda.empty_cache()
            else:
                logger.error(f"🚨 Runtime error in forward pass: {e}")
            return None, None
        except Exception as e:
            logger.error(f"🚨 Unexpected error in loss computation: {e}")
            return None, None

    def _backward_and_step_stable(self, loss: torch.Tensor) -> Tuple[float, bool]:
        """FIXED: Stable backward pass with proper gradient clipping"""
        
        # Scale loss for gradient accumulation
        scaled_loss = loss / self.gradient_accumulation_steps
        
        try:
            if self.fp16:
                # Mixed precision backward
                self.scaler.scale(scaled_loss).backward()
                
                # Only step if we've accumulated enough gradients
                if (self.global_step + 1) % self.gradient_accumulation_steps == 0:
                    # CRITICAL: Unscale gradients before computing norm and clipping
                    self.scaler.unscale_(self.optimizer)
                    
                    # Compute gradient norm BEFORE clipping
                    total_norm = 0.0
                    param_count = 0
                    for p in self.model.parameters():
                        if p.grad is not None:
                            param_norm = p.grad.data.norm(2)
                            total_norm += param_norm.item() ** 2
                            param_count += 1
                    
                    if param_count > 0:
                        grad_norm = total_norm ** 0.5
                    else:
                        grad_norm = 0.0
                    
                    # Apply gradient clipping
                    if self.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), 
                            self.max_grad_norm
                        )
                    
                    # Check for gradient explosion BEFORE stepping
                    is_stable = grad_norm < 100.0  # Threshold for explosion
                    
                    if is_stable and torch.isfinite(torch.tensor(grad_norm)):
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        self.scheduler.step()
                        self.optimizer.zero_grad()
                    else:
                        logger.warning(f"⚠️ Skipping step due to gradient issues: norm={grad_norm:.2f}")
                        self.scaler.update()
                        self.optimizer.zero_grad()
                        return grad_norm, False
                else:
                    # Just accumulate gradients
                    return 0.0, True
                    
            else:
                # Regular precision backward
                scaled_loss.backward()
                
                if (self.global_step + 1) % self.gradient_accumulation_steps == 0:
                    # Compute gradient norm
                    total_norm = 0.0
                    for p in self.model.parameters():
                        if p.grad is not None:
                            param_norm = p.grad.data.norm(2)
                            total_norm += param_norm.item() ** 2
                    grad_norm = total_norm ** 0.5
                    
                    # Check stability before clipping and stepping
                    is_stable = grad_norm < 100.0 and torch.isfinite(torch.tensor(grad_norm))
                    
                    if is_stable:
                        # Apply gradient clipping
                        if self.max_grad_norm > 0:
                            torch.nn.utils.clip_grad_norm_(
                                self.model.parameters(), 
                                self.max_grad_norm
                            )
                        
                        self.optimizer.step()
                        self.scheduler.step()
                        self.optimizer.zero_grad()
                    else:
                        logger.warning(f"⚠️ Skipping step due to gradient explosion: {grad_norm:.2f}")
                        self.optimizer.zero_grad()
                        return grad_norm, False
                else:
                    return 0.0, True
            
            return grad_norm, is_stable
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                logger.error("🚨 GPU OOM during backward pass")
                torch.cuda.empty_cache()
            else:
                logger.error(f"🚨 Runtime error in backward pass: {e}")
            self.optimizer.zero_grad()
            return 0.0, False
        except Exception as e:
            logger.error(f"🚨 Unexpected error in backward pass: {e}")
            self.optimizer.zero_grad()
            return 0.0, False

    def _monitor_training_health(self, loss: float, metrics: Dict[str, float], grad_norm: float) -> bool:
        """Monitor training health and detect issues early"""
        
        if not self.gradient_monitoring:
            return True
        
        is_healthy = True
        
        # 1. GRADIENT HEALTH CHECK
        if grad_norm > 100:
            self.gradient_explosion_count += 1
            self.consecutive_explosions += 1
            logger.error(f"🚨 GRADIENT EXPLOSION: GradNorm={grad_norm:.1f} (#{self.gradient_explosion_count})")
            
            if self.consecutive_explosions >= 5:
                logger.error("🚨 CRITICAL: 5 consecutive gradient explosions!")
                if self.early_stop_on_explosion:
                    logger.error("🛑 Stopping training due to repeated gradient explosions")
                    return False
            is_healthy = False
            
        elif grad_norm > 50:
            logger.warning(f"⚠️ High gradient norm: {grad_norm:.1f}")
            self.consecutive_explosions = max(0, self.consecutive_explosions - 1)
        else:
            self.consecutive_explosions = 0
        
        # 2. LOSS HEALTH CHECK  
        if torch.isnan(torch.tensor(loss)) or torch.isinf(torch.tensor(loss)):
            self.nan_loss_count += 1
            logger.error(f"🚨 NaN/Inf loss detected! (#{self.nan_loss_count})")
            if self.nan_loss_count >= 10:
                logger.error("🛑 Too many NaN losses - stopping training")
                return False
            is_healthy = False
        
        # 3. VELOCITY SIMILARITY HEALTH CHECK
        vel_sim = metrics.get('velocity_similarity', 0)
        if vel_sim < 0.01 and self.global_step > 100:
            logger.warning(f"⚠️ Very low velocity similarity: {vel_sim:.4f}")
        
        # 4. LEARNING RATE CHECK
        current_lr = self.optimizer.param_groups[0]['lr']
        if current_lr < 1e-8:
            logger.warning(f"⚠️ Learning rate extremely low: {current_lr:.2e}")
        
        # 5. PERIODIC HEALTH SUMMARY
        if self.global_step % 100 == 0:
            health_score = 0
            health_score += 1 if loss < 5.0 else 0
            health_score += 1 if grad_norm < 10 else 0  
            health_score += 1 if vel_sim > 0.1 else 0
            health_score += 1 if current_lr > 1e-7 else 0
            
            health_status = ("🟢 HEALTHY" if health_score >= 3 else 
                           "🟡 CONCERNING" if health_score >= 2 else "🔴 CRITICAL")
            
            logger.info(f"📊 Health Check (Step {self.global_step}): {health_status} ({health_score}/4)")
            logger.info(f"   Loss: {loss:.4f}, GradNorm: {grad_norm:.2f}, VelSim: {vel_sim:.4f}, LR: {current_lr:.2e}")
            
            if self.use_wandb:
                wandb.log({
                    "health/score": health_score,
                    "health/gradient_explosions": self.gradient_explosion_count,
                    "health/consecutive_explosions": self.consecutive_explosions,
                    "health/nan_losses": self.nan_loss_count,
                }, step=self.global_step)
        
        return is_healthy

    def _evaluate(self, num_samples: Optional[int] = None) -> Dict[str, float]:
        """Run evaluation with stability checks"""
        if self.eval_dataloader is None:
            return {}
        
        if num_samples is None:
            num_samples = self.eval_num_samples
        
        logger.info(f"Starting stable evaluation with {num_samples} samples")
        
        self.model.eval()
        
        all_similarities = []
        all_mse_losses = []
        all_generated_norms = []
        all_target_norms = []
        samples_processed = 0
        
        eval_start_time = time.time()
        
        try:
            with torch.no_grad():
                for batch_idx, batch in enumerate(self.eval_dataloader):
                    if samples_processed >= num_samples:
                        break
                    
                    try:
                        eva_features = batch['encoder_hidden_states'].to(self.device)
                        target_clip = batch['clip_embeddings'].to(self.device)
                        
                        # Check input health
                        if not self._check_tensor_health(eva_features, "eval_eva_features"):
                            continue
                        if not self._check_tensor_health(target_clip, "eval_target_clip"):
                            continue
                        
                        # Generate CLIP embeddings
                        generated_clip = self.model.generate(
                            eva_features=eva_features,
                            num_inference_steps=self.eval_inference_steps,
                        )
                        
                        # Check generated output health
                        if not self._check_tensor_health(generated_clip, "eval_generated_clip"):
                            continue
                        
                        # Compute metrics
                        target_norm = F.normalize(target_clip, p=2, dim=-1)
                        generated_norm = F.normalize(generated_clip, p=2, dim=-1)
                        similarity = F.cosine_similarity(generated_norm, target_norm, dim=-1)
                        per_image_similarity = similarity.mean(dim=1)
                        
                        mse_loss = F.mse_loss(generated_clip, target_clip, reduction='none').mean(dim=(1, 2))
                        
                        generated_norms = torch.norm(generated_clip, dim=-1).mean(dim=1)
                        target_norms = torch.norm(target_clip, dim=-1).mean(dim=1)
                        
                        all_similarities.append(per_image_similarity.cpu())
                        all_mse_losses.append(mse_loss.cpu())
                        all_generated_norms.append(generated_norms.cpu())
                        all_target_norms.append(target_norms.cpu())
                        samples_processed += eva_features.shape[0]
                    
                    except Exception as e:
                        logger.warning(f"⚠️ Error in evaluation batch {batch_idx}: {e}")
                        continue
        
        except Exception as e:
            logger.error(f"❌ Error in evaluation loop: {e}")
            return {}
        
        finally:
            self.model.train()
        
        if not all_similarities:
            logger.warning("No evaluation samples processed successfully")
            return {}
        
        try:
            all_sims = torch.cat(all_similarities)
            all_mse = torch.cat(all_mse_losses)
            all_gen_norms = torch.cat(all_generated_norms)
            all_tgt_norms = torch.cat(all_target_norms)
            
            eval_time = time.time() - eval_start_time
            
            eval_metrics = {
                'eval_clip_similarity': all_sims.mean().item(),
                'eval_clip_similarity_std': all_sims.std().item(),
                'eval_mse_loss': all_mse.mean().item(),
                'eval_high_quality': (all_sims > 0.7).float().mean().item(),
                'eval_very_high_quality': (all_sims > 0.8).float().mean().item(),
                'eval_excellent_quality': (all_sims > 0.9).float().mean().item(),
                'eval_samples': samples_processed,
                'eval_time_seconds': eval_time,
                'eval_generated_norm_mean': all_gen_norms.mean().item(),
                'eval_target_norm_mean': all_tgt_norms.mean().item(),
                'eval_norm_ratio': all_gen_norms.mean().item() / (all_tgt_norms.mean().item() + 1e-8),
            }
            
            logger.info(f"✅ Evaluation completed: {samples_processed} samples, similarity: {eval_metrics['eval_clip_similarity']:.4f}")
            
            return eval_metrics
            
        except Exception as e:
            logger.error(f"❌ Error processing evaluation results: {e}")
            return {}

    def _log_metrics(self, loss: float, metrics: Dict[str, float], grad_norm: float, step_time: float = 0.0):
        """Enhanced logging with stability metrics"""
        
        # Store metrics
        self.loss_history.append(loss)
        if 'velocity_similarity' in metrics:
            self.similarity_history.append(metrics['velocity_similarity'])
        self.lr_history.append(self.optimizer.param_groups[0]['lr'])
        self.grad_norm_history.append(grad_norm)
        
        # Update best metrics
        if 'velocity_similarity' in metrics:
            if metrics['velocity_similarity'] > self.best_eval_similarity:
                self.best_eval_similarity = metrics['velocity_similarity']
        
        if loss < self.best_loss:
            self.best_loss = loss
        
        # Console logging
        if self.global_step % self.log_every_n_steps == 0:
            vel_sim = metrics.get('velocity_similarity', 0)
            
            # Determine status based on gradient norm
            if grad_norm > 100:
                status = "🚨 EXPLOSION"
            elif grad_norm > 50:
                status = "⚠️ HIGH"
            elif grad_norm < 0.001:
                status = "⚠️ LOW"
            else:
                status = "✅ STABLE"
            
            log_msg = (f"Step {self.global_step}: Loss={loss:.6f}, VelSim={vel_sim:.4f}, "
                      f"GradNorm={grad_norm:.3f} {status}, LR={self.optimizer.param_groups[0]['lr']:.2e}")
            
            if step_time > 0:
                log_msg += f", Time={step_time:.2f}s"
            
            logger.info(log_msg)
        
        # WandB logging
        if self.use_wandb:
            wandb_metrics = {
                "train/loss": loss,
                "train/grad_norm": grad_norm,
                "train/learning_rate": self.optimizer.param_groups[0]['lr'],
                "train/epoch": self.current_epoch,
                "train/step": self.global_step,
                "stability/gradient_explosions_total": self.gradient_explosion_count,
                "stability/consecutive_explosions": self.consecutive_explosions,
                "stability/nan_losses_total": self.nan_loss_count,
            }
            
            if step_time > 0:
                wandb_metrics["timing/step_time"] = step_time
            
            if metrics:
                for key, value in metrics.items():
                    if isinstance(value, (int, float)) and not math.isnan(value):
                        wandb_metrics[f"train/{key}"] = value
            
            # Moving averages
            if len(self.loss_history) > 0:
                wandb_metrics["train/loss_ma"] = np.mean(list(self.loss_history))
            if len(self.similarity_history) > 0:
                wandb_metrics["train/similarity_ma"] = np.mean(list(self.similarity_history))
            if len(self.grad_norm_history) > 0:
                wandb_metrics["train/grad_norm_ma"] = np.mean(list(self.grad_norm_history))
            
            wandb_metrics["train/best_loss"] = self.best_loss
            wandb_metrics["train/best_similarity"] = self.best_eval_similarity
            
            if torch.cuda.is_available():
                wandb_metrics["system/gpu_memory_allocated"] = torch.cuda.memory_allocated() / 1e9
                wandb_metrics["system/gpu_memory_reserved"] = torch.cuda.memory_reserved() / 1e9
            
            wandb.log(wandb_metrics, step=self.global_step)

    def _save_checkpoint(self):
        """Save model checkpoint with stability metrics"""
        checkpoint_path = self.output_dir / f"checkpoint_step_{self.global_step}.pt"
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'global_step': self.global_step,
            'current_epoch': self.current_epoch,
            'best_eval_similarity': self.best_eval_similarity,
            'best_loss': self.best_loss,
            'loss_history': list(self.loss_history),
            'similarity_history': list(self.similarity_history),
            'grad_norm_history': list(self.grad_norm_history),
            'gradient_explosion_count': self.gradient_explosion_count,
            'nan_loss_count': self.nan_loss_count,
            'experiment_type': 'blip3o_clip_stable',
            'stability_config': {
                'max_grad_norm': self.max_grad_norm,
                'learning_rate': self.learning_rate,
                'warmup_steps': self.warmup_steps,
                'gradient_accumulation_steps': self.gradient_accumulation_steps,
            }
        }
        
        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"💾 Checkpoint saved: {checkpoint_path}")
        
        if self.use_wandb:
            wandb.log({
                "checkpoint/saved": True,
                "checkpoint/step": self.global_step,
                "checkpoint/path": str(checkpoint_path),
            }, step=self.global_step)

    def train(self) -> Dict[str, Any]:
        """Main training loop with stability monitoring"""
        logger.info("🚀 Starting stable BLIP3-o training with gradient explosion fixes...")
        logger.info(f"  Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        logger.info(f"  Conservative learning rate: {self.learning_rate:.2e}")
        logger.info(f"  Strict gradient clipping: {self.max_grad_norm}")
        logger.info(f"  Extended warmup: {self.warmup_steps} steps")
        
        if self.use_wandb:
            wandb.log({"setup/training_started": True}, step=0)
        
        self.model.train()
        start_time = time.time()
        
        try:
            for epoch in range(self.num_epochs):
                self.current_epoch = epoch
                logger.info(f"🔄 Starting epoch {epoch + 1}/{self.num_epochs}")
                
                epoch_loss = 0.0
                epoch_steps = 0
                successful_steps = 0
                skipped_steps = 0
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
                        
                        # Compute loss with stability checks
                        loss, metrics = self._compute_loss_stable(batch)
                        if loss is None:
                            skipped_steps += 1
                            if self.skip_nan_loss:
                                logger.warning(f"⚠️ Skipping step {self.global_step} due to loss computation issues")
                                continue
                            else:
                                logger.error("🚨 Loss computation failed - stopping training")
                                break
                        
                        # Backward pass with stability checks
                        grad_norm, step_successful = self._backward_and_step_stable(loss)
                        
                        if not step_successful:
                            skipped_steps += 1
                            logger.warning(f"⚠️ Skipping step {self.global_step} due to gradient issues")
                            continue
                        
                        # Update counters
                        if (self.global_step + 1) % self.gradient_accumulation_steps == 0:
                            successful_steps += 1
                            epoch_loss += loss.item()
                            epoch_steps += 1
                        
                        step_time = time.time() - step_start_time
                        
                        # Health monitoring
                        is_healthy = self._monitor_training_health(
                            loss.item(), metrics or {}, grad_norm
                        )
                        
                        if not is_healthy and self.early_stop_on_explosion:
                            logger.error("🛑 Stopping training due to critical health issues")
                            break
                        
                        # Log metrics
                        self._log_metrics(loss.item(), metrics or {}, grad_norm, step_time)
                        
                        self.global_step += 1
                        
                        # Evaluation
                        if self.global_step % self.eval_every_n_steps == 0:
                            logger.info(f"📊 Running evaluation at step {self.global_step}...")
                            
                            try:
                                eval_metrics = self._evaluate()
                                
                                if eval_metrics:
                                    eval_sim = eval_metrics.get('eval_clip_similarity', 0)
                                    logger.info(f"✅ Evaluation: CLIP similarity = {eval_sim:.4f}")
                                    
                                    if self.use_wandb:
                                        wandb_eval_metrics = {f"eval/{k}": v for k, v in eval_metrics.items()}
                                        wandb.log(wandb_eval_metrics, step=self.global_step)
                                    
                                    if eval_sim > self.best_eval_similarity:
                                        self.best_eval_similarity = eval_sim
                                        logger.info(f"🎉 NEW BEST evaluation similarity: {self.best_eval_similarity:.4f}")
                                        
                                        if self.use_wandb:
                                            wandb.log({
                                                "eval/new_best_similarity": self.best_eval_similarity,
                                                "eval/best_similarity_step": self.global_step,
                                            }, step=self.global_step)
                                else:
                                    logger.warning("Evaluation returned no metrics")
                                    
                            except Exception as e:
                                logger.error(f"❌ Evaluation failed at step {self.global_step}: {e}")
                                if self.use_wandb:
                                    wandb.log({"eval/failed": True, "eval/error": str(e)}, step=self.global_step)
                        
                        # Save checkpoint
                        if self.global_step % self.save_every_n_steps == 0:
                            self._save_checkpoint()
                
                except Exception as e:
                    logger.error(f"❌ Error during epoch {epoch + 1}: {e}")
                    continue
                
                # End of epoch summary
                epoch_time = time.time() - epoch_start_time
                avg_epoch_loss = epoch_loss / max(epoch_steps, 1)
                
                logger.info(f"📊 Epoch {epoch + 1} summary:")
                logger.info(f"  Average loss: {avg_epoch_loss:.6f}")
                logger.info(f"  Best loss: {self.best_loss:.6f}")
                logger.info(f"  Best similarity: {self.best_eval_similarity:.4f}")
                logger.info(f"  Successful steps: {successful_steps}")
                logger.info(f"  Skipped steps: {skipped_steps}")
                logger.info(f"  Epoch time: {epoch_time:.1f}s")
                logger.info(f"  Gradient explosions: {self.gradient_explosion_count}")
                
                if self.use_wandb:
                    wandb.log({
                        "epoch/completed": epoch + 1,
                        "epoch/avg_loss": avg_epoch_loss,
                        "epoch/successful_steps": successful_steps,
                        "epoch/skipped_steps": skipped_steps,
                        "epoch/time_seconds": epoch_time,
                    }, step=self.global_step)
        
        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
            if self.use_wandb:
                wandb.log({"training/interrupted": True}, step=self.global_step)
        except Exception as e:
            logger.error(f"❌ Training failed with error: {e}")
            if self.use_wandb:
                wandb.log({"training/failed": True, "training/error": str(e)}, step=self.global_step)
            raise
        
        finally:
            # Final checkpoint
            self._save_checkpoint()
            
            # Final evaluation
            logger.info("📊 Running final evaluation...")
            try:
                final_eval = self._evaluate(num_samples=self.eval_num_samples * 2)
            except Exception as e:
                logger.error(f"❌ Final evaluation failed: {e}")
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
                'stability_stats': {
                    'gradient_explosion_count': self.gradient_explosion_count,
                    'nan_loss_count': self.nan_loss_count,
                    'max_consecutive_explosions': max(getattr(self, 'max_consecutive_explosions', 0), self.consecutive_explosions),
                },
                'loss_history': list(self.loss_history),
                'similarity_history': list(self.similarity_history),
                'grad_norm_history': list(self.grad_norm_history),
                'experiment_type': 'blip3o_clip_stable',
                'stability_features_used': True,
            }
            
            # Final WandB logging
            if self.use_wandb:
                final_wandb_metrics = {
                    "final/training_completed": True,
                    "final/total_time_seconds": total_time,
                    "final/total_steps": self.global_step,
                    "final/best_loss": self.best_loss,
                    "final/best_eval_similarity": self.best_eval_similarity,
                    "final/gradient_explosions": self.gradient_explosion_count,
                    "final/nan_losses": self.nan_loss_count,
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
            
            # Final status
            logger.info("=" * 80)
            logger.info("🎉 STABLE BLIP3-O TRAINING COMPLETED!")
            logger.info("=" * 80)
            logger.info(f"📊 RESULTS:")
            logger.info(f"  Duration: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
            logger.info(f"  Total steps: {self.global_step}")
            logger.info(f"  Best loss: {self.best_loss:.6f}")
            logger.info(f"  Best CLIP similarity: {self.best_eval_similarity:.4f}")
            logger.info(f"🛡️ STABILITY STATS:")
            logger.info(f"  Gradient explosions: {self.gradient_explosion_count}")
            logger.info(f"  NaN losses: {self.nan_loss_count}")
            
            if final_eval:
                final_sim = final_eval.get('eval_clip_similarity', 0)
                if final_sim > 0.6:
                    logger.info(f"  🎉 EXCELLENT: Final similarity {final_sim:.4f} > 0.6!")
                elif final_sim > 0.5:
                    logger.info(f"  ✅ VERY GOOD: Final similarity {final_sim:.4f} > 0.5!")
                elif final_sim > 0.4:
                    logger.info(f"  ✅ GOOD: Final similarity {final_sim:.4f} > 0.4!")
                elif final_sim > 0.3:
                    logger.info(f"  📈 PROGRESS: Final similarity {final_sim:.4f} > 0.3!")
                else:
                    logger.info(f"  ⚠️ NEEDS MORE TRAINING: Final similarity {final_sim:.4f}")
            
            logger.info("=" * 80)
            
            return summary


def create_stable_clip_trainer(
    model,
    loss_fn,
    train_dataloader,
    eval_dataloader=None,
    learning_rate: float = 5e-5,  # Conservative default
    num_epochs: int = 100,        # Longer training
    max_grad_norm: float = 0.5,   # Stricter clipping
    warmup_steps: int = 2000,     # Extended warmup
    output_dir: str = "./checkpoints",
    use_wandb: bool = False,
    wandb_project: str = "blip3o-clip-stable",
    wandb_run_name: Optional[str] = None,
    wandb_config: Optional[Dict] = None,
    **kwargs
) -> StableBLIP3oCLIPTrainer:
    """Factory function to create stable CLIP trainer with gradient explosion fixes"""
    
    return StableBLIP3oCLIPTrainer(
        model=model,
        loss_fn=loss_fn,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        max_grad_norm=max_grad_norm,
        warmup_steps=warmup_steps,
        output_dir=output_dir,
        use_wandb=use_wandb,
        wandb_project=wandb_project,
        wandb_run_name=wandb_run_name,
        wandb_config=wandb_config,
        **kwargs
    )