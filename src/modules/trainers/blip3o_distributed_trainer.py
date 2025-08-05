#!/usr/bin/env python3
"""
BLIP3-o Distributed Trainer with FSDP
src/modules/trainers/blip3o_distributed_trainer.py

Distributed trainer using FSDP (Fully Sharded Data Parallel) for memory-efficient
multi-GPU training of BLIP3-o models.
"""

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy, CPUOffload
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

# Import FSDP utilities
from src.modules.distributed.fsdp_utils import (
    wrap_model_with_fsdp,
    save_fsdp_checkpoint,
    load_fsdp_checkpoint,
    sync_across_gpus,
    is_master_rank,
    get_world_size,
    get_rank
)

# Import communication utilities
from src.modules.distributed.communication import (
    DistributedCommunicator,
    MetricsAggregator,
    log_distributed_info
)

# WandB import with error handling
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None

logger = logging.getLogger(__name__)


class BLIP3oDistributedTrainer:
    """
    BLIP3-o Distributed Trainer with FSDP Support
    
    Features:
    - FSDP parameter sharding for memory efficiency
    - Distributed evaluation and checkpointing
    - Multi-GPU gradient synchronization
    - Memory-efficient training for large models
    - Smart checkpoint management across nodes
    """
    
    def __init__(
        self,
        model,
        loss_fn,
        train_dataloader,
        eval_dataloader=None,
        
        # Distributed configuration
        world_size: int = 4,
        rank: int = 0,
        use_fsdp: bool = True,
        sharding_strategy: str = "FULL_SHARD",
        cpu_offload: bool = False,
        mixed_precision_fsdp: bool = True,
        
        # Training configuration
        learning_rate: float = 4e-5,
        weight_decay: float = 0.04,
        num_epochs: int = 8,
        warmup_steps: int = 100,
        max_grad_norm: float = 1.0,
        fp16: bool = True,
        
        # Evaluation
        eval_every_n_steps: int = 50,
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
        
        # Output - Enhanced for distributed
        output_dir: str = "./checkpoints",
        temp_checkpoint_dir: Optional[str] = None,
        keep_local_checkpoints: int = 3,
        save_to_temp_every_n_steps: int = 1000,
        
        # WandB configuration (only rank 0)
        use_wandb: bool = False,
        wandb_project: str = "blip3o-clip-fsdp",
        wandb_run_name: Optional[str] = None,
        wandb_config: Optional[Dict] = None,
        
        **kwargs
    ):
        self.world_size = world_size
        self.rank = rank
        self.use_fsdp = use_fsdp
        
        # Convert sharding strategy string to enum
        if isinstance(sharding_strategy, str):
            self.sharding_strategy = getattr(ShardingStrategy, sharding_strategy)
        else:
            self.sharding_strategy = sharding_strategy
        
        self.cpu_offload = cpu_offload
        self.mixed_precision_fsdp = mixed_precision_fsdp
        
        # Store raw model and loss function
        self.raw_model = model
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
        
        # Checkpoint management
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.temp_checkpoint_dir = None
        if temp_checkpoint_dir:
            self.temp_checkpoint_dir = Path(temp_checkpoint_dir)
            self.temp_checkpoint_dir.mkdir(parents=True, exist_ok=True)
            if is_master_rank():
                logger.info(f"🗂️ Temp checkpoint directory: {self.temp_checkpoint_dir}")
        
        self.keep_local_checkpoints = keep_local_checkpoints
        self.save_to_temp_every_n_steps = save_to_temp_every_n_steps
        self.local_checkpoints = []
        
        # Device setup
        self.device = torch.device(f'cuda:{rank}')
        
        # Distributed communication
        self.communicator = DistributedCommunicator(rank, world_size)
        self.metrics_aggregator = MetricsAggregator(rank, world_size)
        
        # WandB configuration (only rank 0)
        self.use_wandb = use_wandb and WANDB_AVAILABLE and is_master_rank()
        self.wandb_project = wandb_project
        self.wandb_run_name = wandb_run_name
        self.wandb_config = wandb_config or {}
        
        # Initialize tracking variables
        self.global_step = 0
        self.current_epoch = 0
        self.best_eval_similarity = 0.0
        self.best_loss = float('inf')
        
        # Enhanced monitoring (per rank)
        self.loss_history = deque(maxlen=1000)
        self.grad_norm_history = deque(maxlen=1000)
        self.lr_history = deque(maxlen=1000)
        
        # Stability tracking
        self.consecutive_failures = 0
        self.emergency_lr_reductions = 0
        self.stability_alerts = 0
        self.last_stable_step = 0
        
        # Setup distributed training
        self._setup_distributed_training()
        
        if is_master_rank():
            logger.info("✅ BLIP3-o Distributed Trainer initialized")
            logger.info(f"  World size: {self.world_size}")
            logger.info(f"  FSDP enabled: {self.use_fsdp}")
            logger.info(f"  Sharding strategy: {self.sharding_strategy}")
            logger.info(f"  Mixed precision: {'BF16' if self.mixed_precision_fsdp else 'FP32'}")
            logger.info(f"  CPU offload: {'Enabled' if self.cpu_offload else 'Disabled'}")

    def _setup_distributed_training(self):
        """Setup distributed training components"""
        
        # Wrap model with FSDP
        if self.use_fsdp:
            self.model = wrap_model_with_fsdp(
                self.raw_model,
                device=self.device,
                sharding_strategy=self.sharding_strategy,
                use_mixed_precision=self.mixed_precision_fsdp,
                cpu_offload=self.cpu_offload
            )
        else:
            self.model = self.raw_model.to(self.device)
        
        # Estimate steps per epoch
        self.estimated_steps_per_epoch = self._estimate_steps_per_epoch()
        
        # Setup optimizer and scheduler
        self._setup_optimizer_and_scheduler()
        
        # Setup mixed precision scaler
        if self.fp16 and not self.use_fsdp:  # FSDP handles mixed precision internally
            self.scaler = torch.amp.GradScaler('cuda')
        else:
            self.scaler = None
        
        # Setup WandB (rank 0 only)
        if self.use_wandb:
            self._setup_wandb()

    def _estimate_steps_per_epoch(self) -> int:
        """Estimate steps per epoch with distributed consideration"""
        try:
            length = len(self.train_dataloader)
            if is_master_rank():
                logger.info(f"Got exact dataloader length: {length}")
            return length
        except TypeError:
            try:
                dataset_length = len(self.train_dataloader.dataset)
                batch_size = getattr(self.train_dataloader, 'batch_size', 1)
                # Account for distributed sampling
                estimated_steps = max(1, dataset_length // (batch_size * self.world_size))
                if is_master_rank():
                    logger.info(f"Estimated steps per epoch (distributed): {estimated_steps}")
                return estimated_steps
            except (TypeError, AttributeError):
                if is_master_rank():
                    logger.warning("Could not estimate steps per epoch, using conservative default: 100")
                return 100

    def _setup_optimizer_and_scheduler(self):
        """Setup optimizer and scheduler for distributed training"""
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
        
        if is_master_rank():
            logger.info(f"✅ Distributed optimizer setup complete")

    def _setup_wandb(self):
        """Setup WandB configuration (rank 0 only)"""
        try:
            wandb_config = {
                'distributed_training': True,
                'world_size': self.world_size,
                'fsdp_enabled': self.use_fsdp,
                'sharding_strategy': str(self.sharding_strategy),
                'mixed_precision_fsdp': self.mixed_precision_fsdp,
                'cpu_offload': self.cpu_offload,
                'learning_rate': self.learning_rate,
                'weight_decay': self.weight_decay,
                'num_epochs': self.num_epochs,
                'warmup_steps': self.warmup_steps,
                'max_grad_norm': self.max_grad_norm,
                'experiment_type': 'blip3o_distributed_NO_NORMALIZATION',
                'normalization': 'DISABLED',
                **self.wandb_config,
            }
            
            self.wandb_run = wandb.init(
                project=self.wandb_project,
                name=self.wandb_run_name,
                config=wandb_config,
                dir=str(self.output_dir),
                resume="allow",
                tags=["blip3o", "fsdp", "distributed", "no_normalization"]
            )
            
            logger.info(f"✅ WandB initialized: {self.wandb_project}")
            
        except Exception as e:
            logger.error(f"❌ Failed to setup WandB: {e}")
            self.use_wandb = False

    def _compute_loss_with_stability_check(self, batch: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, float], bool]:
        """Compute loss with distributed stability checking"""
        try:
            # Move batch to device
            for key, value in batch.items():
                if torch.is_tensor(value):
                    batch[key] = value.to(self.device)
            
            # Forward pass
            if self.fp16 and not self.use_fsdp:
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
            
            # Compute loss
            if self.fp16 and not self.use_fsdp:
                with torch.amp.autocast('cuda'):
                    loss, metrics = self.loss_fn(
                        model_output=model_output,
                        target_samples=batch['clip_embeddings'],
                        timesteps=batch['timestep'],
                        eva_conditioning=batch['encoder_hidden_states'],
                        noise=batch.get('noise'),
                        noisy_input=batch['hidden_states'],
                        return_metrics=True
                    )
            else:
                loss, metrics = self.loss_fn(
                    model_output=model_output,
                    target_samples=batch['clip_embeddings'],
                    timesteps=batch['timestep'],
                    eva_conditioning=batch['encoder_hidden_states'],
                    noise=batch.get('noise'),
                    noisy_input=batch['hidden_states'],
                    return_metrics=True
                )
            
            # Check for loss explosion
            if loss.item() > self.loss_explosion_threshold:
                log_distributed_info(self.rank, self.world_size, 
                                   f"Loss explosion detected: {loss.item():.3f}")
                return loss, metrics or {}, False
            
            return loss, metrics or {}, True
            
        except Exception as e:
            log_distributed_info(self.rank, self.world_size, f"Error in loss computation: {e}")
            return torch.tensor(float('inf')), {}, False

    def _distributed_backward_and_step(self, loss: torch.Tensor) -> Tuple[float, bool]:
        """Distributed backward pass and optimizer step"""
        try:
            # Backward pass (FSDP handles gradient synchronization)
            if self.fp16 and not self.use_fsdp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Compute gradient norm (after FSDP synchronization)
            if self.use_fsdp:
                # For FSDP, we need to unscale gradients first if using scaler
                grad_norm = self.model.clip_grad_norm_(self.max_grad_norm)
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            
            # Optimizer step
            if self.fp16 and not self.use_fsdp:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            
            self.optimizer.zero_grad()
            self.scheduler.step()
            
            return grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm, True
            
        except Exception as e:
            log_distributed_info(self.rank, self.world_size, f"Error in backward pass: {e}")
            return float('inf'), False

    def _distributed_evaluate(self, num_samples: Optional[int] = None) -> Dict[str, float]:
        """Distributed evaluation across all ranks"""
        if self.eval_dataloader is None:
            return {}
        
        if num_samples is None:
            num_samples = self.eval_num_samples
        
        # Divide samples across ranks
        samples_per_rank = max(1, num_samples // self.world_size)
        if self.rank == self.world_size - 1:
            # Last rank gets any remaining samples
            samples_per_rank += num_samples % self.world_size
        
        log_distributed_info(self.rank, self.world_size, 
                           f"Starting evaluation with {samples_per_rank} samples per rank")
        
        self.model.eval()
        
        try:
            local_generated = []
            local_targets = []
            local_samples_processed = 0
            local_errors = 0
            
            with torch.no_grad():
                for batch_idx, batch in enumerate(self.eval_dataloader):
                    if local_samples_processed >= samples_per_rank:
                        break
                    
                    try:
                        eva_features = batch['encoder_hidden_states'].to(self.device)
                        target_clip = batch['clip_embeddings'].to(self.device)
                        
                        # Generate with Heun solver
                        generated_clip = self._safe_distributed_generate(
                            eva_features=eva_features,
                            num_steps=self.eval_inference_steps,
                        )
                        
                        if generated_clip is not None:
                            local_generated.append(generated_clip.cpu())
                            local_targets.append(target_clip.cpu())
                            local_samples_processed += eva_features.shape[0]
                        else:
                            local_errors += 1
                            
                    except Exception as e:
                        local_errors += 1
                        logger.debug(f"Rank {self.rank} evaluation error: {e}")
                        continue
            
            # Compute local metrics
            if local_generated:
                local_generated = torch.cat(local_generated, dim=0)
                local_targets = torch.cat(local_targets, dim=0)
                local_metrics = self._compute_evaluation_metrics(local_generated, local_targets)
                local_metrics.update({
                    'local_samples': local_samples_processed,
                    'local_errors': local_errors,
                })
            else:
                local_metrics = {
                    'local_samples': 0,
                    'local_errors': local_errors,
                    'eval_clip_similarity': 0.0,
                }
            
            # Aggregate metrics across all ranks
            if is_master_rank():
                aggregated_metrics = self.metrics_aggregator.aggregate_metrics(local_metrics)
                
                if aggregated_metrics:
                    # Compute weighted averages
                    total_samples = aggregated_metrics.get('local_samples', 1)
                    if total_samples > 0:
                        # Re-weight similarity by number of samples
                        weighted_similarity = (aggregated_metrics.get('eval_clip_similarity', 0) * 
                                             local_metrics.get('local_samples', 0)) / total_samples
                        aggregated_metrics['eval_clip_similarity'] = weighted_similarity
                    
                    aggregated_metrics['eval_samples'] = total_samples
                    aggregated_metrics['eval_errors'] = aggregated_metrics.get('local_errors', 0)
                    
                    log_distributed_info(self.rank, self.world_size,
                                       f"Distributed evaluation completed: {total_samples} total samples")
                    
                    return aggregated_metrics
                else:
                    return {'eval_error': 'aggregation_failed', 'eval_clip_similarity': 0.0}
            else:
                # Non-master ranks just participate in aggregation
                self.metrics_aggregator.aggregate_metrics(local_metrics)
                return {}
            
        except Exception as e:
            log_distributed_info(self.rank, self.world_size, f"Distributed evaluation failed: {e}")
            return {'eval_error': str(e), 'eval_clip_similarity': 0.0}
        
        finally:
            self.model.train()

    def _safe_distributed_generate(self, eva_features: torch.Tensor, num_steps: int = 50) -> Optional[torch.Tensor]:
        """Generate using model with distributed error handling"""
        try:
            if hasattr(self.model, 'generate'):
                return self.model.generate(
                    eva_features=eva_features,
                    num_inference_steps=num_steps,
                    use_heun=self.use_heun_inference
                )
            else:
                # Fallback to manual generation
                batch_size, seq_len, _ = eva_features.shape
                
                x = torch.randn(
                    batch_size, seq_len, 1024,
                    device=self.device, dtype=eva_features.dtype
                )
                
                timesteps = torch.linspace(1.0, 0.0, num_steps + 1, device=self.device)[:-1]
                
                for i, t in enumerate(timesteps):
                    t_batch = torch.full((batch_size,), t.item(), device=self.device, dtype=eva_features.dtype)
                    
                    if i < len(timesteps) - 1:
                        dt = timesteps[i] - timesteps[i + 1]
                    else:
                        dt = timesteps[i]
                    dt = dt.item()
                    
                    # Simple Euler step
                    velocity = self.model(
                        hidden_states=x,
                        timestep=t_batch,
                        encoder_hidden_states=eva_features,
                        return_dict=False
                    )
                    if isinstance(velocity, dict):
                        velocity = velocity.get('velocity_prediction', list(velocity.values())[0])
                    
                    x = x + dt * velocity
                    x = torch.clamp(x, min=-10.0, max=10.0)
                
                return x
                
        except Exception as e:
            log_distributed_info(self.rank, self.world_size, f"Generation failed: {e}")
            return None

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
            
            # MSE loss
            mse_loss = F.mse_loss(generated, target).item()
            
            return {
                'eval_clip_similarity': similarity.mean().item(),
                'eval_mse_loss': mse_loss,
                'eval_high_quality': high_quality,
                'eval_very_high_quality': very_high_quality,
                'eval_similarity_std': similarity.std().item(),
            }
            
        except Exception as e:
            logger.warning(f"Error computing metrics on rank {self.rank}: {e}")
            return {
                'eval_clip_similarity': 0.0,
                'eval_mse_loss': float('inf'),
            }

    def _save_distributed_checkpoint(self, is_best: bool = False, force_temp: bool = False) -> bool:
        """Save distributed checkpoint using FSDP utilities"""
        if not is_master_rank():
            return True  # Only rank 0 saves checkpoints
        
        try:
            # Determine checkpoint filename
            if is_best:
                checkpoint_filename = f"best_fsdp_checkpoint_step_{self.global_step}.pt"
            else:
                checkpoint_filename = f"fsdp_checkpoint_step_{self.global_step}.pt"
            
            checkpoint_path = self.output_dir / checkpoint_filename
            
            # Additional data for distributed checkpoint
            additional_data = {
                'global_step': self.global_step,
                'current_epoch': self.current_epoch,
                'best_eval_similarity': self.best_eval_similarity,
                'best_loss': self.best_loss,
                'world_size': self.world_size,
                'fsdp_config': {
                    'sharding_strategy': str(self.sharding_strategy),
                    'cpu_offload': self.cpu_offload,
                    'mixed_precision': self.mixed_precision_fsdp
                },
                'normalization': 'DISABLED',
                'distributed_training': True,
            }
            
            # Save using FSDP utilities
            save_fsdp_checkpoint(
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                scaler=self.scaler,
                checkpoint_path=checkpoint_path,
                global_step=self.global_step,
                additional_data=additional_data,
                save_full_state=True  # For inference compatibility
            )
            
            # Track local checkpoints
            self.local_checkpoints.append(checkpoint_path)
            
            # Save to temp directory if conditions are met
            should_save_to_temp = (
                self.temp_checkpoint_dir and (
                    is_best or 
                    force_temp or 
                    (self.global_step % self.save_to_temp_every_n_steps == 0)
                )
            )
            
            if should_save_to_temp:
                temp_checkpoint_path = self.temp_checkpoint_dir / checkpoint_filename
                import shutil
                shutil.copy2(checkpoint_path, temp_checkpoint_path)
                logger.info(f"📦 Checkpoint copied to temp: {temp_checkpoint_path}")
            
            # Clean up old local checkpoints
            if not is_best and len(self.local_checkpoints) > self.keep_local_checkpoints:
                old_checkpoints = self.local_checkpoints[:-self.keep_local_checkpoints]
                for old_checkpoint in old_checkpoints:
                    try:
                        if old_checkpoint.exists():
                            old_checkpoint.unlink()
                    except Exception as e:
                        logger.warning(f"Could not remove old checkpoint: {e}")
                self.local_checkpoints = self.local_checkpoints[-self.keep_local_checkpoints:]
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Distributed checkpoint save failed: {e}")
            return False

    def train(self) -> Dict[str, Any]:
        """Main distributed training loop"""
        if is_master_rank():
            logger.info("🚀 Starting BLIP3-o distributed training with FSDP...")
            logger.info(f"  World size: {self.world_size}")
            logger.info(f"  FSDP sharding: {self.sharding_strategy}")
            logger.info(f"  Mixed precision: {'BF16' if self.mixed_precision_fsdp else 'FP32'}")
        
        # Synchronize all ranks before starting
        self.communicator.barrier()
        
        self.model.train()
        start_time = time.time()
        
        try:
            for epoch in range(self.num_epochs):
                self.current_epoch = epoch
                
                # Set epoch for distributed sampler
                if hasattr(self.train_dataloader.sampler, 'set_epoch'):
                    self.train_dataloader.sampler.set_epoch(epoch)
                
                if is_master_rank():
                    logger.info(f"Starting distributed epoch {epoch + 1}/{self.num_epochs}")
                
                epoch_loss = 0.0
                epoch_steps = 0
                epoch_failures = 0
                
                for batch_idx, batch in enumerate(self.train_dataloader):
                    step_start_time = time.time()
                    
                    # Compute loss with stability checks
                    loss, metrics, is_stable = self._compute_loss_with_stability_check(batch)
                    
                    if not is_stable:
                        epoch_failures += 1
                        continue
                    
                    # Distributed backward pass
                    grad_norm, step_success = self._distributed_backward_and_step(loss)
                    
                    if not step_success:
                        epoch_failures += 1
                        continue
                    
                    # Update tracking
                    epoch_loss += loss.item()
                    epoch_steps += 1
                    self.global_step += 1
                    
                    # Sync loss across ranks for logging
                    synced_loss = sync_across_gpus(loss.detach())
                    
                    # Update best loss
                    if synced_loss.item() < self.best_loss:
                        self.best_loss = synced_loss.item()
                    
                    # Log to WandB (rank 0 only)
                    if self.use_wandb and self.global_step % self.log_every_n_steps == 0:
                        wandb.log({
                            "train/loss": synced_loss.item(),
                            "train/grad_norm": grad_norm,
                            "train/learning_rate": self.optimizer.param_groups[0]['lr'],
                            "train/epoch": self.current_epoch,
                            "distributed/world_size": self.world_size,
                            "distributed/rank": self.rank,
                        }, step=self.global_step)
                    
                    # Console logging (rank 0 only)
                    if is_master_rank() and self.global_step % self.log_every_n_steps == 0:
                        logger.info(f"Step {self.global_step}: Loss={synced_loss.item():.6f}, "
                                  f"GradNorm={grad_norm:.3f}, Failures={epoch_failures}")
                    
                    # Distributed evaluation
                    if self.global_step % self.eval_every_n_steps == 0:
                        if is_master_rank():
                            logger.info(f"Running distributed evaluation at step {self.global_step}...")
                        
                        eval_metrics = self._distributed_evaluate()
                        
                        if is_master_rank() and eval_metrics and 'eval_clip_similarity' in eval_metrics:
                            similarity = eval_metrics['eval_clip_similarity']
                            logger.info(f"✅ Distributed evaluation: CLIP similarity = {similarity:.4f}")
                            
                            if self.use_wandb:
                                wandb_eval = {f"eval/{k.replace('eval_', '')}": v for k, v in eval_metrics.items() 
                                            if isinstance(v, (int, float)) and not math.isnan(v)}
                                wandb.log(wandb_eval, step=self.global_step)
                            
                            # Check if this is the best model
                            is_best = similarity > self.best_eval_similarity
                            if is_best:
                                self.best_eval_similarity = similarity
                                logger.info(f"🎉 NEW BEST distributed similarity: {similarity:.4f}")
                                # Save best checkpoint
                                self._save_distributed_checkpoint(is_best=True)
                    
                    # Regular checkpoint saving
                    if self.global_step % self.save_every_n_steps == 0:
                        self._save_distributed_checkpoint(is_best=False)
                
                # End of epoch sync
                self.communicator.barrier()
                
                if is_master_rank():
                    avg_loss = epoch_loss / max(epoch_steps, 1)
                    logger.info(f"Epoch {epoch + 1} completed: avg_loss={avg_loss:.6f}, failures={epoch_failures}")
            
            # Final evaluation and checkpoint
            if is_master_rank():
                logger.info("Running final distributed evaluation...")
            
            final_eval = self._distributed_evaluate(num_samples=self.eval_num_samples * 2)
            self._save_distributed_checkpoint(force_temp=True)
            
            total_time = time.time() - start_time
            
            # Training summary
            summary = {
                'training_completed': True,
                'total_time_seconds': total_time,
                'total_steps': self.global_step,
                'best_eval_similarity': self.best_eval_similarity,
                'best_loss': self.best_loss,
                'world_size': self.world_size,
                'fsdp_enabled': self.use_fsdp,
                'final_eval': final_eval,
                'distributed': True,
            }
            
            if self.use_wandb:
                wandb.log({
                    "final/training_completed": True,
                    "final/total_time_seconds": total_time,
                    "final/best_eval_similarity": self.best_eval_similarity,
                    "final/distributed_training": True,
                    "final/world_size": self.world_size,
                }, step=self.global_step)
                wandb.finish()
            
            if is_master_rank():
                logger.info("🎉 Distributed training completed!")
                logger.info(f"  Total time: {total_time:.1f} seconds")
                logger.info(f"  Best similarity: {self.best_eval_similarity:.4f}")
                logger.info(f"  FSDP training successful across {self.world_size} GPUs")
            
            return summary
            
        except Exception as e:
            if is_master_rank():
                logger.error(f"❌ Distributed training failed: {e}")
            if self.use_wandb:
                wandb.finish()
            raise


def create_distributed_clip_trainer(
    model,
    loss_fn,
    train_dataloader,
    eval_dataloader=None,
    world_size: int = 4,
    rank: int = 0,
    use_fsdp: bool = True,
    sharding_strategy: str = "FULL_SHARD",
    cpu_offload: bool = False,
    mixed_precision_fsdp: bool = True,
    output_dir: str = "./checkpoints",
    temp_checkpoint_dir: Optional[str] = None,
    use_wandb: bool = False,
    wandb_project: str = "blip3o-clip-fsdp",
    **kwargs
) -> BLIP3oDistributedTrainer:
    """Factory function to create distributed CLIP trainer with FSDP"""
    
    return BLIP3oDistributedTrainer(
        model=model,
        loss_fn=loss_fn,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        world_size=world_size,
        rank=rank,
        use_fsdp=use_fsdp,
        sharding_strategy=sharding_strategy,
        cpu_offload=cpu_offload,
        mixed_precision_fsdp=mixed_precision_fsdp,
        output_dir=output_dir,
        temp_checkpoint_dir=temp_checkpoint_dir,
        use_wandb=use_wandb,
        wandb_project=wandb_project,
        **kwargs
    )