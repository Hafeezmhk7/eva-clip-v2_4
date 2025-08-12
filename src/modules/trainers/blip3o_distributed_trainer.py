"""
STANDALONE BLIP3-o Distributed Trainer - NO CIRCULAR IMPORTS
src/modules/trainers/blip3o_distributed_trainer.py

MAJOR FIXES:
- Completely standalone - no imports from distributed module
- All FSDP utilities included directly
- Avoids all circular import issues
- Simplified and robust
"""

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    BackwardPrefetch,
    ShardingStrategy,
    CPUOffload,
    FullStateDictConfig,
    StateDictType
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from typing import Dict, Any, Optional, Tuple
import logging
import time
from pathlib import Path
import json
import os
import warnings
from functools import partial
from datetime import timedelta

logger = logging.getLogger(__name__)


# =============================================================================
# STANDALONE FSDP UTILITIES (NO EXTERNAL IMPORTS)
# =============================================================================

def setup_environment_variables_standalone():
    """Setup environment variables to avoid warnings"""
    if 'TRANSFORMERS_CACHE' in os.environ and 'HF_HOME' not in os.environ:
        os.environ['HF_HOME'] = os.environ['TRANSFORMERS_CACHE']
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.environ['WANDB_SILENT'] = 'true'
    os.environ['PYTHONWARNINGS'] = 'ignore::UserWarning'
    os.environ['NCCL_DEBUG'] = 'WARN'
    os.environ['NCCL_TIMEOUT'] = '1800'
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'


def get_fsdp_sharding_policy_standalone():
    """Get FSDP sharding policy"""
    try:
        from src.modules.models.blip3o_dit import StableDiTBlock3D
        dit_block_class = StableDiTBlock3D
    except ImportError:
        dit_block_class = torch.nn.TransformerEncoderLayer
        logger.warning("Using fallback transformer layer for FSDP policy")
    
    auto_wrap_policy = partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={dit_block_class},
    )
    return auto_wrap_policy


def create_fsdp_mixed_precision_policy_standalone(use_mixed_precision: bool = True) -> Optional[MixedPrecision]:
    """Create mixed precision policy for FSDP"""
    if not use_mixed_precision:
        return None
    
    return MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
        keep_low_precision_grads=True,
    )


def wrap_model_with_fsdp_standalone(
    model: torch.nn.Module,
    device: torch.device,
    sharding_strategy: ShardingStrategy = ShardingStrategy.FULL_SHARD,
    use_mixed_precision: bool = True,
    cpu_offload: bool = False,
) -> FSDP:
    """Wrap model with FSDP - standalone version"""
    
    # Move model to device first
    logger.info(f"Moving model to device {device} before FSDP wrapping...")
    model = model.to(device)
    
    # Ensure all parameters are on correct device
    for param in model.parameters():
        if param.device != device:
            param.data = param.data.to(device)
            if param.grad is not None:
                param.grad = param.grad.to(device)
    
    # Ensure all buffers are on correct device
    for buffer in model.buffers():
        if buffer.device != device:
            buffer.data = buffer.data.to(device)
    
    logger.info(f"✅ Model moved to {device}")
    
    # Get policies
    auto_wrap_policy = get_fsdp_sharding_policy_standalone()
    mixed_precision_policy = create_fsdp_mixed_precision_policy_standalone(use_mixed_precision)
    cpu_offload_policy = CPUOffload(offload_params=True) if cpu_offload else None
    
    # Wrap model with complete warning suppression
    logger.info("Wrapping model with FSDP...")
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        fsdp_model = FSDP(
            model,
            auto_wrap_policy=auto_wrap_policy,
            mixed_precision=mixed_precision_policy,
            sharding_strategy=sharding_strategy,
            cpu_offload=cpu_offload_policy,
            backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
            limit_all_gathers=True,
            use_orig_params=False,
            sync_module_states=True,
        )
    
    if dist.get_rank() == 0:
        logger.info("✅ Model wrapped with FSDP successfully")
        logger.info(f"   Sharding strategy: {sharding_strategy}")
        logger.info(f"   Mixed precision: {'BF16' if use_mixed_precision else 'FP32'}")
        logger.info(f"   CPU offload: {'Enabled' if cpu_offload else 'Disabled'}")
    
    return fsdp_model


def save_fsdp_checkpoint_standalone(
    model: FSDP,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    scaler: Optional[torch.amp.GradScaler],
    checkpoint_path: Path,
    global_step: int,
    additional_data: Optional[Dict[str, Any]] = None,
):
    """Save FSDP checkpoint - standalone version"""
    
    if dist.get_rank() != 0:
        return
    
    logger.info(f"Saving checkpoint: {checkpoint_path}")
    
    checkpoint_data = {
        'global_step': global_step,
        'fsdp_model': True,
        'version': 'STANDALONE_v1',
    }
    
    if additional_data:
        checkpoint_data.update(additional_data)
    
    # Save model state dict
    with FSDP.state_dict_type(
        model,
        StateDictType.FULL_STATE_DICT,
        FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    ):
        checkpoint_data['model_state_dict'] = model.state_dict()
    
    # Save other states
    if hasattr(optimizer, 'state_dict'):
        checkpoint_data['optimizer_state_dict'] = optimizer.state_dict()
    if scheduler is not None:
        checkpoint_data['scheduler_state_dict'] = scheduler.state_dict()
    if scaler is not None:
        checkpoint_data['scaler_state_dict'] = scaler.state_dict()
    
    # Save checkpoint
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint_data, checkpoint_path)
    
    file_size_mb = checkpoint_path.stat().st_size / (1024 * 1024)
    logger.info(f"✅ Checkpoint saved: {checkpoint_path.name} ({file_size_mb:.1f} MB)")


def is_master_rank_standalone() -> bool:
    """Check if current rank is master"""
    return not dist.is_initialized() or dist.get_rank() == 0


def get_world_size_standalone() -> int:
    """Get world size"""
    return dist.get_world_size() if dist.is_initialized() else 1


def get_rank_standalone() -> int:
    """Get current rank"""
    return dist.get_rank() if dist.is_initialized() else 0


# =============================================================================
# STANDALONE DISTRIBUTED TRAINER
# =============================================================================

class BLIP3oDistributedTrainer:
    """
    STANDALONE BLIP3-o Distributed Trainer - NO CIRCULAR IMPORTS
    
    All dependencies included directly to avoid import issues
    """
    
    def __init__(
        self,
        model,
        loss_fn,
        train_dataloader,
        eval_dataloader=None,
        
        # Distributed configuration
        world_size: int = 2,
        rank: int = 0,
        use_fsdp: bool = True,
        sharding_strategy: str = "FULL_SHARD",
        cpu_offload: bool = False,
        mixed_precision_fsdp: bool = True,
        
        # Training configuration
        learning_rate: float = 4e-5,
        weight_decay: float = 0.04,
        num_epochs: int = 1,
        warmup_steps: int = 100,
        max_grad_norm: float = 1.0,
        fp16: bool = True,
        
        # Evaluation
        eval_every_n_steps: int = 1000,
        eval_num_samples: int = 10,
        eval_inference_steps: int = 20,
        use_heun_inference: bool = True,
        
        # Logging and checkpointing
        log_every_n_steps: int = 10,
        save_every_n_steps: int = 500,
        output_dir: str = "./checkpoints",
        
        # WandB (disabled)
        use_wandb: bool = False,
        wandb_project: str = "blip3o-clip-fsdp-fixed",
        wandb_run_name: Optional[str] = None,
        wandb_config: Optional[Dict] = None,
        
        # Testing and stability
        progress_tracking: bool = True,
        max_batches_per_epoch: Optional[int] = None,
        batch_timeout_seconds: int = 60,
        enable_recovery_mode: bool = True,
        
        **kwargs
    ):
        # Setup environment first
        setup_environment_variables_standalone()
        
        self.world_size = world_size
        self.rank = rank
        self.use_fsdp = use_fsdp
        
        # Convert sharding strategy
        if isinstance(sharding_strategy, str):
            self.sharding_strategy = getattr(ShardingStrategy, sharding_strategy)
        else:
            self.sharding_strategy = sharding_strategy
        
        self.cpu_offload = cpu_offload
        self.mixed_precision_fsdp = mixed_precision_fsdp
        
        # Store components
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
        
        # Logging config
        self.log_every_n_steps = log_every_n_steps
        self.save_every_n_steps = save_every_n_steps
        
        # Testing config
        self.progress_tracking = progress_tracking
        self.max_batches_per_epoch = max_batches_per_epoch
        self.batch_timeout_seconds = batch_timeout_seconds
        self.enable_recovery_mode = enable_recovery_mode
        
        # Output directory
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Device setup
        self.device = torch.device(f'cuda:{rank}')
        torch.cuda.set_device(rank)
        
        # WandB configuration
        self.use_wandb = use_wandb and is_master_rank_standalone()
        self.wandb_project = wandb_project
        self.wandb_run_name = wandb_run_name
        self.wandb_config = wandb_config or {}
        
        # Training state
        self.global_step = 0
        self.current_epoch = 0
        self.best_loss = float('inf')
        
        # Error tracking
        self.batch_failures = 0
        self.consecutive_failures = 0
        
        # Setup distributed training
        self._setup_distributed_training()
        
        if is_master_rank_standalone():
            logger.info("✅ STANDALONE Distributed Trainer initialized")
            logger.info(f"  World size: {self.world_size}")
            logger.info(f"  FSDP enabled: {self.use_fsdp}")
            logger.info(f"  Sharding strategy: {self.sharding_strategy}")
            logger.info(f"  Mixed precision: {'BF16' if self.mixed_precision_fsdp else 'FP32'}")
            logger.info(f"  Max batches/epoch: {self.max_batches_per_epoch or 'Unlimited'}")

    def _setup_distributed_training(self):
        """Setup distributed training components"""
        
        # Wrap model with FSDP
        if self.use_fsdp:
            if is_master_rank_standalone():
                logger.info(f"Wrapping model with FSDP on device {self.device}")
            
            self.model = wrap_model_with_fsdp_standalone(
                self.raw_model,
                device=self.device,
                sharding_strategy=self.sharding_strategy,
                use_mixed_precision=self.mixed_precision_fsdp,
                cpu_offload=self.cpu_offload
            )
        else:
            self.model = self.raw_model.to(self.device)
        
        # Setup optimizer and scheduler
        self._setup_optimizer_and_scheduler()
        
        # Setup mixed precision scaler
        if self.fp16 and not (self.use_fsdp and self.mixed_precision_fsdp):
            self.scaler = torch.amp.GradScaler('cuda')
        else:
            self.scaler = None
        
        # Setup WandB if enabled
        if self.use_wandb:
            self._setup_wandb()

    def _setup_optimizer_and_scheduler(self):
        """Setup optimizer and scheduler"""
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Estimate total steps
        try:
            if hasattr(self.train_dataloader, '__len__'):
                steps_per_epoch = len(self.train_dataloader)
            else:
                steps_per_epoch = self.max_batches_per_epoch or 100
        except:
            steps_per_epoch = self.max_batches_per_epoch or 100
        
        total_steps = steps_per_epoch * self.num_epochs
        
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

    def _setup_wandb(self):
        """Setup WandB (if enabled)"""
        try:
            import wandb
            
            wandb_config = {
                'world_size': self.world_size,
                'fsdp_enabled': self.use_fsdp,
                'sharding_strategy': str(self.sharding_strategy),
                'learning_rate': self.learning_rate,
                'num_epochs': self.num_epochs,
                'max_batches_per_epoch': self.max_batches_per_epoch,
                'experiment_type': 'blip3o_distributed_STANDALONE',
                **self.wandb_config,
            }
            
            self.wandb_run = wandb.init(
                project=self.wandb_project,
                name=self.wandb_run_name,
                config=wandb_config,
                dir=str(self.output_dir),
                tags=["blip3o", "fsdp", "distributed", "standalone"]
            )
            
            logger.info(f"✅ WandB initialized")
            
        except Exception as e:
            logger.warning(f"⚠️ WandB setup failed: {e}")
            self.use_wandb = False

    def _process_batch(self, batch: Dict[str, Any]) -> Tuple[Optional[torch.Tensor], Optional[Dict[str, float]], bool]:
        """Process a single batch"""
        
        try:
            # Move batch to device
            for key, value in batch.items():
                if torch.is_tensor(value):
                    batch[key] = value.to(self.device, non_blocking=True)
            
            # Forward pass
            use_autocast = self.fp16 and not (self.use_fsdp and self.mixed_precision_fsdp)
            
            if use_autocast:
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
            if use_autocast:
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
            
            # Validate loss
            if torch.isnan(loss) or torch.isinf(loss) or loss.item() > 100.0:
                logger.warning(f"[Rank {self.rank}] Invalid loss: {loss.item()}")
                return None, metrics or {}, False
            
            return loss, metrics or {}, True
            
        except Exception as e:
            logger.warning(f"[Rank {self.rank}] Error in batch processing: {e}")
            return None, {}, False

    def _backward_step(self, loss: torch.Tensor) -> Tuple[float, bool]:
        """Backward pass with gradient clipping"""
        
        try:
            # Backward pass
            use_scaler = self.scaler is not None
            
            if use_scaler:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Gradient clipping
            if self.use_fsdp:
                grad_norm = self.model.clip_grad_norm_(self.max_grad_norm)
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            
            # Optimizer step
            if use_scaler:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            
            self.optimizer.zero_grad()
            self.scheduler.step()
            
            return grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm, True
            
        except Exception as e:
            logger.warning(f"[Rank {self.rank}] Error in backward pass: {e}")
            return float('inf'), False

    def _save_checkpoint(self, is_best: bool = False) -> bool:
        """Save checkpoint safely"""
        if not is_master_rank_standalone():
            return True
        
        try:
            checkpoint_filename = f"{'best_' if is_best else ''}checkpoint_step_{self.global_step}.pt"
            checkpoint_path = self.output_dir / checkpoint_filename
            
            additional_data = {
                'global_step': self.global_step,
                'current_epoch': self.current_epoch,
                'best_loss': self.best_loss,
                'world_size': self.world_size,
                'batch_failures': self.batch_failures,
                'version': 'STANDALONE_v1',
            }
            
            if self.use_fsdp:
                save_fsdp_checkpoint_standalone(
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    scaler=self.scaler,
                    checkpoint_path=checkpoint_path,
                    global_step=self.global_step,
                    additional_data=additional_data,
                )
            else:
                # Regular checkpoint
                checkpoint = {
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    **additional_data
                }
                if self.scaler:
                    checkpoint['scaler_state_dict'] = self.scaler.state_dict()
                torch.save(checkpoint, checkpoint_path)
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Checkpoint save failed: {e}")
            return False

    def train(self) -> Dict[str, Any]:
        """Main training loop - STANDALONE VERSION"""
        
        if is_master_rank_standalone():
            logger.info("🚀 Starting STANDALONE distributed training...")
            logger.info(f"  World size: {self.world_size}")
            logger.info(f"  FSDP sharding: {self.sharding_strategy}")
            logger.info(f"  Max batches/epoch: {self.max_batches_per_epoch or 'Unlimited'}")
        
        self.model.train()
        start_time = time.time()
        
        try:
            for epoch in range(self.num_epochs):
                self.current_epoch = epoch
                
                if is_master_rank_standalone():
                    logger.info(f"Starting epoch {epoch + 1}/{self.num_epochs}")
                
                epoch_loss = 0.0
                epoch_steps = 0
                epoch_failures = 0
                
                # Create dataloader iterator
                try:
                    dataloader_iter = iter(self.train_dataloader)
                    batch_count = 0
                    successful_batches = 0
                    
                    while True:
                        # Check batch limit
                        if self.max_batches_per_epoch and batch_count >= self.max_batches_per_epoch:
                            if is_master_rank_standalone():
                                logger.info(f"Reached batch limit: {self.max_batches_per_epoch}")
                            break
                        
                        # Get next batch
                        try:
                            batch = next(dataloader_iter)
                            batch_count += 1
                        except StopIteration:
                            if is_master_rank_standalone():
                                logger.info(f"Epoch {epoch + 1} completed: {batch_count} batches")
                            break
                        except Exception as e:
                            logger.error(f"[Rank {self.rank}] Error getting batch: {e}")
                            epoch_failures += 1
                            if epoch_failures > 10:
                                logger.error(f"[Rank {self.rank}] Too many failures, stopping")
                                break
                            continue
                        
                        # Process batch
                        loss, metrics, success = self._process_batch(batch)
                        
                        if not success:
                            self.batch_failures += 1
                            self.consecutive_failures += 1
                            epoch_failures += 1
                            continue
                        
                        # Backward pass
                        grad_norm, backward_success = self._backward_step(loss)
                        
                        if not backward_success:
                            self.batch_failures += 1
                            self.consecutive_failures += 1
                            continue
                        
                        # Success - update tracking
                        self.consecutive_failures = 0
                        successful_batches += 1
                        epoch_loss += loss.item()
                        epoch_steps += 1
                        self.global_step += 1
                        
                        # Update best loss
                        if loss.item() < self.best_loss:
                            self.best_loss = loss.item()
                        
                        # Log to WandB
                        if self.use_wandb and self.global_step % self.log_every_n_steps == 0:
                            try:
                                import wandb
                                wandb.log({
                                    "train/loss": loss.item(),
                                    "train/grad_norm": grad_norm,
                                    "train/learning_rate": self.optimizer.param_groups[0]['lr'],
                                    "train/epoch": self.current_epoch,
                                    "distributed/world_size": self.world_size,
                                }, step=self.global_step)
                            except:
                                pass
                        
                        # Console logging
                        if is_master_rank_standalone() and self.global_step % self.log_every_n_steps == 0:
                            logger.info(f"Step {self.global_step}: Loss={loss.item():.6f}, "
                                      f"GradNorm={grad_norm:.3f}, Success={successful_batches}/{batch_count}")
                        
                        # Save checkpoints
                        if self.global_step % self.save_every_n_steps == 0:
                            self._save_checkpoint(is_best=False)
                
                except Exception as e:
                    logger.error(f"❌ Error during epoch {epoch + 1}: {e}")
                    continue
                
                # End of epoch summary
                if is_master_rank_standalone():
                    success_rate = successful_batches / max(batch_count, 1) * 100
                    avg_loss = epoch_loss / max(successful_batches, 1)
                    logger.info(f"Epoch {epoch + 1} summary:")
                    logger.info(f"  Processed: {batch_count} batches, {successful_batches} successful")
                    logger.info(f"  Success rate: {success_rate:.1f}%")
                    logger.info(f"  Average loss: {avg_loss:.6f}")
                    logger.info(f"  Best loss: {self.best_loss:.6f}")
            
            # Final checkpoint
            self._save_checkpoint(is_best=False)
            
            total_time = time.time() - start_time
            
            # Training summary
            summary = {
                'training_completed': True,
                'total_time_seconds': total_time,
                'total_steps': self.global_step,
                'best_loss': self.best_loss,
                'world_size': self.world_size,
                'batch_failures': self.batch_failures,
                'version': 'STANDALONE_v1',
            }
            
            if self.use_wandb:
                try:
                    import wandb
                    wandb.log({
                        "final/training_completed": True,
                        "final/total_time_seconds": total_time,
                        "final/best_loss": self.best_loss,
                        "final/total_steps": self.global_step,
                        "final/world_size": self.world_size,
                    }, step=self.global_step)
                    wandb.finish()
                except:
                    pass
            
            if is_master_rank_standalone():
                logger.info("🎉 STANDALONE distributed training completed!")
                logger.info(f"  Total time: {total_time:.1f} seconds")
                logger.info(f"  Total steps: {self.global_step}")
                logger.info(f"  Best loss: {self.best_loss:.6f}")
                logger.info(f"  ✅ NO CIRCULAR IMPORT ISSUES!")
            
            return summary
            
        except Exception as e:
            if is_master_rank_standalone():
                logger.error(f"❌ Training failed: {e}")
            raise
        
        finally:
            # Cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


# Factory function for creating trainer
def create_distributed_clip_trainer(
    model,
    loss_fn,
    train_dataloader,
    eval_dataloader=None,
    world_size: int = 2,
    rank: int = 0,
    use_fsdp: bool = True,
    sharding_strategy: str = "FULL_SHARD",
    cpu_offload: bool = False,
    mixed_precision_fsdp: bool = True,
    output_dir: str = "./checkpoints",
    use_wandb: bool = False,
    wandb_project: str = "blip3o-clip-fsdp-standalone",
    progress_tracking: bool = True,
    max_batches_per_epoch: Optional[int] = None,
    **kwargs
) -> BLIP3oDistributedTrainer:
    """Factory function to create STANDALONE distributed trainer"""
    
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
        use_wandb=use_wandb,
        wandb_project=wandb_project,
        progress_tracking=progress_tracking,
        max_batches_per_epoch=max_batches_per_epoch,
        **kwargs
    )


# Aliases for backward compatibility
FixedBLIP3oDistributedTrainer = BLIP3oDistributedTrainer
create_fixed_distributed_clip_trainer = create_distributed_clip_trainer