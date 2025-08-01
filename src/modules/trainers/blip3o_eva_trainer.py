#!/usr/bin/env python3
"""
Fixed BLIP3-o Trainer for Spherical EVA-CLIP Denoising
Key fixes:
1. Proper spherical flow matching training loop
2. EVA denoising evaluation metrics
3. Better gradient flow monitoring
4. Comprehensive debugging and validation
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

logger = logging.getLogger(__name__)


class SphericalEVATrainer:
    """
    Fixed trainer for spherical EVA denoising with comprehensive monitoring
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
        max_grad_norm: float = 1.0,
        fp16: bool = False,
        # Evaluation
        eval_every_n_steps: int = 100,
        eval_num_samples: int = 500,
        eval_inference_steps: int = 50,
        # Debugging
        debug_mode: bool = False,
        overfit_test_size: Optional[int] = None,
        log_every_n_steps: int = 10,
        save_every_n_steps: int = 500,
        # Output
        output_dir: str = "./checkpoints",
        # Device
        device: Optional[torch.device] = None,
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
        
        # Debugging config
        self.debug_mode = debug_mode
        self.overfit_test_size = overfit_test_size
        self.log_every_n_steps = log_every_n_steps
        self.save_every_n_steps = save_every_n_steps
        
        # Output
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Device
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.model = self.model.to(self.device)
        
        # Setup optimizer and scheduler
        self._setup_optimizer_and_scheduler()
        
        # Setup mixed precision
        if self.fp16:
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None
        
        # Tracking variables
        self.global_step = 0
        self.current_epoch = 0
        self.best_eval_similarity = 0.0
        self.best_loss = float('inf')
        
        # Metrics tracking
        self.loss_history = deque(maxlen=1000)
        self.similarity_history = deque(maxlen=1000)
        self.eval_similarity_history = deque(maxlen=1000)
        self.lr_history = deque(maxlen=1000)
        self.grad_norm_history = deque(maxlen=1000)
        self.sphere_violation_history = deque(maxlen=1000)
        
        # Overfitting test data
        self.overfit_batch = None
        if self.overfit_test_size:
            self._prepare_overfit_test()
        
        logger.info("Spherical EVA Denoising Trainer initialized")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  Task: EVA-CLIP Spherical Denoising")
        logger.info(f"  Input: Noisy EVA embeddings [B, N, 4096]")
        logger.info(f"  Conditioning: Clean EVA embeddings [B, N, 4096]")
        logger.info(f"  Target: Clean EVA embeddings [B, N, 4096]")
        logger.info(f"  Learning rate: {self.learning_rate}")
        logger.info(f"  Epochs: {self.num_epochs}")
        logger.info(f"  Prediction type: {getattr(model.config, 'prediction_type', 'velocity')}")
        logger.info(f"  Overfit test: {self.overfit_test_size if self.overfit_test_size else 'Disabled'}")
        logger.info(f"  Mixed precision: {self.fp16}")

    def _setup_optimizer_and_scheduler(self):
        """Setup optimizer and learning rate scheduler"""
        # Use AdamW with weight decay
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.95),  # Slightly different betas for spherical flow
            eps=1e-8
        )
        
        # Setup learning rate scheduler with warmup
        try:
            dataloader_length = len(self.train_dataloader)
            total_steps = dataloader_length * self.num_epochs
        except (TypeError, AttributeError):
            # Fallback for IterableDataset
            logger.warning("Cannot determine exact dataloader length, using estimate")
            estimated_samples_per_epoch = 10000  # Conservative estimate
            batch_size = getattr(self.train_dataloader, 'batch_size', 16)
            estimated_batches_per_epoch = estimated_samples_per_epoch // batch_size
            total_steps = estimated_batches_per_epoch * self.num_epochs
        
        if self.warmup_steps > 0:
            # Warmup + Cosine decay
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
            # Just cosine decay
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=total_steps,
                eta_min=self.learning_rate * 0.01
            )
        
        logger.info(f"Optimizer and scheduler setup complete")
        logger.info(f"  Estimated total steps: {total_steps}")
        logger.info(f"  Warmup steps: {self.warmup_steps}")

    def _prepare_overfit_test(self):
        """Prepare overfitting test batch"""
        logger.info(f"Preparing overfitting test with {self.overfit_test_size} samples...")
        
        try:
            # Get first batch and repeat it
            first_batch = next(iter(self.train_dataloader))
            
            # Trim to desired size
            actual_size = min(self.overfit_test_size, first_batch['batch_size'])
            
            self.overfit_batch = {}
            for key, value in first_batch.items():
                if torch.is_tensor(value) and value.dim() > 0:
                    self.overfit_batch[key] = value[:actual_size].clone().detach()
                elif isinstance(value, list):
                    self.overfit_batch[key] = value[:actual_size]
                else:
                    self.overfit_batch[key] = value
            
            # Update batch size
            self.overfit_batch['batch_size'] = actual_size
            
            logger.info(f"Overfitting test prepared with {actual_size} samples")
            
        except Exception as e:
            logger.error(f"Failed to prepare overfitting test: {e}")
            self.overfit_batch = None

    def _compute_loss(self, batch: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute loss for a batch"""
        # Move batch to device
        for key, value in batch.items():
            if torch.is_tensor(value):
                batch[key] = value.to(self.device)
        
        # Use overfit batch if specified
        if self.overfit_batch is not None:
            # Move overfit batch to device
            for key, value in self.overfit_batch.items():
                if torch.is_tensor(value):
                    batch[key] = value.to(self.device)
                else:
                    batch[key] = value
        
        # Extract inputs for spherical EVA denoising
        x_t = batch['hidden_states']                    # [B, N, 4096] - Current flow state
        timestep = batch['timestep']                    # [B] - Timesteps
        clean_eva = batch['encoder_hidden_states']      # [B, N, 4096] - Clean EVA conditioning
        target_eva = batch['clean_eva_embeddings']      # [B, N, 4096] - Clean EVA target
        velocity_target = batch.get('velocity_target') # [B, N, 4096] - Velocity target
        noise = batch.get('noise')                      # [B, N, 4096] - Noise
        
        # Forward pass
        if self.fp16:
            with torch.cuda.amp.autocast():
                model_output = self.model(
                    hidden_states=x_t,
                    timestep=timestep,
                    encoder_hidden_states=clean_eva,
                    return_dict=False
                )
                
                # Compute spherical flow matching loss
                loss, metrics = self.loss_fn(
                    model_output=model_output,
                    target_samples=target_eva,
                    timesteps=timestep,
                    conditioning=clean_eva,
                    noise=noise,
                    x_t=x_t,
                    return_metrics=True
                )
        else:
            model_output = self.model(
                hidden_states=x_t,
                timestep=timestep,
                encoder_hidden_states=clean_eva,
                return_dict=False
            )
            
            loss, metrics = self.loss_fn(
                model_output=model_output,
                target_samples=target_eva,
                timesteps=timestep,
                conditioning=clean_eva,
                noise=noise,
                x_t=x_t,
                return_metrics=True
            )
        
        return loss, metrics

    def _backward_and_step(self, loss: torch.Tensor) -> float:
        """Backward pass and optimizer step"""
        # Backward pass
        if self.fp16:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Compute gradient norm before clipping
        grad_norm = 0.0
        for param in self.model.parameters():
            if param.grad is not None:
                grad_norm += param.grad.data.norm(2).item() ** 2
        grad_norm = grad_norm ** 0.5
        
        # Gradient clipping
        if self.max_grad_norm > 0:
            if self.fp16:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            else:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        
        # Optimizer step
        if self.fp16:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        
        self.optimizer.zero_grad()
        self.scheduler.step()
        
        return grad_norm

    def _evaluate(self, num_samples: Optional[int] = None) -> Dict[str, float]:
        """Run spherical EVA denoising evaluation"""
        if self.eval_dataloader is None:
            return {}
        
        if num_samples is None:
            num_samples = self.eval_num_samples
        
        self.model.eval()
        
        all_similarities = []
        all_angular_distances = []
        all_sphere_violations = []
        samples_processed = 0
        
        with torch.no_grad():
            for batch in self.eval_dataloader:
                if samples_processed >= num_samples:
                    break
                
                # Move to device
                noisy_eva = batch['noisy_eva_embeddings'].to(self.device)
                clean_eva = batch['clean_eva_embeddings'].to(self.device)
                
                # Denoise using the model
                denoised_eva = self.model.denoise(
                    noisy_eva=noisy_eva,
                    clean_eva_conditioning=clean_eva,  # Use clean as conditioning
                    num_inference_steps=self.eval_inference_steps,
                )
                
                # Compute similarity
                clean_eva_norm = F.normalize(clean_eva, p=2, dim=-1)
                denoised_eva_norm = F.normalize(denoised_eva, p=2, dim=-1)
                
                # Cosine similarity (main metric)
                similarity = F.cosine_similarity(denoised_eva_norm, clean_eva_norm, dim=-1)
                per_image_similarity = similarity.mean(dim=1)
                
                # Angular distance
                cos_sim_clamped = torch.clamp(similarity, -1 + 1e-7, 1 - 1e-7)
                angular_distance = torch.acos(cos_sim_clamped).mean(dim=1)
                
                # Sphere constraint violation
                denoised_norms = torch.norm(denoised_eva, dim=-1)
                sphere_violation = torch.abs(denoised_norms - 1.0).mean(dim=1)
                
                all_similarities.append(per_image_similarity.cpu())
                all_angular_distances.append(angular_distance.cpu())
                all_sphere_violations.append(sphere_violation.cpu())
                samples_processed += noisy_eva.shape[0]
        
        self.model.train()
        
        if not all_similarities:
            return {}
        
        all_sims = torch.cat(all_similarities)
        all_angular = torch.cat(all_angular_distances)
        all_violations = torch.cat(all_sphere_violations)
        
        return {
            'eval_eva_similarity': all_sims.mean().item(),
            'eval_eva_similarity_std': all_sims.std().item(),
            'eval_angular_distance': all_angular.mean().item(),
            'eval_sphere_violation': all_violations.mean().item(),
            'eval_high_quality': (all_sims > 0.7).float().mean().item(),
            'eval_very_high_quality': (all_sims > 0.8).float().mean().item(),
            'eval_excellent_quality': (all_sims > 0.9).float().mean().item(),
            'eval_samples': samples_processed,
            'eval_similarity_min': all_sims.min().item(),
            'eval_similarity_max': all_sims.max().item(),
        }

    def _log_metrics(self, loss: float, metrics: Dict[str, float], grad_norm: float):
        """Log training metrics"""
        # Store metrics
        self.loss_history.append(loss)
        if 'prediction_similarity' in metrics:
            self.similarity_history.append(metrics['prediction_similarity'])
        if 'eval_similarity' in metrics:
            self.eval_similarity_history.append(metrics['eval_similarity'])
        if 'sphere_violation' in metrics:
            self.sphere_violation_history.append(metrics['sphere_violation'])
        
        self.lr_history.append(self.optimizer.param_groups[0]['lr'])
        self.grad_norm_history.append(grad_norm)
        
        # Update best metrics
        eval_sim = metrics.get('eval_similarity', metrics.get('prediction_similarity', 0))
        if eval_sim > self.best_eval_similarity:
            self.best_eval_similarity = eval_sim
        
        if loss < self.best_loss:
            self.best_loss = loss
        
        # Log to console
        if self.global_step % self.log_every_n_steps == 0:
            log_msg = f"Step {self.global_step}: Loss={loss:.6f}"
            
            if 'prediction_similarity' in metrics:
                pred_sim = metrics['prediction_similarity']
                eval_sim = metrics.get('eval_similarity', pred_sim)
                quality = metrics.get('quality_assessment', 'unknown')
                log_msg += f", PredSim={pred_sim:.4f}, EvalSim={eval_sim:.4f} ({quality})"
            
            if 'sphere_violation' in metrics:
                sphere_viol = metrics['sphere_violation']
                log_msg += f", SphereViol={sphere_viol:.6f}"
            
            log_msg += f", GradNorm={grad_norm:.3f}"
            log_msg += f", LR={self.optimizer.param_groups[0]['lr']:.2e}"
            
            if self.overfit_batch is not None:
                log_msg += " [OVERFIT TEST]"
            
            logger.info(log_msg)
            
            # Detailed logging in debug mode
            if self.debug_mode:
                logger.info(f"  Detailed metrics:")
                for key, value in metrics.items():
                    if isinstance(value, (int, float)):
                        logger.info(f"    {key}: {value:.6f}")

    def _save_checkpoint(self):
        """Save model checkpoint"""
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
            'eval_similarity_history': list(self.eval_similarity_history),
        }
        
        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved: {checkpoint_path}")

    def train(self) -> Dict[str, Any]:
        """Main training loop"""
        logger.info("Starting spherical EVA denoising training...")
        logger.info(f"  Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        # Handle dataloader length safely
        try:
            steps_per_epoch = len(self.train_dataloader)
            total_training_steps = steps_per_epoch * self.num_epochs
            logger.info(f"  Training steps per epoch: {steps_per_epoch}")
            logger.info(f"  Total training steps: {total_training_steps}")
        except (TypeError, AttributeError):
            logger.info(f"  Training steps per epoch: Unknown (IterableDataset)")
            logger.info(f"  Total training steps: Estimated")
        
        if self.overfit_batch is not None:
            logger.info(f"  OVERFITTING TEST MODE: Using {self.overfit_batch['batch_size']} samples")
        
        self.model.train()
        start_time = time.time()
        
        try:
            for epoch in range(self.num_epochs):
                self.current_epoch = epoch
                logger.info(f"Starting epoch {epoch + 1}/{self.num_epochs}")
                
                epoch_loss = 0.0
                epoch_steps = 0
                
                for batch_idx, batch in enumerate(self.train_dataloader):
                    step_start_time = time.time()
                    
                    # Compute loss
                    try:
                        loss, metrics = self._compute_loss(batch)
                    except Exception as e:
                        logger.error(f"Error computing loss at step {self.global_step}: {e}")
                        continue
                    
                    # Backward pass
                    try:
                        grad_norm = self._backward_and_step(loss)
                    except Exception as e:
                        logger.error(f"Error in backward pass at step {self.global_step}: {e}")
                        continue
                    
                    # Update metrics
                    epoch_loss += loss.item()
                    epoch_steps += 1
                    self.global_step += 1
                    
                    # Log metrics
                    self._log_metrics(loss.item(), metrics or {}, grad_norm)
                    
                    # Run evaluation
                    if self.global_step % self.eval_every_n_steps == 0:
                        logger.info(f"Running evaluation at step {self.global_step}...")
                        eval_metrics = self._evaluate()
                        
                        if eval_metrics:
                            logger.info(f"Evaluation results:")
                            for key, value in eval_metrics.items():
                                logger.info(f"  {key}: {value:.4f}")
                            
                            # Update best eval similarity
                            if eval_metrics.get('eval_eva_similarity', 0) > self.best_eval_similarity:
                                self.best_eval_similarity = eval_metrics['eval_eva_similarity']
                                logger.info(f"New best EVA similarity: {self.best_eval_similarity:.4f}")
                    
                    # Save checkpoint
                    if self.global_step % self.save_every_n_steps == 0:
                        self._save_checkpoint()
                    
                    # Check for early success in overfitting test
                    if (self.overfit_batch is not None and 
                        metrics and 
                        metrics.get('eval_similarity', 0) > 0.9):
                        logger.info("🎉 OVERFITTING TEST PASSED! Model can learn effectively.")
                        break
                
                # End of epoch logging
                avg_epoch_loss = epoch_loss / max(epoch_steps, 1)
                logger.info(f"Epoch {epoch + 1} completed:")
                logger.info(f"  Average loss: {avg_epoch_loss:.6f}")
                logger.info(f"  Best loss: {self.best_loss:.6f}")
                logger.info(f"  Best similarity: {self.best_eval_similarity:.4f}")
                
                # Early stopping for overfitting test
                if (self.overfit_batch is not None and 
                    len(self.eval_similarity_history) > 0 and 
                    max(self.eval_similarity_history[-10:]) > 0.9):
                    logger.info("Overfitting test completed successfully!")
                    break
        
        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
        except Exception as e:
            logger.error(f"Training failed with error: {e}")
            raise
        
        finally:
            # Final checkpoint
            self._save_checkpoint()
            
            # Final evaluation
            logger.info("Running final evaluation...")
            final_eval = self._evaluate(num_samples=self.eval_num_samples * 2)
            
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
                'overfit_test': self.overfit_batch is not None,
                'overfit_success': (self.overfit_batch is not None and 
                                  len(self.eval_similarity_history) > 0 and 
                                  max(self.eval_similarity_history) > 0.8),
                'loss_history': list(self.loss_history),
                'similarity_history': list(self.similarity_history),
                'eval_similarity_history': list(self.eval_similarity_history),
                'lr_history': list(self.lr_history),
                'grad_norm_history': list(self.grad_norm_history),
                'sphere_violation_history': list(self.sphere_violation_history),
            }
            
            # Save training summary
            summary_path = self.output_dir / "training_summary.json"
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)
            
            logger.info("Training completed!")
            logger.info(f"  Total time: {total_time:.1f} seconds")
            logger.info(f"  Total steps: {self.global_step}")
            logger.info(f"  Best loss: {self.best_loss:.6f}")
            logger.info(f"  Best EVA similarity: {self.best_eval_similarity:.4f}")
            
            if final_eval:
                logger.info(f"  Final evaluation:")
                for key, value in final_eval.items():
                    logger.info(f"    {key}: {value:.4f}")
            
            if self.overfit_batch is not None:
                success = summary['overfit_success']
                logger.info(f"  Overfitting test: {'✅ PASSED' if success else '❌ FAILED'}")
            
            return summary


def create_spherical_eva_trainer(
    model,
    loss_fn,
    train_dataloader,
    eval_dataloader=None,
    learning_rate: float = 1e-4,
    num_epochs: int = 10,
    output_dir: str = "./checkpoints",
    overfit_test_size: Optional[int] = None,
    debug_mode: bool = False,
    **kwargs
) -> SphericalEVATrainer:
    """Factory function to create spherical EVA trainer"""
    
    return SphericalEVATrainer(
        model=model,
        loss_fn=loss_fn,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        output_dir=output_dir,
        overfit_test_size=overfit_test_size,
        debug_mode=debug_mode,
        **kwargs
    )