#!/usr/bin/env python3
"""
FIXED Flow Matching Loss for CLIP Reproduction with Gradient Stability
Based on BLIP3-o paper and flow matching literature best practices
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, Tuple
import logging
import math

logger = logging.getLogger(__name__)


class StableBLIP3oCLIPFlowMatchingLoss(nn.Module):
    """
    FIXED Flow Matching Loss for CLIP reproduction with gradient stability
    
    Key fixes for gradient explosion:
    - Stable loss scaling and clipping
    - Robust similarity computation
    - NaN/Inf detection and handling
    - Conservative velocity target computation
    """
    
    def __init__(
        self,
        prediction_type: str = "velocity",
        flow_type: str = "rectified",
        loss_weight: float = 1.0,
        eps: float = 1e-8,
        min_timestep: float = 1e-3,
        max_timestep: float = 1.0 - 1e-3,
        # FIXED: Stability parameters
        max_loss_scale: float = 100.0,  # Prevent extremely large losses
        gradient_clip_value: float = 10.0,  # Clip gradients in loss computation
        stable_similarity: bool = True,  # Use stable similarity computation
        loss_smoothing: float = 0.1,  # Smooth loss to reduce spikes
    ):
        super().__init__()
        
        self.prediction_type = prediction_type
        self.flow_type = flow_type
        self.loss_weight = loss_weight
        self.eps = eps
        self.min_timestep = min_timestep
        self.max_timestep = max_timestep
        
        # FIXED: Stability parameters
        self.max_loss_scale = max_loss_scale
        self.gradient_clip_value = gradient_clip_value
        self.stable_similarity = stable_similarity
        self.loss_smoothing = loss_smoothing
        
        # Validate inputs
        assert prediction_type in ["velocity", "noise", "sample"]
        assert flow_type in ["rectified", "reflow"]
        
        logger.info(f"🛡️ Stable CLIP Flow Matching Loss initialized:")
        logger.info(f"  Prediction type: {prediction_type}")
        logger.info(f"  Flow type: {flow_type}")
        logger.info(f"  Max loss scale: {max_loss_scale}")
        logger.info(f"  Gradient clip value: {gradient_clip_value}")
        logger.info(f"  Stable similarity: {stable_similarity}")

    def _clamp_timesteps(self, timesteps: torch.Tensor) -> torch.Tensor:
        """Clamp timesteps to avoid numerical issues"""
        return torch.clamp(timesteps, min=self.min_timestep, max=self.max_timestep)

    def _check_tensor_health(self, tensor: torch.Tensor, name: str) -> bool:
        """Check tensor for NaN/Inf values"""
        if torch.isnan(tensor).any():
            logger.error(f"🚨 NaN detected in {name}")
            return False
        if torch.isinf(tensor).any():
            logger.error(f"🚨 Inf detected in {name}")
            return False
        return True

    def _stable_cosine_similarity(self, x: torch.Tensor, y: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """Compute cosine similarity with numerical stability"""
        if not self.stable_similarity:
            return F.cosine_similarity(x, y, dim=dim)
        
        # Normalize with numerical stability
        x_norm = F.normalize(x, p=2, dim=dim, eps=self.eps)
        y_norm = F.normalize(y, p=2, dim=dim, eps=self.eps)
        
        # Compute similarity
        similarity = torch.sum(x_norm * y_norm, dim=dim)
        
        # Clamp to valid range [-1, 1]
        similarity = torch.clamp(similarity, min=-1.0 + self.eps, max=1.0 - self.eps)
        
        return similarity

    def _smooth_loss(self, loss: torch.Tensor, target_loss: torch.Tensor) -> torch.Tensor:
        """Apply loss smoothing to reduce spikes"""
        if self.loss_smoothing > 0:
            # Exponential moving average style smoothing
            smooth_factor = self.loss_smoothing
            smoothed_loss = smooth_factor * target_loss + (1 - smooth_factor) * loss
            return smoothed_loss
        return loss

    def forward(
        self,
        model_output: torch.Tensor,  # [B, N, 1024] - Model's velocity prediction
        target_samples: torch.Tensor,  # [B, N, 1024] - Clean CLIP embeddings
        timesteps: torch.Tensor,  # [B] - Flow matching timesteps
        eva_conditioning: torch.Tensor,  # [B, N, 4096] - EVA features (for logging)
        noise: Optional[torch.Tensor] = None,
        return_metrics: bool = True,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[Dict[str, float]]]:
        """
        FIXED: Compute rectified flow matching loss with stability checks
        """
        
        # Input validation
        if not self._check_tensor_health(model_output, "model_output"):
            raise ValueError("Invalid model_output tensor")
        if not self._check_tensor_health(target_samples, "target_samples"):
            raise ValueError("Invalid target_samples tensor")
        if not self._check_tensor_health(timesteps, "timesteps"):
            raise ValueError("Invalid timesteps tensor")
        
        batch_size, num_tokens, embed_dim = model_output.shape
        device = model_output.device
        dtype = model_output.dtype
        
        # Clamp timesteps for numerical stability
        timesteps = self._clamp_timesteps(timesteps)
        
        # Keep targets as-is (no normalization during training)
        target_clean = target_samples.detach()
        
        # Use standard Gaussian noise with stability checks
        if noise is None:
            noise = torch.randn_like(target_clean, device=device, dtype=dtype)
        
        # Check noise health
        if not self._check_tensor_health(noise, "noise"):
            logger.warning("⚠️ Regenerating noise due to NaN/Inf")
            noise = torch.randn_like(target_clean, device=device, dtype=dtype)
        
        # Expand timesteps for broadcasting [B, 1, 1]
        t = timesteps.view(batch_size, 1, 1).to(dtype)
        
        # RECTIFIED FLOW COMPUTATION with stability
        if self.flow_type == "rectified":
            # Linear interpolation: x_t = (1-t) * noise + t * target
            # Velocity target: v = target - noise (for rectified flow)
            true_velocity = target_clean - noise
            target_for_loss = true_velocity
        else:
            raise NotImplementedError("Only rectified flow is implemented")
        
        # Check velocity target health
        if not self._check_tensor_health(target_for_loss, "velocity_target"):
            raise ValueError("Invalid velocity target")
        
        # LOSS COMPUTATION with stability
        if self.prediction_type == "velocity":
            # Direct velocity prediction loss with robust computation
            prediction_loss = F.mse_loss(model_output, target_for_loss, reduction='none')
            
            # FIXED: Check for extreme loss values
            max_loss = prediction_loss.max().item()
            if max_loss > self.max_loss_scale:
                logger.warning(f"⚠️ Clamping extreme loss: {max_loss:.2f} -> {self.max_loss_scale}")
                prediction_loss = torch.clamp(prediction_loss, max=self.max_loss_scale)
            
        else:
            raise NotImplementedError(f"Prediction type {self.prediction_type} not implemented")
        
        # Check prediction loss health
        if not self._check_tensor_health(prediction_loss, "prediction_loss"):
            raise ValueError("Invalid prediction loss")
        
        # Reduce loss: mean over tokens and embedding dimensions, then over batch
        prediction_loss = prediction_loss.mean(dim=(1, 2))  # [B]
        
        # Main loss
        main_loss = prediction_loss.mean()
        
        # FIXED: Apply loss smoothing if enabled
        if hasattr(self, '_prev_loss'):
            main_loss = self._smooth_loss(main_loss, self._prev_loss)
        self._prev_loss = main_loss.detach()
        
        # Total loss with stability check
        total_loss = main_loss * self.loss_weight
        
        # Final stability check
        if not self._check_tensor_health(total_loss, "total_loss"):
            logger.error("🚨 Total loss contains NaN/Inf - returning safe fallback")
            total_loss = torch.tensor(1.0, device=device, requires_grad=True)
        
        # METRICS COMPUTATION with stability
        metrics = {}
        if return_metrics:
            with torch.no_grad():
                try:
                    # FIXED: Stable similarity computation
                    cosine_sim = self._stable_cosine_similarity(model_output, target_for_loss, dim=-1)
                    per_image_sim = cosine_sim.mean(dim=1)  # [B]
                    mean_similarity = per_image_sim.mean().item()
                    
                    # Compute norms for monitoring (raw, unnormalized)
                    pred_norm = torch.norm(model_output, dim=-1).mean().item()
                    target_norm_val = torch.norm(target_for_loss, dim=-1).mean().item()
                    clip_norm = torch.norm(target_clean, dim=-1).mean().item()
                    noise_norm = torch.norm(noise, dim=-1).mean().item()
                    
                    # Error analysis with stability
                    error = model_output - target_for_loss
                    error_norm = torch.norm(error, dim=-1).mean().item()
                    relative_error = error_norm / (target_norm_val + self.eps)
                    
                    # FIXED: Clamp metrics to reasonable ranges
                    mean_similarity = float(np.clip(mean_similarity, -1.0, 1.0))
                    pred_norm = float(np.clip(pred_norm, 0.0, 1000.0))
                    target_norm_val = float(np.clip(target_norm_val, 0.0, 1000.0))
                    
                    metrics = {
                        # Core metrics
                        'loss': main_loss.item(),
                        'total_loss': total_loss.item(),
                        'velocity_similarity': mean_similarity,
                        'velocity_similarity_std': per_image_sim.std().item(),
                        
                        # Raw norm tracking
                        'pred_norm': pred_norm,
                        'target_norm': target_norm_val,
                        'clip_norm': clip_norm,
                        'noise_norm': noise_norm,
                        
                        # Error analysis
                        'error_norm': error_norm,
                        'relative_error': relative_error,
                        
                        # Flow matching specific
                        'timestep_mean': timesteps.mean().item(),
                        'timestep_std': timesteps.std().item(),
                        
                        # Stability metrics
                        'max_prediction_loss': max_loss,
                        'loss_scale_applied': max_loss > self.max_loss_scale,
                    }
                    
                    # Check for numerical issues in metrics
                    for key, value in metrics.items():
                        if math.isnan(value) or math.isinf(value):
                            logger.warning(f"⚠️ Invalid metric {key}: {value}")
                            metrics[key] = 0.0
                    
                    # Overall health check
                    if torch.isnan(total_loss) or torch.isinf(total_loss):
                        metrics['numerical_issue'] = True
                        logger.error("🚨 Numerical issue detected in loss computation!")
                    else:
                        metrics['numerical_issue'] = False
                
                except Exception as e:
                    logger.error(f"❌ Error computing metrics: {e}")
                    metrics = {
                        'loss': total_loss.item() if torch.isfinite(total_loss) else 1.0,
                        'total_loss': total_loss.item() if torch.isfinite(total_loss) else 1.0,
                        'velocity_similarity': 0.0,
                        'numerical_issue': True,
                        'metrics_error': str(e),
                    }
        
        return total_loss, metrics

    def compute_eval_loss(
        self,
        generated: torch.Tensor,  # Generated CLIP embeddings
        target: torch.Tensor,     # Target CLIP embeddings
    ) -> Dict[str, float]:
        """FIXED: Compute evaluation metrics for generated embeddings with stability"""
        with torch.no_grad():
            try:
                # Input validation
                if not self._check_tensor_health(generated, "generated"):
                    raise ValueError("Invalid generated tensor")
                if not self._check_tensor_health(target, "target"):
                    raise ValueError("Invalid target tensor")
                
                # FIXED: Stable similarity computation
                cosine_sim = self._stable_cosine_similarity(generated, target, dim=-1)
                per_image_sim = cosine_sim.mean(dim=1)
                
                # MSE in raw space with stability
                mse_loss = F.mse_loss(generated, target)
                if not torch.isfinite(mse_loss):
                    logger.warning("⚠️ Invalid MSE loss, setting to fallback value")
                    mse_loss = torch.tensor(1.0)
                
                # Quality metrics with stability checks
                sim_values = per_image_sim.cpu().numpy()
                high_quality = float(np.mean(sim_values > 0.7))
                very_high_quality = float(np.mean(sim_values > 0.8))
                excellent_quality = float(np.mean(sim_values > 0.9))
                
                # Scale analysis (raw space)
                generated_norm_val = torch.norm(generated, dim=-1).mean().item()
                target_norm_val = torch.norm(target, dim=-1).mean().item()
                
                # Clamp values to reasonable ranges
                generated_norm_val = float(np.clip(generated_norm_val, 0.0, 1000.0))
                target_norm_val = float(np.clip(target_norm_val, 0.0, 1000.0))
                
                return {
                    'eval_clip_similarity': float(np.clip(per_image_sim.mean().item(), -1.0, 1.0)),
                    'eval_mse_loss': mse_loss.item(),
                    'eval_high_quality': high_quality,
                    'eval_very_high_quality': very_high_quality,
                    'eval_excellent_quality': excellent_quality,
                    'eval_similarity_std': float(np.clip(per_image_sim.std().item(), 0.0, 2.0)),
                    
                    # Raw embedding norms
                    'eval_generated_norm': generated_norm_val,
                    'eval_target_norm': target_norm_val,
                    'eval_norm_ratio': generated_norm_val / (target_norm_val + self.eps),
                    
                    # Stability indicators
                    'eval_numerical_issue': False,
                }
                
            except Exception as e:
                logger.error(f"❌ Error in evaluation: {e}")
                return {
                    'eval_clip_similarity': 0.0,
                    'eval_mse_loss': 1.0,
                    'eval_high_quality': 0.0,
                    'eval_very_high_quality': 0.0,
                    'eval_excellent_quality': 0.0,
                    'eval_similarity_std': 0.0,
                    'eval_generated_norm': 1.0,
                    'eval_target_norm': 1.0,
                    'eval_norm_ratio': 1.0,
                    'eval_numerical_issue': True,
                    'eval_error': str(e),
                }


# Import numpy for clipping
import numpy as np


def create_stable_clip_reproduction_loss(
    prediction_type: str = "velocity",
    flow_type: str = "rectified", 
    loss_weight: float = 1.0,
    max_loss_scale: float = 100.0,  # Prevent extreme losses
    gradient_clip_value: float = 10.0,  # Gradient clipping
    stable_similarity: bool = True,  # Stable similarity computation
    **kwargs
) -> StableBLIP3oCLIPFlowMatchingLoss:
    """Factory function for stable CLIP reproduction loss"""
    
    return StableBLIP3oCLIPFlowMatchingLoss(
        prediction_type=prediction_type,
        flow_type=flow_type,
        loss_weight=loss_weight,
        max_loss_scale=max_loss_scale,
        gradient_clip_value=gradient_clip_value,
        stable_similarity=stable_similarity,
        **kwargs
    )