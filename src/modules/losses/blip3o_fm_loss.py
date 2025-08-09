#!/usr/bin/env python3
"""
FIXED Flow Matching Loss WITHOUT CLIP Normalization
src/modules/losses/blip3o_fm_loss.py

FIX: Added proper alias for distributed trainer compatibility
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, Tuple
import logging
import math

logger = logging.getLogger(__name__)


class SemanticPreservingFlowMatchingLoss(nn.Module):
    """
    Flow Matching Loss without CLIP normalization
    All computations work directly with raw CLIP embeddings
    """
    
    def __init__(
        self,
        prediction_type: str = "velocity",
        flow_type: str = "rectified",
        
        # Loss weights
        velocity_weight: float = 1.0,
        
        # Timestep configuration
        use_timestep_weighting: bool = True,
        early_timestep_threshold: float = 0.3,
        late_timestep_threshold: float = 0.5,
        
        # Stability parameters
        eps: float = 1e-8,
        min_timestep: float = 1e-3,
        max_timestep: float = 1.0 - 1e-3,
        max_loss_value: float = 100.0,
        gradient_clip_value: float = 10.0,
        adaptive_scaling: bool = True,
        robust_similarity: bool = True,
    ):
        super().__init__()
        
        self.prediction_type = prediction_type
        self.flow_type = flow_type
        
        # Loss weights
        self.velocity_weight = velocity_weight
        
        # Timestep weighting
        self.use_timestep_weighting = use_timestep_weighting
        self.early_timestep_threshold = early_timestep_threshold
        self.late_timestep_threshold = late_timestep_threshold
        
        # Stability parameters
        self.eps = eps
        self.min_timestep = min_timestep
        self.max_timestep = max_timestep
        self.max_loss_value = max_loss_value
        self.gradient_clip_value = gradient_clip_value
        self.adaptive_scaling = adaptive_scaling
        self.robust_similarity = robust_similarity
        
        # Adaptive scaling state
        self.loss_scale_factor = 1.0
        self.recent_losses = []
        self.max_recent_losses = 100
        
        # Validate inputs
        assert prediction_type in ["velocity", "noise", "sample"]
        assert flow_type in ["rectified", "reflow"]
        
        logger.info(f"✅ Flow Matching Loss initialized (NO NORMALIZATION):")
        logger.info(f"  Prediction type: {prediction_type}")
        logger.info(f"  Flow type: {flow_type}")
        logger.info(f"  Weights - Velocity: {velocity_weight}")

    def _clamp_timesteps(self, timesteps: torch.Tensor) -> torch.Tensor:
        """Clamp timesteps to avoid numerical issues"""
        return torch.clamp(timesteps, min=self.min_timestep, max=self.max_timestep)

    def _check_tensor_health(self, tensor: torch.Tensor, name: str) -> bool:
        """Check tensor for NaN/Inf and return health status"""
        if torch.isnan(tensor).any():
            logger.warning(f"⚠️ NaN detected in {name}")
            return False
        if torch.isinf(tensor).any():
            logger.warning(f"⚠️ Inf detected in {name}")
            return False
        return True

    def _robust_normalize(self, tensor: torch.Tensor, dim: int = -1, eps: float = None) -> torch.Tensor:
        """Robust normalization with overflow protection"""
        if eps is None:
            eps = self.eps
        
        # Compute norm with protection against overflow
        norm = torch.norm(tensor, p=2, dim=dim, keepdim=True)
        norm = torch.clamp(norm, min=eps, max=1e6)
        
        normalized = tensor / norm
        
        # Additional safety check
        if not self._check_tensor_health(normalized, "normalized_tensor"):
            logger.warning("⚠️ Normalization failed, returning unit vector")
            # Fallback: return unit vector in last dimension
            unit_tensor = torch.zeros_like(tensor)
            unit_tensor[..., 0] = 1.0
            return unit_tensor
        
        return normalized

    def _robust_cosine_similarity(self, x: torch.Tensor, y: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """Robust cosine similarity computation"""
        if self.robust_similarity:
            # Normalize both tensors robustly
            x_norm = self._robust_normalize(x, dim=dim)
            y_norm = self._robust_normalize(y, dim=dim)
            
            # Compute similarity
            similarity = torch.sum(x_norm * y_norm, dim=dim)
            
            # Clamp to valid range [-1, 1]
            similarity = torch.clamp(similarity, min=-1.0 + self.eps, max=1.0 - self.eps)
        else:
            # Standard cosine similarity
            similarity = F.cosine_similarity(x, y, dim=dim)
        
        return similarity

    def _compute_timestep_weights(self, timesteps: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute timestep-dependent weights"""
        if not self.use_timestep_weighting:
            batch_size = timesteps.shape[0]
            device = timesteps.device
            return {
                'velocity_weights': torch.ones(batch_size, device=device),
            }
        
        # Velocity loss: constant throughout
        velocity_weights = torch.ones_like(timesteps)
        
        return {
            'velocity_weights': velocity_weights
        }

    def _compute_predicted_clean(
        self, 
        model_output: torch.Tensor, 
        noisy_input: torch.Tensor, 
        timesteps: torch.Tensor, 
        noise: torch.Tensor
    ) -> torch.Tensor:
        """Compute predicted clean embeddings from model output"""
        
        if self.prediction_type == "velocity":
            # For rectified flow: x_0 = x_t - (1-t) * v_pred
            t_expanded = timesteps.view(-1, 1, 1)
            predicted_clean = noisy_input - (1 - t_expanded) * model_output
            
        elif self.prediction_type == "noise":
            # For noise prediction: x_0 = x_t - noise_pred
            predicted_clean = noisy_input - model_output
            
        elif self.prediction_type == "sample":
            # Direct prediction
            predicted_clean = model_output
            
        else:
            raise ValueError(f"Unknown prediction type: {self.prediction_type}")
        
        return predicted_clean

    def _update_adaptive_scaling(self, loss_value: float):
        """Update adaptive loss scaling based on recent loss values"""
        if not self.adaptive_scaling:
            return
        
        self.recent_losses.append(loss_value)
        if len(self.recent_losses) > self.max_recent_losses:
            self.recent_losses.pop(0)
        
        if len(self.recent_losses) >= 10:
            # Check for loss explosion
            recent_mean = sum(self.recent_losses[-10:]) / 10
            if recent_mean > 10.0:
                self.loss_scale_factor *= 0.9
                logger.warning(f"⚠️ High loss detected, reducing scale factor to {self.loss_scale_factor:.3f}")
            elif recent_mean < 0.1:
                self.loss_scale_factor *= 1.01
                self.loss_scale_factor = min(self.loss_scale_factor, 2.0)

    def forward(
        self,
        model_output: torch.Tensor,
        target_samples: torch.Tensor,  # Raw CLIP embeddings
        timesteps: torch.Tensor,
        eva_conditioning: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
        noisy_input: Optional[torch.Tensor] = None,
        return_metrics: bool = True,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[Dict[str, float]]]:
        """
        Multi-component flow matching loss without normalization
        All computations work directly with raw CLIP embeddings
        """
        
        batch_size, num_tokens, embed_dim = model_output.shape
        device = model_output.device
        dtype = model_output.dtype
        
        # Check input tensors
        if not self._check_tensor_health(model_output, "model_output"):
            logger.error("❌ Unhealthy model output detected!")
            return torch.tensor(float('inf'), device=device), {}
        
        if not self._check_tensor_health(target_samples, "target_samples"):
            logger.error("❌ Unhealthy target samples detected!")
            return torch.tensor(float('inf'), device=device), {}
        
        # Clamp timesteps
        timesteps = self._clamp_timesteps(timesteps)
        
        # Use provided noise or create new
        if noise is None:
            noise = torch.randn_like(target_samples, device=device, dtype=dtype)
        
        # Expand timesteps for broadcasting
        t = timesteps.view(batch_size, 1, 1).to(dtype)
        
        # RECTIFIED FLOW COMPUTATION
        if self.flow_type == "rectified":
            # Linear interpolation: x_t = (1-t) * noise + t * target
            if noisy_input is None:
                noisy_input = (1 - t) * noise + t * target_samples
            
            # Velocity target: v = target - noise
            velocity_target = target_samples - noise
        else:
            raise NotImplementedError("Only rectified flow is implemented")
        
        # Get timestep weights
        weights = self._compute_timestep_weights(timesteps)
        
        # COMPONENT 1: VELOCITY PREDICTION LOSS
        if self.prediction_type == "velocity":
            velocity_loss = F.mse_loss(model_output, velocity_target, reduction='none')
        else:
            raise NotImplementedError(f"Prediction type {self.prediction_type} not implemented")
        
        # Apply timestep weighting and reduce
        velocity_loss = velocity_loss.mean(dim=(1, 2))  # [B]
        velocity_loss = (velocity_loss * weights['velocity_weights']).mean()
        
        # Clamp velocity loss
        velocity_loss = torch.clamp(velocity_loss, max=self.max_loss_value)
        
        # Compute predicted clean embeddings for metrics
        predicted_clean = self._compute_predicted_clean(
            model_output, noisy_input, timesteps, noise
        )
        
        # Apply adaptive scaling
        scale_factor = self.loss_scale_factor if self.adaptive_scaling else 1.0
        
        # TOTAL LOSS COMBINATION
        total_loss = scale_factor * (self.velocity_weight * velocity_loss)
        
        # Final safety checks
        if not self._check_tensor_health(total_loss, "total_loss"):
            logger.error("❌ Unhealthy total loss detected! Using fallback loss.")
            total_loss = torch.tensor(1.0, device=device, dtype=dtype, requires_grad=True)
        
        # Clamp final loss to prevent training instability
        total_loss = torch.clamp(total_loss, max=self.max_loss_value)
        
        # Update adaptive scaling
        self._update_adaptive_scaling(total_loss.item())
        
        # METRICS COMPUTATION
        metrics = {}
        if return_metrics:
            with torch.no_grad():
                try:
                    # Velocity similarities
                    pred_vel_norm = self._robust_normalize(model_output, dim=-1)
                    target_vel_norm = self._robust_normalize(velocity_target, dim=-1)
                    velocity_cosine_sim = self._robust_cosine_similarity(pred_vel_norm, target_vel_norm, dim=-1)
                    mean_velocity_similarity = velocity_cosine_sim.mean().item()
                    
                    # Clean embedding similarities
                    pred_clean_norm = self._robust_normalize(predicted_clean, dim=-1)
                    target_clean_norm = self._robust_normalize(target_samples, dim=-1)
                    clean_cosine_sim = self._robust_cosine_similarity(pred_clean_norm, target_clean_norm, dim=-1)
                    mean_clean_similarity = clean_cosine_sim.mean().item()
                    
                    # Per-image similarities
                    pred_per_img = predicted_clean.mean(dim=1)
                    target_per_img = target_samples.mean(dim=1)
                    per_image_sim = self._robust_cosine_similarity(pred_per_img, target_per_img, dim=-1)
                    mean_per_image_similarity = per_image_sim.mean().item()
                    
                    # Norms for monitoring
                    pred_norm_val = torch.norm(model_output, dim=-1).mean().item()
                    target_norm_val = torch.norm(velocity_target, dim=-1).mean().item()
                    
                    # Error analysis
                    velocity_error = model_output - velocity_target
                    velocity_error_norm = torch.norm(velocity_error, dim=-1).mean().item()
                    
                    metrics = {
                        # Loss components
                        'velocity_loss': velocity_loss.item(),
                        'total_loss': total_loss.item(),
                        
                        # Similarity metrics
                        'velocity_similarity': mean_velocity_similarity,
                        'clean_similarity': mean_clean_similarity,
                        'per_image_similarity': mean_per_image_similarity,
                        
                        # Distribution tracking
                        'velocity_similarity_std': velocity_cosine_sim.std().item(),
                        'clean_similarity_std': clean_cosine_sim.std().item(),
                        'per_image_similarity_std': per_image_sim.std().item(),
                        
                        # Error analysis
                        'velocity_error_norm': velocity_error_norm,
                        'pred_velocity_norm': pred_norm_val,
                        'target_velocity_norm': target_norm_val,
                        
                        # Timestep analysis
                        'timestep_mean': timesteps.mean().item(),
                        'timestep_std': timesteps.std().item(),
                        
                        # Stability metrics
                        'loss_scale_factor': self.loss_scale_factor,
                        'numerical_stability': 1.0,
                        'adaptive_scaling_active': float(self.adaptive_scaling),
                        
                        # Disconnect detection
                        'velocity_clean_disconnect': abs(mean_velocity_similarity - mean_clean_similarity),
                        'disconnect_detected': (
                            mean_velocity_similarity > 0.3 and 
                            mean_clean_similarity < mean_velocity_similarity - 0.15
                        ),
                    }
                    
                except Exception as e:
                    logger.warning(f"⚠️ Error computing metrics: {e}")
                    metrics = {
                        'total_loss': total_loss.item(),
                        'metrics_error': str(e),
                        'numerical_stability': 0.0,
                    }
        
        return total_loss, metrics

    def compute_eval_loss(
        self,
        generated: torch.Tensor,
        target: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Compute evaluation metrics without denormalization
        All computations work directly with raw CLIP embeddings
        """
        with torch.no_grad():
            eval_metrics = {}
            
            # Check tensor health first
            if not self._check_tensor_health(generated, "generated") or not self._check_tensor_health(target, "target"):
                return {
                    'eval_error': 'unhealthy_tensors',
                    'eval_clip_similarity': 0.0,
                    'eval_mse_loss': float('inf'),
                }
            
            # Compute metrics directly in raw CLIP space
            eval_metrics.update(self._compute_similarity_metrics(generated, target))
            
            # Set primary metrics
            eval_metrics.update({
                'eval_clip_similarity': eval_metrics.get('clip_similarity', 0.0),
                'eval_mse_loss': eval_metrics.get('mse_loss', float('inf')),
                'eval_high_quality': eval_metrics.get('high_quality', 0.0),
                'eval_very_high_quality': eval_metrics.get('very_high_quality', 0.0),
                'eval_excellent_quality': eval_metrics.get('excellent_quality', 0.0),
                'eval_similarity_std': eval_metrics.get('similarity_std', 0.0),
            })
            
            return eval_metrics
    
    def _compute_similarity_metrics(
        self, 
        generated: torch.Tensor, 
        target: torch.Tensor,
    ) -> Dict[str, float]:
        """Compute similarity metrics in raw CLIP space"""
        
        try:
            # Per-image similarity (primary metric)
            generated_per_img = generated.mean(dim=1)
            target_per_img = target.mean(dim=1)
            
            # Use robust cosine similarity
            similarity = self._robust_cosine_similarity(generated_per_img, target_per_img, dim=-1)
            
            # Per-token similarity
            token_similarity = self._robust_cosine_similarity(generated, target, dim=-1).mean(dim=1)
            
            # MSE loss with clamping
            mse_loss = F.mse_loss(generated, target)
            mse_loss = min(mse_loss.item(), 1000.0)
            
            # Quality metrics
            high_quality = (similarity > 0.7).float().mean().item()
            very_high_quality = (similarity > 0.8).float().mean().item()
            excellent_quality = (similarity > 0.9).float().mean().item()
            
            # Norm analysis with protection
            generated_norm_val = torch.norm(generated, dim=-1).mean().item()
            target_norm_val = torch.norm(target, dim=-1).mean().item()
            
            # Protect against division by zero
            norm_ratio = generated_norm_val / max(target_norm_val, 1e-8)
            
            return {
                'clip_similarity': similarity.mean().item(),
                'token_similarity': token_similarity.mean().item(),
                'mse_loss': mse_loss,
                'high_quality': high_quality,
                'very_high_quality': very_high_quality,
                'excellent_quality': excellent_quality,
                'similarity_std': similarity.std().item(),
                'generated_norm': generated_norm_val,
                'target_norm': target_norm_val,
                'norm_ratio': norm_ratio,
            }
            
        except Exception as e:
            logger.warning(f"⚠️ Error computing similarity metrics: {e}")
            return {
                'clip_similarity': 0.0,
                'mse_loss': float('inf'),
                'error': str(e),
            }


# FIXED: Add alias for distributed trainer compatibility
BLIP3oCLIPFlowMatchingLoss = SemanticPreservingFlowMatchingLoss


def create_clip_reproduction_loss(
    prediction_type: str = "velocity",
    flow_type: str = "rectified", 
    velocity_weight: float = 1.0,
    use_timestep_weighting: bool = True,
    max_loss_value: float = 100.0,
    adaptive_scaling: bool = True,
    robust_similarity: bool = True,
    **kwargs
) -> SemanticPreservingFlowMatchingLoss:
    """Factory function for CLIP reproduction loss without normalization"""
    
    return SemanticPreservingFlowMatchingLoss(
        prediction_type=prediction_type,
        flow_type=flow_type,
        velocity_weight=velocity_weight,
        use_timestep_weighting=use_timestep_weighting,
        max_loss_value=max_loss_value,
        adaptive_scaling=adaptive_scaling,
        robust_similarity=robust_similarity,
        **kwargs
    )