#!/usr/bin/env python3
"""
Clean Flow Matching Loss for CLIP Reproduction
Simple rectified flow matching aligned with BLIP3-o paper
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)


class BLIP3oCLIPFlowMatchingLoss(nn.Module):
    """
    Clean Flow Matching Loss for CLIP reproduction
    Rectified flow matching implementation following BLIP3-o paper
    """
    
    def __init__(
        self,
        prediction_type: str = "velocity",
        flow_type: str = "rectified",
        loss_weight: float = 1.0,
        eps: float = 1e-8,
        min_timestep: float = 1e-3,
        max_timestep: float = 1.0 - 1e-3,
    ):
        super().__init__()
        
        self.prediction_type = prediction_type
        self.flow_type = flow_type
        self.loss_weight = loss_weight
        self.eps = eps
        self.min_timestep = min_timestep
        self.max_timestep = max_timestep
        
        # Validate inputs
        assert prediction_type in ["velocity", "noise", "sample"]
        assert flow_type in ["rectified", "reflow"]
        
        logger.info(f"Clean CLIP Flow Matching Loss initialized:")
        logger.info(f"  Prediction type: {prediction_type}")
        logger.info(f"  Flow type: {flow_type}")

    def _clamp_timesteps(self, timesteps: torch.Tensor) -> torch.Tensor:
        """Clamp timesteps to avoid numerical issues"""
        return torch.clamp(timesteps, min=self.min_timestep, max=self.max_timestep)

    def forward(
        self,
        model_output: torch.Tensor,  # [B, N, 1024] - Model's prediction
        target_samples: torch.Tensor,  # [B, N, 1024] - Clean CLIP embeddings
        timesteps: torch.Tensor,  # [B] - Flow matching timesteps
        eva_conditioning: torch.Tensor,  # [B, N, 4096] - EVA features (for logging)
        noise: Optional[torch.Tensor] = None,
        return_metrics: bool = True,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[Dict[str, float]]]:
        """
        Compute rectified flow matching loss
        """
        
        batch_size, num_tokens, embed_dim = model_output.shape
        device = model_output.device
        dtype = model_output.dtype
        
        # Clamp timesteps
        timesteps = self._clamp_timesteps(timesteps)
        
        # Keep targets as-is (no normalization during training)
        target_clean = target_samples.detach()
        
        # Use standard Gaussian noise
        if noise is None:
            noise = torch.randn_like(target_clean, device=device, dtype=dtype)
        
        # Expand timesteps for broadcasting [B, 1, 1]
        t = timesteps.view(batch_size, 1, 1).to(dtype)
        
        # RECTIFIED FLOW COMPUTATION
        if self.flow_type == "rectified":
            # Linear interpolation: x_t = (1-t) * noise + t * target
            # Velocity target: v = target - noise (for rectified flow)
            true_velocity = target_clean - noise
            target_for_loss = true_velocity
        else:
            raise NotImplementedError("Only rectified flow is implemented")
        
        # LOSS COMPUTATION
        if self.prediction_type == "velocity":
            # Direct velocity prediction loss
            prediction_loss = F.mse_loss(model_output, target_for_loss, reduction='none')
        else:
            raise NotImplementedError(f"Prediction type {self.prediction_type} not implemented")
        
        # Reduce loss: mean over tokens and embedding dimensions, then over batch
        prediction_loss = prediction_loss.mean(dim=(1, 2))  # [B]
        
        # Main loss
        main_loss = prediction_loss.mean()
        
        # Total loss
        total_loss = main_loss * self.loss_weight
        
        # METRICS COMPUTATION
        metrics = {}
        if return_metrics:
            with torch.no_grad():
                # Normalize only for cosine similarity computation
                pred_normalized = F.normalize(model_output, p=2, dim=-1)
                target_norm = F.normalize(target_for_loss, p=2, dim=-1)
                
                # Cosine similarity (requires normalization)
                cosine_sim = F.cosine_similarity(pred_normalized, target_norm, dim=-1)
                per_image_sim = cosine_sim.mean(dim=1)  # [B]
                mean_similarity = per_image_sim.mean().item()
                
                # Compute norms for monitoring (raw, unnormalized)
                pred_norm = torch.norm(model_output, dim=-1).mean().item()
                target_norm_val = torch.norm(target_for_loss, dim=-1).mean().item()
                clip_norm = torch.norm(target_clean, dim=-1).mean().item()
                noise_norm = torch.norm(noise, dim=-1).mean().item()
                
                # Error analysis
                error = model_output - target_for_loss
                error_norm = torch.norm(error, dim=-1).mean().item()
                relative_error = error_norm / (target_norm_val + self.eps)
                
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
                }
                
                # Check for numerical issues
                if torch.isnan(total_loss) or torch.isinf(total_loss):
                    metrics['numerical_issue'] = True
                    logger.error("Numerical issue detected in loss computation!")
        
        return total_loss, metrics

    def compute_eval_loss(
        self,
        generated: torch.Tensor,  # Generated CLIP embeddings
        target: torch.Tensor,     # Target CLIP embeddings
    ) -> Dict[str, float]:
        """Compute evaluation metrics for generated embeddings"""
        with torch.no_grad():
            # Normalize only for cosine similarity computation
            generated_norm = F.normalize(generated, p=2, dim=-1)
            target_norm = F.normalize(target, p=2, dim=-1)
            
            # Cosine similarity
            cosine_sim = F.cosine_similarity(generated_norm, target_norm, dim=-1)
            per_image_sim = cosine_sim.mean(dim=1)
            
            # MSE in raw space
            mse_loss = F.mse_loss(generated, target)
            
            # Quality metrics
            high_quality = (per_image_sim > 0.7).float().mean().item()
            very_high_quality = (per_image_sim > 0.8).float().mean().item()
            excellent_quality = (per_image_sim > 0.9).float().mean().item()
            
            # Scale analysis (raw space)
            generated_norm_val = torch.norm(generated, dim=-1).mean().item()
            target_norm_val = torch.norm(target, dim=-1).mean().item()
            
            return {
                'eval_clip_similarity': per_image_sim.mean().item(),
                'eval_mse_loss': mse_loss.item(),
                'eval_high_quality': high_quality,
                'eval_very_high_quality': very_high_quality,
                'eval_excellent_quality': excellent_quality,
                'eval_similarity_std': per_image_sim.std().item(),
                
                # Raw embedding norms
                'eval_generated_norm': generated_norm_val,
                'eval_target_norm': target_norm_val,
                'eval_norm_ratio': generated_norm_val / (target_norm_val + 1e-8),
            }


def create_clip_reproduction_loss(
    prediction_type: str = "velocity",
    flow_type: str = "rectified", 
    loss_weight: float = 1.0,
    **kwargs
) -> BLIP3oCLIPFlowMatchingLoss:
    """Factory function for CLIP reproduction loss"""
    
    return BLIP3oCLIPFlowMatchingLoss(
        prediction_type=prediction_type,
        flow_type=flow_type,
        loss_weight=loss_weight,
        **kwargs
    )