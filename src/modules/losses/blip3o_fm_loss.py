#!/usr/bin/env python3
"""
UPDATED Flow Matching Loss with Semantic-Preserving Components
src/modules/losses/blip3o_fm_loss.py

Key improvements:
1. Multi-component loss combining velocity + semantic + cosine terms
2. Timestep-aware weighting for better training dynamics
3. Perceptual distance preservation
4. Better numerical stability
5. Enhanced metrics tracking
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
    IMPROVED Flow Matching Loss with Semantic Preservation
    
    This loss addresses the velocity-semantic alignment disconnect by:
    1. Standard velocity prediction loss
    2. Semantic consistency loss for later timesteps
    3. Cosine similarity preservation
    4. Timestep-adaptive weighting
    """
    
    def __init__(
        self,
        prediction_type: str = "velocity",
        flow_type: str = "rectified",
        
        # Loss component weights
        velocity_weight: float = 1.0,
        semantic_weight: float = 0.1,
        cosine_weight: float = 0.05,
        
        # Timestep weighting
        use_timestep_weighting: bool = True,
        early_timestep_threshold: float = 0.3,
        late_timestep_threshold: float = 0.7,
        
        # Stability parameters
        eps: float = 1e-8,
        min_timestep: float = 1e-3,
        max_timestep: float = 1.0 - 1e-3,
        
        # Advanced features
        use_lpips_loss: bool = False,
        normalize_targets: bool = False,
    ):
        super().__init__()
        
        self.prediction_type = prediction_type
        self.flow_type = flow_type
        
        # Loss weights
        self.velocity_weight = velocity_weight
        self.semantic_weight = semantic_weight
        self.cosine_weight = cosine_weight
        
        # Timestep weighting
        self.use_timestep_weighting = use_timestep_weighting
        self.early_timestep_threshold = early_timestep_threshold
        self.late_timestep_threshold = late_timestep_threshold
        
        # Stability
        self.eps = eps
        self.min_timestep = min_timestep
        self.max_timestep = max_timestep
        
        # Advanced features
        self.use_lpips_loss = use_lpips_loss
        self.normalize_targets = normalize_targets
        
        # Validate inputs
        assert prediction_type in ["velocity", "noise", "sample"]
        assert flow_type in ["rectified", "reflow"]
        
        # Initialize LPIPS if requested
        if self.use_lpips_loss:
            try:
                import lpips
                self.lpips_fn = lpips.LPIPS(net='alex')
                logger.info("✅ LPIPS loss initialized")
            except ImportError:
                logger.warning("LPIPS not available, falling back to MSE")
                self.use_lpips_loss = False
        
        logger.info(f"✅ IMPROVED Flow Matching Loss initialized:")
        logger.info(f"  Prediction type: {prediction_type}")
        logger.info(f"  Flow type: {flow_type}")
        logger.info(f"  Weights - Velocity: {velocity_weight}, Semantic: {semantic_weight}, Cosine: {cosine_weight}")
        logger.info(f"  Timestep weighting: {use_timestep_weighting}")

    def _clamp_timesteps(self, timesteps: torch.Tensor) -> torch.Tensor:
        """Clamp timesteps to avoid numerical issues"""
        return torch.clamp(timesteps, min=self.min_timestep, max=self.max_timestep)

    def _compute_timestep_weights(self, timesteps: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute timestep-dependent weights for different loss components"""
        if not self.use_timestep_weighting:
            batch_size = timesteps.shape[0]
            device = timesteps.device
            return {
                'velocity_weights': torch.ones(batch_size, device=device),
                'semantic_weights': torch.ones(batch_size, device=device),
                'cosine_weights': torch.ones(batch_size, device=device),
            }
        
        # Velocity loss: constant throughout
        velocity_weights = torch.ones_like(timesteps)
        
        # Semantic loss: focus on later timesteps where prediction is more accurate
        semantic_weights = torch.where(
            timesteps > self.late_timestep_threshold,
            torch.ones_like(timesteps),
            torch.zeros_like(timesteps)
        )
        
        # Cosine loss: emphasize middle timesteps
        cosine_weights = torch.where(
            (timesteps > self.early_timestep_threshold) & (timesteps < self.late_timestep_threshold),
            torch.ones_like(timesteps),
            torch.ones_like(timesteps) * 0.5  # Reduced weight for early/late
        )
        
        return {
            'velocity_weights': velocity_weights,
            'semantic_weights': semantic_weights,
            'cosine_weights': cosine_weights,
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

    def forward(
        self,
        model_output: torch.Tensor,  # [B, N, 1024] - Model's prediction
        target_samples: torch.Tensor,  # [B, N, 1024] - Clean CLIP embeddings
        timesteps: torch.Tensor,  # [B] - Flow matching timesteps
        eva_conditioning: torch.Tensor,  # [B, N, 4096] - EVA features (for logging)
        noise: Optional[torch.Tensor] = None,
        noisy_input: Optional[torch.Tensor] = None,
        return_metrics: bool = True,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[Dict[str, float]]]:
        """
        Compute multi-component flow matching loss with semantic preservation
        """
        
        batch_size, num_tokens, embed_dim = model_output.shape
        device = model_output.device
        dtype = model_output.dtype
        
        # Clamp timesteps
        timesteps = self._clamp_timesteps(timesteps)
        
        # Normalize targets if requested
        if self.normalize_targets:
            target_samples = F.normalize(target_samples, p=2, dim=-1)
        
        # Use provided noise or create new
        if noise is None:
            noise = torch.randn_like(target_samples, device=device, dtype=dtype)
        
        # Expand timesteps for broadcasting [B, 1, 1]
        t = timesteps.view(batch_size, 1, 1).to(dtype)
        
        # RECTIFIED FLOW COMPUTATION
        if self.flow_type == "rectified":
            # Linear interpolation: x_t = (1-t) * noise + t * target
            if noisy_input is None:
                noisy_input = (1 - t) * noise + t * target_samples
            
            # Velocity target: v = target - noise (for rectified flow)
            true_velocity = target_samples - noise
            velocity_target = true_velocity
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
        
        # COMPONENT 2: SEMANTIC CONSISTENCY LOSS
        predicted_clean = self._compute_predicted_clean(
            model_output, noisy_input, timesteps, noise
        )
        
        # Only compute semantic loss for valid timesteps
        semantic_mask = weights['semantic_weights'] > 0
        if semantic_mask.any():
            if self.use_lpips_loss and hasattr(self, 'lpips_fn'):
                # LPIPS loss (perceptually motivated)
                try:
                    # Reshape for LPIPS: [B*N, 1, 1, D] (treating embedding as 1D image)
                    pred_clean_lpips = predicted_clean[semantic_mask].view(-1, 1, 1, embed_dim)
                    target_lpips = target_samples[semantic_mask].view(-1, 1, 1, embed_dim)
                    
                    # Normalize to [-1, 1] range for LPIPS
                    pred_clean_lpips = 2 * (pred_clean_lpips - pred_clean_lpips.min()) / (pred_clean_lpips.max() - pred_clean_lpips.min()) - 1
                    target_lpips = 2 * (target_lpips - target_lpips.min()) / (target_lpips.max() - target_lpips.min()) - 1
                    
                    semantic_loss = self.lpips_fn(pred_clean_lpips, target_lpips).mean()
                except Exception as e:
                    logger.warning(f"LPIPS computation failed: {e}, falling back to MSE")
                    semantic_loss = F.mse_loss(
                        predicted_clean[semantic_mask], 
                        target_samples[semantic_mask]
                    )
            else:
                # Standard MSE loss
                semantic_loss = F.mse_loss(
                    predicted_clean[semantic_mask], 
                    target_samples[semantic_mask]
                )
        else:
            semantic_loss = torch.tensor(0.0, device=device, dtype=dtype)
        
        # COMPONENT 3: COSINE SIMILARITY PRESERVATION
        cosine_mask = weights['cosine_weights'] > 0
        if cosine_mask.any():
            pred_norm = F.normalize(predicted_clean[cosine_mask], p=2, dim=-1)
            target_norm = F.normalize(target_samples[cosine_mask], p=2, dim=-1)
            cosine_sim = F.cosine_similarity(pred_norm, target_norm, dim=-1)
            cosine_loss = 1.0 - cosine_sim.mean()
        else:
            cosine_loss = torch.tensor(0.0, device=device, dtype=dtype)
        
        # TOTAL LOSS COMBINATION
        total_loss = (
            self.velocity_weight * velocity_loss + 
            self.semantic_weight * semantic_loss +
            self.cosine_weight * cosine_loss
        )
        
        # METRICS COMPUTATION
        metrics = {}
        if return_metrics:
            with torch.no_grad():
                # Normalize for similarity computation
                pred_normalized = F.normalize(model_output, p=2, dim=-1)
                target_norm = F.normalize(velocity_target, p=2, dim=-1)
                
                # Velocity cosine similarity
                velocity_cosine_sim = F.cosine_similarity(pred_normalized, target_norm, dim=-1)
                per_image_velocity_sim = velocity_cosine_sim.mean(dim=1)  # [B]
                mean_velocity_similarity = per_image_velocity_sim.mean().item()
                
                # Predicted clean embedding similarity
                if semantic_mask.any():
                    pred_clean_norm = F.normalize(predicted_clean, p=2, dim=-1)
                    target_clean_norm = F.normalize(target_samples, p=2, dim=-1)
                    clean_cosine_sim = F.cosine_similarity(pred_clean_norm, target_clean_norm, dim=-1)
                    per_image_clean_sim = clean_cosine_sim.mean(dim=1)  # [B]
                    mean_clean_similarity = per_image_clean_sim.mean().item()
                else:
                    mean_clean_similarity = 0.0
                
                # Compute norms for monitoring
                pred_norm_val = torch.norm(model_output, dim=-1).mean().item()
                target_norm_val = torch.norm(velocity_target, dim=-1).mean().item()
                clean_pred_norm = torch.norm(predicted_clean, dim=-1).mean().item()
                clean_target_norm = torch.norm(target_samples, dim=-1).mean().item()
                
                # Error analysis
                velocity_error = model_output - velocity_target
                velocity_error_norm = torch.norm(velocity_error, dim=-1).mean().item()
                velocity_relative_error = velocity_error_norm / (target_norm_val + self.eps)
                
                # Clean embedding error
                clean_error = predicted_clean - target_samples
                clean_error_norm = torch.norm(clean_error, dim=-1).mean().item()
                clean_relative_error = clean_error_norm / (clean_target_norm + self.eps)
                
                # Timestep analysis
                active_semantic_ratio = semantic_mask.float().mean().item()
                active_cosine_ratio = cosine_mask.float().mean().item()
                
                metrics = {
                    # Main losses
                    'velocity_loss': velocity_loss.item(),
                    'semantic_loss': semantic_loss.item() if isinstance(semantic_loss, torch.Tensor) else 0.0,
                    'cosine_loss': cosine_loss.item() if isinstance(cosine_loss, torch.Tensor) else 0.0,
                    'total_loss': total_loss.item(),
                    
                    # Velocity metrics
                    'velocity_similarity': mean_velocity_similarity,
                    'velocity_similarity_std': per_image_velocity_sim.std().item(),
                    'velocity_error_norm': velocity_error_norm,
                    'velocity_relative_error': velocity_relative_error,
                    
                    # Clean embedding metrics (CRITICAL FOR EVALUATION)
                    'clean_similarity': mean_clean_similarity,
                    'clean_error_norm': clean_error_norm,
                    'clean_relative_error': clean_relative_error,
                    
                    # Norm tracking
                    'pred_velocity_norm': pred_norm_val,
                    'target_velocity_norm': target_norm_val,
                    'pred_clean_norm': clean_pred_norm,
                    'target_clean_norm': clean_target_norm,
                    
                    # Timestep analysis
                    'timestep_mean': timesteps.mean().item(),
                    'timestep_std': timesteps.std().item(),
                    'active_semantic_ratio': active_semantic_ratio,
                    'active_cosine_ratio': active_cosine_ratio,
                    
                    # Weight tracking
                    'velocity_weight_mean': weights['velocity_weights'].mean().item(),
                    'semantic_weight_mean': weights['semantic_weights'].mean().item(),
                    'cosine_weight_mean': weights['cosine_weights'].mean().item(),
                }
                
                # Check for numerical issues
                if torch.isnan(total_loss) or torch.isinf(total_loss):
                    metrics['numerical_issue'] = True
                    logger.error("❌ Numerical issue detected in loss computation!")
                    
                    # Debug info
                    logger.error(f"  Velocity loss: {velocity_loss.item()}")
                    logger.error(f"  Semantic loss: {semantic_loss.item() if isinstance(semantic_loss, torch.Tensor) else 0.0}")
                    logger.error(f"  Cosine loss: {cosine_loss.item() if isinstance(cosine_loss, torch.Tensor) else 0.0}")
        
        return total_loss, metrics

    def compute_eval_loss(
        self,
        generated: torch.Tensor,  # Generated CLIP embeddings (potentially normalized)
        target: torch.Tensor,     # Target CLIP embeddings (potentially normalized)
        denormalize_fn: Optional[callable] = None,  # Function to denormalize embeddings
    ) -> Dict[str, float]:
        """
        Compute comprehensive evaluation metrics for generated embeddings
        
        Args:
            generated: Generated embeddings [B, N, D]
            target: Target embeddings [B, N, D]
            denormalize_fn: Optional function to convert to original CLIP space
        """
        with torch.no_grad():
            eval_metrics = {}
            
            # Denormalize if function provided
            if denormalize_fn is not None:
                try:
                    generated_denorm = denormalize_fn(generated)
                    target_denorm = denormalize_fn(target)
                    
                    # Compute metrics in original space
                    eval_metrics.update(self._compute_similarity_metrics(
                        generated_denorm, target_denorm, prefix="denorm_"
                    ))
                except Exception as e:
                    logger.warning(f"Denormalization failed: {e}")
            
            # Compute metrics in current space (normalized)
            eval_metrics.update(self._compute_similarity_metrics(
                generated, target, prefix="norm_"
            ))
            
            # Use denormalized metrics as primary if available, otherwise normalized
            primary_prefix = "denorm_" if denormalize_fn is not None and f"denorm_clip_similarity" in eval_metrics else "norm_"
            
            # Set primary evaluation metrics
            eval_metrics.update({
                'eval_clip_similarity': eval_metrics[f'{primary_prefix}clip_similarity'],
                'eval_mse_loss': eval_metrics[f'{primary_prefix}mse_loss'],
                'eval_high_quality': eval_metrics[f'{primary_prefix}high_quality'],
                'eval_very_high_quality': eval_metrics[f'{primary_prefix}very_high_quality'],
                'eval_excellent_quality': eval_metrics[f'{primary_prefix}excellent_quality'],
                'eval_similarity_std': eval_metrics[f'{primary_prefix}similarity_std'],
                'eval_generated_norm': eval_metrics[f'{primary_prefix}generated_norm'],
                'eval_target_norm': eval_metrics[f'{primary_prefix}target_norm'],
                'eval_norm_ratio': eval_metrics[f'{primary_prefix}norm_ratio'],
            })
            
            return eval_metrics
    
    def _compute_similarity_metrics(
        self, 
        generated: torch.Tensor, 
        target: torch.Tensor, 
        prefix: str = ""
    ) -> Dict[str, float]:
        """Compute similarity metrics between generated and target embeddings"""
        
        # Normalize for cosine similarity computation
        generated_norm = F.normalize(generated, p=2, dim=-1)
        target_norm = F.normalize(target, p=2, dim=-1)
        
        # Cosine similarity
        similarity = F.cosine_similarity(generated_norm, target_norm, dim=-1)
        per_image_sim = similarity.mean(dim=1)
        
        # MSE loss in current space
        mse_loss = F.mse_loss(generated, target)
        
        # Quality metrics
        high_quality = (per_image_sim > 0.7).float().mean().item()
        very_high_quality = (per_image_sim > 0.8).float().mean().item()
        excellent_quality = (per_image_sim > 0.9).float().mean().item()
        
        # Norm analysis
        generated_norm_val = torch.norm(generated, dim=-1).mean().item()
        target_norm_val = torch.norm(target, dim=-1).mean().item()
        
        # L2 distance
        l2_distance = torch.norm(generated - target, p=2, dim=-1).mean().item()
        
        # Dot product (unnormalized similarity)
        dot_product = (generated * target).sum(dim=-1).mean().item()
        
        return {
            f'{prefix}clip_similarity': per_image_sim.mean().item(),
            f'{prefix}mse_loss': mse_loss.item(),
            f'{prefix}high_quality': high_quality,
            f'{prefix}very_high_quality': very_high_quality,
            f'{prefix}excellent_quality': excellent_quality,
            f'{prefix}similarity_std': per_image_sim.std().item(),
            f'{prefix}generated_norm': generated_norm_val,
            f'{prefix}target_norm': target_norm_val,
            f'{prefix}norm_ratio': generated_norm_val / (target_norm_val + 1e-8),
            f'{prefix}l2_distance': l2_distance,
            f'{prefix}dot_product': dot_product,
        }


def create_improved_clip_reproduction_loss(
    prediction_type: str = "velocity",
    flow_type: str = "rectified", 
    velocity_weight: float = 1.0,
    semantic_weight: float = 0.5,
    cosine_weight: float = 0.2,
    use_timestep_weighting: bool = True,
    **kwargs
) -> SemanticPreservingFlowMatchingLoss:
    """Factory function for improved CLIP reproduction loss"""
    
    return SemanticPreservingFlowMatchingLoss(
        prediction_type=prediction_type,
        flow_type=flow_type,
        velocity_weight=velocity_weight,
        semantic_weight=semantic_weight,
        cosine_weight=cosine_weight,
        use_timestep_weighting=use_timestep_weighting,
        **kwargs
    )


# Backward compatibility aliases
BLIP3oCLIPFlowMatchingLoss = SemanticPreservingFlowMatchingLoss
create_clip_reproduction_loss = create_improved_clip_reproduction_loss