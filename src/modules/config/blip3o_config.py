"""
Clean BLIP3-o Configuration for CLIP Reproduction
Simplified configuration aligned with BLIP3-o paper
"""

from transformers import PretrainedConfig
from typing import Dict, Any, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


class BLIP3oCLIPDiTConfig(PretrainedConfig):
    """
    Clean configuration class for BLIP3-o CLIP reproduction DiT model.
    
    This configuration follows the BLIP3-o paper architecture with:
    - Patch-level training on 256 EVA tokens (4096-dim)
    - CLIP embedding reproduction (256 tokens, 1024-dim)  
    - Flow matching training objective
    - 3D RoPE and Grouped-Query Attention
    - Sandwich normalization (RMSNorm)
    """
    
    model_type = "blip3o_clip_dit"
    
    def __init__(
        self,
        # Model architecture
        hidden_size: int = 768,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        num_key_value_heads: int = 4,  # For grouped-query attention
        intermediate_size: int = 3072,
        
        # Input/output dimensions (BLIP3-o specific)
        eva_embedding_size: int = 4096,  # EVA-CLIP dimension (conditioning)
        clip_embedding_size: int = 1024,  # CLIP dimension (target)
        num_tokens: int = 256,  # 16x16 = 256 patches (or 257 with CLS)
        
        # Training configuration
        max_position_embeddings: int = 256,  # 16x16 patches
        dropout_prob: float = 0.0,  # Disabled for better training
        
        # Normalization (BLIP3-o uses RMSNorm)
        rms_norm_eps: float = 1e-6,
        use_rms_norm: bool = True,
        
        # Attention configuration (BLIP3-o specific)
        attention_dropout: float = 0.0,
        use_3d_rope: bool = True,  # 3D Rotary Position Embedding
        rope_theta: float = 10000.0,
        image_size: int = 224,
        patch_size: int = 14,
        
        # Flow matching parameters
        prediction_type: str = "velocity",
        
        # Training optimizations
        use_gradient_checkpointing: bool = False,
        training_mode: str = "patch_only",
        zero_init_output: bool = True,  # Zero init output for flow matching
        
        # BLIP3-o specific features
        use_sandwich_norm: bool = True,  # Sandwich normalization
        use_grouped_query_attention: bool = True,
        
        **kwargs,
    ):
        super().__init__(**kwargs)
        
        # Core architecture
        self.hidden_size = int(hidden_size)
        self.num_hidden_layers = int(num_hidden_layers)
        self.num_attention_heads = int(num_attention_heads)
        self.num_key_value_heads = int(num_key_value_heads)
        self.intermediate_size = int(intermediate_size)
        
        # Input/output dimensions
        self.eva_embedding_size = int(eva_embedding_size)
        self.clip_embedding_size = int(clip_embedding_size)
        self.num_tokens = int(num_tokens)
        
        # Training configuration
        self.max_position_embeddings = int(max_position_embeddings)
        self.dropout_prob = float(dropout_prob)
        
        # Normalization
        self.rms_norm_eps = float(rms_norm_eps)
        self.use_rms_norm = bool(use_rms_norm)
        
        # Attention
        self.attention_dropout = float(attention_dropout)
        self.use_3d_rope = bool(use_3d_rope)
        self.rope_theta = float(rope_theta)
        self.image_size = int(image_size)
        self.patch_size = int(patch_size)
        
        # Flow matching
        self.prediction_type = str(prediction_type)
        
        # Training optimizations
        self.use_gradient_checkpointing = bool(use_gradient_checkpointing)
        self.training_mode = str(training_mode)
        self.zero_init_output = bool(zero_init_output)
        
        # BLIP3-o specific
        self.use_sandwich_norm = bool(use_sandwich_norm)
        self.use_grouped_query_attention = bool(use_grouped_query_attention)
        
        # Calculate grid size for 3D RoPE
        self.grid_size = self.image_size // self.patch_size  # 224 // 14 = 16
        
        # Validate configuration
        self._validate_config()
    
    def _validate_config(self):
        """Validate configuration parameters"""
        validation_errors = []
        
        # Check head dimension compatibility
        if self.hidden_size % self.num_attention_heads != 0:
            validation_errors.append(
                f"hidden_size ({self.hidden_size}) must be divisible by "
                f"num_attention_heads ({self.num_attention_heads})"
            )
        
        # Check grouped-query attention compatibility
        if self.use_grouped_query_attention:
            if self.num_attention_heads % self.num_key_value_heads != 0:
                validation_errors.append(
                    f"num_attention_heads ({self.num_attention_heads}) must be divisible by "
                    f"num_key_value_heads ({self.num_key_value_heads}) for grouped-query attention"
                )
        
        # Check number of tokens
        if self.num_tokens not in [256, 257]:
            validation_errors.append(f"num_tokens must be 256 or 257, got {self.num_tokens}")
        
        # Check prediction type
        if self.prediction_type not in ["velocity", "epsilon"]:
            validation_errors.append(f"prediction_type must be 'velocity' or 'epsilon', got {self.prediction_type}")
        
        # Validate embedding dimensions
        if self.eva_embedding_size <= 0:
            validation_errors.append(f"eva_embedding_size must be positive, got {self.eva_embedding_size}")
        if self.clip_embedding_size <= 0:
            validation_errors.append(f"clip_embedding_size must be positive, got {self.clip_embedding_size}")
        
        # Raise error if validation fails
        if validation_errors:
            error_msg = "Configuration validation failed:\n" + "\n".join(f"  • {err}" for err in validation_errors)
            logger.error(f"❌ {error_msg}")
            raise ValueError(error_msg)
        
        logger.info("✅ Configuration validation passed")


def get_blip3o_clip_config(
    model_size: str = "base",
    training_mode: str = "patch_only",
    **kwargs
) -> BLIP3oCLIPDiTConfig:
    """
    Get predefined BLIP3-o configuration for CLIP reproduction.
    
    Args:
        model_size: Model size - "tiny", "small", "base", "large"
        training_mode: "patch_only" (256 tokens) or "cls_patch" (257 tokens)
        **kwargs: Additional configuration overrides
        
    Returns:
        BLIP3oCLIPDiTConfig instance
    """
    # Predefined configurations optimized for BLIP3-o architecture
    configs = {
        "tiny": {
            "hidden_size": 384,
            "num_hidden_layers": 6,
            "num_attention_heads": 6,
            "num_key_value_heads": 2,  # Grouped-query attention
            "intermediate_size": 1536,
        },
        "small": {
            "hidden_size": 512,
            "num_hidden_layers": 8,
            "num_attention_heads": 8,
            "num_key_value_heads": 4,
            "intermediate_size": 2048,
        },
        "base": {
            "hidden_size": 768,
            "num_hidden_layers": 12,
            "num_attention_heads": 12,
            "num_key_value_heads": 4,
            "intermediate_size": 3072,
        },
        "large": {
            "hidden_size": 1024,
            "num_hidden_layers": 16,
            "num_attention_heads": 16,
            "num_key_value_heads": 8,
            "intermediate_size": 4096,
        },
    }
    
    if model_size not in configs:
        raise ValueError(f"Unknown model size: {model_size}. Available: {list(configs.keys())}")
    
    # Get base config
    config_dict = configs[model_size].copy()
    
    # Set token count based on training mode
    config_dict["num_tokens"] = 257 if training_mode == "cls_patch" else 256
    config_dict["max_position_embeddings"] = max(config_dict["num_tokens"], 257)
    config_dict["training_mode"] = training_mode
    
    # BLIP3-o specific defaults
    config_dict.update({
        "use_3d_rope": True,
        "use_sandwich_norm": True,
        "use_grouped_query_attention": True,
        "use_rms_norm": True,
        "zero_init_output": True,
        "dropout_prob": 0.0,  # Disable dropout
        "attention_dropout": 0.0,
    })
    
    # Apply additional overrides
    config_dict.update(kwargs)
    
    return BLIP3oCLIPDiTConfig(**config_dict)


@dataclass
class FlowMatchingConfig:
    """Configuration for flow matching training for CLIP reproduction"""
    prediction_type: str = "velocity"
    flow_type: str = "rectified"
    loss_scale: float = 1.0
    
    # Stability parameters
    min_timestep: float = 1e-3
    max_timestep: float = 1.0 - 1e-3


@dataclass  
class TrainingConfig:
    """Configuration for training parameters"""
    num_epochs: int = 20
    batch_size: int = 16
    eval_batch_size: int = 16
    learning_rate: float = 1e-4  # Conservative for stability
    weight_decay: float = 0.01
    warmup_steps: int = 100
    lr_scheduler_type: str = "cosine"
    gradient_accumulation_steps: int = 2
    fp16: bool = True
    dataloader_num_workers: int = 0
    
    # Evaluation parameters
    eval_every_n_steps: int = 50
    eval_num_samples: int = 15
    eval_inference_steps: int = 20
    
    # Robustness
    skip_corrupted_samples: bool = True
    validate_tensor_shapes: bool = True
    max_grad_norm: float = 1.0


@dataclass
class EvaluationConfig:
    """Configuration for evaluation parameters"""
    eval_every_n_steps: int = 50
    eval_num_samples: int = 15
    eval_batch_size: int = 16
    eval_inference_steps: int = 20
    normalize_embeddings: bool = True
    
    # Quality thresholds for CLIP reproduction
    high_quality_threshold: float = 0.7
    very_high_quality_threshold: float = 0.8
    excellent_quality_threshold: float = 0.9


def create_config_from_args(args) -> tuple:
    """Create configurations from command line arguments"""
    
    model_config = get_blip3o_clip_config(
        model_size=getattr(args, 'model_size', 'base'),
        training_mode=getattr(args, 'training_mode', 'patch_only'),
        use_gradient_checkpointing=getattr(args, 'gradient_checkpointing', False),
    )
    
    flow_config = FlowMatchingConfig(
        prediction_type="velocity",
        flow_type="rectified",
        loss_scale=1.0,
    )
    
    training_config = TrainingConfig(
        num_epochs=getattr(args, 'num_epochs', 10),
        batch_size=getattr(args, 'batch_size', 8),
        learning_rate=getattr(args, 'learning_rate', 1e-4),
        gradient_accumulation_steps=getattr(args, 'gradient_accumulation_steps', 2),
        fp16=getattr(args, 'fp16', True),
        eval_every_n_steps=getattr(args, 'eval_every_n_steps', 50),
        eval_num_samples=getattr(args, 'eval_num_samples', 15),
        eval_inference_steps=getattr(args, 'eval_inference_steps', 20),
    )
    
    eval_config = EvaluationConfig(
        eval_every_n_steps=getattr(args, 'eval_every_n_steps', 50),
        eval_num_samples=getattr(args, 'eval_num_samples', 15),
        eval_inference_steps=getattr(args, 'eval_inference_steps', 20),
    )
    
    return model_config, flow_config, training_config, eval_config