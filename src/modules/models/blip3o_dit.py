#!/usr/bin/env python3
"""
CORRECTED BLIP3-o DiT Model for CLIP Reproduction with SwiGLU
Now properly aligned with Lumina DiT architecture from BLIP3-O paper
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, Tuple, Union
import math
import logging
from transformers import PreTrainedModel, PretrainedConfig

logger = logging.getLogger(__name__)


class BLIP3oCLIPDiTConfig(PretrainedConfig):
    """Configuration for BLIP3-o CLIP DiT model"""
    model_type = "blip3o_clip_dit"
    
    def __init__(
        self,
        hidden_size: int = 768,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        num_key_value_heads: int = 4,
        intermediate_size: int = 3072,  # For SwiGLU, typically hidden_size * 8 // 3
        eva_embedding_size: int = 4096,
        clip_embedding_size: int = 1024,
        num_tokens: int = 256,
        max_position_embeddings: int = 256,
        # 3D RoPE parameters
        use_3d_rope: bool = True,
        rope_theta: float = 10000.0,
        image_size: int = 224,
        patch_size: int = 14,
        # Sandwich normalization
        use_sandwich_norm: bool = True,
        rms_norm_eps: float = 1e-6,
        dropout_prob: float = 0.0,
        attention_dropout: float = 0.0,
        initializer_range: float = 0.02,
        use_gradient_checkpointing: bool = False,
        training_mode: str = "patch_only",
        **kwargs
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.intermediate_size = intermediate_size
        self.eva_embedding_size = eva_embedding_size
        self.clip_embedding_size = clip_embedding_size
        self.num_tokens = num_tokens
        self.max_position_embeddings = max_position_embeddings
        
        # 3D RoPE
        self.use_3d_rope = use_3d_rope
        self.rope_theta = rope_theta
        self.image_size = image_size
        self.patch_size = patch_size
        
        # Sandwich normalization
        self.use_sandwich_norm = use_sandwich_norm
        self.rms_norm_eps = rms_norm_eps
        
        self.dropout_prob = dropout_prob
        self.attention_dropout = attention_dropout
        self.initializer_range = initializer_range
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.training_mode = training_mode
        
        # Calculate grid size for 3D RoPE
        self.grid_size = image_size // patch_size  # 224 // 14 = 16


class RMSNorm(nn.Module):
    """RMS Normalization"""
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class Rotary3DEmbedding(nn.Module):
    """3D Rotary Position Embedding for BLIP3-o"""
    def __init__(
        self, 
        dim: int, 
        grid_size: int = 16,
        max_position_embeddings: int = 256, 
        base: float = 10000,
        use_3d: bool = True
    ):
        super().__init__()
        self.dim = dim
        self.grid_size = grid_size
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.use_3d = use_3d
        
        if use_3d:
            assert dim % 4 == 0, "Dimension must be divisible by 4 for 3D RoPE"
            self.dim_h = dim // 4
            self.dim_w = dim // 4
            self.dim_d = dim // 2
        else:
            self.dim_h = dim // 2
            self.dim_w = dim // 2
            self.dim_d = 0
        
        self._create_frequency_tensors()

    def _create_frequency_tensors(self):
        """Create frequency tensors for each spatial dimension"""
        if self.use_3d:
            inv_freq_h = 1.0 / (self.base ** (torch.arange(0, self.dim_h, 2).float() / self.dim_h))
            self.register_buffer("inv_freq_h", inv_freq_h, persistent=False)
            
            inv_freq_w = 1.0 / (self.base ** (torch.arange(0, self.dim_w, 2).float() / self.dim_w))
            self.register_buffer("inv_freq_w", inv_freq_w, persistent=False)
            
            inv_freq_d = 1.0 / (self.base ** (torch.arange(0, self.dim_d, 2).float() / self.dim_d))
            self.register_buffer("inv_freq_d", inv_freq_d, persistent=False)
        else:
            inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
            self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x: torch.Tensor, seq_len: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if seq_len is None:
            seq_len = x.shape[1]
        
        device = x.device
        
        if not self.use_3d:
            t = torch.arange(seq_len, device=device, dtype=torch.float32)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            return emb.cos().unsqueeze(0), emb.sin().unsqueeze(0)
        
        batch_size = x.shape[0]
        has_cls = seq_len == self.grid_size * self.grid_size + 1
        start_idx = 1 if has_cls else 0
        spatial_len = seq_len - start_idx
        
        pos_embeddings = []
        
        if has_cls:
            cls_emb = torch.zeros(1, self.dim, device=device)
            pos_embeddings.append(cls_emb)
        
        grid_h, grid_w = int(math.sqrt(spatial_len)), int(math.sqrt(spatial_len))
        
        pos_h = torch.arange(grid_h, device=device, dtype=torch.float32)
        freqs_h = torch.einsum("i,j->ij", pos_h, self.inv_freq_h)
        
        pos_w = torch.arange(grid_w, device=device, dtype=torch.float32)
        freqs_w = torch.einsum("i,j->ij", pos_w, self.inv_freq_w)
        
        depth_scale = torch.zeros(1, device=device, dtype=torch.float32)
        freqs_d = torch.einsum("i,j->ij", depth_scale, self.inv_freq_d)
        
        for h in range(grid_h):
            for w in range(grid_w):
                h_emb = torch.cat((freqs_h[h], freqs_h[h]), dim=-1)
                w_emb = torch.cat((freqs_w[w], freqs_w[w]), dim=-1)
                d_emb = torch.cat((freqs_d[0], freqs_d[0]), dim=-1)
                
                combined_emb = torch.cat([h_emb, w_emb, d_emb], dim=0)
                pos_embeddings.append(combined_emb)
        
        all_pos_emb = torch.stack(pos_embeddings, dim=0)
        cos_emb = all_pos_emb.cos().unsqueeze(0)
        sin_emb = all_pos_emb.sin().unsqueeze(0)
        
        return cos_emb, sin_emb


def apply_rotary_pos_emb_3d(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """Apply 3D rotary position embedding"""
    def rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class TimestepEmbedder(nn.Module):
    """Timestep embedding for BLIP3-o"""
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.hidden_size = hidden_size
        self.frequency_embedding_size = frequency_embedding_size
        
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        
        nn.init.xavier_uniform_(self.mlp[0].weight)
        nn.init.zeros_(self.mlp[0].bias)
        nn.init.xavier_uniform_(self.mlp[2].weight)
        nn.init.zeros_(self.mlp[2].bias)

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        """Create sinusoidal timestep embeddings"""
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq)


class Attention3D(nn.Module):
    """Multi-head attention with 3D RoPE for BLIP3-o"""
    def __init__(self, config: BLIP3oCLIPDiTConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        
        assert self.hidden_size % self.num_heads == 0
        
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        
        self.rotary_emb = Rotary3DEmbedding(
            self.head_dim,
            grid_size=config.grid_size,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
            use_3d=config.use_3d_rope,
        )
        
        self.dropout = nn.Dropout(config.attention_dropout)
        self._init_weights()
    
    def _init_weights(self):
        """Initialize attention weights"""
        for module in [self.q_proj, self.k_proj, self.v_proj]:
            nn.init.xavier_uniform_(module.weight)
        nn.init.xavier_uniform_(self.o_proj.weight, gain=1.0 / math.sqrt(self.config.num_hidden_layers))
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        bsz, q_len, _ = hidden_states.size()
        
        if key_value_states is not None:
            kv_seq_len = key_value_states.shape[1]
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(key_value_states)
            value_states = self.v_proj(key_value_states)
        else:
            kv_seq_len = q_len
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)
        
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, kv_seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, kv_seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        
        if key_value_states is None:
            cos, sin = self.rotary_emb(hidden_states, seq_len=kv_seq_len)
            query_states, key_states = apply_rotary_pos_emb_3d(query_states, key_states, cos, sin)
        
        key_states = key_states.repeat_interleave(self.num_key_value_groups, dim=1)
        value_states = value_states.repeat_interleave(self.num_key_value_groups, dim=1)
        
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = self.dropout(attn_weights)
        
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, self.hidden_size)
        
        return self.o_proj(attn_output)


class SwiGLUMLP(nn.Module):
    """
    CORRECTED: SwiGLU MLP to match Lumina DiT architecture
    
    SwiGLU formula: SwiGLU(x) = SiLU(x @ W_gate) ⊗ (x @ W_up) @ W_down
    where ⊗ is element-wise multiplication (gating mechanism)
    
    This uses 3 linear transformations instead of 2, following the Lumina DiT specification.
    """
    def __init__(self, config: BLIP3oCLIPDiTConfig):
        super().__init__()
        
        # SwiGLU requires 3 linear layers
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)  # W_gate
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)    # W_up  
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)  # W_down
        
        # SwiGLU uses SiLU (Swish with β=1) as the activation function
        self.act_fn = nn.SiLU()
        self.dropout = nn.Dropout(config.dropout_prob)
        
        # Initialize weights following Lumina DiT practices
        nn.init.xavier_uniform_(self.gate_proj.weight)
        nn.init.xavier_uniform_(self.up_proj.weight)
        nn.init.xavier_uniform_(self.down_proj.weight, gain=1.0 / math.sqrt(config.num_hidden_layers))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        SwiGLU forward pass:
        1. Gate path: x -> W_gate -> SiLU activation
        2. Up path: x -> W_up (no activation)  
        3. Element-wise multiplication (gating): gate ⊗ up
        4. Down projection: gated -> W_down -> output
        """
        # Gate branch with SiLU activation
        gate = self.act_fn(self.gate_proj(x))
        
        # Up branch without activation
        up = self.up_proj(x)
        
        # Element-wise multiplication (the "gating" mechanism)
        gated = gate * up
        
        # Final down projection
        return self.dropout(self.down_proj(gated))


class AdaLN(nn.Module):
    """Adaptive Layer Normalization for timestep conditioning"""
    def __init__(self, hidden_size: int, conditioning_size: int, eps: float = 1e-6):
        super().__init__()
        self.norm = RMSNorm(hidden_size, eps)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(conditioning_size, 2 * hidden_size, bias=True)
        )
        
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(self, x: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
        if conditioning.dim() == 2:
            conditioning = conditioning.unsqueeze(1)
        
        shift, scale = self.adaLN_modulation(conditioning).chunk(2, dim=-1)
        
        if shift.shape[1] == 1 and x.shape[1] > 1:
            shift = shift.expand(-1, x.shape[1], -1)
            scale = scale.expand(-1, x.shape[1], -1)
        
        normalized = self.norm(x)
        return normalized * (1 + scale) + shift


class DiTBlock3D(nn.Module):
    """BLIP3-o DiT transformer block with 3D RoPE, Sandwich Normalization, and SwiGLU"""
    def __init__(self, config: BLIP3oCLIPDiTConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.use_sandwich_norm = config.use_sandwich_norm
        
        self.self_attn = Attention3D(config)
        self.cross_attn = Attention3D(config)
        
        # CORRECTED: Use SwiGLU MLP instead of regular MLP
        self.mlp = SwiGLUMLP(config)
        
        self.eva_proj = nn.Linear(config.eva_embedding_size, config.hidden_size, bias=True)
        nn.init.xavier_uniform_(self.eva_proj.weight)
        nn.init.zeros_(self.eva_proj.bias)
        
        if config.use_sandwich_norm:
            self.self_attn_pre_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.self_attn_post_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.self_attn_ada_ln_pre = AdaLN(config.hidden_size, config.hidden_size)
            self.self_attn_ada_ln_post = AdaLN(config.hidden_size, config.hidden_size)
            
            self.cross_attn_pre_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.cross_attn_post_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.cross_attn_ada_ln_pre = AdaLN(config.hidden_size, config.hidden_size)
            self.cross_attn_ada_ln_post = AdaLN(config.hidden_size, config.hidden_size)
            
            self.mlp_pre_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.mlp_post_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.mlp_ada_ln_pre = AdaLN(config.hidden_size, config.hidden_size)
            self.mlp_ada_ln_post = AdaLN(config.hidden_size, config.hidden_size)
        else:
            self.norm1 = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.norm2 = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.norm3 = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.ada_ln1 = AdaLN(config.hidden_size, config.hidden_size)
            self.ada_ln2 = AdaLN(config.hidden_size, config.hidden_size)
            self.ada_ln3 = AdaLN(config.hidden_size, config.hidden_size)

    def forward(
        self, 
        hidden_states: torch.Tensor, 
        encoder_hidden_states: torch.Tensor,
        timestep_emb: torch.Tensor
    ) -> torch.Tensor:
        
        if self.use_sandwich_norm:
            # Self-attention with sandwich norm
            residual = hidden_states
            hidden_states = self.self_attn_pre_norm(hidden_states)
            hidden_states = self.self_attn_ada_ln_pre(hidden_states, timestep_emb)
            hidden_states = self.self_attn(hidden_states)
            hidden_states = self.self_attn_post_norm(hidden_states)
            hidden_states = self.self_attn_ada_ln_post(hidden_states, timestep_emb)
            hidden_states = residual + hidden_states
            
            # Cross-attention with sandwich norm
            residual = hidden_states
            hidden_states = self.cross_attn_pre_norm(hidden_states)
            hidden_states = self.cross_attn_ada_ln_pre(hidden_states, timestep_emb)
            eva_features = self.eva_proj(encoder_hidden_states)
            hidden_states = self.cross_attn(hidden_states, key_value_states=eva_features)
            hidden_states = self.cross_attn_post_norm(hidden_states)
            hidden_states = self.cross_attn_ada_ln_post(hidden_states, timestep_emb)
            hidden_states = residual + hidden_states
            
            # SwiGLU MLP with sandwich norm
            residual = hidden_states
            hidden_states = self.mlp_pre_norm(hidden_states)
            hidden_states = self.mlp_ada_ln_pre(hidden_states, timestep_emb)
            hidden_states = self.mlp(hidden_states)  # Now uses SwiGLU
            hidden_states = self.mlp_post_norm(hidden_states)
            hidden_states = self.mlp_ada_ln_post(hidden_states, timestep_emb)
            hidden_states = residual + hidden_states
            
        else:
            # Standard pre-norm pattern
            residual = hidden_states
            hidden_states = self.norm1(hidden_states)
            hidden_states = self.ada_ln1(hidden_states, timestep_emb)
            hidden_states = self.self_attn(hidden_states)
            hidden_states = residual + hidden_states
            
            residual = hidden_states
            hidden_states = self.norm2(hidden_states)
            hidden_states = self.ada_ln2(hidden_states, timestep_emb)
            eva_features = self.eva_proj(encoder_hidden_states)
            hidden_states = self.cross_attn(hidden_states, key_value_states=eva_features)
            hidden_states = residual + hidden_states
            
            residual = hidden_states
            hidden_states = self.norm3(hidden_states)
            hidden_states = self.ada_ln3(hidden_states, timestep_emb)
            hidden_states = self.mlp(hidden_states)  # Now uses SwiGLU
            hidden_states = residual + hidden_states
        
        return hidden_states


class BLIP3oCLIPDiTModel(PreTrainedModel):
    """
    CORRECTED BLIP3-o DiT Model with SwiGLU - Now Fully Aligned with Lumina DiT Architecture
    
    Key corrections:
    - ✅ 3D Rotary Position Embedding
    - ✅ Sandwich Normalization (RMSNorm)  
    - ✅ Grouped-Query Attention
    - ✅ SwiGLU activation (CORRECTED!)
    - ✅ Rectified Flow Matching
    - ✅ Cross-attention conditioning
    """
    
    config_class = BLIP3oCLIPDiTConfig
    supports_gradient_checkpointing = True
    
    def __init__(self, config: BLIP3oCLIPDiTConfig):
        super().__init__(config)
        self.config = config
        self.gradient_checkpointing = False
        
        self.input_proj = nn.Linear(config.clip_embedding_size, config.hidden_size, bias=True)
        self.timestep_embedder = TimestepEmbedder(config.hidden_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, config.max_position_embeddings, config.hidden_size))
        
        self.blocks = nn.ModuleList([
            DiTBlock3D(config) for _ in range(config.num_hidden_layers)
        ])
        
        if config.use_sandwich_norm:
            self.output_pre_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.output_adaln_pre = AdaLN(config.hidden_size, config.hidden_size)
        else:
            self.output_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.output_adaln = AdaLN(config.hidden_size, config.hidden_size)
        
        self.output_proj = nn.Linear(config.hidden_size, config.clip_embedding_size, bias=True)
        
        self._init_weights()
        
        logger.info(f"CORRECTED BLIP3-o CLIP DiT model with SwiGLU initialized: {self.get_num_parameters():,} parameters")
        logger.info(f"  ✅ 3D RoPE: {config.use_3d_rope}")
        logger.info(f"  ✅ Sandwich Normalization: {config.use_sandwich_norm}")
        logger.info(f"  ✅ Grouped-Query Attention: {config.num_attention_heads}/{config.num_key_value_heads} heads")
        logger.info(f"  ✅ SwiGLU activation: Now properly implemented!")

    def _init_weights(self):
        """Initialize model weights"""
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)
        nn.init.normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.output_proj.weight, std=0.02)
        nn.init.zeros_(self.output_proj.bias)

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.gradient_checkpointing = True

    def gradient_checkpointing_disable(self):
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        return_dict: bool = True,
        **kwargs
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward pass"""
        batch_size, seq_len, _ = hidden_states.shape
        
        x = self.input_proj(hidden_states)
        
        if not self.config.use_3d_rope and seq_len <= self.config.max_position_embeddings:
            x = x + self.pos_embed[:, :seq_len, :]
        
        timestep_emb = self.timestep_embedder(timestep)
        
        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(
                    block, x, encoder_hidden_states, timestep_emb, use_reentrant=False
                )
            else:
                x = block(x, encoder_hidden_states, timestep_emb)
        
        if self.config.use_sandwich_norm:
            x = self.output_pre_norm(x)
            x = self.output_adaln_pre(x, timestep_emb)
            velocity_pred = self.output_proj(x)
        else:
            x = self.output_norm(x)
            x = self.output_adaln(x, timestep_emb)
            velocity_pred = self.output_proj(x)
        
        if return_dict:
            return {"velocity_prediction": velocity_pred, "hidden_states": x}
        return velocity_pred

    @torch.no_grad()
    def generate(
        self,
        eva_features: torch.Tensor,
        num_inference_steps: int = 50,
        generator: Optional[torch.Generator] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Clean generation with simple linear timestep schedule and Euler steps
        Following BLIP3-o paper implementation
        """
        device = eva_features.device
        batch_size, num_tokens, _ = eva_features.shape
        
        # Start from noise
        x = torch.randn(
            batch_size, num_tokens, self.config.clip_embedding_size,
            device=device, generator=generator, dtype=eva_features.dtype
        )
        
        # Simple linear timestep schedule from 1 to 0
        timesteps = torch.linspace(1.0, 0.0, num_inference_steps + 1, device=device)[:-1]
        
        # Forward ODE integration with Euler method
        for i, t in enumerate(timesteps):
            t_batch = torch.full((batch_size,), t.item(), device=device, dtype=eva_features.dtype)
            
            # Get velocity prediction
            velocity = self.forward(
                hidden_states=x,
                timestep=t_batch,
                encoder_hidden_states=eva_features,
                return_dict=False
            )
            
            # Compute step size
            if i < len(timesteps) - 1:
                dt = timesteps[i] - timesteps[i + 1]
            else:
                dt = timesteps[i]
            
            # Euler step: x_{t+dt} = x_t + dt * v_t
            x = x + dt * velocity
        
        return x
    
    def get_num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_clip_reproduction_model(
    config: Optional[BLIP3oCLIPDiTConfig] = None,
    training_mode: str = "patch_only",
    model_size: str = "base",
    use_3d_rope: bool = True,
    use_sandwich_norm: bool = True,
    **kwargs
) -> BLIP3oCLIPDiTModel:
    """Create CORRECTED CLIP reproduction model with SwiGLU"""
    
    if config is None:
        size_configs = {
            "tiny": {
                "hidden_size": 384, 
                "num_hidden_layers": 6, 
                "num_attention_heads": 6, 
                "num_key_value_heads": 2,
                "intermediate_size": 384 * 8 // 3  # Adjusted for SwiGLU (3 linear layers)
            },
            "small": {
                "hidden_size": 512, 
                "num_hidden_layers": 8, 
                "num_attention_heads": 8, 
                "num_key_value_heads": 4,
                "intermediate_size": 512 * 8 // 3  # Adjusted for SwiGLU
            },
            "base": {
                "hidden_size": 768, 
                "num_hidden_layers": 12, 
                "num_attention_heads": 12, 
                "num_key_value_heads": 4,
                "intermediate_size": 768 * 8 // 3  # Adjusted for SwiGLU (= 2048)
            },
            "large": {
                "hidden_size": 1024, 
                "num_hidden_layers": 16, 
                "num_attention_heads": 16, 
                "num_key_value_heads": 8,
                "intermediate_size": 1024 * 8 // 3  # Adjusted for SwiGLU
            },
        }
        
        model_config = size_configs[model_size].copy()
        model_config.update({
            "num_tokens": 257 if training_mode == "cls_patch" else 256,
            "training_mode": training_mode,
            "eva_embedding_size": 4096,
            "clip_embedding_size": 1024,
            "use_3d_rope": use_3d_rope,
            "use_sandwich_norm": use_sandwich_norm,
            **kwargs
        })
        
        config = BLIP3oCLIPDiTConfig(**model_config)
    
    return BLIP3oCLIPDiTModel(config)