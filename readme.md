# CLIP Reproduction from EVA Embeddings with BLIP3-o DiT

This implementation reproduces **CLIP embeddings [B, N, 1024]** from **EVA embeddings [B, N, 4096]** using a BLIP3-o Diffusion Transformer (DiT) with **minimal normalization approach**.

## 🔄 Key Change from Original Implementation

**Original**: EVA reproduction (CLIP → EVA)
- Conditioning: CLIP embeddings [B, N, 1024]
- Target: EVA embeddings [B, N, 4096]

**This Implementation**: CLIP reproduction (EVA → CLIP)
- Conditioning: EVA embeddings [B, N, 4096] 
- Target: CLIP embeddings [B, N, 1024]

## 🚫 Minimal Normalization Approach

Unlike the original implementation, this version uses **minimal normalization**:
- **During Training**: No L2 normalization of embeddings
- **During Evaluation**: Normalization ONLY for cosine similarity computation
- **Philosophy**: Let the model learn natural embedding distributions

This approach tests whether the model can learn effectively in raw embedding space without forced normalization constraints.

## 📁 File Structure

```
├── blip3o_clip_dit.py           # BLIP3-o DiT model for CLIP reproduction
├── blip3o_clip_loss.py          # Flow matching loss with minimal normalization
├── blip3o_clip_dataset.py       # Dataset and dataloaders with no normalization
├── blip3o_clip_trainer.py       # Training loop with minimal normalization
├── blip3o_clip_config.py        # Configuration management
├── train_clip_reproduction.py   # Main training script
├── train_clip_repro.job         # SLURM job script
└── README.md                    # This file
```

## 🏗️ Architecture

**BLIP3-o DiT Features:**
- **Input**: Noisy CLIP embeddings [B, N, 1024] + timesteps
- **Conditioning**: EVA embeddings [B, N, 4096] via cross-attention
- **Output**: Velocity for CLIP embeddings [B, N, 1024]
- **Components**: 3D RoPE, Grouped-Query Attention, Sandwich Normalization (RMSNorm)

**Flow Matching:**
- **Method**: Rectified Flow 
- **Objective**: v = x₁ - x₀ (velocity field)
- **Interpolation**: x_t = (1-t) * noise + t * clean_clip

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Load modules (on Snellius)
module load 2024
module load Miniconda3/24.7.1-0
module load CUDA/12.6.0

# Activate environment
source activate eva_clip_env

# Install dependencies
pip install torch transformers datasets numpy tqdm
```

### 2. Data Preparation

Ensure you have embeddings in the expected format:
```
embeddings_dir/
├── embeddings_shard_00000_patch_only.pkl
├── embeddings_shard_00001_patch_only.pkl
└── ...
```

Each shard should contain:
```python
{
    'eva_blip3o_embeddings': torch.Tensor,    # [N, 256, 4096] - EVA embeddings
    'clip_blip3o_embeddings': torch.Tensor,   # [N, 256, 1024] - CLIP embeddings  
    'captions': List[str],                     # [N] - Captions
    'keys': List[str]                          # [N] - Sample keys
}
```

### 3. Training

**Interactive Training:**
```bash
python train_clip_reproduction.py \
    --chunked_embeddings_dir /path/to/embeddings \
    --output_dir ./checkpoints/clip_repro \
    --model_size base \
    --training_mode patch_only \
    --batch_size 8 \
    --num_epochs 10 \
    --learning_rate 5e-4 \
    --max_shards 2 \
    --overfit_test_size 20 \
    --debug_mode
```

**SLURM Job:**
```bash
sbatch train_clip_repro.job
```

### 4. Monitor Progress

Check training logs:
```bash
tail -f clip_reproduction_training.log
```

## 📊 Expected Training Behavior

### With Minimal Normalization:
- **✅ Loss**: Should decrease steadily from first epoch
- **✅ Velocity Similarity**: Increases from ~0.01 to >0.1
- **✅ Gradients**: Non-zero and stable throughout training
- **✅ CLIP Similarity**: >0.1 (good), >0.4 (excellent) during evaluation
- **✅ Embedding Norms**: Will vary naturally (not forced to 1.0)

### Overfitting Test:
If enabled with `--overfit_test_size 20`:
- Should achieve >0.8 similarity on test samples
- Validates that architecture can learn effectively

## ⚙️ Configuration Options

### Model Sizes:
- **tiny**: 384D, 6L, 6H (2M params)
- **small**: 512D, 8L, 8H (4M params) 
- **base**: 768D, 12L, 12H (8M params)
- **large**: 1024D, 16L, 16H (16M params)

### Training Modes:
- **patch_only**: 256 tokens (patches only)
- **cls_patch**: 257 tokens (CLS + patches)

### Key Hyperparameters:
```python
learning_rate: 5e-4       # Conservative for stability
batch_size: 8            # Memory-efficient  
weight_decay: 0.01       # Standard regularization
warmup_steps: 100        # Quick warmup
max_grad_norm: 1.0       # Gradient clipping
```

## 🔍 Evaluation Metrics

The model is evaluated using:

1. **Cosine Similarity** (normalized for similarity only):
   ```python
   clip_sim = F.cosine_similarity(
       F.normalize(generated_clip, p=2, dim=-1),
       F.normalize(target_clip, p=2, dim=-1), 
       dim=-1
   ).mean()
   ```

2. **Quality Thresholds**:
   - High Quality: >0.7 similarity
   - Very High Quality: >0.8 similarity
   - Excellent Quality: >0.9 similarity

3. **MSE Loss** (in raw space, no normalization)

## 🧪 Debugging and Testing

### Overfitting Test:
```bash
python train_clip_reproduction.py \
    --overfit_test_size 5 \
    --batch_size 2 \
    --num_epochs 50 \
    --debug_mode
```

### Debug Mode Features:
- Detailed gradient norm tracking
- Raw embedding norm monitoring
- Step-by-step loss analysis
- Memory usage reporting

### Common Issues:

1. **NaN Loss**: Check learning rate, try fp16=False
2. **Zero Gradients**: Verify data flow, check model initialization
3. **Memory Issues**: Reduce batch_size or use gradient_checkpointing
4. **Low Similarity**: Try longer training or different hyperparameters

## 📈 Results Analysis

### Training Outputs:
```
checkpoints/clip_repro_YYYYMMDD_HHMMSS/
├── checkpoint_step_500.pt      # Model checkpoints
├── training_summary.json       # Training metrics
├── experiment_config.json      # Full configuration
└── final_summary.json         # Final results
```

### Key Metrics to Monitor:
- **Loss trajectory**: Should decrease consistently
- **Velocity similarity**: Measures prediction quality
- **CLIP similarity**: Final reproduction quality
- **Gradient norms**: Should be stable, non-zero
- **Embedding norms**: Natural variation (no forced normalization)

## 🔬 Comparison with Normalized Approach

This minimal normalization approach can be compared with the original normalized approach:

| Aspect | Minimal Normalization | Full Normalization |
|--------|----------------------|-------------------|
| Training | Raw embedding space | L2 normalized space |
| Loss computation | Raw MSE | Normalized MSE |
| Data flow | Natural distributions | Unit sphere constraint |
| Evaluation | Normalize for similarity only | Normalize throughout |

## 🛠️ Advanced Usage

### Custom Configuration:
```python
from blip3o_clip_config import get_blip3o_clip_config

config = get_blip3o_clip_config(
    model_size="base",
    training_mode="patch_only",
    hidden_size=768,
    num_hidden_layers=12
)
```

### Memory Optimization:
```python
from blip3o_clip_config import get_memory_optimized_config

model_size, config, memory_usage = get_memory_optimized_config(
    available_memory_gb=16.0,
    target_batch_size=8
)
```

## 📝 Citation

If you use this implementation, please cite:

```bibtex
@article{blip3o_clip_reproduction,
  title={CLIP Reproduction from EVA Embeddings using BLIP3-o DiT with Minimal Normalization},
  author={Your Name},
  year={2024},
  note={Implementation for cross-modal embedding translation}
}
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test with overfitting test
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🔗 Related Work

- **BLIP3-o Paper**: [Reference to original paper]
- **EVA-CLIP**: [BAAI/EVA-CLIP-8B](https://huggingface.co/BAAI/EVA-CLIP-8B)
- **CLIP**: [openai/clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14)
- **Rectified Flow**: [Flow matching methodology]

---

**Happy Training! 🚀**

For questions or issues, please check the logs first, then open an issue with:
- Training command used
- Error messages from logs
- System configuration
- Data format details