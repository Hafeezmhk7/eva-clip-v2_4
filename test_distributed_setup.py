#!/usr/bin/env python3
"""
Test Distributed Setup Script
test_distributed_setup.py

Run this before submitting the job to catch any issues early.
Usage: python test_distributed_setup.py
"""

import os
import sys
import torch
from pathlib import Path
import traceback

def setup_environment():
    """Setup environment variables"""
    os.environ['PYTHONWARNINGS'] = 'ignore::UserWarning'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.environ['WANDB_SILENT'] = 'true'
    
    # Fix transformers cache warning
    if 'TRANSFORMERS_CACHE' in os.environ and 'HF_HOME' not in os.environ:
        os.environ['HF_HOME'] = os.environ['TRANSFORMERS_CACHE']

def test_imports():
    """Test all required imports"""
    print("🧪 Testing imports...")
    
    import_tests = [
        ("PyTorch", "torch", None),
        ("Distributed Dataset", "src.modules.datasets.blip3o_distributed_dataset", "DistributedBLIP3oCLIPReproductionDataset"),
        ("Model", "src.modules.models.blip3o_dit", "create_improved_clip_reproduction_model"),
        ("Loss", "src.modules.losses.blip3o_fm_loss", "create_clip_reproduction_loss"),
        ("Distributed Trainer", "src.modules.trainers.blip3o_distributed_trainer", "BLIP3oDistributedTrainer"),
        ("FSDP Utils", "src.modules.distributed.fsdp_utils", "setup_distributed_environment"),
        ("Base Dataset", "src.modules.datasets.blip3o_dataset", "clip_reproduction_collate_fn"),
    ]
    
    for name, module_path, class_name in import_tests:
        try:
            if class_name:
                module = __import__(module_path, fromlist=[class_name])
                getattr(module, class_name)
            else:
                __import__(module_path)
            print(f"  ✅ {name}")
        except Exception as e:
            print(f"  ❌ {name}: {e}")
            return False
    
    return True

def test_cuda():
    """Test CUDA availability"""
    print("\n🖥️ Testing CUDA...")
    
    if not torch.cuda.is_available():
        print("  ❌ CUDA not available")
        return False
    
    gpu_count = torch.cuda.device_count()
    print(f"  ✅ CUDA available with {gpu_count} GPUs")
    
    for i in range(min(gpu_count, 4)):
        gpu_name = torch.cuda.get_device_name(i)
        total_memory = torch.cuda.get_device_properties(i).total_memory / 1e9
        print(f"    GPU {i}: {gpu_name} ({total_memory:.1f} GB)")
    
    return True

def test_embeddings_directory():
    """Test embeddings directory"""
    print("\n📁 Testing embeddings directory...")
    
    # Default path from the user's error logs
    embeddings_dir = Path("/scratch-shared/azadaianchuk1/blip3o_workspace/embeddings/patch_embeddings_short_256")
    
    if not embeddings_dir.exists():
        print(f"  ❌ Directory does not exist: {embeddings_dir}")
        
        # Try to find alternatives
        parent_dir = embeddings_dir.parent
        if parent_dir.exists():
            print(f"  💡 Parent directory exists: {parent_dir}")
            try:
                contents = list(parent_dir.iterdir())
                print(f"  💡 Parent directory contains {len(contents)} items:")
                for item in contents[:5]:
                    print(f"     {item.name}")
            except Exception as e:
                print(f"     Could not list contents: {e}")
        return False
    
    print(f"  ✅ Directory exists: {embeddings_dir}")
    
    # Check for .pkl files
    pkl_files = list(embeddings_dir.glob("*.pkl"))
    if not pkl_files:
        print(f"  ❌ No .pkl files found")
        return False
    
    print(f"  ✅ Found {len(pkl_files)} .pkl files")
    
    # Test loading one file
    try:
        import pickle
        test_file = pkl_files[0]
        print(f"  🧪 Testing file: {test_file.name}")
        
        with open(test_file, 'rb') as f:
            data = pickle.load(f)
        
        required_keys = ['clip_blip3o_embeddings', 'eva_blip3o_embeddings', 'captions']
        for key in required_keys:
            if key not in data:
                print(f"    ❌ Missing key: {key}")
                return False
        
        print(f"    ✅ File structure valid")
        print(f"    ✅ CLIP shape: {data['clip_blip3o_embeddings'].shape}")
        print(f"    ✅ EVA shape: {data['eva_blip3o_embeddings'].shape}")
        
    except Exception as e:
        print(f"  ❌ Error loading file: {e}")
        return False
    
    return True

def test_model_creation():
    """Test model creation"""
    print("\n🏗️ Testing model creation...")
    
    try:
        from src.modules.models.blip3o_dit import create_improved_clip_reproduction_model
        
        model = create_improved_clip_reproduction_model(
            model_size="base",
            training_mode="patch_only",
            use_eva_adapter=True,
        )
        
        param_count = sum(p.numel() for p in model.parameters())
        print(f"  ✅ Model created with {param_count:,} parameters")
        
        # Test forward pass
        batch_size = 2
        seq_len = 256
        eva_emb = torch.randn(batch_size, seq_len, 4096)
        hidden_states = torch.randn(batch_size, seq_len, 1024)
        timestep = torch.randn(batch_size)
        
        with torch.no_grad():
            output = model(
                hidden_states=hidden_states,
                timestep=timestep,
                encoder_hidden_states=eva_emb,
                return_dict=False
            )
        
        print(f"  ✅ Forward pass successful: {output.shape}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Model creation failed: {e}")
        traceback.print_exc()
        return False

def test_dataset_creation():
    """Test dataset creation"""
    print("\n📊 Testing dataset creation...")
    
    try:
        from src.modules.datasets.blip3o_distributed_dataset import DistributedBLIP3oCLIPReproductionDataset
        
        embeddings_dir = "/scratch-shared/azadaianchuk1/blip3o_workspace/embeddings/patch_embeddings_short_256"
        
        dataset = DistributedBLIP3oCLIPReproductionDataset(
            chunked_embeddings_dir=embeddings_dir,
            training_mode="patch_only",
            max_shards=1,  # Just test with 1 shard
            world_size=1,
            rank=0,
            max_samples_per_epoch=5,  # Very limited for testing
        )
        
        print(f"  ✅ Dataset created with {len(dataset):,} estimated samples")
        
        # Test iteration
        iterator = iter(dataset)
        sample = next(iterator)
        
        print(f"  ✅ Sample iteration successful")
        print(f"    EVA shape: {sample['eva_embeddings'].shape}")
        print(f"    CLIP shape: {sample['clip_embeddings'].shape}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Dataset creation failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("🧪 BLIP3-o Distributed Training Setup Test")
    print("=" * 60)
    
    # Setup environment
    setup_environment()
    
    # Add current directory to path
    sys.path.insert(0, str(Path(__file__).parent))
    
    tests = [
        ("Imports", test_imports),
        ("CUDA", test_cuda),
        ("Embeddings Directory", test_embeddings_directory),
        ("Model Creation", test_model_creation),
        ("Dataset Creation", test_dataset_creation),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n❌ {test_name} test crashed: {e}")
            traceback.print_exc()
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 Test Summary:")
    print("=" * 60)
    
    all_passed = True
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}: {test_name}")
        if not passed:
            all_passed = False
    
    print("=" * 60)
    
    if all_passed:
        print("🎉 ALL TESTS PASSED!")
        print("✅ Ready to submit distributed training job")
        print("\nNext steps:")
        print("  1. Submit job: sbatch job_scripts/train_blip3o_distributed.job")
        print("  2. Monitor: tail -f slurm_out/blip3o_distributed_fixed_*.out")
        print("  3. Check results in ./checkpoints/")
    else:
        print("❌ SOME TESTS FAILED!")
        print("⚠️ Fix the issues above before submitting the job")
        print("\nCommon solutions:")
        print("  • Import errors: Check file paths and module structure")
        print("  • CUDA errors: Load CUDA module and check GPU availability")
        print("  • Path errors: Verify embeddings directory path")
        print("  • Permission errors: Check file permissions")
    
    print("=" * 60)
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)