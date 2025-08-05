"""
UPDATED Download BLIP3o datasets with support for multiple dataset types
Supports both short and long caption datasets with structured organization
Place this file in: src/data_hand/download_data.py

NEW FEATURES:
- Support for both short_caption and long_caption datasets
- Organized folder structure for different datasets
- Better disk space checking and quota management
- Parallel download support
"""

import os
import sys
from pathlib import Path
from huggingface_hub import hf_hub_download, list_repo_files
from tqdm import tqdm
import argparse
import shutil
import logging
import concurrent.futures
import threading
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

# Dataset configurations
DATASET_CONFIGS = {
    'short_caption': {
        'repo_id': 'BLIP3o/BLIP3o-Pretrain-Short-Caption',
        'description': 'BLIP3-o Pretrain Short Caption Dataset',
        'estimated_size_per_shard_gb': 1.2,
        'max_shards': 2000,
    },
    'long_caption': {
        'repo_id': 'BLIP3o/BLIP3o-Pretrain-Long-Caption',
        'description': 'BLIP3-o Pretrain Long Caption Dataset',
        'estimated_size_per_shard_gb': 1.5,
        'max_shards': 1500,
    }
}

def get_project_root():
    """Get the project root directory"""
    current_file = Path(__file__).resolve()
    # Go up from src/data_hand/download_data.py to project root
    return current_file.parent.parent.parent

def setup_temp_manager():
    """Setup temp manager for structured directory management."""
    try:
        # Add utils to path
        project_root = get_project_root()
        sys.path.insert(0, str(project_root / "src" / "modules" / "utils"))
        
        from temp_manager import setup_snellius_environment
        manager = setup_snellius_environment("blip3o_workspace")
        return manager
    except ImportError as e:
        print(f"⚠️  Temp manager not available: {e}")
        print("Using fallback directories")
        return None

def get_temp_directory(dataset_type: str):
    """Get the temp directory path for a specific dataset type"""
    # Try to use temp manager first
    temp_manager = setup_temp_manager()
    if temp_manager:
        datasets_dir = temp_manager.get_datasets_dir()
        return datasets_dir / dataset_type
    
    # FIXED: Better fallback to proper Snellius directories
    # First try environment variables set by job script
    if "BLIP3O_DATASETS" in os.environ:
        temp_dir = Path(os.environ["BLIP3O_DATASETS"]) / dataset_type
        print(f"📁 Using BLIP3O_DATASETS: {temp_dir}")
        return temp_dir
    
    # Try Snellius scratch directories
    user = os.environ.get("USER", "user")
    
    # Check for scratch-shared
    if Path("/scratch-shared").exists():
        temp_dir = Path("/scratch-shared") / user / "blip3o_workspace" / "datasets" / dataset_type
        print(f"📁 Using scratch-shared: {temp_dir}")
        return temp_dir
    
    # Fallback to environment variables
    if "TMPDIR" in os.environ:
        temp_dir = Path(os.environ["TMPDIR"]) / "blip3o_data" / dataset_type
        print(f"📁 Using TMPDIR: {temp_dir}")
    elif "SCRATCH_SHARED" in os.environ:
        temp_dir = Path(os.environ["SCRATCH_SHARED"]) / user / "blip3o_data" / dataset_type
        print(f"📁 Using SCRATCH_SHARED env var: {temp_dir}")
    else:
        # AVOID home directory to prevent quota issues
        temp_dir = Path("/tmp") / user / "blip3o_data" / dataset_type
        print(f"⚠️  Using /tmp fallback: {temp_dir}")
        print("⚠️  Consider setting proper scratch directories")
    
    return temp_dir

def check_disk_space(target_dir: Path, required_gb: float) -> bool:
    """Check if there's enough disk space for download."""
    try:
        # Get available space
        total, used, free = shutil.disk_usage(target_dir.parent)
        free_gb = free / (1024**3)
        total_gb = total / (1024**3)
        used_percent = (used / total) * 100
        
        print(f"💾 Disk space check for {target_dir.parent}:")
        print(f"   Total: {total_gb:.1f} GB")
        print(f"   Used: {used_percent:.1f}%")
        print(f"   Free: {free_gb:.1f} GB")
        print(f"   Required: {required_gb:.1f} GB")
        
        if free_gb < required_gb:
            print(f"❌ Insufficient disk space!")
            print(f"   Need {required_gb:.1f} GB, but only {free_gb:.1f} GB available")
            return False
        
        # Warning if less than 2x required space
        if free_gb < required_gb * 2:
            print(f"⚠️  Low disk space warning!")
            print(f"   Only {free_gb:.1f} GB available for {required_gb:.1f} GB download")
            print(f"   Consider using fewer shards or cleaning up space")
        
        return True
        
    except Exception as e:
        print(f"⚠️  Could not check disk space: {e}")
        return True  # Continue anyway

def estimate_download_size(dataset_type: str, num_shards: int) -> float:
    """Estimate download size in GB based on dataset type and number of shards."""
    config = DATASET_CONFIGS.get(dataset_type, DATASET_CONFIGS['short_caption'])
    estimated_gb = num_shards * config['estimated_size_per_shard_gb']
    return estimated_gb

def download_single_shard(args_tuple):
    """Download a single shard (for parallel processing)"""
    repo_id, shard_filename, data_dir, shard_idx, force_download = args_tuple
    
    local_file_path = data_dir / shard_filename
    
    # Check if file already exists
    if local_file_path.exists() and not force_download:
        file_size_gb = local_file_path.stat().st_size / (1024**3)
        return {
            'success': True,
            'shard_idx': shard_idx,
            'filename': shard_filename,
            'size_gb': file_size_gb,
            'status': 'already_exists'
        }
    
    try:
        # Download using HuggingFace Hub
        downloaded_path = hf_hub_download(
            repo_id=repo_id,
            filename=shard_filename,
            repo_type="dataset",
            local_dir=str(data_dir),
            local_dir_use_symlinks=False,  # Download actual files, not symlinks
            resume_download=True,  # Resume partial downloads
        )
        
        # Verify download
        if os.path.exists(downloaded_path):
            file_size_gb = os.path.getsize(downloaded_path) / (1024**3)
            return {
                'success': True,
                'shard_idx': shard_idx,
                'filename': shard_filename,
                'size_gb': file_size_gb,
                'status': 'downloaded',
                'path': downloaded_path
            }
        else:
            return {
                'success': False,
                'shard_idx': shard_idx,
                'filename': shard_filename,
                'error': 'File not found after download'
            }
            
    except Exception as e:
        return {
            'success': False,
            'shard_idx': shard_idx,
            'filename': shard_filename,
            'error': str(e)
        }

def download_blip3o_shards(
    dataset_type: str = "short_caption",
    shard_indices=None, 
    data_dir=None, 
    force_download=False, 
    parallel_downloads=4
):
    """
    Download multiple shards with parallel processing and better organization
    
    Args:
        dataset_type: Type of dataset ('short_caption' or 'long_caption')
        shard_indices: List of shard indices to download
        data_dir: Directory to save data. If None, uses temp directory
        force_download: Force re-download even if file exists
        parallel_downloads: Number of parallel downloads
    
    Returns:
        dict: Download results with statistics
    """
    
    # Get dataset configuration
    if dataset_type not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset type: {dataset_type}. Available: {list(DATASET_CONFIGS.keys())}")
    
    config = DATASET_CONFIGS[dataset_type]
    repo_id = config['repo_id']
    max_shards = config['max_shards']
    
    # Setup temp manager for structured storage
    temp_manager = setup_temp_manager()
    
    # Set up paths - prioritize temp manager
    if data_dir is None:
        if temp_manager:
            data_dir = temp_manager.get_datasets_dir() / dataset_type
            print(f"📁 Using temp manager datasets directory: {data_dir}")
        else:
            data_dir = get_temp_directory(dataset_type)
            print(f"📁 Using fallback temp directory: {data_dir}")
    else:
        data_dir = Path(data_dir) / dataset_type
        print(f"📁 Using specified directory: {data_dir}")
    
    # Create data directory
    try:
        data_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        if "Disk quota exceeded" in str(e):
            print(f"❌ DISK QUOTA EXCEEDED when creating directory!")
            print(f"   Target directory: {data_dir}")
            print(f"   Consider using scratch directories instead of home")
            raise
        else:
            print(f"❌ Error creating directory {data_dir}: {e}")
            raise
    
    # Default shard selection
    if shard_indices is None:
        shard_indices = list(range(min(30, max_shards)))  # Default: first 30 shards
    
    # Ensure shard_indices is a list
    if isinstance(shard_indices, int):
        shard_indices = [shard_indices]
    
    # Validate shard indices
    shard_indices = [idx for idx in shard_indices if 0 <= idx < max_shards]
    
    if not shard_indices:
        raise ValueError(f"No valid shard indices provided. Must be in range 0-{max_shards-1}")
    
    # Check disk space before starting download
    estimated_size_gb = estimate_download_size(dataset_type, len(shard_indices))
    if not check_disk_space(data_dir, estimated_size_gb):
        # Try to suggest alternatives
        print(f"\n💡 Suggestions to fix disk space issue:")
        print(f"   1. Use fewer shards: --shards 0 1 2 3 4 (for 5 shards)")
        print(f"   2. Clean up existing files in {data_dir}")
        print(f"   3. Use a different directory with more space")
        if temp_manager:
            print(f"   4. Check temp manager status for disk usage")
        raise RuntimeError("Insufficient disk space for download")
    
    print(f"Downloading {config['description']}")
    print(f"Repository: {repo_id}")
    print(f"Dataset type: {dataset_type}")
    print(f"Shards to download: {shard_indices}")
    print(f"Destination: {data_dir}")
    print(f"Total shards requested: {len(shard_indices)}")
    print(f"Estimated size: {estimated_size_gb:.1f} GB")
    print(f"Parallel downloads: {parallel_downloads}")
    print("=" * 70)
    
    # Prepare download arguments
    download_args = []
    for shard_idx in shard_indices:
        shard_filename = f"{shard_idx:05d}.tar"
        download_args.append((repo_id, shard_filename, data_dir, shard_idx, force_download))
    
    # Execute parallel downloads
    successful_downloads = []
    failed_downloads = []
    total_size_gb = 0
    
    print(f"🚀 Starting parallel download with {parallel_downloads} workers...")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=parallel_downloads) as executor:
        # Submit all download tasks
        future_to_shard = {executor.submit(download_single_shard, args): args[3] for args in download_args}
        
        # Progress bar for downloads
        with tqdm(total=len(download_args), desc="Downloading shards", unit="shard") as pbar:
            for future in concurrent.futures.as_completed(future_to_shard):
                result = future.result()
                
                if result['success']:
                    successful_downloads.append(result)
                    total_size_gb += result['size_gb']
                    
                    status_icon = "✅" if result['status'] == 'downloaded' else "☑️"
                    pbar.set_postfix({
                        'Status': f"{status_icon} {result['filename']} ({result['size_gb']:.2f}GB)"
                    })
                else:
                    failed_downloads.append(result)
                    pbar.set_postfix({
                        'Status': f"❌ {result['filename']} - {result.get('error', 'Unknown error')}"
                    })
                
                pbar.update(1)
    
    print("\n" + "=" * 70)
    print(f"📊 DOWNLOAD SUMMARY:")
    print(f"   Dataset type: {dataset_type}")
    print(f"   Successfully downloaded: {len(successful_downloads)}/{len(shard_indices)} shards")
    print(f"   Total size: {total_size_gb:.2f} GB")
    print(f"   Storage location: {data_dir}")
    print(f"   Estimated total samples: ~{int(total_size_gb * 400000):,}")
    
    if failed_downloads:
        print(f"   Failed downloads: {len(failed_downloads)} shards")
        print(f"   Failed shard indices: {[r['shard_idx'] for r in failed_downloads]}")
        
        # Show specific errors
        print(f"   Error details:")
        for failure in failed_downloads[:5]:  # Show first 5 errors
            print(f"     • Shard {failure['shard_idx']}: {failure.get('error', 'Unknown')}")
    
    if temp_manager:
        print(f"\n🗂️  TEMP MANAGER INFO:")
        print(f"   Storage type: Persistent (scratch-shared)")
        print(f"   Retention policy: 14 days automatic cleanup")
        print(f"   Access across jobs: Yes")
        print(f"   Workspace: {temp_manager.persistent_workspace}")
    
    if successful_downloads:
        print(f"\n✅ Ready for embedding extraction!")
        print(f"   Use these files in extract_embeddings_distributed.py")
        print(f"   Command: python src/modules/extract_embeddings_distributed.py --dataset_type {dataset_type}")
    else:
        print(f"\n❌ No files downloaded successfully")
        if failed_downloads:
            print(f"   This is likely due to network or disk space issues")
    
    # Save file list for embedding extraction
    if successful_downloads:
        try:
            file_list_path = data_dir / "downloaded_shards.txt"
            with open(file_list_path, 'w') as f:
                for result in successful_downloads:
                    if 'path' in result:
                        f.write(f"{result['path']}\n")
                    else:
                        f.write(f"{data_dir / result['filename']}\n")
            print(f"\n📝 File list saved to: {file_list_path}")
            
            # Also save download manifest
            manifest = {
                'dataset_type': dataset_type,
                'repo_id': repo_id,
                'download_timestamp': time.time(),
                'total_shards': len(successful_downloads),
                'total_size_gb': total_size_gb,
                'successful_downloads': successful_downloads,
                'failed_downloads': failed_downloads,
                'shard_indices': shard_indices,
            }
            
            manifest_path = data_dir / "download_manifest.json"
            import json
            with open(manifest_path, 'w') as f:
                json.dump(manifest, f, indent=2)
            print(f"📝 Download manifest saved to: {manifest_path}")
            
        except Exception as e:
            print(f"⚠️  Could not save file list: {e}")
    
    return {
        'dataset_type': dataset_type,
        'successful_downloads': successful_downloads,
        'failed_downloads': failed_downloads,
        'total_size_gb': total_size_gb,
        'data_dir': str(data_dir)
    }

def list_available_files(dataset_type: str = "short_caption"):
    """List all available files in the repository"""
    if dataset_type not in DATASET_CONFIGS:
        print(f"❌ Unknown dataset type: {dataset_type}")
        return []
    
    config = DATASET_CONFIGS[dataset_type]
    repo_id = config['repo_id']
    
    try:
        print(f"🔍 Available files in {repo_id}:")
        files = list_repo_files(repo_id, repo_type="dataset")
        
        tar_files = [f for f in files if f.endswith('.tar')]
        tar_files.sort()
        
        print(f"   Found {len(tar_files)} tar files:")
        for i, filename in enumerate(tar_files):
            shard_num = filename.replace('.tar', '')
            print(f"     {i:2d}. {filename} (shard {int(shard_num)})")
        
        return tar_files
        
    except Exception as e:
        print(f"❌ Error listing files: {e}")
        return []

def show_dataset_info():
    """Show information about available datasets"""
    print("📊 Available BLIP3-o Datasets:")
    print("=" * 50)
    
    for dataset_type, config in DATASET_CONFIGS.items():
        print(f"\n🗂️  Dataset: {dataset_type}")
        print(f"   Description: {config['description']}")
        print(f"   Repository: {config['repo_id']}")
        print(f"   Max shards: {config['max_shards']}")
        print(f"   Est. size per shard: {config['estimated_size_per_shard_gb']:.1f} GB")
        print(f"   Total est. size: {config['max_shards'] * config['estimated_size_per_shard_gb']:.1f} GB")

def main():
    """Main function for command line usage"""
    parser = argparse.ArgumentParser(
        description="Download BLIP3o datasets with support for multiple dataset types",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/data_hand/download_data.py --dataset_type short_caption --shards 0 1 2 3 4
  python src/data_hand/download_data.py --dataset_type long_caption --shards 0 1 2
  python src/data_hand/download_data.py --list --dataset_type short_caption
  python src/data_hand/download_data.py --info
        """
    )
    
    parser.add_argument(
        "--dataset_type",
        type=str,
        default="short_caption",
        choices=list(DATASET_CONFIGS.keys()),
        help="Type of dataset to download"
    )
    
    parser.add_argument(
        "--shards",
        type=int,
        nargs='+',
        default=None,
        help="Shard indices to download (default: first 30 shards)"
    )
    
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Directory to save data (default: use temp manager)"
    )
    
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force download even if file exists"
    )
    
    parser.add_argument(
        "--parallel_downloads",
        type=int,
        default=4,
        help="Number of parallel downloads"
    )
    
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available files in the repository"
    )
    
    parser.add_argument(
        "--info",
        action="store_true",
        help="Show information about available datasets"
    )
    
    args = parser.parse_args()
    
    if args.info:
        show_dataset_info()
        return
    
    if args.list:
        list_available_files(args.dataset_type)
        return
    
    print(f"🚀 BLIP3-o Dataset Downloader")
    print(f"Dataset type: {args.dataset_type}")
    print(f"Description: {DATASET_CONFIGS[args.dataset_type]['description']}")
    print("=" * 60)
    
    try:
        # Download the shards
        results = download_blip3o_shards(
            dataset_type=args.dataset_type,
            shard_indices=args.shards,
            data_dir=args.data_dir,
            force_download=args.force,
            parallel_downloads=args.parallel_downloads
        )
        
        if results['successful_downloads']:
            print(f"\n🎉 SUCCESS! Downloaded {len(results['successful_downloads'])} shards")
            print(f"   Dataset type: {results['dataset_type']}")
            print(f"   Total size: {results['total_size_gb']:.1f} GB")
            print(f"   Location: {results['data_dir']}")
            print(f"\n📋 Next steps:")
            print(f"1. Extract embeddings:")
            print(f"   python src/modules/extract_embeddings_distributed.py --dataset_type {args.dataset_type}")
            print(f"2. Start training:")
            print(f"   torchrun --nproc_per_node=4 train_dit_distributed.py --chunked_embeddings_dir <embeddings_dir>")
        else:
            print(f"\n❌ Download failed. Please check the error messages above.")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n❌ Download failed with error: {e}")
        if "Disk quota exceeded" in str(e):
            print(f"\n💡 Disk quota exceeded solutions:")
            print(f"   1. Use scratch-shared instead of home directory")
            print(f"   2. Clean up existing files")
            print(f"   3. Download fewer shards")
            print(f"   4. Set BLIP3O_DATASETS to a scratch directory")
        sys.exit(1)

if __name__ == "__main__":
    main()