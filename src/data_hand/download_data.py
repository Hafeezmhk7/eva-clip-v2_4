"""
FIXED Download BLIP3o-Pretrain-Short-Caption dataset with better Snellius support
Supports multiple shards and structured temp directory storage with quota management
Place this file in: src/data_hand/download_data.py

FIXES:
- Better disk space checking before download
- Proper temp manager integration
- Avoids home directory quota issues
- Better error handling for disk quota exceeded
"""

import os
import sys
from pathlib import Path
from huggingface_hub import hf_hub_download, list_repo_files
from tqdm import tqdm
import argparse
import shutil
import logging

logger = logging.getLogger(__name__)

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

def get_temp_directory():
    """Get the temp directory path for Snellius or other systems"""
    # Try to use temp manager first
    temp_manager = setup_temp_manager()
    if temp_manager:
        return temp_manager.get_datasets_dir()
    
    # FIXED: Better fallback to proper Snellius directories
    # First try environment variables set by job script
    if "BLIP3O_DATASETS" in os.environ:
        temp_dir = Path(os.environ["BLIP3O_DATASETS"])
        print(f"📁 Using BLIP3O_DATASETS: {temp_dir}")
        return temp_dir
    
    # Try Snellius scratch directories
    user = os.environ.get("USER", "user")
    
    # Check for scratch-shared
    if Path("/scratch-shared").exists():
        temp_dir = Path("/scratch-shared") / user / "blip3o_workspace" / "datasets"
        print(f"📁 Using scratch-shared: {temp_dir}")
        return temp_dir
    
    # Fallback to environment variables
    if "TMPDIR" in os.environ:
        temp_dir = Path(os.environ["TMPDIR"]) / "blip3o_data"
        print(f"📁 Using TMPDIR: {temp_dir}")
    elif "SCRATCH_SHARED" in os.environ:
        temp_dir = Path(os.environ["SCRATCH_SHARED"]) / user / "blip3o_data"
        print(f"📁 Using SCRATCH_SHARED env var: {temp_dir}")
    else:
        # AVOID home directory to prevent quota issues
        temp_dir = Path("/tmp") / user / "blip3o_data"
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

def estimate_download_size(num_shards: int) -> float:
    """Estimate download size in GB based on number of shards."""
    # Based on empirical data: ~1.2 GB per shard on average
    estimated_gb = num_shards * 1.2
    return estimated_gb

def download_blip3o_shards(shard_indices=None, data_dir=None, force_download=False, max_shards=2000):
    """
    FIXED: Download multiple shards with better disk space management
    
    Args:
        shard_indices (list): List of shard indices to download (0-11). If None, downloads all.
        data_dir (str): Directory to save data. If None, uses temp directory
        force_download (bool): Force re-download even if file exists
        max_shards (int): Maximum number of shards available
    
    Returns:
        list: Paths to downloaded files
    """
    
    # Setup temp manager for structured storage
    temp_manager = setup_temp_manager()
    
    # Set up paths - prioritize temp manager
    if data_dir is None:
        if temp_manager:
            data_dir = temp_manager.get_datasets_dir()
            print(f"📁 Using temp manager datasets directory: {data_dir}")
        else:
            data_dir = get_temp_directory()
            print(f"📁 Using fallback temp directory: {data_dir}")
    else:
        data_dir = Path(data_dir)
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
        shard_indices = list(range(min(30, max_shards)))  # Default: first 30 shards for ~100k samples
    
    # Ensure shard_indices is a list
    if isinstance(shard_indices, int):
        shard_indices = [shard_indices]
    
    # Validate shard indices
    shard_indices = [idx for idx in shard_indices if 0 <= idx < max_shards]
    
    if not shard_indices:
        raise ValueError(f"No valid shard indices provided. Must be in range 0-{max_shards-1}")
    
    # FIXED: Check disk space before starting download
    estimated_size_gb = estimate_download_size(len(shard_indices))
    if not check_disk_space(data_dir, estimated_size_gb):
        # Try to suggest alternatives
        print(f"\n💡 Suggestions to fix disk space issue:")
        print(f"   1. Use fewer shards: --shards 0 1 2 3 4 (for 5 shards)")
        print(f"   2. Clean up existing files in {data_dir}")
        print(f"   3. Use a different directory with more space")
        if temp_manager:
            print(f"   4. Check temp manager status for disk usage")
        raise RuntimeError("Insufficient disk space for download")
    
    # Dataset info
    repo_id = "BLIP3o/BLIP3o-Pretrain-Short-Caption"
    
    print(f"Downloading BLIP3o Short Caption Dataset")
    print(f"Repository: {repo_id}")
    print(f"Shards to download: {shard_indices}")
    print(f"Destination: {data_dir}")
    print(f"Total shards requested: {len(shard_indices)}")
    print(f"Estimated size: {estimated_size_gb:.1f} GB")
    
    if temp_manager:
        print(f"Storage type: Structured temp management")
        print(f"Retention: 14 days (scratch-shared)")
        # Show current disk usage
        usage = temp_manager.get_disk_usage()
        datasets_usage = usage.get('datasets', {})
        if datasets_usage.get('exists', False):
            print(f"Current datasets size: {datasets_usage.get('total_size_gb', 0):.2f} GB")
    else:
        print(f"Storage type: Basic temp directory")
    
    print("=" * 70)
    
    downloaded_files = []
    total_size_gb = 0
    failed_downloads = []
    
    for shard_idx in shard_indices:
        shard_filename = f"{shard_idx:05d}.tar"
        local_file_path = data_dir / shard_filename
        
        print(f"\n📥 Processing shard {shard_idx}: {shard_filename}")
        
        # Check if file already exists
        if local_file_path.exists() and not force_download:
            file_size_gb = local_file_path.stat().st_size / (1024**3)
            print(f"✅ File already exists: {local_file_path}")
            print(f"   Size: {file_size_gb:.2f} GB")
            downloaded_files.append(str(local_file_path))
            total_size_gb += file_size_gb
            continue
        
        # Check disk space before each download
        remaining_shards = len(shard_indices) - len(downloaded_files) - len(failed_downloads)
        estimated_remaining_gb = remaining_shards * 1.2
        
        if not check_disk_space(data_dir, estimated_remaining_gb):
            print(f"⚠️  Stopping download due to insufficient disk space")
            print(f"   Successfully downloaded {len(downloaded_files)} shards")
            print(f"   Remaining shards would need ~{estimated_remaining_gb:.1f} GB")
            break
        
        try:
            print(f"🔄 Downloading {shard_filename}...")
            
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
                print(f"✅ Download successful!")
                print(f"   File path: {downloaded_path}")
                print(f"   File size: {file_size_gb:.2f} GB")
                
                downloaded_files.append(downloaded_path)
                total_size_gb += file_size_gb
                
                # Estimate number of samples
                estimated_samples = int(file_size_gb * 400000 / 1.0)  # Rough estimate
                print(f"   Estimated samples: ~{estimated_samples:,}")
            else:
                print(f"❌ Download failed: File not found at {downloaded_path}")
                failed_downloads.append(shard_idx)
                
        except Exception as e:
            print(f"❌ Download error for shard {shard_idx}: {e}")
            
            # Check if it's a disk quota error
            if "Disk quota exceeded" in str(e) or "No space left" in str(e):
                print(f"💾 DISK QUOTA/SPACE ERROR!")
                print(f"   Downloaded {len(downloaded_files)} shards successfully")
                print(f"   Total downloaded: {total_size_gb:.2f} GB")
                print(f"   Consider using fewer shards or cleaning up space")
                break
            else:
                failed_downloads.append(shard_idx)
                continue
        
        # Show disk usage update if using temp manager
        if temp_manager and shard_idx % 5 == 0:  # Every 5 downloads
            try:
                usage = temp_manager.get_disk_usage()
                datasets_usage = usage.get('datasets', {})
                if datasets_usage.get('exists', False):
                    current_size = datasets_usage.get('total_size_gb', 0)
                    print(f"   💾 Current datasets storage: {current_size:.2f} GB")
            except Exception as e:
                # Don't fail the whole download for status check issues
                pass
    
    print("\n" + "=" * 70)
    print(f"📊 DOWNLOAD SUMMARY:")
    print(f"   Successfully downloaded: {len(downloaded_files)}/{len(shard_indices)} shards")
    print(f"   Total size: {total_size_gb:.2f} GB")
    print(f"   Storage location: {data_dir}")
    print(f"   Estimated total samples: ~{int(total_size_gb * 400000):,}")
    
    if failed_downloads:
        print(f"   Failed downloads: {len(failed_downloads)} shards")
        print(f"   Failed shard indices: {failed_downloads}")
    
    if temp_manager:
        print(f"\n🗂️  TEMP MANAGER INFO:")
        print(f"   Storage type: Persistent (scratch-shared)")
        print(f"   Retention policy: 14 days automatic cleanup")
        print(f"   Access across jobs: Yes")
        print(f"   Workspace: {temp_manager.persistent_workspace}")
        
        # Show final disk usage
        try:
            usage = temp_manager.get_disk_usage()
            for name, info in usage.items():
                if info.get('exists', False) and 'datasets' in name:
                    size_gb = info.get('total_size_gb', 0)
                    file_count = info.get('file_count', 0)
                    print(f"   {name}: {size_gb:.2f} GB ({file_count} files)")
        except Exception as e:
            # Don't fail for status issues
            pass
    
    if downloaded_files:
        print(f"\n✅ Ready for embedding extraction!")
        print(f"   Use these files in extract_embeddings_g.py")
        print(f"   Files are in structured temp storage with 14-day retention")
    else:
        print(f"\n❌ No files downloaded successfully")
        if failed_downloads:
            print(f"   This is likely due to disk space issues")
            print(f"   Try downloading fewer shards or cleaning up space")
    
    # Save file list for embedding extraction
    if downloaded_files:
        try:
            file_list_path = data_dir / "downloaded_shards.txt"
            with open(file_list_path, 'w') as f:
                for file_path in downloaded_files:
                    f.write(f"{file_path}\n")
            print(f"\n📝 File list saved to: {file_list_path}")
        except Exception as e:
            print(f"⚠️  Could not save file list: {e}")
    
    return downloaded_files

def list_available_files(repo_id="BLIP3o/BLIP3o-Pretrain-Short-Caption"):
    """List all available files in the repository"""
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

def verify_downloads(file_paths):
    """Verify the downloaded files"""
    print(f"\n🧪 Verifying {len(file_paths)} downloaded files...")
    
    valid_files = []
    total_size = 0
    
    for file_path in file_paths:
        if not os.path.exists(file_path):
            print(f"❌ File not found: {file_path}")
            continue
        
        file_size = os.path.getsize(file_path)
        if file_size < 1024 * 1024:  # Less than 1MB is suspicious
            print(f"❌ File seems too small: {file_path} ({file_size} bytes)")
            continue
        
        print(f"✅ {Path(file_path).name}: {file_size / (1024**3):.2f} GB")
        valid_files.append(file_path)
        total_size += file_size
    
    print(f"\n📊 Verification summary:")
    print(f"   Valid files: {len(valid_files)}/{len(file_paths)}")
    print(f"   Total size: {total_size / (1024**3):.2f} GB")
    
    return valid_files

def show_temp_info():
    """Show information about temp directory setup"""
    temp_manager = setup_temp_manager()
    
    if temp_manager:
        print("\n🗂️  TEMP MANAGER STATUS:")
        temp_manager.print_status()
        
        # Show disk quota safety
        safety = temp_manager.check_disk_quota_safety()
        print(f"\n🛡️  Disk quota safety: {safety['status']}")
        if safety['warnings']:
            for warning in safety['warnings']:
                print(f"   ⚠️  {warning}")
        if safety['recommendations']:
            for rec in safety['recommendations']:
                print(f"   💡 {rec}")
    else:
        print("\n📁 FALLBACK TEMP DIRECTORIES:")
        temp_dir = get_temp_directory()
        print(f"   Datasets directory: {temp_dir}")
        
        # Check available space if possible
        try:
            total, used, free = shutil.disk_usage(temp_dir.parent)
            print(f"   Available space: {free / (1024**3):.1f} GB")
            print(f"   Used: {(used/total)*100:.1f}%")
        except:
            print("   Cannot determine available space")

def main():
    """Main function for command line usage"""
    parser = argparse.ArgumentParser(
        description="Download BLIP3o-Pretrain-Short-Caption dataset with better disk space management",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/data_hand/download_data.py                           # Download first 30 shards to temp
  python src/data_hand/download_data.py --shards 0 1 2 3 4        # Download specific shards
  python src/data_hand/download_data.py --shards 0 --data_dir /tmp # Download to specific directory
  python src/data_hand/download_data.py --list                    # List available files
  python src/data_hand/download_data.py --all                     # Download ALL shards (be careful!)
  python src/data_hand/download_data.py --info                    # Show temp directory info
        """
    )
    
    parser.add_argument(
        "--shards",
        type=int,
        nargs='+',
        default=None,
        help="Shard indices to download (default: first 30 shards)"
    )
    
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download ALL available shards (warning: very large!)"
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
        "--list",
        action="store_true",
        help="List available files in the repository"
    )
    
    parser.add_argument(
        "--verify",
        nargs='+',
        help="Verify specific downloaded files"
    )
    
    parser.add_argument(
        "--info",
        action="store_true",
        help="Show temp directory information"
    )
    
    args = parser.parse_args()
    
    if args.info:
        show_temp_info()
        return
    
    if args.list:
        list_available_files()
        return
    
    if args.verify:
        verify_downloads(args.verify)
        return
    
    # Show temp directory info first
    show_temp_info()
    
    # Determine which shards to download
    if args.all:
        # Download all available shards (get list first)
        print("⚠️  WARNING: Downloading ALL shards will use significant storage!")
        
        # Estimate space needed
        available_files = list_available_files()
        estimated_gb = len(available_files) * 1.2
        print(f"   Estimated space needed: {estimated_gb:.1f} GB")
        
        response = input("Continue? (y/N): ")
        if response.lower() != 'y':
            print("Cancelled.")
            return
        
        shard_indices = list(range(len(available_files)))
    else:
        shard_indices = args.shards
    
    # Estimate space needed and warn user
    if shard_indices:
        estimated_gb = estimate_download_size(len(shard_indices))
        print(f"\n💾 SPACE ESTIMATE:")
        print(f"   Shards to download: {len(shard_indices)}")
        print(f"   Estimated space needed: {estimated_gb:.1f} GB")
        
        # Check with temp manager
        temp_manager = setup_temp_manager()
        if temp_manager:
            usage = temp_manager.get_disk_usage()
            
            # Check workspace space
            workspace_info = usage.get('workspace_system', {})
            if workspace_info and 'free_gb' in workspace_info:
                available_gb = workspace_info['free_gb']
                print(f"   Available space: {available_gb:.1f} GB")
                
                if estimated_gb > available_gb * 0.8:  # 80% threshold
                    print(f"   ⚠️  WARNING: May not have enough space!")
                    print(f"   Consider downloading fewer shards")
                    response = input("Continue anyway? (y/N): ")
                    if response.lower() != 'y':
                        print("Cancelled.")
                        return
    
    # Download the shards
    try:
        downloaded_files = download_blip3o_shards(
            shard_indices=shard_indices,
            data_dir=args.data_dir,
            force_download=args.force
        )
        
        if downloaded_files:
            print(f"\n🎉 SUCCESS! Downloaded {len(downloaded_files)} shards")
            print(f"\n📋 Next steps:")
            print(f"1. Extract embeddings: python src/modules/extract_embeddings_g.py")
            print(f"2. Start training with structured temp management")
            print(f"3. Files are in structured temp storage with proper retention")
            
            temp_manager = setup_temp_manager()
            if temp_manager:
                print(f"\n🗂️  TEMP MANAGER BENEFITS:")
                print(f"   ✅ Structured storage in persistent workspace")
                print(f"   ✅ 14-day retention (scratch-shared)")
                print(f"   ✅ Accessible across different jobs")
                print(f"   ✅ Automatic disk usage monitoring")
                print(f"   ✅ Proper cache management")
            
        else:
            print(f"\n❌ Download failed. Please check the error messages above.")
            print(f"   Common issues:")
            print(f"   • Disk quota exceeded (use scratch directories)")
            print(f"   • Network issues (try again later)")
            print(f"   • Insufficient disk space (download fewer shards)")
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