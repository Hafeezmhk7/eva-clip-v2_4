#!/usr/bin/env python3
"""
TAR File Diagnostic Script
Analyzes TAR files to understand sample counts and file types
"""

import tarfile
import sys
from pathlib import Path
from collections import defaultdict, Counter
from PIL import Image
import io

def analyze_tar_file(tar_path):
    """Analyze a single TAR file"""
    print(f"\n🔍 Analyzing: {Path(tar_path).name}")
    print("=" * 60)
    
    try:
        with tarfile.open(tar_path, 'r') as tar:
            all_members = tar.getmembers()
            
            # Basic stats
            total_files = len(all_members)
            total_dirs = sum(1 for m in all_members if m.isdir())
            total_regular_files = sum(1 for m in all_members if m.isfile())
            
            print(f"📊 Total TAR members: {total_files}")
            print(f"   Directories: {total_dirs}")
            print(f"   Regular files: {total_regular_files}")
            
            # File extension analysis
            extensions = Counter()
            sizes = defaultdict(list)
            
            for member in all_members:
                if member.isfile():
                    ext = Path(member.name).suffix.lower()
                    if not ext:
                        ext = "no_extension"
                    extensions[ext] += 1
                    sizes[ext].append(member.size)
            
            print(f"\n📁 File Extensions:")
            for ext, count in extensions.most_common():
                avg_size = sum(sizes[ext]) / len(sizes[ext]) if sizes[ext] else 0
                print(f"   {ext:15s}: {count:6d} files (avg {avg_size/1024:.1f} KB)")
            
            # Focus on image files
            image_extensions = ['.jpg', '.jpeg', '.png', '.webp']
            image_members = []
            
            for member in all_members:
                if member.isfile():
                    if any(ext in member.name.lower() for ext in image_extensions):
                        image_members.append(member)
            
            print(f"\n🖼️  Image Files Found: {len(image_members)}")
            
            # Validate images (sample first 50 to avoid taking too long)
            sample_size = min(50, len(image_members))
            valid_images = 0
            invalid_images = 0
            validation_errors = Counter()
            
            print(f"🔬 Validating {sample_size} sample images...")
            
            for i, member in enumerate(image_members[:sample_size]):
                try:
                    file_obj = tar.extractfile(member)
                    if file_obj is None:
                        invalid_images += 1
                        validation_errors["no_file_object"] += 1
                        continue
                    
                    image_data = file_obj.read()
                    if not image_data or len(image_data) == 0:
                        invalid_images += 1
                        validation_errors["empty_file"] += 1
                        continue
                    
                    # Try to load and verify image
                    test_image = Image.open(io.BytesIO(image_data))
                    test_image.verify()  # This is the strict validation
                    
                    valid_images += 1
                    
                except Exception as e:
                    invalid_images += 1
                    error_type = type(e).__name__
                    validation_errors[error_type] += 1
            
            validation_rate = (valid_images / sample_size) * 100 if sample_size > 0 else 0
            
            print(f"   ✅ Valid images: {valid_images}/{sample_size} ({validation_rate:.1f}%)")
            print(f"   ❌ Invalid images: {invalid_images}/{sample_size}")
            
            if validation_errors:
                print(f"   🔍 Validation errors:")
                for error, count in validation_errors.items():
                    print(f"      {error}: {count}")
            
            # Estimate total valid images
            estimated_valid = int(len(image_members) * (validation_rate / 100))
            print(f"\n📈 Estimation:")
            print(f"   Total image files: {len(image_members)}")
            print(f"   Estimated valid: {estimated_valid}")
            print(f"   Per GPU (3 GPUs): {estimated_valid // 3}")
            
            # Check for common dataset patterns
            print(f"\n🔍 Sample file names:")
            for member in image_members[:10]:
                print(f"   {member.name}")
            if len(image_members) > 10:
                print(f"   ... and {len(image_members) - 10} more")
                
    except Exception as e:
        print(f"❌ Error analyzing TAR file: {e}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_tar.py <tar_file_or_directory>")
        print("Examples:")
        print("  python analyze_tar.py /path/to/00000.tar")
        print("  python analyze_tar.py /scratch-shared/user/blip3o_workspace/datasets/")
        sys.exit(1)
    
    path = Path(sys.argv[1])
    
    if path.is_file() and path.suffix == '.tar':
        # Analyze single TAR file
        analyze_tar_file(path)
    elif path.is_dir():
        # Analyze all TAR files in directory
        tar_files = list(path.glob("*.tar"))
        if not tar_files:
            print(f"❌ No TAR files found in {path}")
            sys.exit(1)
        
        print(f"🎯 Found {len(tar_files)} TAR files")
        
        # Analyze first few files for quick overview
        for i, tar_file in enumerate(tar_files[:3]):
            analyze_tar_file(tar_file)
            if i < 2:  # Don't add separator after last file
                print("\n" + "🔄" * 60)
        
        if len(tar_files) > 3:
            print(f"\n💡 Analyzed first 3 files. Run on individual files for full analysis.")
    else:
        print(f"❌ Path not found or not a TAR file: {path}")
        sys.exit(1)

if __name__ == "__main__":
    main()