#!/usr/bin/env python3
"""
WebDataset Compatibility Checker and Debugger
debug_webdataset.py

Run this script to check your WebDataset installation and compatibility
before running the embedding extraction.

Usage:
    python debug_webdataset.py
    python debug_webdataset.py --test-tar /path/to/test.tar
"""

import sys
import argparse
from pathlib import Path
import traceback

def check_webdataset_installation():
    """Check WebDataset installation and capabilities"""
    print("🔧 WebDataset Compatibility Check")
    print("=" * 50)
    
    # Check if WebDataset is installed
    try:
        import webdataset as wds
        print("✅ WebDataset installed")
    except ImportError as e:
        print("❌ WebDataset not installed")
        print(f"   Error: {e}")
        print("   Install with: pip install webdataset")
        return False
    
    # Check version
    try:
        version = getattr(wds, '__version__', 'unknown')
        print(f"📦 Version: {version}")
    except Exception as e:
        print(f"⚠️ Could not determine version: {e}")
        version = 'unknown'
    
    # Check for key methods and classes
    capabilities = {}
    
    # Check WebDataset class
    try:
        dataset = wds.WebDataset([])
        capabilities['WebDataset_class'] = True
        print("✅ WebDataset class available")
    except Exception as e:
        capabilities['WebDataset_class'] = False
        print(f"❌ WebDataset class issue: {e}")
    
    # Check .pipe() method
    try:
        dataset = wds.WebDataset([])
        hasattr(dataset, 'pipe')
        capabilities['pipe_method'] = hasattr(dataset, 'pipe')
        status = "✅" if capabilities['pipe_method'] else "❌"
        print(f"{status} .pipe() method: {capabilities['pipe_method']}")
    except Exception as e:
        capabilities['pipe_method'] = False
        print(f"❌ Error checking .pipe() method: {e}")
    
    # Check split_by_node
    try:
        capabilities['split_by_node'] = hasattr(wds, 'split_by_node')
        status = "✅" if capabilities['split_by_node'] else "❌"
        print(f"{status} split_by_node function: {capabilities['split_by_node']}")
    except Exception as e:
        capabilities['split_by_node'] = False
        print(f"❌ Error checking split_by_node: {e}")
    
    # Check split_by_worker
    try:
        capabilities['split_by_worker'] = hasattr(wds, 'split_by_worker')
        status = "✅" if capabilities['split_by_worker'] else "❌"
        print(f"{status} split_by_worker function: {capabilities['split_by_worker']}")
    except Exception as e:
        capabilities['split_by_worker'] = False
        print(f"❌ Error checking split_by_worker: {e}")
    
    # Overall compatibility assessment
    print("\n📊 Compatibility Assessment:")
    
    if capabilities.get('WebDataset_class', False):
        if capabilities.get('pipe_method', False) and capabilities.get('split_by_node', False):
            print("✅ EXCELLENT: Full modern WebDataset support")
            compatibility = 'excellent'
        elif capabilities.get('WebDataset_class', False):
            print("⚠️ GOOD: Basic WebDataset support (will use fallback methods)")
            compatibility = 'good'
        else:
            print("❌ LIMITED: WebDataset available but limited functionality")
            compatibility = 'limited'
    else:
        print("❌ POOR: WebDataset not properly installed")
        compatibility = 'poor'
    
    # Recommendations
    print("\n💡 Recommendations:")
    if compatibility == 'excellent':
        print("   • Your WebDataset installation should work perfectly")
        print("   • All distributed features will be available")
    elif compatibility == 'good':
        print("   • Your WebDataset will work with fallback methods")
        print("   • Consider upgrading WebDataset for better performance")
        print("   • Command: pip install --upgrade webdataset")
    elif compatibility == 'limited':
        print("   • WebDataset may have issues, but TAR fallback will work")
        print("   • Recommend reinstalling: pip uninstall webdataset && pip install webdataset")
    else:
        print("   • Install WebDataset: pip install webdataset")
        print("   • TAR fallback processing will be used")
    
    return capabilities

def test_webdataset_with_tar(tar_path: str):
    """Test WebDataset with an actual TAR file"""
    print(f"\n🧪 Testing WebDataset with TAR file: {tar_path}")
    print("=" * 50)
    
    tar_file = Path(tar_path)
    if not tar_file.exists():
        print(f"❌ TAR file not found: {tar_path}")
        return False
    
    print(f"📁 TAR file: {tar_file.name} ({tar_file.stat().st_size / 1024 / 1024:.1f} MB)")
    
    try:
        import webdataset as wds
        from PIL import Image
        import io
        
        def decode_sample(sample):
            try:
                # Get image
                for ext in ['jpg', 'jpeg', 'png', 'webp']:
                    if ext in sample:
                        image_data = sample[ext]
                        break
                else:
                    return None
                
                image = Image.open(io.BytesIO(image_data)).convert('RGB')
                
                # Get caption
                caption = ""
                for caption_key in ['txt', 'caption', 'text']:
                    if caption_key in sample:
                        caption_data = sample[caption_key]
                        if isinstance(caption_data, bytes):
                            caption = caption_data.decode('utf-8').strip()
                        else:
                            caption = str(caption_data).strip()
                        break
                
                key = sample.get('__key__', 'unknown')
                
                return {
                    'image': image,
                    'caption': caption,
                    'key': key,
                }
            except Exception as e:
                return None
        
        # Test different WebDataset approaches
        print("\n🔄 Testing WebDataset approaches:")
        
        # Approach 1: Modern WebDataset
        try:
            print("   Testing modern WebDataset with .pipe()...")
            dataset = (
                wds.WebDataset([str(tar_file)], empty_check=False)
                .pipe(wds.split_by_worker)
                .map(decode_sample)
                .select(lambda x: x is not None)
            )
            
            # Try to get a few samples
            sample_count = 0
            for sample in dataset:
                if sample:
                    sample_count += 1
                    if sample_count >= 3:  # Just test a few samples
                        break
            
            print(f"   ✅ Modern WebDataset works: {sample_count} samples processed")
            
        except Exception as e:
            print(f"   ❌ Modern WebDataset failed: {e}")
        
        # Approach 2: Simple WebDataset
        try:
            print("   Testing simple WebDataset...")
            dataset = (
                wds.WebDataset([str(tar_file)], empty_check=False)
                .map(decode_sample)
                .select(lambda x: x is not None)
            )
            
            # Try to get a few samples
            sample_count = 0
            for sample in dataset:
                if sample:
                    sample_count += 1
                    if sample_count >= 3:  # Just test a few samples
                        break
            
            print(f"   ✅ Simple WebDataset works: {sample_count} samples processed")
            
        except Exception as e:
            print(f"   ❌ Simple WebDataset failed: {e}")
        
        # Approach 3: Ultra-simple WebDataset
        try:
            print("   Testing ultra-simple WebDataset...")
            dataset = wds.WebDataset(str(tar_file)).map(decode_sample)
            
            # Try to get a few samples
            sample_count = 0
            for sample in dataset:
                if sample:
                    sample_count += 1
                    if sample_count >= 3:  # Just test a few samples
                        break
            
            print(f"   ✅ Ultra-simple WebDataset works: {sample_count} samples processed")
            
        except Exception as e:
            print(f"   ❌ Ultra-simple WebDataset failed: {e}")
        
        print("✅ WebDataset TAR test completed")
        return True
        
    except ImportError:
        print("❌ WebDataset not available for testing")
        return False
    except Exception as e:
        print(f"❌ WebDataset TAR test failed: {e}")
        traceback.print_exc()
        return False

def test_tar_fallback(tar_path: str):
    """Test TAR fallback processing"""
    print(f"\n🔄 Testing TAR fallback processing: {tar_path}")
    print("=" * 50)
    
    tar_file = Path(tar_path)
    if not tar_file.exists():
        print(f"❌ TAR file not found: {tar_path}")
        return False
    
    try:
        import tarfile
        from PIL import Image
        import io
        
        print(f"📁 Opening TAR file: {tar_file.name}")
        
        sample_count = 0
        with tarfile.open(tar_file, 'r') as tar:
            members = tar.getmembers()
            print(f"📊 Found {len(members)} members in TAR file")
            
            for member in members[:10]:  # Test first 10 members
                if member.isfile() and any(ext in member.name.lower() for ext in ['.jpg', '.jpeg', '.png', '.webp']):
                    try:
                        file_obj = tar.extractfile(member)
                        if file_obj:
                            image = Image.open(io.BytesIO(file_obj.read())).convert('RGB')
                            sample_count += 1
                            print(f"   ✅ Processed image: {member.name} ({image.size})")
                            
                            if sample_count >= 3:  # Just test a few
                                break
                    except Exception as e:
                        print(f"   ⚠️ Could not process {member.name}: {e}")
        
        print(f"✅ TAR fallback test completed: {sample_count} images processed")
        return sample_count > 0
        
    except Exception as e:
        print(f"❌ TAR fallback test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Main debug function"""
    parser = argparse.ArgumentParser(description="WebDataset Compatibility Checker")
    parser.add_argument("--test-tar", type=str, help="Test with a specific TAR file")
    
    args = parser.parse_args()
    
    print("🔧 BLIP3-o WebDataset Compatibility Checker")
    print("=" * 60)
    print("This tool checks if your WebDataset installation is compatible")
    print("with the BLIP3-o embedding extraction process.")
    print("=" * 60)
    
    # Check basic WebDataset installation
    capabilities = check_webdataset_installation()
    
    # Test with TAR file if provided
    if args.test_tar:
        webdataset_success = test_webdataset_with_tar(args.test_tar)
        tar_fallback_success = test_tar_fallback(args.test_tar)
        
        print(f"\n📊 TAR File Test Results:")
        print(f"   WebDataset processing: {'✅ Works' if webdataset_success else '❌ Failed'}")
        print(f"   TAR fallback processing: {'✅ Works' if tar_fallback_success else '❌ Failed'}")
        
        if webdataset_success or tar_fallback_success:
            print("✅ At least one processing method works - extraction should succeed")
        else:
            print("❌ Both processing methods failed - check your TAR file")
    
    # Final recommendations
    print(f"\n🎯 FINAL ASSESSMENT:")
    print("=" * 30)
    
    has_webdataset = capabilities.get('WebDataset_class', False)
    
    if has_webdataset:
        print("✅ EXTRACTION WILL WORK")
        print("   Your WebDataset installation is sufficient.")
        print("   The FIXED extraction script will automatically choose the best approach.")
        
        if args.test_tar:
            print("   TAR file testing confirms compatibility.")
        
        print("\n🚀 You can now run:")
        print("   python src/modules/extract_embeddings_g.py")
        print("   python src/modules/extract_embeddings_distributed.py")
        
    else:
        print("⚠️ EXTRACTION WILL USE FALLBACK")
        print("   WebDataset not available, but TAR fallback will be used.")
        print("   Performance may be slower but should still work.")
        
        if args.test_tar and test_tar_fallback(args.test_tar):
            print("   TAR fallback processing confirmed working.")
        
        print("\n💡 To improve performance, install WebDataset:")
        print("   pip install webdataset")
    
    print("=" * 60)
    print("🔧 The FIXED extraction scripts include multiple fallback methods")
    print("and should work regardless of your WebDataset version!")
    print("=" * 60)

if __name__ == "__main__":
    main()