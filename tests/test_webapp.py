#!/usr/bin/env python3
"""
Simple test script to validate the PSGAN web application setup
"""

import os
import sys
from pathlib import Path

def test_directories():
    """Test if all required directories exist"""
    print("🔍 Checking directories...")
    
    required_dirs = [
        "uploads",
        "results", 
        "saved_uploads",
        "saved_results",
        "templates",
        "static/images",
        "assets/images/makeup"
    ]
    
    all_good = True
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            print(f"✅ {dir_path} exists")
        else:
            print(f"❌ {dir_path} missing")
            all_good = False
    
    return all_good

def test_templates():
    """Test if required templates exist"""
    print("\n🔍 Checking templates...")
    
    required_templates = [
        "templates/makeup_app.html",
        "templates/gallery.html"
    ]
    
    all_good = True
    for template_path in required_templates:
        if os.path.exists(template_path):
            print(f"✅ {template_path} exists")
        else:
            print(f"❌ {template_path} missing")
            all_good = False
    
    return all_good

def test_reference_images():
    """Test if reference images are available"""
    print("\n🔍 Checking reference images...")
    
    makeup_dir = "assets/images/makeup"
    if not os.path.exists(makeup_dir):
        print(f"❌ Reference directory {makeup_dir} missing")
        return False
    
    image_files = [f for f in os.listdir(makeup_dir) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if image_files:
        print(f"✅ Found {len(image_files)} reference image(s):")
        for img in image_files[:5]:  # Show first 5
            print(f"   - {img}")
        if len(image_files) > 5:
            print(f"   ... and {len(image_files) - 5} more")
        return True
    else:
        print(f"⚠️  No reference images found in {makeup_dir}")
        return False

def test_imports():
    """Test if all required packages can be imported"""
    print("\n🔍 Testing imports...")
    
    imports_to_test = [
        ("flask", "Flask"),
        ("PIL", "Pillow"),
        ("torch", "PyTorch"),
        ("torchvision", "Torchvision"),
        ("cv2", "OpenCV"),
        ("numpy", "NumPy"),
        ("dlib", "dlib"),
    ]
    
    all_good = True
    for module_name, display_name in imports_to_test:
        try:
            __import__(module_name)
            print(f"✅ {display_name} import successful")
        except ImportError as e:
            print(f"❌ {display_name} import failed: {e}")
            all_good = False
    
    return all_good

def test_psgan_imports():
    """Test PSGAN-specific imports"""
    print("\n🔍 Testing PSGAN imports...")
    
    try:
        from psgan import Inference, PostProcess
        print("✅ PSGAN Inference and PostProcess import successful")
        
        from setup import setup_config, setup_argparser
        print("✅ PSGAN setup imports successful")
        
        return True
    except ImportError as e:
        print(f"❌ PSGAN import failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 PSGAN Web Application Setup Test")
    print("=" * 50)
    
    tests = [
        test_directories,
        test_templates,
        test_reference_images,
        test_imports,
        test_psgan_imports
    ]
    
    all_passed = True
    for test_func in tests:
        if not test_func():
            all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 All tests passed! Your PSGAN web app should be ready to run.")
        print("💡 You can now start the app with: python app.py")
    else:
        print("❌ Some tests failed. Please fix the issues above.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
