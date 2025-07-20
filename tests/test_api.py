#!/usr/bin/env python3
"""
Demo script to test image upload and result saving functionality
"""

import requests
from PIL import Image
import io
import os

def create_test_image(filename, size=(256, 256), color=(255, 192, 203)):
    """Create a simple test image"""
    img = Image.new('RGB', size, color)
    img.save(filename)
    print(f"✅ Created test image: {filename}")

def test_upload_api():
    """Test the upload API endpoint"""
    # Create test images
    create_test_image('test_source.jpg', color=(255, 192, 203))  # Pink
    
    print("\n🧪 Testing upload API...")
    
    try:
        url = 'http://localhost:5000/upload'
        
        # Prepare files
        with open('test_source.jpg', 'rb') as source_file:
            files = {
                'source_image': ('test_source.jpg', source_file, 'image/jpeg')
            }
            
            data = {
                'reference_option': 'gallery',
                'selected_reference': 'vFG586.png'  # Use the reference image we saw
            }
            
            response = requests.post(url, files=files, data=data)
            
        if response.status_code == 200:
            result = response.json()
            if result.get('success'):
                print("✅ Upload API test successful!")
                print(f"   Session ID: {result.get('session_id')}")
                print(f"   Result filename: {result.get('result_filename')}")
                print(f"   Timestamp: {result.get('timestamp')}")
                
                # Check if files were saved
                saved_paths = result.get('saved_paths', {})
                for path_type, path in saved_paths.items():
                    if path and os.path.exists(path):
                        print(f"   ✅ {path_type.title()} saved: {path}")
                    else:
                        print(f"   ❌ {path_type.title()} not saved properly")
                
                return result.get('session_id')
            else:
                print(f"❌ API returned error: {result.get('error')}")
                return None
        else:
            print(f"❌ HTTP error {response.status_code}: {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ Upload test failed: {e}")
        return None
    finally:
        # Clean up test image
        if os.path.exists('test_source.jpg'):
            os.remove('test_source.jpg')

def test_gallery_api():
    """Test the gallery API endpoint"""
    print("\n🧪 Testing gallery API...")
    
    try:
        response = requests.get('http://localhost:5000/gallery')
        
        if response.status_code == 200:
            print("✅ Gallery API accessible")
            # Check if it's HTML content
            if 'html' in response.headers.get('content-type', '').lower():
                print("   ✅ Gallery page returns HTML content")
            else:
                print("   ⚠️  Gallery page doesn't return HTML")
        else:
            print(f"❌ Gallery API failed: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Gallery test failed: {e}")

def test_view_result_api(session_id):
    """Test viewing a specific result"""
    if not session_id:
        print("\n⚠️  Skipping view result test - no session ID")
        return
        
    print(f"\n🧪 Testing view result API for session: {session_id}")
    
    try:
        url = f'http://localhost:5000/view_result/{session_id}'
        response = requests.get(url)
        
        if response.status_code == 200:
            result = response.json()
            if result.get('success'):
                print("✅ View result API successful!")
                print(f"   Metadata: {result.get('metadata', {}).get('processing_time')}")
                
                images = result.get('images', {})
                for img_type, img_data in images.items():
                    if img_data:
                        print(f"   ✅ {img_type.title()} image available")
                    else:
                        print(f"   ❌ {img_type.title()} image missing")
            else:
                print(f"❌ API returned error: {result.get('error')}")
        else:
            print(f"❌ HTTP error {response.status_code}")
            
    except Exception as e:
        print(f"❌ View result test failed: {e}")

def test_health_api():
    """Test the health check endpoint"""
    print("\n🧪 Testing health API...")
    
    try:
        response = requests.get('http://localhost:5000/health')
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Health API successful!")
            print(f"   Status: {result.get('status')}")
            print(f"   Models: {result.get('models')}")
            print(f"   Timestamp: {result.get('timestamp')}")
        else:
            print(f"❌ Health API failed: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Health test failed: {e}")

def main():
    """Run all API tests"""
    print("🧪 PSGAN Web Application API Testing")
    print("=" * 50)
    
    # Test health first
    test_health_api()
    
    # Test main upload functionality
    session_id = test_upload_api()
    
    # Test gallery
    test_gallery_api()
    
    # Test viewing specific result
    test_view_result_api(session_id)
    
    print("\n" + "=" * 50)
    print("🎉 API testing completed!")
    print("💡 Check the saved_uploads/ and saved_results/ folders for saved files")

if __name__ == "__main__":
    main()
