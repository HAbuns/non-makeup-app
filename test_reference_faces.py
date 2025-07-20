#!/usr/bin/env python3
"""
Test script to check if reference faces are loaded correctly
"""
import os
import sys

# Add paths for the clean project structure
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(project_root, 'src', 'models'))
sys.path.append(os.path.join(project_root, 'src', 'utils'))

# Configuration
REFERENCE_FACES_FOLDER = os.path.join(project_root, 'temp', 'datasets', 'reference_faces')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def test_reference_faces():
    """Test if reference faces are accessible"""
    print(f"Checking reference faces folder: {REFERENCE_FACES_FOLDER}")
    print(f"Folder exists: {os.path.exists(REFERENCE_FACES_FOLDER)}")
    
    if not os.path.exists(REFERENCE_FACES_FOLDER):
        print("ERROR: Reference faces folder does not exist!")
        return False
    
    # Get current reference face files
    current_files = {}
    image_count = 0
    
    for filename in os.listdir(REFERENCE_FACES_FOLDER):
        if allowed_file(filename):
            file_path = os.path.join(REFERENCE_FACES_FOLDER, filename)
            current_files[filename] = os.path.getmtime(file_path)
            image_count += 1
            print(f"Found image: {filename}")
    
    print(f"\nTotal reference face images found: {image_count}")
    
    if image_count == 0:
        print("ERROR: No reference face images found!")
        return False
    
    print("SUCCESS: Reference faces are accessible!")
    return True

if __name__ == "__main__":
    test_reference_faces()
