#!/usr/bin/env python3
"""
AI Makeup Transfer Application Launcher
Quick start script for the clean project structure
"""

import os
import sys

# Get project root directory
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Add necessary paths
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src', 'models'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src', 'utils'))

def main():
    print("🎨 Starting AI Makeup Transfer Web Application...")
    print(f"📁 Project Root: {PROJECT_ROOT}")
    
    # Change to project root directory
    os.chdir(PROJECT_ROOT)
    
    # Import and run the app
    try:
        from src.api.app import app
        print("✅ Successfully imported Flask app")
        print("🚀 Starting server on http://localhost:5000")
        print("💄 Makeup transfer app is ready!")
        app.run(host='0.0.0.0', port=5000, debug=True)
    except ImportError as e:
        print(f"❌ Error importing app: {e}")
        print("🔧 Please check if all dependencies are installed:")
        print("   pip install -r requirements.txt")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error starting app: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
