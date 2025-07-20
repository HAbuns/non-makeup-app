#!/usr/bin/env python3
"""
AI Makeup Transfer Web Application - Main Entry Point
Clean and organized project structure
"""

import os
import sys

# Get project root and add paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src', 'models'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src', 'utils'))

def main():
    # Change to project root directory
    os.chdir(PROJECT_ROOT)
    
    # Import and run the app
    from src.api.app import app
    app.run(host='0.0.0.0', port=5000, debug=True)

if __name__ == '__main__':
    main()
