#!/bin/bash

# Exit on error
set -e

echo "==================================="
echo "Building app.py (Main App)"
echo "==================================="

# Set cache directory
export HF_HOME=/workspace/.cache

# Clean previous builds
rm -rf build dist_app

# Run PyInstaller with your spec file
pyinstaller --clean app.spec

# Move to correct location
if [ -d "dist/my_app" ]; then
    mkdir -p dist_app
    mv dist/my_app dist_app/
    rm -rf dist
fi

echo ""
echo "✓ Build complete!"
echo "  Executable: ./dist_app/my_app/my_app"
echo ""