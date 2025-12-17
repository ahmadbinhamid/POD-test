#!/bin/bash

# Exit on error
set -e

echo "==================================="
echo "Building pixtral.py"
echo "==================================="

# Set cache directory
export HF_HOME=/workspace/.cache

# Clean previous builds
rm -rf build dist_pixtral

# Run PyInstaller with your spec file
pyinstaller --clean pixtral.spec

# Move to correct location
if [ -d "dist/my_app" ]; then
    mkdir -p dist_pixtral
    mv dist/my_app dist_pixtral/
    rm -rf dist
fi

echo ""
echo "✓ Build complete!"
echo "  Executable: ./dist_pixtral/my_app/my_app"
echo ""