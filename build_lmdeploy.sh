#!/bin/bash
 
# Exit on error
set -e
 
echo "==================================="
echo "Building lmdeploy_app.py"
echo "==================================="
 
# Set cache directory
export HF_HOME=/workspace/.cache
 
# Clean previous builds
echo "Cleaning previous builds..."
rm -rf build dist_lmdeploy dist
 
# Run PyInstaller with your spec file
echo "Running PyInstaller..."
pyinstaller --clean lmdeploy_app.spec
 
# Move to correct location
if [ -d "dist/my_app" ]; then
    echo "Moving build to dist_lmdeploy..."
    mkdir -p dist_lmdeploy
    mv dist/my_app dist_lmdeploy/
    rm -rf dist
fi
 
echo ""
echo "✓ Build complete!"
echo "  Executable: ./dist_lmdeploy/my_app/my_app"
echo ""
echo "To run the application:"
echo "  cd dist_lmdeploy/my_app"
echo "  ./my_app"
echo ""