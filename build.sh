#!/bin/bash
# Build script for Ray Tracing App
# Creates a standalone executable using PyInstaller

echo "🦁 Building Ray Tracing App..."
echo "================================"

# Activate virtual environment
source .venv/bin/activate

# Clean previous builds
echo "🐸 Cleaning previous builds..."
rm -rf build dist

# Build with PyInstaller
echo "🦊 Building executable..."
pyinstaller ray_tracing_app.spec

# Check if build was successful
if [ -d "dist/RayTracingApp.app" ]; then
    echo ""
    echo "✅ Build successful!"
    echo "================================"
    echo "🐻 macOS App Bundle created at:"
    echo "   dist/RayTracingApp.app"
    echo ""
    echo "To run the app:"
    echo "   open dist/RayTracingApp.app"
    echo ""
    echo "To distribute:"
    echo "   1. Compress the .app bundle: zip -r RayTracingApp.zip dist/RayTracingApp.app"
    echo "   2. Share the .zip file with users"
    echo "   3. Users can extract and double-click to run"
    echo "================================"
elif [ -f "dist/RayTracingApp" ]; then
    echo ""
    echo "✅ Build successful!"
    echo "================================"
    echo "🐻 Executable created at:"
    echo "   dist/RayTracingApp"
    echo ""
    echo "To run:"
    echo "   ./dist/RayTracingApp"
    echo "================================"
else
    echo ""
    echo "❌ Build failed! Check the output above for errors."
    exit 1
fi
