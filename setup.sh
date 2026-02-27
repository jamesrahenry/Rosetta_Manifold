#!/bin/bash
# Setup script for Rosetta Manifold project

set -e

echo "=== Rosetta Manifold Setup ==="
echo ""

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Check if conda is available
if command -v conda &> /dev/null; then
    echo "Conda detected"
    echo ""
    echo "Recommended: Create a new conda environment"
    echo "  conda create -n platonic python=3.10"
    echo "  conda activate platonic"
    echo ""
else
    echo "Conda not detected - using system Python"
fi

# Install requirements
echo "Installing Python dependencies..."
pip install -r requirements.txt

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Next steps:"
echo "  1. Configure Opik (if using): ./infra/opik/opik.sh up && ./infra/opik/opik.sh configure"
echo "  2. Verify installation: python src/verify_setup.py"
echo "  3. Run Phase 2 extraction: python src/extract_vectors.py --model llama3"
echo ""
