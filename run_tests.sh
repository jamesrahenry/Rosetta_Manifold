#!/bin/bash
# Test runner for Phase 2 implementation

set -e

echo "============================================================"
echo "Rosetta Manifold - Phase 2 Test Suite"
echo "============================================================"
echo ""

# Test 1: Mathematical functions (no dependencies)
echo "=== Test 1: Mathematical Functions ==="
echo "Testing DoM, LAT, and cosine similarity implementations..."
echo ""
python tests/test_math_only.py
echo ""

# Test 2: Check if dependencies are installed
echo "=== Test 2: Dependency Check ==="
echo "Checking for PyTorch and TransformerLens..."
echo ""

if python -c "import torch; import transformer_lens" 2>/dev/null; then
    echo "✓ All dependencies installed"
    echo ""

    # Test 3: Smoke tests (requires dependencies)
    echo "=== Test 3: Smoke Tests ==="
    echo "Testing imports and basic functionality..."
    echo ""
    python tests/test_smoke.py
    echo ""

    # Test 4: Full unit tests (requires dependencies)
    echo "=== Test 4: Phase 2 Unit Tests ==="
    echo "Running comprehensive unit tests..."
    echo ""
    python tests/test_extract_vectors.py
    echo ""

    # Test 5: Phase 3 unit tests
    echo "=== Test 5: Phase 3 Unit Tests ==="
    echo "Running ablation tests..."
    echo ""
    python tests/test_ablate_vectors.py
    echo ""

else
    echo "⚠ PyTorch/TransformerLens not installed"
    echo "  Install with: pip install -r requirements.txt"
    echo "  Skipping import and unit tests"
    echo ""
fi

# Test 6: Code structure check
echo "=== Test 6: Code Structure ==="
echo "Verifying file structure..."
echo ""

files=(
    "src/extract_vectors.py"
    "src/ablate_vectors.py"
    "src/generate_dataset.py"
    "src/upload_to_opik.py"
    "requirements.txt"
    "setup.sh"
    "run_phase2.sh"
    "run_phase3.sh"
    "docs/Phase2_Usage.md"
    "docs/Phase3_Usage.md"
    "PHASE2_SUMMARY.md"
    "PHASE3_SUMMARY.md"
    "PROJECT_SUMMARY.md"
)

all_good=true
for file in "${files[@]}"; do
    if [ -f "$file" ]; then
        echo "  ✓ $file"
    else
        echo "  ✗ $file (missing)"
        all_good=false
    fi
done

if $all_good; then
    echo ""
    echo "✓ All required files present"
fi

echo ""
echo "============================================================"
echo "Test Suite Complete!"
echo "============================================================"
echo ""
echo "Next steps:"
echo "  1. Install dependencies: ./setup.sh"
echo "  2. Verify setup: python src/verify_setup.py"
echo "  3. Run Phase 2: ./run_phase2.sh all"
echo ""
