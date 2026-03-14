#!/bin/bash
# Check status of comprehensive testing

echo "═══════════════════════════════════════════════════════════════"
echo "  Comprehensive Test Status"
echo "═══════════════════════════════════════════════════════════════"
echo ""

# Find latest test directory
LATEST_DIR=$(ls -dt results/comprehensive_test_* 2>/dev/null | head -1)

if [ -z "$LATEST_DIR" ]; then
    echo "No test runs found."
    exit 0
fi

echo "Latest test run: $LATEST_DIR"
echo ""

# Check if process is running
if ps aux | grep -q "run_comprehensive_test.sh" | grep -v grep; then
    echo "Status: ⏳ RUNNING"
else
    echo "Status: ✅ COMPLETE"
fi

echo ""
echo "Progress:"
echo "─────────────────────────────────────────────────────────────"

# Count completed models
completed=0
total=4

for model in gpt2 qwen2-0.5b qwen2-1.5b phi2; do
    if [ -f "$LATEST_DIR/${model}.ablation.json" ]; then
        echo "  ✓ $model - COMPLETE"
        completed=$((completed + 1))
    elif [ -f "$LATEST_DIR/${model}.vectors.json" ]; then
        echo "  ⏳ $model - Extraction done, running ablation..."
    elif [ -f "$LATEST_DIR/${model}.extraction.log" ]; then
        echo "  ⏳ $model - Running extraction..."
    else
        echo "  ⏸  $model - Pending"
    fi
done

echo ""
echo "Completed: $completed / $total models"
echo ""

# Show summary if complete
if [ -f "$LATEST_DIR/SUMMARY.md" ]; then
    echo "═══════════════════════════════════════════════════════════════"
    echo "  SUMMARY REPORT AVAILABLE"
    echo "═══════════════════════════════════════════════════════════════"
    echo ""
    echo "View full report:"
    echo "  cat $LATEST_DIR/SUMMARY.md"
    echo ""
    echo "Quick stats:"
    grep -A 2 "Phase 3" "$LATEST_DIR/SUMMARY.md" | grep -E "Reduction|KL Divergence" | head -8
fi

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "To monitor live:"
echo "  tail -f /tmp/claude-1000/-home-jhenry-Source-Rosetta-Manifold/tasks/bdc4af7.output"
echo ""
