#!/bin/bash
# Quick Validation Runner - Automatic PYTHONPATH setup
# Works on Windows (Git Bash) and Linux/macOS

set -e  # Exit on error

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Phase 3 Quick Validation Runner${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Set PYTHONPATH to current directory
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
echo -e "${GREEN}✓${NC} PYTHONPATH set to: $(pwd)"
echo ""

# Verify src module can be imported
if python -c "from src.data.edge_features import EdgeFeatureComputer" 2>/dev/null; then
    echo -e "${GREEN}✓${NC} src module import successful"
else
    echo -e "${RED}✗${NC} Cannot import src module"
    echo -e "${YELLOW}Make sure you're in services/gnn_service directory${NC}"
    exit 1
fi
echo ""

# Function to run step 1
run_step1() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}STEP 1: CSV Inspection${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo ""
    
    python scripts/inspect_csv_sensors.py \
        --input data/bim_comprehensive.csv \
        --output data/analysis
    
    echo ""
    echo -e "${GREEN}✓ Step 1 complete!${NC}"
    echo ""
}

# Function to run step 2
run_step2() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}STEP 2: Graph Conversion (8D → 14D)${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo ""
    
    python scripts/convert_graphs_to_14d.py \
        --input data/gnn_graphs_multilabel.pt \
        --edge-specs data/edge_specifications.json \
        --output data/gnn_graphs_v2_14d_test.pt \
        --max-samples 200
    
    echo ""
    echo -e "${GREEN}✓ Step 2 complete!${NC}"
    echo ""
}

# Function to run step 3
run_step3() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}STEP 3: Model Test (14D edges)${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo ""
    
    python test_14d_model.py
    
    echo ""
    echo -e "${GREEN}✓ Step 3 complete!${NC}"
    echo ""
}

# Main execution
if [ "$1" = "step1" ]; then
    run_step1
elif [ "$1" = "step2" ]; then
    run_step2
elif [ "$1" = "step3" ]; then
    run_step3
else
    # Run all steps
    run_step1
    run_step2
    run_step3
    
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}✓ ALL VALIDATION STEPS COMPLETE!${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""
    echo -e "${BLUE}Phase 3.1 Components Validated:${NC}"
    echo -e "  ${GREEN}✓${NC} 14D edge features loaded"
    echo -e "  ${GREEN}✓${NC} Model accepts 14D edges"
    echo -e "  ${GREEN}✓${NC} Forward pass successful"
    echo -e "  ${GREEN}✓${NC} Output shapes correct"
    echo -e "  ${GREEN}✓${NC} Output values in valid range"
    echo -e "  ${GREEN}✓${NC} Batch inference works"
    echo ""
    echo -e "${BLUE}🚀 Ready for:${NC}"
    echo -e "  - Full dataset conversion"
    echo -e "  - Model retraining (v2.0.0)"
    echo -e "  - Production deployment"
    echo ""
fi
