#!/bin/bash
# Training script for GNN Service

set -e

echo "=== GNN Training Pipeline ==="
echo ""

# Configuration
EQUIPMENT_IDS="EX-2024-ABC123 EX-2024-DEF456 EX-2024-GHI789"
START_DATE="2024-01-01"
END_DATE="2024-12-31"
VAL_SPLIT=0.2

echo "Equipment IDs: $EQUIPMENT_IDS"
echo "Date range: $START_DATE to $END_DATE"
echo "Validation split: $VAL_SPLIT"
echo ""

# Check CUDA availability
if python -c "import torch; print(torch.cuda.is_available())" | grep -q "True"; then
    echo "✅ CUDA available"
    python -c "import torch; print(f'CUDA version: {torch.version.cuda}')"
    python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"
else
    echo "⚠️  CUDA not available, training on CPU"
fi

echo ""
echo "Starting training..."
echo ""

# Run training
python -m gnn_service.train \
    --equipment-ids $EQUIPMENT_IDS \
    --start-date $START_DATE \
    --end-date $END_DATE \
    --val-split $VAL_SPLIT

echo ""
echo "=== Training Complete ==="
echo ""
echo "Model saved to: ./models/gnn_classifier_best.ckpt"
echo "Logs saved to: ./logs/"
echo ""
echo "To start inference server:"
echo "  python -m gnn_service.main"
echo ""
