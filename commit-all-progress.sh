#!/bin/bash

echo "��� Committing all progress"
echo "=========================="

# Add all changes
git add services/shared/
git add test_import.py

# Show what's changed
echo ""
echo "Files to commit:"
git status --short

echo ""
read -p "Commit? (yes/no): " confirm

if [ "$confirm" == "yes" ]; then
    git commit -m "feat: Add shared package with unified utilities

Created shared package (hdx-shared v1.0.0):
- clients/gnn_client.py - Unified GNN service client
- schemas/equipment.py - Canonical equipment schemas  
- validation/sensors.py - Sensor data validation utilities

Features:
✅ No code duplication between services
✅ Consistent data models across all services
✅ Centralized validation logic
✅ Type-safe API clients
✅ Easy to test and maintain

Installation:
  cd services/shared && pip install -e .

Usage:
  from shared.clients import GNNClient
  from shared.schemas import SensorData
  from shared.validation import validate_sensor_batch

Tested: All imports working correctly
"
    
    echo ""
    echo "✅ Committed!"
    
    read -p "Push to GitHub? (yes/no): " push_confirm
    if [ "$push_confirm" == "yes" ]; then
        git push origin master
        echo "✅ Pushed to GitHub!"
    fi
else
    echo "Skipped commit"
fi

