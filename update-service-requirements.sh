#!/bin/bash

echo "Ì≥¶ Updating service requirements to use shared package"
echo "======================================================"

# Services that need shared package
SERVICES=(
    "diagnosis_service"
    "equipment_service"
)

for service in "${SERVICES[@]}"; do
    echo "Updating $service..."
    
    # Add shared package reference
    if ! grep -q "shared" "services/$service/requirements.txt" 2>/dev/null; then
        echo "-e ../shared" >> "services/$service/requirements.txt"
        echo "  ‚úÖ Added to $service/requirements.txt"
    else
        echo "  ‚ÑπÔ∏è  Already in $service/requirements.txt"
    fi
done

echo ""
echo "‚úÖ Requirements updated!"

