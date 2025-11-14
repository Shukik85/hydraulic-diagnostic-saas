#!/bin/bash

echo "Ì≥ù Committing current clean state"
echo "=================================="

# Add all changes
git add .

# Commit
git commit -m "chore: Clean cache, Phase 0 architecture stable

- Removed Python cache (__pycache__, *.pyc)
- Removed test cache (.pytest_cache, .mypy_cache)
- All services intact and operational
- Architecture: Django Admin + FastAPI Gateway + Phase 0 Microservices

Services:
- backend/ (Django Admin)
- backend_fastapi/ (API Gateway)
- equipment_service/ (Phase 0 CRUD)
- diagnosis_service/ (Phase 0 Orchestrator)
- gnn_service/ (Phase 0 ML)
- rag_service/ (Phase 0 AI)
- simulator/ (Data Generator)
- frontend/ (Nuxt UI)
- shared/ (Common utilities)

Next: Consolidate shared code & update docker-compose
"

# Show status
echo ""
echo "Git status:"
git status

# Push
read -p "Push to remote? (yes/no): " push_confirm
if [ "$push_confirm" == "yes" ]; then
    git push origin master
    echo "‚úÖ Pushed to remote!"
fi

