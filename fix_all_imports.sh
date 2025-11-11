#!/bin/bash
# Fix all relative imports in the project

echo "í´§ Fixing relative imports across the project..."
echo ""

# Function to fix imports in a directory
fix_imports() {
    local dir=$1
    local name=$2
    
    echo "Processing $name..."
    
    find "$dir" -name "*.py" -type f \
        ! -path "*/.venv/*" \
        ! -path "*/alembic/versions/*" \
        ! -path "*/__pycache__/*" \
        -exec sed -i 's/from \.\.\./from /g' {} \;
    
    find "$dir" -name "*.py" -type f \
        ! -path "*/.venv/*" \
        ! -path "*/alembic/versions/*" \
        ! -path "*/__pycache__/*" \
        -exec sed -i 's/from \.\./from /g' {} \;
    
    local count=$(grep -r "from \.\." "$dir" --include="*.py" 2>/dev/null | \
                  grep -v "__pycache__" | grep -v ".venv" | wc -l)
    
    echo "  âœ“ $name: $count relative imports remaining"
}

# Fix all services
fix_imports "services/backend_fastapi" "Backend FastAPI"
fix_imports "services/gnn_service" "GNN Service"
fix_imports "services/rag_service" "RAG Service"
fix_imports "training" "Training Scripts"

echo ""
echo "âœ… Done! Run 'docker-compose build' to rebuild services."
