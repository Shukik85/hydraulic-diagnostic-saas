#!/bin/bash

echo "í·¹ Cleanup: Cache & temporary files only"
echo "========================================="

# Python cache
echo "Removing Python cache..."
find services/ -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null
find services/ -type f -name "*.pyc" -delete
find services/ -type f -name "*.pyo" -delete

# Pytest cache
find services/ -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null

# MyPy cache
find services/ -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null

# Old files
find services/ -name "*.old" -delete 2>/dev/null
find services/ -name "*.backup" -delete 2>/dev/null
find services/ -name "*~" -delete 2>/dev/null

# Node modules cache (if any old ones)
find services/frontend/ -name ".nuxt" -type d -exec rm -rf {} + 2>/dev/null

echo ""
echo "âœ… Cache cleanup complete!"
echo ""
echo "All services remain intact:"
ls -1 services/

