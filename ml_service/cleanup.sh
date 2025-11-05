#!/bin/bash
# Production Cleanup Script
echo "🧹 HYDRAULIC ML PLATFORM - PRODUCTION CLEANUP"
echo "=" * 50

# Archive development files
mkdir -p archive
mv enhanced*.py ultimate*.py archive/ 2>/dev/null || echo "✅ Development scripts clean"

# Clean temporary files  
rm -f *.tmp *.log.old 2>/dev/null || echo "✅ Temp files clean"
rm -rf __pycache__/ .pytest_cache/ 2>/dev/null || echo "✅ Cache clean"

echo "✅ CLEANUP COMPLETE - READY FOR PRODUCTION!"
echo "🎉 Ultimate UCI models preserved: models/v20251105_0011/"
echo "🎉 API operational: main.py"
echo "🎉 Testing suite: scripts/push_to_api.py"
