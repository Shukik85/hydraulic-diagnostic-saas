#!/bin/bash

echo "í³‹ FastAPI Main Endpoints Audit"
echo "================================"

echo ""
echo "Routers:"
grep -r "@router\." services/backend_fastapi/app/routers/ | grep -E "(get|post|put|delete)" | sed 's/:.*@/@/' | sort | uniq

echo ""
echo "API endpoints:"
grep -r "@app\." services/backend_fastapi/api/ | grep -E "(get|post|put|delete)" | sed 's/:.*@/@/' | sort | uniq

echo ""
echo "Services:"
ls -1 services/backend_fastapi/services/*.py | xargs basename -a

