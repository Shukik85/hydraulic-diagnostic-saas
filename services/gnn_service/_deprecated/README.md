# Deprecated/Legacy Code Archive

⚠️ **WARNING:** This directory contains legacy code that is **NO LONGER MAINTAINED** and should **NOT** be used in production.

## Contents

### `api_old/`
Old FastAPI application (port 8002)
- `main.py` - Legacy API endpoints
- `middleware.py` - Request ID tracking middleware

**Why deprecated:**
- Replaced by modern `app/main.py` (port 8000)
- Used outdated project structure
- Inferior error handling

**Migration path:** See `../app/main.py` for current implementation

### `legacy_docs/`
Obsolete documentation and tracking files
- `MIGRATION_SUMMARY.md` - Historical migration notes (2024)
- `ISSUE_95_CHECKLIST.md` - Old issue checklist
- `SETUP_VALIDATION.md` - Outdated setup guide

**Note:** Current setup validation is in main README.md

### `legacy_tests/`
Standalone test files not integrated into pytest
- `test_14d_model.py` - Old model testing script

**Migration path:** Tests moved to `../tests/` directory with proper organization

---

## 📚 Historical Reference

To understand how the project evolved:
1. Read main `../README.md` for current state
2. Check `../STRUCTURE.md` for architecture
3. Review `../CHANGELOG.md` for version history
4. See `../docs/` for current documentation

---

## 🔄 If You Need to Restore

These files are preserved in git history. To restore:
```bash
git log --all --full-history -- "services/gnn_service/api/main.py"
git checkout <commit-hash> -- "services/gnn_service/api/main.py"
```

---

**Last updated:** 2025-12-12  
**Archive created as part of:** Production readiness cleanup
