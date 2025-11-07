# ml_service Directory Structure Reorganization
# See conversation for approved structure and file destinations.

# 1. Production code
mkdir -p src/api src/models src/services src/data src/utils

# 2. Scripts and utilities
mkdir -p scripts/train scripts/test scripts/deploy scripts/data

# 3. Tests
mkdir -p tests/unit tests/integration

# 4. Documentation
mkdir -p docs

# 5. Move production files
mv config.py src/config.py

# Assume models/, services/, data/, api/ already exist and are production code
# Otherwise move/rename as needed

# 6. Move scripts
mv train_real_production_models.py scripts/train/train_production.py
mv quick_test.py scripts/test/quick_test.py
mv test_uci_loader.py tests/integration/test_data_loader.py
mv validate_setup.py scripts/test/validate_setup.py
mv make_uci_dataset.py scripts/data/make_uci_dataset.py
mv cleanup.sh scripts/deploy/cleanup.sh

# 7. Move docs
mv TRAINING.md docs/training.md
mv TESTING.md docs/testing.md
mv REAL_DATA_TRAINING.md docs/real_data_training.md
mv production_plan.md docs/production_plan.md

# Entry points stay in root (main.py, simple_predict.py)
# Update all import paths accordingly in the Python codebase.

# 8. Update all Python files for new import paths, e.g.
# from config import settings -> from src.config import settings
