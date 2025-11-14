#!/bin/bash

echo "í³¦ Consolidating Shared Code - Phase 1"
echo "======================================"

# Create shared structure
echo "Creating shared/ structure..."
mkdir -p services/shared/{clients,schemas,validation,utils,middleware}

# Create __init__.py files
touch services/shared/__init__.py
touch services/shared/clients/__init__.py
touch services/shared/schemas/__init__.py
touch services/shared/validation/__init__.py
touch services/shared/utils/__init__.py
touch services/shared/middleware/__init__.py

echo "âœ… Structure created!"

# Create setup.py for shared package
cat > services/shared/setup.py << 'SETUP'
from setuptools import setup, find_packages

setup(
    name="hdx-shared",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "pydantic>=2.12.0",
        "httpx>=0.25.0",
    ],
)
SETUP

echo "âœ… setup.py created!"

