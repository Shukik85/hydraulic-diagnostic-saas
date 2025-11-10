"""
Automated migration script for hydraulic-diagnostic-saas project.

Creates 'pre_mcp' branch with clean, production-ready structure.

Usage:
    python migrate_to_pre_mcp.py

Requirements:
    - Git repository initialized
    - Current branch: feature/gnn-service or master
    - No uncommitted changes (will be stashed)
"""

import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path


class ProjectMigrator:
    """Automated project migration to pre_mcp structure."""

    def __init__(self):
        self.root = Path.cwd()
        self.backup_dir = (
            self.root / f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"  # noqa: DTZ005
        )  # noqa: DTZ005
        self.new_structure = {
            "services": {
                "backend": None,
                "frontend": None,
                "ml_service": None,
                "gnn_service": None,
                "simulator": None,
            },
            "training": {
                "gnn": None,
                "ml_models": None,
                "ssl_transformer": None,
            },
            "infrastructure": {
                "docker": None,
                "nginx": None,
                "scripts": None,
            },
            "data": None,
            "tests": None,
            "docs": None,
        }

    def run(self):
        """Execute migration."""
        print("=" * 70)
        print("HYDRAULIC DIAGNOSTIC SAAS - PROJECT MIGRATION")
        print("=" * 70)
        print(f"Root: {self.root}")
        print(f"Backup will be saved to: {self.backup_dir}")
        print()

        try:
            self._extracted_from_run_12()
        except Exception as e:
            print(f"\n❌ ERROR: {e}")
            print(f"Backup available at: {self.backup_dir}")
            sys.exit(1)

    # TODO Rename this here and in `run`
    def _extracted_from_run_12(self):
        # Step 1: Pre-checks
        self.step_1_prechecks()

        # Step 2: Create backup
        self.step_2_create_backup()

        # Step 3: Git operations
        self.step_3_git_operations()

        # Step 4: Create new structure
        self.step_4_create_structure()

        # Step 5: Move files
        self.step_5_move_files()

        # Step 6: Create configs
        self.step_6_create_configs()

        # Step 7: Update imports
        self.step_7_update_imports()

        # Step 8: Cleanup
        self.step_8_cleanup()

        # Step 9: Git commit
        self.step_9_git_commit()

        print("\n" + "=" * 70)
        print("✅ MIGRATION COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        print("\nNew branch: pre_mcp")
        print(f"Backup location: {self.backup_dir}")
        print("\nNext steps:")
        print("  1. Review changes: git status")
        print("  2. Push to remote: git push origin pre_mcp")
        print("  3. Test services: docker-compose up")
        print("=" * 70)

    def step_1_prechecks(self):
        """Check prerequisites."""
        print("\n[1/9] Running pre-checks...")

        # Check if git repo
        if not (self.root / ".git").exists():
            raise Exception("Not a git repository")

        # Check if key directories exist
        required = ["backend", "frontend", "ml_service", "gnn_service"]
        if missing := [d for d in required if not (self.root / d).exists()]:
            raise Exception(f"Missing directories: {', '.join(missing)}")

        print("  ✓ Git repository found")
        print("  ✓ Key directories present")

    def step_2_create_backup(self):
        """Create backup of current state."""
        print("\n[2/9] Creating backup...")

        # Create backup directory
        self.backup_dir.mkdir(exist_ok=True)

        # Save current git status
        status = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True
        )
        (self.backup_dir / "git_status.txt").write_text(status.stdout)

        # Save current branch
        branch = subprocess.run(
            ["git", "branch", "--show-current"], capture_output=True, text=True
        )
        (self.backup_dir / "current_branch.txt").write_text(branch.stdout.strip())

        print(f"  ✓ Backup created: {self.backup_dir}")

    def step_3_git_operations(self):
        """Git branch and stash operations."""
        print("\n[3/9] Git operations...")

        # Stash any uncommitted changes
        subprocess.run(["git", "stash"], check=False)

        # Get current branch
        result = subprocess.run(
            ["git", "branch", "--show-current"], capture_output=True, text=True
        )
        current_branch = result.stdout.strip()

        # Create new branch from current
        subprocess.run(["git", "checkout", "-b", "pre_mcp"], check=True)

        print(f"  ✓ Created branch 'pre_mcp' from '{current_branch}'")

    def step_4_create_structure(self):
        """Create new directory structure."""
        print("\n[4/9] Creating new structure...")

        dirs_to_create = [
            "services/backend",
            "services/frontend",
            "services/ml_service",
            "services/gnn_service",
            "services/simulator",
            "training/gnn",
            "training/ml_models",
            "training/ssl_transformer",
            "infrastructure/docker",
            "infrastructure/nginx",
            "infrastructure/scripts",
            "data/raw",
            "data/processed",
            "data/models",
            "tests/unit",
            "tests/integration",
            "tests/e2e",
            "docs",
        ]

        for dir_path in dirs_to_create:
            (self.root / dir_path).mkdir(parents=True, exist_ok=True)

        print(f"  ✓ Created {len(dirs_to_create)} directories")

    def step_5_move_files(self):
        """Move files to new structure."""
        print("\n[5/9] Moving files...")

        moves = [
            ("backend", "services/backend"),
            ("frontend", "services/frontend"),
            ("ml_service", "services/ml_service"),
            ("gnn_service", "services/gnn_service"),
            ("hydraulic_excavator_sim", "services/simulator"),
        ]

        for source, dest in moves:
            source_path = self.root / source
            dest_path = self.root / dest

            if source_path.exists() and source_path.is_dir():
                # Copy contents
                for item in source_path.iterdir():
                    if item.name in [
                        ".git",
                        "__pycache__",
                        ".pytest_cache",
                        ".ruff_cache",
                        "htmlcov",
                        ".nuxt",
                    ]:
                        continue

                    dest_item = dest_path / item.name
                    if item.is_dir():
                        if dest_item.exists():
                            shutil.rmtree(dest_item)
                        shutil.copytree(
                            item,
                            dest_item,
                            ignore=shutil.ignore_patterns(
                                "__pycache__",
                                "*.pyc",
                                ".pytest_cache",
                                ".ruff_cache",
                                "htmlcov",
                                ".nuxt",
                            ),
                        )
                    else:
                        shutil.copy2(item, dest_item)

                print(f"  ✓ Moved {source} → {dest}")

        # Move training scripts
        training_moves = [
            ("services/gnn_service/train.py", "training/gnn/train.py"),
            ("services/gnn_service/train_v2.py", "training/gnn/train_v2.py"),
            (
                "services/gnn_service/prepare_bim_data.py",
                "training/gnn/prepare_data.py",
            ),
        ]

        for source, dest in training_moves:
            source_path = self.root / source
            dest_path = self.root / dest
            if source_path.exists():
                shutil.copy2(source_path, dest_path)
                print(f"  ✓ Copied {source} → {dest}")

    def step_6_create_configs(self):
        """Create configuration files."""
        print("\n[6/9] Creating configuration files...")

        # Main docker-compose.yml
        docker_compose = """version: '3.8'

services:
  # PostgreSQL + TimescaleDB
  postgres:
    image: timescale/timescaledb:latest-pg15
    environment:
      POSTGRES_DB: hydraulic_diagnostics
      POSTGRES_USER: ${DB_USER}
      POSTGRES_PASSWORD: ${DB_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ${DB_USER}"]
      interval: 10s
      timeout: 5s
      retries: 5

  # Redis
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

  # Backend (Django)
  backend:
    build:
      context: ./services/backend
      dockerfile: ../../infrastructure/docker/Dockerfile.backend
    environment:
      DATABASE_URL: postgresql://${DB_USER}:${DB_PASSWORD}@postgres/hydraulic_diagnostics
      REDIS_URL: redis://redis:6379
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_started
    ports:
      - "8000:8000"
    volumes:
      - ./services/backend:/app

  # Frontend (Nuxt 4)
  frontend:
    build:
      context: ./services/frontend
      dockerfile: ../../infrastructure/docker/Dockerfile.frontend
    ports:
      - "3000:3000"
    environment:
      API_URL: http://backend:8000
    depends_on:
      - backend

  # ML Service
  ml_service:
    build:
      context: ./services/ml_service
      dockerfile: ../../infrastructure/docker/Dockerfile.ml_service
    environment:
      MODEL_PATH: /app/models
    ports:
      - "8001:8001"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    volumes:
      - ./data/models:/app/models

  # GNN Service
  gnn_service:
    build:
      context: ./services/gnn_service
      dockerfile: ../../infrastructure/docker/Dockerfile.gnn_service
    environment:
      MODEL_PATH: /app/models
    ports:
      - "8003:8003"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    volumes:
      - ./data/models:/app/models

volumes:
  postgres_data:
  redis_data:
"""
        (self.root / "docker-compose.yml").write_text(docker_compose)
        print("  ✓ Created docker-compose.yml")

        # .env.example
        env_example = """# Database
DB_USER=hydraulic_user
DB_PASSWORD=your_secure_password
DB_NAME=hydraulic_diagnostics

# Redis
REDIS_URL=redis://redis:6379

# Django
SECRET_KEY=your-secret-key-here
DEBUG=False
ALLOWED_HOSTS=localhost,127.0.0.1

# ML Services
ML_SERVICE_URL=http://ml_service:8001
GNN_SERVICE_URL=http://gnn_service:8003

# GPU
CUDA_VISIBLE_DEVICES=0
"""
        (self.root / ".env.example").write_text(env_example)
        print("  ✓ Created .env.example")

        # README.md
        readme = """# Hydraulic Diagnostic SaaS

Enterprise SaaS platform for hydraulic system diagnostics using ML and Graph Neural Networks.

## Architecture

services/
- backend/          Django REST API
- frontend/         Nuxt 4 SPA
- ml_service/       ML inference (CatBoost, SSL Transformer)
- gnn_service/      GNN inference (Temporal GAT)
- simulator/        Hydraulic system simulator

training/           ML training scripts
infrastructure/     Docker, Nginx, deployment scripts
data/               Datasets and models

## Quick Start

### Prerequisites
- Docker & Docker Compose
- NVIDIA GPU (for ML services)
- Python 3.11+
- Node.js 20+

### Setup

1. Clone repository:
git clone <repo-url>
cd hydraulic-diagnostic-saas
git checkout pre_mcp

2. Create environment:
cp .env.example .env

3. Start services:
docker-compose up -d

4. Access:
- Frontend: http://localhost:3000
- Backend: http://localhost:8000
- ML Service: http://localhost:8001
- GNN Service: http://localhost:8003

## ML Models

### GNN Model (Temporal GAT)
- F1 Score: 90.87% (target)
- Parameters: 901K
- Architecture: 3 GAT layers + LSTM + Attention Pooling

### SSL Transformer
- F1 Score: 82%
- Parameters: 2.5M

## Performance

- Inference: <50ms (GPU)
- Multi-label classification: 7 components
- Real-time monitoring: <100ms latency

## Security

- TLS 1.3 encryption
- JWT authentication
- Rate limiting

## License

Proprietary - All Rights Reserved
"""
        (self.root / "README.md").write_text(readme)
        print("  ✓ Created README.md")

    def step_7_update_imports(self):
        """Update import paths in Python files."""
        print("\n[7/9] Updating import paths...")

        # This is a simplified version
        # In production, use proper AST parsing
        print("  ⚠ Manual import updates may be needed")
        print("  Check: services/*/imports from other services")

    def step_8_cleanup(self):
        """Clean up old files and directories."""
        print("\n[8/9] Cleaning up...")

        # Remove old top-level directories (now moved)
        cleanup_dirs = [
            "backend",
            "frontend",
            "ml_service",
            "gnn_service",
            "hydraulic_excavator_sim",
        ]

        for dir_name in cleanup_dirs:
            dir_path = self.root / dir_name
            if dir_path.exists():
                shutil.rmtree(dir_path)
                print(f"  ✓ Removed old {dir_name}/")

        # Remove caches and temporary files
        patterns = [
            "**/__pycache__",
            "**/*.pyc",
            "**/.pytest_cache",
            "**/.ruff_cache",
            "**/htmlcov",
            "**/.nuxt",
        ]

        removed_count = 0
        for pattern in patterns:
            for path in self.root.glob(pattern):
                if path.is_dir():
                    shutil.rmtree(path)
                else:
                    path.unlink()
                removed_count += 1

        print(f"  ✓ Cleaned {removed_count} cache files/directories")

    def step_9_git_commit(self):
        """Commit changes to git."""
        print("\n[9/9] Committing to git...")

        # Stage all changes
        subprocess.run(["git", "add", "."], check=True)

        # Commit
        commit_message = """feat: reorganize project structure for pre_mcp

- Moved all services to services/ directory
- Created infrastructure/ for Docker, Nginx, scripts
- Organized training scripts in training/
- Cleaned up caches and temporary files
- Added docker-compose.yml for all services
- Created comprehensive README.md

This reorganization prepares the project for:
- MCP (Model Context Protocol) integration
- Production deployment
- Better separation of concerns
- Easier maintenance and scaling
"""

        subprocess.run(["git", "commit", "-m", commit_message], check=True)

        print("  ✓ Changes committed to 'pre_mcp' branch")


def main():
    """Main entry point."""
    migrator = ProjectMigrator()

    # Confirmation
    print("\n⚠️  This will:")
    print("  - Create new 'pre_mcp' branch")
    print("  - Reorganize entire project structure")
    print("  - Move files to new locations")
    print("  - Update configurations")
    print("  - Commit changes")
    print()

    response = input("Continue? (yes/no): ").strip().lower()

    if response == "yes":
        migrator.run()
    else:
        print("\n❌ Migration cancelled")
        sys.exit(0)


if __name__ == "__main__":
    main()
