#!/usr/bin/env python3
"""
Скрипт реорганизации backend структуры.

Что делает:
1. Переносит apps/* на уровень выше (backend/apps/users -> backend/users)
2. Переименовывает core -> config
3. Обновляет все импорты и ссылки
4. Упрощает INSTALLED_APPS

Использование:
    python scripts/reorganize_backend.py [--dry-run]
"""

import os
import re
import shutil
from pathlib import Path
from typing import List, Tuple
import argparse


class BackendReorganizer:
    """Reorganize backend structure to be simpler and cleaner."""

    def __init__(self, backend_root: Path, dry_run: bool = False):
        self.backend_root = backend_root
        self.dry_run = dry_run
        self.apps_to_move = ["users", "diagnostics", "sensors", "rag_assistant"]
        self.changes_log: List[str] = []

    def log(self, message: str) -> None:
        """Log a change."""
        print(f"{('[DRY RUN] ' if self.dry_run else ''}{message}")
        self.changes_log.append(message)

    def move_apps_to_root(self) -> None:
        """Переносит apps/* на уровень выше."""
        apps_dir = self.backend_root / "apps"
        if not apps_dir.exists():
            self.log("⚠️  apps/ directory not found, skipping")
            return

        for app_name in self.apps_to_move:
            src = apps_dir / app_name
            dst = self.backend_root / app_name

            if not src.exists():
                self.log(f"⚠️  {src} not found, skipping")
                continue

            if dst.exists():
                self.log(f"⚠️  {dst} already exists, skipping")
                continue

            self.log(f"📁 Moving {src} -> {dst}")
            if not self.dry_run:
                shutil.move(str(src), str(dst))

        # Remove empty apps directory
        if not self.dry_run and apps_dir.exists():
            try:
                apps_dir.rmdir()
                self.log("🗑️  Removed empty apps/ directory")
            except OSError:
                self.log("⚠️  apps/ not empty, keeping it")

    def rename_core_to_config(self) -> None:
        """Переименовывает core/ -> config/."""
        src = self.backend_root / "core"
        dst = self.backend_root / "config"

        if not src.exists():
            self.log("⚠️  core/ not found, skipping")
            return

        if dst.exists():
            self.log("⚠️  config/ already exists, skipping")
            return

        self.log(f"📁 Renaming {src} -> {dst}")
        if not self.dry_run:
            shutil.move(str(src), str(dst))

    def update_imports(self) -> None:
        """Обновляет импорты в всех .py файлах."""
        patterns = [
            (r'from apps\.([\w_]+)', r'from \1'),  # from apps.users -> from users
            (r'import apps\.([\w_]+)', r'import \1'),  # import apps.users -> import users
            (r'from core\.', r'from config.'),  # from core. -> from config.
            (r'import core\.', r'import config.'),  # import core. -> import config.
            (r'"apps\.([\w_]+)', r'"\1'),  # "apps.users" -> "users"
            (r"'apps\.([\w_]+)", r"'\1"),  # 'apps.users' -> 'users'
        ]

        python_files = list(self.backend_root.rglob("*.py"))
        self.log(f"🔍 Found {len(python_files)} Python files to update")

        for py_file in python_files:
            try:
                content = py_file.read_text(encoding="utf-8")
                original = content

                for pattern, replacement in patterns:
                    content = re.sub(pattern, replacement, content)

                if content != original:
                    self.log(f"✏️  Updating imports in {py_file.relative_to(self.backend_root)}")
                    if not self.dry_run:
                        py_file.write_text(content, encoding="utf-8")

            except Exception as e:
                self.log(f"❌ Error processing {py_file}: {e}")

    def update_installed_apps(self) -> None:
        """Упрощает INSTALLED_APPS в settings.py."""
        settings_file = self.backend_root / "config" / "settings.py"
        if not settings_file.exists():
            settings_file = self.backend_root / "core" / "settings.py"

        if not settings_file.exists():
            self.log("⚠️  settings.py not found")
            return

        content = settings_file.read_text(encoding="utf-8")
        original = content

        # Simplify LOCAL_APPS
        local_apps_pattern = r'LOCAL_APPS = \[([^\]]+)\]'
        
        def simplify_apps(match):
            apps_content = match.group(1)
            # Extract app names from "apps.name.apps.NameConfig" -> "name"
            simple_apps = []
            for app in re.findall(r'"apps\.(\w+)\.apps\.\w+Config"', apps_content):
                simple_apps.append(f'    "{app}"')
            
            if not simple_apps:
                # Fallback: just remove "apps." prefix
                apps_content = re.sub(r'"apps\.([\w_]+)\.apps\.(\w+Config)"', r'"\1"', apps_content)
                return f"LOCAL_APPS = [{apps_content}]"
            
            return f"LOCAL_APPS = [\n{',\n'.join(simple_apps)},\n]"

        content = re.sub(local_apps_pattern, simplify_apps, content, flags=re.DOTALL)

        # Update manage.py references
        content = content.replace('"core.settings"', '"config.settings"')
        content = content.replace("'core.settings'", "'config.settings'")

        if content != original:
            self.log(f"✏️  Simplifying INSTALLED_APPS in {settings_file.name}")
            if not self.dry_run:
                settings_file.write_text(content, encoding="utf-8")

    def update_manage_py(self) -> None:
        """Обновляет manage.py."""
        manage_file = self.backend_root / "manage.py"
        if not manage_file.exists():
            return

        content = manage_file.read_text(encoding="utf-8")
        original = content

        content = content.replace('"core.settings"', '"config.settings"')
        content = content.replace("'core.settings'", "'config.settings'")

        if content != original:
            self.log("✏️  Updating manage.py")
            if not self.dry_run:
                manage_file.write_text(content, encoding="utf-8")

    def update_asgi_wsgi(self) -> None:
        """Обновляет asgi.py и wsgi.py."""
        for filename in ["asgi.py", "wsgi.py"]:
            for location in ["config", "core"]:
                file_path = self.backend_root / location / filename
                if file_path.exists():
                    content = file_path.read_text(encoding="utf-8")
                    original = content

                    content = content.replace('"core.settings"', '"config.settings"')
                    content = content.replace("'core.settings'", "'config.settings'")

                    if content != original:
                        self.log(f"✏️  Updating {file_path.relative_to(self.backend_root)}")
                        if not self.dry_run:
                            file_path.write_text(content, encoding="utf-8")

    def cleanup_pycache(self) -> None:
        """Удаляет все __pycache__ директории."""
        pycache_dirs = list(self.backend_root.rglob("__pycache__"))
        self.log(f"🗑️  Found {len(pycache_dirs)} __pycache__ directories to remove")

        for pycache_dir in pycache_dirs:
            if not self.dry_run:
                shutil.rmtree(pycache_dir)

    def run(self) -> None:
        """Execute the reorganization."""
        self.log("🚀 Starting backend reorganization...")
        self.log(f"📂 Backend root: {self.backend_root}")
        self.log("")

        steps = [
            ("Moving apps to root level", self.move_apps_to_root),
            ("Renaming core to config", self.rename_core_to_config),
            ("Updating imports", self.update_imports),
            ("Simplifying INSTALLED_APPS", self.update_installed_apps),
            ("Updating manage.py", self.update_manage_py),
            ("Updating ASGI/WSGI", self.update_asgi_wsgi),
            ("Cleaning up __pycache__", self.cleanup_pycache),
        ]

        for step_name, step_func in steps:
            self.log(f"\n{'='*60}")
            self.log(f"📋 {step_name}")
            self.log(f"{'='*60}")
            try:
                step_func()
            except Exception as e:
                self.log(f"❌ Error in step '{step_name}': {e}")
                if not self.dry_run:
                    raise

        self.log("\n" + "="*60)
        self.log("✅ Reorganization complete!")
        self.log("="*60)

        if self.dry_run:
            self.log("\n⚠️  This was a DRY RUN. No changes were made.")
            self.log("Run without --dry-run to apply changes.")


def main():
    parser = argparse.ArgumentParser(description="Reorganize backend structure")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes"
    )
    parser.add_argument(
        "--backend-root",
        type=Path,
        default=Path(__file__).parent.parent / "backend",
        help="Path to backend directory"
    )

    args = parser.parse_args()

    if not args.backend_root.exists():
        print(f"❌ Backend root not found: {args.backend_root}")
        return 1

    reorganizer = BackendReorganizer(args.backend_root, dry_run=args.dry_run)
    reorganizer.run()

    return 0


if __name__ == "__main__":
    exit(main())
