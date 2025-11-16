#!/usr/bin/env python
"""Script to automatically fix RUF012 errors (add ClassVar annotations)."""

from __future__ import annotations

import re
from pathlib import Path


def fix_model_ordering(content: str) -> str:
    """Fix ordering = [...] to use ClassVar."""
    # Match ordering = [...]
    pattern = r'(\s+)ordering = (\[.+?\])'
    replacement = r'\1ordering: ClassVar[list[str]] = \2'
    return re.sub(pattern, replacement, content)


def fix_model_indexes(content: str) -> str:
    """Fix indexes = [...] to use ClassVar."""
    # Match indexes = [...] (multiline)
    pattern = r'(\s+)indexes = (\[(?:[^\[\]]*|\[.*?\])*\])'
    replacement = r'\1indexes: ClassVar[list] = \2'
    return re.sub(pattern, replacement, content, flags=re.DOTALL)


def add_classvar_import(content: str) -> str:
    """Add ClassVar import if not present."""
    if 'from typing import' in content and 'ClassVar' not in content:
        # Add ClassVar to existing typing import
        content = re.sub(
            r'from typing import ([^\n]+)',
            lambda m: f"from typing import ClassVar, {m.group(1)}",
            content,
            count=1,
        )
    elif 'ClassVar' not in content:
        # Add new typing import after __future__
        if 'from __future__' in content:
            content = re.sub(
                r'(from __future__ import [^\n]+\n)',
                r'\1\nfrom typing import ClassVar\n',
                content,
                count=1,
            )
        else:
            # Add at the beginning
            content = f"from typing import ClassVar\n\n{content}"
    return content


def fix_file(filepath: Path) -> bool:
    """Fix RUF012 errors in a single file.

    Returns:
        True if file was modified, False otherwise
    """
    content = filepath.read_text(encoding='utf-8')
    original = content

    # Apply fixes
    content = fix_model_ordering(content)
    content = fix_model_indexes(content)

    # Add ClassVar import if we made changes
    if content != original:
        content = add_classvar_import(content)
        filepath.write_text(content, encoding='utf-8')
        return True
    return False


def main() -> None:
    """Fix all models.py files."""
    backend_dir = Path(__file__).parent
    apps_dir = backend_dir / 'apps'

    modified = []
    for models_file in apps_dir.rglob('models.py'):
        if fix_file(models_file):
            modified.append(models_file.relative_to(backend_dir))
            print(f"✓ Fixed: {models_file.relative_to(backend_dir)}")

    if modified:
        print(f"\n✓ Fixed {len(modified)} files")
    else:
        print("✓ No files needed fixing")


if __name__ == '__main__':
    main()
