#!/usr/bin/env python
"""Add ClassVar imports to all models.py and admin.py files."""
from pathlib import Path
import re


def add_classvar_import(filepath: Path) -> bool:
    """Add ClassVar import if ClassVar is used but not imported."""
    content = filepath.read_text(encoding='utf-8')
    
    # Check if ClassVar is used but not imported
    has_classvar_usage = 'ClassVar' in content or ': ClassVar[' in content
    has_classvar_import = False
    
    if 'from typing import' in content:
        typing_imports = content.split('from typing import')[1].split('\n')[0]
        has_classvar_import = 'ClassVar' in typing_imports
    
    if not has_classvar_usage or has_classvar_import:
        return False
    
    original = content
    
    # Add to existing typing import
    if 'from typing import' in content:
        content = re.sub(
            r'from typing import ([^\n]+)',
            lambda m: f"from typing import ClassVar, {m.group(1)}" if 'ClassVar' not in m.group(1) else m.group(0),
            content,
            count=1
        )
    else:
        # Add new import after __future__ if exists
        if 'from __future__' in content:
            content = re.sub(
                r'(from __future__ import [^\n]+\n)',
                r'\1\nfrom typing import ClassVar\n',
                content,
                count=1
            )
        else:
            # Find first import and add before it
            lines = content.split('\n')
            for i, line in enumerate(lines):
                if line.startswith('from ') or line.startswith('import '):
                    lines.insert(i, 'from typing import ClassVar\n')
                    break
            content = '\n'.join(lines)
    
    if content != original:
        filepath.write_text(content, encoding='utf-8')
        return True
    return False


def main():
    """Fix all models.py and admin.py files."""
    backend_dir = Path('.')
    fixed = []
    
    for pattern in ['**/models.py', '**/admin.py']:
        for filepath in backend_dir.glob(pattern):
            if 'migrations' in str(filepath):
                continue
            if add_classvar_import(filepath):
                fixed.append(filepath)
                print(f"✓ Fixed imports: {filepath}")
    
    if fixed:
        print(f"\n✓ Fixed {len(fixed)} files")
    else:
        print("\n✓ No files needed fixing")


if __name__ == '__main__':
    main()
