#!/usr/bin/env python
"""Apply all remaining code quality fixes."""

import re
from pathlib import Path


def fix_docs_views():
    """Fix RUF059 in docs/views.py - unused 'created' variable."""
    file = Path("apps/docs/views.py")
    content = file.read_text(encoding="utf-8")

    # Replace 'created' with '_'
    content = content.replace(
        "progress, created = UserProgress.objects.get_or_create",
        "progress, _ = UserProgress.objects.get_or_create",
    )

    file.write_text(content, encoding="utf-8")
    print("✓ Fixed apps/docs/views.py")


def add_classvar_to_admin(filepath):
    """Add ClassVar annotations to admin file."""
    content = filepath.read_text(encoding="utf-8")
    original = content

    # Add ClassVar import if not present
    if "ClassVar" not in content:
        if "from typing import" in content:
            content = re.sub(
                r"from typing import ([^\n]+)", r"from typing import ClassVar, \1", content, count=1
            )
        else:
            # Add after imports
            lines = content.split("\n")
            for i, line in enumerate(lines):
                if line.startswith("from django.contrib"):
                    lines.insert(i, "from typing import ClassVar\n")
                    break
            content = "\n".join(lines)

    # Fix all list attributes
    patterns = [
        (
            r"(\s+)(list_display|list_filter|search_fields|readonly_fields|ordering|actions|inlines|fields|autocomplete_fields|prepopulated_fields) = (\[|\{|\"|\()",
            r"\1\2: ClassVar = \3",
        ),
    ]

    for pattern, repl in patterns:
        content = re.sub(pattern, repl, content)

    if content != original:
        filepath.write_text(content, encoding="utf-8")
        return True
    return False


def fix_all_admins():
    """Fix all admin.py files."""
    fixed = []
    for admin_file in Path(".").rglob("admin.py"):
        if "migrations" in str(admin_file):
            continue
        if add_classvar_to_admin(admin_file):
            fixed.append(admin_file)
            print(f"✓ Fixed {admin_file}")
    return fixed


def fix_docs_models():
    """Fix unique_together in docs/models.py."""
    file = Path("apps/docs/models.py")
    content = file.read_text(encoding="utf-8")

    # Add ClassVar import
    if "ClassVar" not in content.split("from typing")[0] if "from typing" in content else True:
        content = re.sub(r"from typing import", r"from typing import ClassVar,", content, count=1)

    # Fix unique_together
    content = re.sub(r"(\s+)unique_together = (\[)", r"\1unique_together: ClassVar = \2", content)

    file.write_text(content, encoding="utf-8")
    print("✓ Fixed apps/docs/models.py")


def add_noqa_to_unused_args():
    """Add noqa comments to unused arguments."""
    files_to_fix = [
        (
            "apps/gnn_config/admin.py",
            206,
            "def has_delete_permission(self, request, obj=None):  # noqa: ARG002",
        ),
        (
            "apps/equipment/admin.py",
            30,
            "def has_delete_permission(self, request, obj=None):  # noqa: ARG002",
        ),
    ]

    for filepath, line_num, new_line in files_to_fix:
        file = Path(filepath)
        lines = file.read_text(encoding="utf-8").split("\n")

        # Find and replace the line
        for i, line in enumerate(lines):
            if i + 1 == line_num and "has_delete_permission" in line:
                lines[i] = "    " + new_line
                break

        file.write_text("\n".join(lines), encoding="utf-8")
        print(f"✓ Added noqa to {filepath}")


def ignore_cyrillic_in_script():
    """Add noqa to cyrillic strings in test script."""
    file = Path("apps/gnn_config/scripts/ab_test_batch.py")
    if not file.exists():
        return

    content = file.read_text(encoding="utf-8")

    # Add noqa to lines with cyrillic
    content = content.replace(
        'f.write(f"\\nВсего протестировано: {len(rows)} записей\\n")',  # noqa: RUF001
    )
    content = content.replace(
        'print(f"\\nОтчет готов: {REPORT_CSV}, {REPORT_MD}")',  # noqa: RUF001
    )

    file.write_text(content, encoding="utf-8")
    print("✓ Fixed cyrillic in ab_test_batch.py")


def main():
    """Run all fixes."""
    print("Applying all code quality fixes...")
    print()

    fix_docs_views()
    fix_docs_models()
    admins = fix_all_admins()
    add_noqa_to_unused_args()
    ignore_cyrillic_in_script()

    print()
    print(f"✓ Fixed {len(admins)} admin files")
    print("✓ All automatic fixes applied!")
    print()
    print("Next steps:")
    print("1. Run: python fix_ruff_errors.py")
    print("2. Run: python fix_classvar_imports.py")
    print("3. Run: ruff format .")
    print("4. Run: ruff check .")


if __name__ == "__main__":
    main()
