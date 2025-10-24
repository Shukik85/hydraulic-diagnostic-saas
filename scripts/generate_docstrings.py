"""Модуль конфигурации инструментов разработки."""

import ast
import os
from pathlib import Path
import subprocess


def get_changed_py_files(base_branch: str = "HEAD~1") -> list[str]:
    """Возвращает список изменённых .py файлов относительно base branch.

    Args:
        base_branch: Базовая ветка для сравнения.

    Returns:
        Список путей к изменённым Python файлам.

    """
    result = subprocess.run(
        ["git", "diff", "--name-only", base_branch, "--", "*.py"],
        capture_output=True,
        text=True,
        check=True,
    )
    if result.returncode != 0:
        print("Ошибка при вызове git diff:", result.stderr)
        return []
    files = result.stdout.strip().split("\n")
    return [f for f in files if f and f.endswith(".py")]


def generate_google_docstring(func_def: ast.FunctionDef, base_indent: int) -> str:
    """Создаёт Google стиль докстринга с правильным отступом.

    Args:
        func_def: AST узел функции.
        base_indent: Базовый отступ для докстринга.

    Returns:
        Отформатированный докстринг.

    """
    indent = " " * base_indent
    indent_inner = indent + " " * 4

    params = []
    for arg in func_def.args.args:
        params.append(f"{indent_inner}{arg.arg} (TYPE): описание.")

    docstring_lines = [f'{indent}"""Краткое описание функции.\n']
    if params:
        docstring_lines.append(f"{indent}Args:")
        docstring_lines.extend(params)
        docstring_lines.append("")
    if func_def.returns:
        docstring_lines.append(f"{indent}Returns:")
        docstring_lines.append(f"{indent_inner}TYPE: описание.\n")
    docstring_lines.append(f'{indent}"""')

    return "\n".join(docstring_lines)


def add_docstrings_to_functions(file_content: str) -> str | None:
    """Парсит файл, добавляет докстринги к функциям без них.

    Args:
        file_content: Содержимое Python файла.

    Returns:
        Модифицированное содержимое или None если изменений нет.

    """
    try:
        tree = ast.parse(file_content)
    except SyntaxError:
        print("Ошибка синтаксиса в файле")
        return None

    lines = file_content.split("\n")
    modified = False

    # Обратный порядок для корректной вставки по строкам
    for node in reversed(tree.body):
        if isinstance(node, ast.FunctionDef) & ast.get_docstring(node) is None:
            insert_line = node.body[0].lineno - 1

            if insert_line < len(lines):
                base_indent = len(lines[insert_line]) - len(
                    lines[insert_line].lstrip()
                )
                docstring = generate_google_docstring(node, base_indent)
                lines.insert(insert_line, docstring)
                modified = True

    if modified:
        return "\n".join(lines)
    return None


def process_file(file_path: str) -> None:
    """Обрабатывает отдельный Python файл.

    Args:
        file_path: Путь к файлу для обработки.

    """
    print(f"Обрабатываю {file_path}")
    try:
        with Path(file_path).open(encoding="utf-8") as f:
            content = f.read()

        new_content = add_docstrings_to_functions(content)
        if new_content:
            with Path(file_path).open("w", encoding="utf-8") as f:
                f.write(new_content)
            print(f"Докстринги добавлены: {file_path}")
        else:
            print(f"Изменения не требуются: {file_path}")
    except Exception as e:
        print(f"Ошибка при обработке {file_path}: {e}")


def main() -> None:
    """Основная функция скрипта."""
    base_branch = "HEAD~1"
    changed_files = get_changed_py_files(base_branch)
    if not changed_files:
        print("Нет изменений в .py файлах относительно", base_branch)
        # Обрабатываем все файлы в backend/ если нет изменений
        for root, _, files in os.walk("backend"):
            for file in files:
                if file.endswith(".py"):
                    filepath = Path(root) / file
                    process_file(filepath)
        return

    for file_path in changed_files:
        if Path("file.py").exists(file_path):
            process_file(file_path)
        else:
            print(f"Файл не найден: {file_path}")


if __name__ == "__main__":
    main()
