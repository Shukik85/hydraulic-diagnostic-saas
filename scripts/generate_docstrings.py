import ast
import os
import subprocess


def get_changed_py_files(base_branch):  # Добавляем параметр base_branch
"""Краткое описание функции.

Args:
    base_branch (TYPE): описание.

"""
    result = subprocess.run(
        ["git", "diff", "--name-only", base_branch, "--", "*.py"],  # Используем base_branch вместо "HEAD~1"
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        print("Ошибка при вызове git diff:", result.stderr)
        return []
    files = result.stdout.strip().split("\n")
    return [f for f in files if f]

def generate_google_docstring(func_def):
    """Примитивный генератор докстринга в Google формате по ast.FunctionDef."""
    params = []
    for arg in func_def.args.args:
        params.append(f"{arg.arg} (TYPE): описание.")  # Убрали лишний "Args:" и отступ

    docstring_lines = [
        '"""Краткое описание функции.',
        ''
    ]

    if params:
        docstring_lines.append("Args:")
        for param in params:
            docstring_lines.append(f"    {param}")
        docstring_lines.append('')

    if func_def.returns:
        docstring_lines.append("Returns:")
        docstring_lines.append("    TYPE: описание.")
        docstring_lines.append('')

    docstring_lines.append('"""')
    return "\n".join(docstring_lines)

def add_docstrings_to_functions(file_content):
    """Парсит файл, добавляет докстринги к функциям без них."""
    try:
        tree = ast.parse(file_content)
    except SyntaxError:
        print("Ошибка синтаксиса в файле, пропускаю...")
        return None

    lines = file_content.split("\n")
    modified = False

    for node in reversed(tree.body):
        if isinstance(node, ast.FunctionDef):
            if ast.get_docstring(node) is None:
                # Получаем отступ от первой строки функции
                func_line = node.lineno - 1
                indent = len(lines[func_line]) - len(lines[func_line].lstrip())

                # Генерируем докстринг с правильным отступом
                docstring = generate_google_docstring(node)
                indented_docstring = "\n".join(" " * indent + line for line in docstring.splitlines())

                # Вставляем после сигнатуры функции
                insert_pos = func_line + 1
                lines.insert(insert_pos, indented_docstring)
                modified = True

    return "\n".join(lines) if modified else None

def process_file(file_path):
"""Краткое описание функции.

Args:
    file_path (TYPE): описание.

"""
    print(f"Обрабатываю {file_path}")
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
    except FileNotFoundError:
        print(f"Файл не найден: {file_path}")
        return

    new_content = add_docstrings_to_functions(content)
    if new_content:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(new_content)
        print(f"Докстринги добавлены: {file_path}")
    else:
        print(f"Изменения не требуются: {file_path}")

def main():
"""Краткое описание функции.

"""
    base_branch = "HEAD~1"
    changed_files = get_changed_py_files(base_branch)
    if not changed_files:
        print("Нет изменений в .py файлах относительно", base_branch)
        return

    for file_path in changed_files:
        if os.path.exists(file_path):
            process_file(file_path)
        else:
            print(f"Файл не найден: {file_path}")


if __name__ == "__main__":
    main()
