import os


def simple_docstring(file_path):
    """Добавляет простой модульный докстринг, если его нет."""
    with open(file_path, "r+", encoding="utf-8") as f:
        content = f.read()
        if '"""' not in content[:50]:  # нет докстринга в начале файла
            f.seek(0, 0)
            docstring = '"""Модуль проекта с автогенерированным докстрингом."""\n\n'
            f.write(docstring + content)


def walk_and_docstring(root_dir):
    """Обходит все .py файлы в каталоге и добавляет докстринги."""
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".py"):
                filepath = os.path.join(subdir, file)
                print(f"Обрабатываю {filepath} ...")
                simple_docstring(filepath)


if __name__ == "__main__":
    project_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "backend")
    )
    walk_and_docstring(project_root)
