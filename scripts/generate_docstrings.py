import ast
import os
import traceback
from pathlib import Path
from typing import List, Dict, Optional


class DjangoAwareDocstringGenerator:
    """Генератор докстрингов с учетом Django-специфики"""
    
    def analyze_function(self, func_def: ast.FunctionDef, file_path: str) -> Dict:
        """Анализирует функцию с учетом контекста Django"""
        try:
            info = {
                'name': func_def.name,
                'args': [],
                'returns': None,
                'description': self._infer_django_description(func_def, file_path),
                'raises': []
            }
            
            # Анализ аргументов для Django views
            if hasattr(func_def, 'args') and func_def.args:
                for arg in func_def.args.args:
                    if hasattr(arg, 'arg') and arg.arg != 'self' and arg.arg != 'cls':
                        arg_info = self._analyze_django_argument(arg, file_path)
                        info['args'].append(arg_info)
            
            # Для Django моделей и форм определяем возвращаемые значения
            if 'model' in file_path or 'form' in file_path:
                info['returns'] = {
                    'type': 'None',
                    'description': 'None'
                }
            elif 'view' in file_path and func_def.name == 'get':
                info['returns'] = {
                    'type': 'HttpResponse',
                    'description': 'HTTP ответ'
                }
            
            return info
        except Exception as e:
            print(f"⚠️  Ошибка анализа функции {getattr(func_def, 'name', 'unknown')}: {e}")
            return {
                'name': getattr(func_def, 'name', 'unknown'),
                'args': [],
                'returns': None,
                'description': 'Функция требует документации',
                'raises': []
            }
    
    def _infer_django_description(self, func_def: ast.FunctionDef, file_path: str) -> str: # noqa C901
        """Определяет описание с учетом Django контекста"""
        name = getattr(func_def, 'name', 'function')
        
        # Django models
        if 'model' in file_path.lower():
            if name == 'save':
                return "Сохраняет объект модели в базу данных"
            elif name == 'delete':
                return "Удаляет объект модели из базы данных"
            elif name == '__str__':
                return "Возвращает строковое представление объекта"
            elif name.startswith('get_'):
                return f"Получает {name[4:].replace('_', ' ')}"
            elif name.startswith('clean_'):
                return f"Валидирует поле {name[6:].replace('_', ' ')}"
        
        # Django views
        elif 'view' in file_path.lower():
            if name == 'get':
                return "Обрабатывает GET запрос"
            elif name == 'post':
                return "Обрабатывает POST запрос"
            elif name == 'form_valid':
                return "Вызывается при валидной форме"
            elif name == 'form_invalid':
                return "Вызывается при невалидной форме"
        
        # Django forms
        elif 'form' in file_path.lower():
            if name == 'clean':
                return "Валидирует форму"
            elif name.startswith('clean_'):
                return f"Валидирует поле {name[6:].replace('_', ' ')}"
        
        # Django management commands
        elif 'management' in file_path.lower() and 'command' in file_path.lower():
            if name == 'handle':
                return "Основной метод выполнения команды"
        
        # Общие шаблоны
        if name.startswith('get_'):
            return f"Получает {name[4:].replace('_', ' ')}"
        elif name.startswith('set_'):
            return f"Устанавливает {name[4:].replace('_', ' ')}"
        elif name.startswith('is_'):
            return f"Проверяет, является ли {name[3:].replace('_', ' ')}"
        elif name.startswith('create_'):
            return f"Создает {name[7:].replace('_', ' ')}"
        
        return f"Выполняет {name.replace('_', ' ')}"
    
    def _analyze_django_argument(self, arg: ast.arg, file_path: str) -> Dict:
        """Анализирует аргументы с учетом Django контекста"""
        arg_name = arg.arg
        
        # Django views аргументы
        if 'view' in file_path.lower():
            if arg_name == 'request':
                return {'name': arg_name, 'type': 'HttpRequest', 'description': 'HTTP запрос'}
            elif arg_name == 'pk':
                return {'name': arg_name, 'type': 'int', 'description': 'Первичный ключ объекта'}
            elif arg_name == 'args':
                return {'name': arg_name, 'type': 'list', 'description': 'Позиционные аргументы'}
            elif arg_name == 'kwargs':
                return {'name': arg_name, 'type': 'dict', 'description': 'Именованные аргументы'}
        
        # Django forms аргументы
        elif 'form' in file_path.lower():
            if arg_name == 'data':
                return {'name': arg_name, 'type': 'dict', 'description': 'Данные формы'}
            elif arg_name == 'files':
                return {'name': arg_name, 'type': 'dict', 'description': 'Загруженные файлы'}
        
        # Общие шаблоны
        if arg_name.endswith('_id'):
            return {'name': arg_name, 'type': 'int', 'description': f'Идентификатор {arg_name[:-3]}'}
        elif arg_name.endswith('_list'):
            return {'name': arg_name, 'type': 'list', 'description': f'Список {arg_name[:-5]}'}
        
        return {'name': arg_name, 'type': 'Any', 'description': f'Параметр {arg_name}'}
    
    def generate_google_docstring(self, func_info: Dict) -> str:
        """Генерирует докстринг в Google формате"""
        try:
            lines = [f'"""{func_info["description"]}']
            lines.append('')
            
            if func_info['args']:
                lines.append('Args:')
                for arg in func_info['args']:
                    lines.append(f"    {arg['name']} ({arg['type']}): {arg['description']}")
                lines.append('')
            
            if func_info['returns']:
                lines.append('Returns:')
                lines.append(f"    {func_info['returns']['type']}: {func_info['returns']['description']}")
                lines.append('')
            
            if func_info['raises']:
                lines.append('Raises:')
                for exc in func_info['raises']:
                    lines.append(f"    {exc['type']}: {exc['description']}")
                lines.append('')
            
            lines.append('"""')
            return '\n'.join(lines)
        except Exception as e:
            print(f"⚠️  Ошибка генерации докстринга: {e}")
            return '"""Базовая документация функции."""'


def find_django_py_files(directory: str) -> List[Path]:
    """Находит все .py файлы в Django проекте, исключая миграции и т.д."""
    backend_path = Path(directory)
    if not backend_path.exists():
        print(f"❌ Директория {directory} не существует")
        return []
    
    # Исключаем директории которые не нужно обрабатывать
    exclude_dirs = {'migrations', '__pycache__', 'tests', 'test', 'fixtures', 'static', 'media'}
    
    py_files = []
    for py_file in backend_path.rglob("*.py"):
        # Пропускаем исключенные директории
        if any(exclude_dir in py_file.parts for exclude_dir in exclude_dirs):
            continue
        
        # Пропускаем файлы которые могут быть проблемными
        if py_file.name in {'settings.py', 'wsgi.py', 'asgi.py', 'urls.py'}:
            continue
            
        py_files.append(py_file)
    
    print(f"📁 Найдено {len(py_files)} .py файлов в {directory} (исключены миграции и системные файлы)")
    return py_files


def safe_parse_content(content: str, file_path: str) -> Optional[ast.AST]:
    """Безопасный парсинг содержимого файла с обработкой ошибок"""
    try:
        return ast.parse(content)
    except SyntaxError as e:
        print(f"⚠️  Синтаксическая ошибка в файле {file_path}: {e}")
        return None
    except Exception as e:
        print(f"⚠️  Ошибка парсинга файла {file_path}: {e}")
        return None


def should_skip_function(func_name: str, file_path: str, node: ast.FunctionDef) -> bool:
    """Определяет, нужно ли пропустить функцию (системные и специальные функции)"""
    skip_functions = {
        '__init__', '__new__', '__class__', '__doc__', '__module__', 
        '__weakref__', '__dict__', 'Meta', 'Objects'
    }
    
    # Пропускаем системные функции
    if func_name in skip_functions:
        return True
    
    # Пропускаем функции в миграциях
    if 'migration' in file_path:
        return True
    
    # Пропускаем функции без тела (только докстринг)
    if len(node.body) == 1:
        first_stmt = node.body[0]
        if isinstance(first_stmt, ast.Expr) and isinstance(first_stmt.value, ast.Str):
            # Функция содержит только докстринг - пропускаем чтобы не нарушить структуру
            return True
    
    return False


def ensure_function_has_body(lines: List[str], func_node: ast.FunctionDef, indent: int) -> List[str]:
    """Убеждается, что у функции есть тело после докстринга"""
    def_line_idx = func_node.lineno - 1
    
    # Ищем следующую строку после докстринга (или после объявления функции)
    # которая имеет правильный отступ и не является пустой или комментарием
    search_start = def_line_idx + 1
    has_body = False
    
    for i in range(search_start, min(search_start + 5, len(lines))):
        line = lines[i].rstrip()
        if line and not line.startswith(' ') and not line.startswith('\t'):
            # Нашли строку без отступа - конец функции
            break
        if line and (line.startswith(' ' * indent) or line.startswith('\t')):
            # Нашли строку с отступом - есть тело функции
            has_body = True
            break
        if line and (line.lstrip().startswith('#') or line.lstrip().startswith('"""')):
            # Комментарий или другой докстринг - продолжаем поиск
            continue
        if line and len(line.lstrip()) > 0:
            # Любая непустая строка - считаем телом
            has_body = True
            break
    
    if not has_body:
        # Добавляем pass как минимальное тело функции
        pass_line = ' ' * indent + 'pass'
        # Вставляем после докстринга
        insert_pos = def_line_idx + 1
        # Находим позицию после всех строк докстринга
        while insert_pos < len(lines) and (lines[insert_pos].strip().startswith('"""')
                                           or not lines[insert_pos].strip()
                                           or lines[insert_pos].strip().startswith('#')):
            insert_pos += 1
        lines.insert(insert_pos, pass_line)
        print("    ➕ Добавлен 'pass' для обеспечения тела функции")
    
    return lines


def process_django_file(file_path: Path, generator: DjangoAwareDocstringGenerator) -> bool: # noqa C901
    """Обрабатывает Django файл с учетом специфики"""
    print(f"🔍 Обрабатываю {file_path}")
    
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        tree = safe_parse_content(content, str(file_path))
        if tree is None:
            return False
        
        lines = content.split('\n')
        modified = False
        
        # Собираем все функции
        functions = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                functions.append(node)
        
        # Обрабатываем функции в обратном порядке
        for node in reversed(functions):
            try:
                if should_skip_function(node.name, str(file_path), node):
                    continue
                
                # Проверяем наличие докстринга
                existing_docstring = ast.get_docstring(node)
                if existing_docstring is None:
                    # Генерируем новый докстринг
                    func_info = generator.analyze_function(node, str(file_path))
                    docstring = generator.generate_google_docstring(func_info)
                    
                    # Определяем отступ
                    def_line_idx = node.lineno - 1
                    if def_line_idx < len(lines):
                        indent = len(lines[def_line_idx]) - len(lines[def_line_idx].lstrip())
                        
                        # Вставляем докстринг после объявления функции
                        docstring_lines = docstring.split('\n')
                        for i, line in enumerate(docstring_lines):
                            lines.insert(def_line_idx + 1 + i, ' ' * indent + line)
                        
                        # Убеждаемся, что у функции есть тело
                        lines = ensure_function_has_body(lines, node, indent)
                        
                        modified = True
                        print(f"  ✅ Добавлен докстринг для: {node.name}")
                        
            except Exception as e:
                print(f"⚠️  Ошибка обработки функции {node.name}: {e}")
                print(traceback.format_exc())
        
        if modified:
            # Создаем backup
            backup_path = file_path.with_suffix(file_path.suffix + '.backup')
            try:
                with open(backup_path, 'w', encoding='utf-8') as f:
                    f.write(content)
            except Exception:
                pass
            
            # Записываем изменения
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(lines))
                
                # Удаляем backup после успешной записи
                try:
                    if backup_path.exists():
                        backup_path.unlink()
                except Exception:
                    pass
                
                print(f"✅ Файл обновлен: {file_path}")
                return True
                
            except Exception as e:
                print(f"❌ Ошибка записи файла {file_path}: {e}")
                # Восстанавливаем из backup
                try:
                    if backup_path.exists():
                        with open(backup_path, 'r', encoding='utf-8') as f:
                            backup_content = f.read()
                        with open(file_path, 'w', encoding='utf-8') as f:
                            f.write(backup_content)
                        backup_path.unlink()
                except Exception:
                    pass
                return False
        else:
            print(f"📭 Файл не требует изменений: {file_path}")
            return False
            
    except Exception as e:
        print(f"❌ Критическая ошибка при обработке {file_path}: {e}")
        print(traceback.format_exc())
        return False


def validate_python_syntax(file_path: Path) -> bool:
    """Проверяет синтаксис Python файла после изменений"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Пытаемся скомпилировать код для проверки синтаксиса
        compile(content, str(file_path), 'exec')
        return True
    except SyntaxError as e:
        print(f"❌ Синтаксическая ошибка в {file_path}: {e}")
        return False
    except Exception as e:
        print(f"⚠️  Ошибка проверки синтаксиса {file_path}: {e}")
        return True  # Пропускаем непроверяемые файлы


def main():
    """Основная функция для Django проекта"""
    print("🚀 ЗАПУСК ГЕНЕРАЦИИ ДОКСТРИНГОВ ДЛЯ DJANGO ПРОЕКТА")
    print("📁 Целевая директория: /backend")
    print("⚡ Учитывается Django-специфика (модели, views, формы и т.д.)")
    print("🔧 Автоматически добавляются тела функций при необходимости")
    
    # Запрашиваем подтверждение
    confirm = input("Продолжить? (y/N): ").strip().lower()
    if confirm != 'y':
        print("❌ Отменено пользователем")
        return
    
    backend_dir = "backend"
    if not os.path.exists(backend_dir):
        print(f"❌ Директория {backend_dir} не найдена!")
        print("💡 Убедитесь, что скрипт запускается из корня проекта")
        return
    
    # Находим все файлы (исключая миграции)
    py_files = find_django_py_files(backend_dir)
    if not py_files:
        print("❌ Не найдено .py файлов для обработки")
        return
    
    # Ограничиваем количество файлов для первого запуска
    if len(py_files) > 20:
        print(f"⚠️  Найдено {len(py_files)} файлов. Обрабатываю первые 20")
        py_files = py_files[:20]
    
    # Создаем генератор
    generator = DjangoAwareDocstringGenerator()
    
    # Обрабатываем файлы
    successful = 0
    failed = 0
    syntax_errors = 0
    
    for file_path in py_files:
        if process_django_file(file_path, generator):
            # Проверяем синтаксис после изменений
            if validate_python_syntax(file_path):
                successful += 1
            else:
                syntax_errors += 1
                failed += 1
        else:
            failed += 1
    
    print("\n🎉 ОБРАБОТКА ЗАВЕРШЕНА!")
    print("📊 Статистика:")
    print(f"   ✅ Успешно обработано: {successful} файлов")
    print(f"   ❌ Синтаксические ошибки: {syntax_errors} файлов")
    print(f"   ❌ Не удалось обработать: {failed - syntax_errors} файлов")
    print(f"   📁 Всего файлов: {len(py_files)}")
    
    if syntax_errors > 0:
        print(f"\n⚠️  ВНИМАНИЕ: {syntax_errors} файлов содержат синтаксические ошибки после обработки!")
        print("💡 Рекомендуется проверить эти файлы вручную и исправить ошибки")
    
    if successful > 0:
        print("\n💡 Рекомендации:")
        print("   1. Проверьте Django проект: python backend/manage.py check")
        print("   2. Проверьте изменения: git status")
        print("   3. Запустите линтер: python -m ruff check backend/")
        print("   4. При необходимости отредактируйте докстринги")
        print("   5. Закоммитите изменения")


if __name__ == "__main__":
    main()
