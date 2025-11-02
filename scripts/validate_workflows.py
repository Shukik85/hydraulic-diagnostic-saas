#!/usr/bin/env python3
"""
Workflow Validation Script
Проверяет GitHub Actions workflow на обязательные ключи и опасные команды.
"""

import yaml
import sys
import re
from pathlib import Path
from typing import List, Dict, Tuple

class WorkflowValidator:
    """GitHub Actions workflow validator"""
    
    # Обязательные ключи в workflow
    REQUIRED_KEYS = ['name', 'on', 'jobs']
    
    # Опасные команды, которые могут нанести вред
    DANGEROUS_PATTERNS = [
        r'rm\s+-rf\s+/',  # Удаление корневых папок
        r'dd\s+if=',      # Опасные операции с дисками
        r'format\s+[cC]:',  # Форматирование системных дисков
        r'>\s*/dev/',     # Перенаправление в системные файлы
        r'sudo\s+chmod\s+777',  # Опасные права
        r'\$\{\{.*secrets.*\}\}.*echo',  # Леак secrets через echo
        r'curl.*\|.*sudo',  # Опасные pipe команды
    ]
    
    # Опасные секреты/права, которые не должны быть в обычных workflow
    SENSITIVE_PERMISSIONS = [
        'contents: write',
        'admin',
        'repo-token',
    ]
    
    def __init__(self):
        self.errors = []
        self.warnings = []
    
    def validate_workflow(self, file_path: Path) -> Tuple[bool, List[str], List[str]]:
        """
        Валидирует workflow файл
        
        Returns:
            (is_valid, errors, warnings)
        """
        self.errors = []
        self.warnings = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                workflow = yaml.safe_load(content)
        except yaml.YAMLError as e:
            self.errors.append(f"YAML syntax error: {e}")
            return False, self.errors, self.warnings
        except Exception as e:
            self.errors.append(f"File read error: {e}")
            return False, self.errors, self.warnings
        
        # Проверка обязательных ключей
        self._check_required_keys(workflow)
        
        # Проверка опасных команд
        self._check_dangerous_commands(content)
        
        # Проверка прав и permissions
        self._check_permissions(workflow)
        
        # Проверка лучших практик
        self._check_best_practices(workflow)
        
        is_valid = len(self.errors) == 0
        return is_valid, self.errors, self.warnings
    
    def _check_required_keys(self, workflow: Dict):
        """Проверяет наличие обязательных ключей"""
        missing = [key for key in self.REQUIRED_KEYS if key not in workflow]
        if missing:
            self.errors.append(f"Missing required keys: {missing}")
        
        # Проверка структуры jobs
        if 'jobs' in workflow:
            jobs = workflow['jobs']
            if not isinstance(jobs, dict) or not jobs:
                self.errors.append("'jobs' must be a non-empty dictionary")
            else:
                for job_name, job_config in jobs.items():
                    if not isinstance(job_config, dict):
                        self.errors.append(f"Job '{job_name}' must be a dictionary")
                    elif 'runs-on' not in job_config:
                        self.errors.append(f"Job '{job_name}' missing 'runs-on'")
    
    def _check_dangerous_commands(self, content: str):
        """Проверяет наличие опасных команд"""
        for pattern in self.DANGEROUS_PATTERNS:
            matches = re.findall(pattern, content, re.IGNORECASE | re.MULTILINE)
            if matches:
                self.errors.append(f"Dangerous command pattern detected: {pattern}")
                for match in matches:
                    self.errors.append(f"  Found: '{match}'")
    
    def _check_permissions(self, workflow: Dict):
        """Проверяет права и permissions"""
        # Проверка глобальных permissions
        if 'permissions' in workflow:
            perms = workflow['permissions']
            if isinstance(perms, dict):
                for perm, value in perms.items():
                    if perm == 'contents' and value == 'write':
                        self.warnings.append(
                            "Global 'contents: write' permission detected - consider limiting to specific jobs"
                        )
        
        # Проверка permissions в jobs
        if 'jobs' in workflow:
            for job_name, job_config in workflow['jobs'].items():
                if isinstance(job_config, dict) and 'permissions' in job_config:
                    perms = job_config['permissions']
                    if isinstance(perms, dict):
                        for perm, value in perms.items():
                            if perm == 'contents' and value == 'write':
                                self.warnings.append(
                                    f"Job '{job_name}' has 'contents: write' - ensure this is necessary"
                                )
    
    def _check_best_practices(self, workflow: Dict):
        """Проверяет соблюдение лучших практик"""
        # Проверка версий actions
        if 'jobs' in workflow:
            for job_name, job_config in workflow['jobs'].items():
                if isinstance(job_config, dict) and 'steps' in job_config:
                    steps = job_config['steps']
                    if isinstance(steps, list):
                        for i, step in enumerate(steps):
                            if isinstance(step, dict) and 'uses' in step:
                                action = step['uses']
                                # Проверка на latest теги
                                if '@latest' in action or '@main' in action or '@master' in action:
                                    self.warnings.append(
                                        f"Job '{job_name}' step {i+1}: Using unpinned version '{action}' - consider using specific version"
                                    )
        
        # Проверка concurrency
        if 'concurrency' not in workflow:
            self.warnings.append(
                "Consider adding 'concurrency' group to prevent parallel runs"
            )

def main():
    """CLI interface для валидации workflow"""
    import argparse
    
    parser = argparse.ArgumentParser(description="GitHub Workflow Validator")
    parser.add_argument("files", nargs="*", help="Workflow files to validate")
    parser.add_argument("--strict", action="store_true", help="Treat warnings as errors")
    
    args = parser.parse_args()
    
    if not args.files:
        # Автоматически найти все workflow файлы
        workflows_dir = Path('.github/workflows')
        if workflows_dir.exists():
            args.files = list(workflows_dir.glob('*.yml')) + list(workflows_dir.glob('*.yaml'))
        else:
            print("❌ .github/workflows directory not found", file=sys.stderr)
            sys.exit(1)
    
    validator = WorkflowValidator()
    all_valid = True
    
    for file_path in args.files:
        file_path = Path(file_path)
        if not file_path.exists():
            print(f"❌ {file_path}: File not found", file=sys.stderr)
            all_valid = False
            continue
        
        print(f"🔍 Validating {file_path}...")
        
        is_valid, errors, warnings = validator.validate_workflow(file_path)
        
        if errors:
            print(f"❌ {file_path}: ERRORS found")
            for error in errors:
                print(f"  • {error}")
            all_valid = False
        
        if warnings:
            print(f"⚠️ {file_path}: WARNINGS found")
            for warning in warnings:
                print(f"  • {warning}")
            
            if args.strict:
                all_valid = False
        
        if not errors and not warnings:
            print(f"✅ {file_path}: Valid")
        elif not errors:
            print(f"✅ {file_path}: Valid (with warnings)")
        
        print()
    
    if all_valid:
        print("✅ All workflows are valid!")
        sys.exit(0)
    else:
        print("❌ Some workflows have issues")
        sys.exit(1)

if __name__ == "__main__":
    main()