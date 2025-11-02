#!/usr/bin/env python3
"""
Bot Risk Classifier - Smart Auto-Approval System
Классифицирует операции на SAFE (автоматические) и RISKY (требуют approval)
"""

import re
import json
from typing import Dict, List, Tuple
from pathlib import Path

class BotRiskClassifier:
    # Операции, которые выполняются автоматически (безопасные)
    SAFE_AUTO_APPROVE = {
        "update_documentation": {
            "patterns": [r"\.md$", r"README", r"CHANGELOG", r"docs/"],
            "max_lines": 500,
            "description": "Обновление документации"
        },
        "add_tests": {
            "patterns": [r"test_.*\.py$", r".*_test\.py$", r"tests/", r"__tests__/"],
            "max_lines": 200,
            "description": "Добавление тестов"
        },
        "fix_lint_errors": {
            "patterns": [r"# ruff: noqa", r"# fmt: off", r"# type: ignore"],
            "content_only": True,
            "description": "Исправления линтеров"
        },
        "update_dependencies": {
            "patterns": [r"requirements.*\.txt$", r"package\.json$", r"pyproject\.toml$"],
            "max_lines": 100,
            "description": "Обновление зависимостей"
        },
        "add_comments": {
            "content_patterns": [r"^\s*#", r"^\s*//", r"^\s*/\*", r"^\s*\*"],
            "description": "Добавление комментариев"
        },
        "update_configs": {
            "patterns": [r"\.editorconfig$", r"\.gitignore$", r"\.prettierrc$"],
            "max_lines": 50,
            "description": "Обновление конфигов"
        }
    }
    
    # Операции, требующие manual approval (критичные)
    RISKY_REQUIRE_APPROVAL = {
        "modify_workflows": {
            "patterns": [r"\.github/workflows/.*\.yml$"],
            "description": "Изменение GitHub Actions"
        },
        "database_migrations": {
            "patterns": [r"migrations/.*\.py$", r"migrate", r"schema"],
            "description": "Миграции базы данных"
        },
        "delete_files": {
            "actions": ["delete"],
            "description": "Удаление файлов"
        },
        "production_configs": {
            "patterns": [r"docker-compose.*\.yml$", r"Dockerfile", r"\.env"],
            "description": "Продакшен конфиги"
        },
        "security_sensitive": {
            "content_patterns": [
                r"password", r"token", r"key", r"secret", r"api_key",
                r"GITHUB_TOKEN", r"SECRET_KEY", r"DATABASE_URL"
            ],
            "description": "Чувствительная информация"
        },
        "core_system_files": {
            "patterns": [r"manage\.py$", r"wsgi\.py$", r"settings.*\.py$"],
            "description": "Системные файлы"
        }
    }
    
    def __init__(self):
        self.operation_log = []
    
    def classify_operation(self, operation: Dict) -> Tuple[str, str, Dict]:
        """
        Классифицирует операцию как SAFE или RISKY
        
        Returns:
            (risk_level, reason, details)
        """
        files = operation.get("files", [])
        action = operation.get("action", "update")
        
        # Проверка на RISKY операции (приоритет)
        for risk_type, config in self.RISKY_REQUIRE_APPROVAL.items():
            if self._matches_risk_pattern(files, action, config):
                return "RISKY", f"Detected {risk_type}: {config['description']}", {
                    "risk_type": risk_type,
                    "requires_approval": True,
                    "auto_execute": False
                }
        
        # Проверка на SAFE операции
        for safe_type, config in self.SAFE_AUTO_APPROVE.items():
            if self._matches_safe_pattern(files, action, config):
                return "SAFE", f"Auto-approved {safe_type}: {config['description']}", {
                    "risk_type": safe_type,
                    "requires_approval": False,
                    "auto_execute": True
                }
        
        # По умолчанию требуем approval для неизвестных операций
        return "UNKNOWN", "Unknown operation type - requires manual approval", {
            "risk_type": "unknown",
            "requires_approval": True,
            "auto_execute": False
        }
    
    def _matches_risk_pattern(self, files: List, action: str, config: Dict) -> bool:
        """Проверяет соответствие рискованным паттернам"""
        
        # Проверка действия
        if "actions" in config and action in config["actions"]:
            return True
            
        # Проверка путей файлов
        if "patterns" in config:
            for file_info in files:
                path = file_info.get("path", "")
                for pattern in config["patterns"]:
                    if re.search(pattern, path, re.IGNORECASE):
                        return True
        
        # Проверка содержимого
        if "content_patterns" in config:
            for file_info in files:
                content = file_info.get("content", "")
                for pattern in config["content_patterns"]:
                    if re.search(pattern, content, re.IGNORECASE):
                        return True
        
        return False
    
    def _matches_safe_pattern(self, files: List, action: str, config: Dict) -> bool:
        """Проверяет соответствие безопасным паттернам"""
        
        # Проверка ограничений по строкам
        max_lines = config.get("max_lines", float('inf'))
        
        # Проверка путей файлов
        if "patterns" in config:
            matches = 0
            for file_info in files:
                path = file_info.get("path", "")
                content = file_info.get("content", "")
                
                for pattern in config["patterns"]:
                    if re.search(pattern, path, re.IGNORECASE):
                        # Проверка лимита строк
                        if len(content.splitlines()) > max_lines:
                            return False
                        matches += 1
            
            if matches > 0:
                return True
        
        # Проверка содержимого (только добавления комментариев и т.д.)
        if "content_patterns" in config:
            for file_info in files:
                content = file_info.get("content", "")
                lines = content.splitlines()
                
                comment_lines = 0
                for line in lines:
                    for pattern in config["content_patterns"]:
                        if re.search(pattern, line):
                            comment_lines += 1
                            break
                
                # Если большинство строк - комментарии
                if len(lines) > 0 and comment_lines / len(lines) > 0.6:
                    return True
        
        return False
    
    def generate_operation_summary(self, operation: Dict) -> str:
        """Генерирует человеко-читаемое описание операции"""
        files = operation.get("files", [])
        action = operation.get("action", "update")
        
        if not files:
            return f"Empty operation ({action})"
        
        file_summary = []
        for file_info in files:
            path = file_info.get("path", "unknown")
            size = len(file_info.get("content", "").splitlines())
            file_summary.append(f"{path} ({size} lines)")
        
        return f"{action.upper()}: {', '.join(file_summary[:3])}" + \
               (f" + {len(file_summary)-3} more" if len(file_summary) > 3 else "")

def main():
    """CLI interface для тестирования классификатора"""
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description="Bot Risk Classifier")
    parser.add_argument("--test", action="store_true", help="Run test cases")
    parser.add_argument("--operation", help="JSON operation to classify")
    
    args = parser.parse_args()
    classifier = BotRiskClassifier()
    
    if args.test:
        # Тестовые случаи
        test_cases = [
            {
                "action": "update",
                "files": [{"path": "README.md", "content": "# Test\nDocumentation update"}]
            },
            {
                "action": "create", 
                "files": [{"path": ".github/workflows/test.yml", "content": "name: Test"}]
            },
            {
                "action": "delete",
                "files": [{"path": "old_file.py", "content": ""}]
            }
        ]
        
        for i, test_op in enumerate(test_cases):
            risk, reason, details = classifier.classify_operation(test_op)
            summary = classifier.generate_operation_summary(test_op)
            print(f"Test {i+1}: {risk} - {reason}")
            print(f"  Summary: {summary}")
            print(f"  Details: {details}")
            print()
    
    elif args.operation:
        try:
            operation = json.loads(args.operation)
            risk, reason, details = classifier.classify_operation(operation)
            summary = classifier.generate_operation_summary(operation)
            
            result = {
                "risk_level": risk,
                "reason": reason,
                "details": details,
                "summary": summary
            }
            print(json.dumps(result, indent=2))
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()