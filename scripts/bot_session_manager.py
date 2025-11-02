#!/usr/bin/env python3
"""
Bot Session Manager
Управляет сессиями bot operations: создание, отслеживание, rollback
"""

import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any

class BotSessionManager:
    """Manager для bot operations sessions"""
    
    def __init__(self, operations_dir: str = ".bot-operations"):
        self.operations_dir = Path(operations_dir)
        self.operations_dir.mkdir(exist_ok=True)
        self.session_file = self.operations_dir / "session.json"
        self.operations_log = self.operations_dir / "operations.log"
        
    def create_session(self, 
                      goal: str, 
                      duration: str = "4h",
                      auto_approve: List[str] = None) -> Dict[str, Any]:
        """
        Создаёт новую сессию bot operations
        
        Args:
            goal: Цель сессии (например, "timescale-ingestion-mvp")
            duration: Продолжительность (например, "4h", "30m")
            auto_approve: Список автоматически одобряемых операций
        
        Returns:
            Dict с данными сессии
        """
        if auto_approve is None:
            auto_approve = [
                "update_documentation",
                "add_tests", 
                "fix_lint_errors",
                "add_comments"
            ]
        
        # Парс duration
        duration_seconds = self._parse_duration(duration)
        expires_at = datetime.utcnow() + timedelta(seconds=duration_seconds)
        
        session_data = {
            "session_id": f"session-{int(time.time())}",
            "goal": goal,
            "created_at": datetime.utcnow().isoformat() + "Z",
            "expires_at": expires_at.isoformat() + "Z",
            "duration": duration,
            "auto_approve": auto_approve,
            "status": "active",
            "operations_count": 0,
            "last_operation": None,
            "metadata": {
                "created_by": "bot-hybrid-workflow",
                "version": "1.0.0"
            }
        }
        
        # Сохраняем сессию
        with open(self.session_file, 'w') as f:
            json.dump(session_data, f, indent=2)
        
        self._log_operation("session_created", session_data)
        
        return session_data
    
    def get_current_session(self) -> Optional[Dict[str, Any]]:
        """Получает текущую активную сессию"""
        if not self.session_file.exists():
            return None
        
        try:
            with open(self.session_file, 'r') as f:
                session = json.load(f)
            
            # Проверяем, не истекла ли сессия
            if self._is_session_expired(session):
                self._close_session("expired")
                return None
            
            return session
        
        except (json.JSONDecodeError, KeyError) as e:
            self._log_operation("session_error", {"error": str(e)})
            return None
    
    def is_operation_auto_approved(self, operation_type: str) -> bool:
        """Проверяет, можно ли автоматически одобрить операцию"""
        session = self.get_current_session()
        if not session:
            return False
        
        auto_approve = session.get("auto_approve", [])
        return operation_type in auto_approve
    
    def log_operation(self, 
                     operation_type: str, 
                     operation_data: Dict[str, Any],
                     status: str = "pending") -> str:
        """
        Логирует новую операцию
        
        Returns:
            operation_id для отслеживания
        """
        operation_id = f"op-{int(time.time())}-{hash(str(operation_data)) % 10000}"
        
        operation_record = {
            "operation_id": operation_id,
            "type": operation_type,
            "status": status,
            "created_at": datetime.utcnow().isoformat() + "Z",
            "data": operation_data,
            "rollback_data": self._prepare_rollback_data(operation_data)
        }
        
        # Сохраняем операцию в отдельный файл
        operation_file = self.operations_dir / f"{operation_id}.json"
        with open(operation_file, 'w') as f:
            json.dump(operation_record, f, indent=2)
        
        # Обновляем сессию
        self._update_session_stats(operation_id)
        
        self._log_operation("operation_logged", {
            "operation_id": operation_id,
            "type": operation_type,
            "status": status
        })
        
        return operation_id
    
    def update_operation_status(self, operation_id: str, status: str, result: Dict = None):
        """Обновляет статус операции"""
        operation_file = self.operations_dir / f"{operation_id}.json"
        
        if not operation_file.exists():
            raise ValueError(f"Operation {operation_id} not found")
        
        with open(operation_file, 'r') as f:
            operation = json.load(f)
        
        operation["status"] = status
        operation["updated_at"] = datetime.utcnow().isoformat() + "Z"
        
        if result:
            operation["result"] = result
        
        with open(operation_file, 'w') as f:
            json.dump(operation, f, indent=2)
        
        self._log_operation("operation_updated", {
            "operation_id": operation_id,
            "status": status
        })
    
    def get_rollback_candidates(self, last_n: int = 3) -> List[Dict[str, Any]]:
        """Получает операции для rollback"""
        operation_files = sorted(
            self.operations_dir.glob("op-*.json"),
            key=lambda x: x.stat().st_mtime,
            reverse=True
        )
        
        candidates = []
        for op_file in operation_files[:last_n]:
            try:
                with open(op_file, 'r') as f:
                    operation = json.load(f)
                
                # Можно откатывать только успешные операции
                if operation.get("status") == "completed":
                    candidates.append(operation)
            
            except (json.JSONDecodeError, KeyError):
                continue
        
        return candidates
    
    def close_session(self, reason: str = "manual"):
        """Закрывает текущую сессию"""
        self._close_session(reason)
    
    def get_session_summary(self) -> Dict[str, Any]:
        """Получает сводку по текущей сессии"""
        session = self.get_current_session()
        if not session:
            return {"status": "no_active_session"}
        
        # Считаем статистику операций
        operation_files = list(self.operations_dir.glob("op-*.json"))
        
        stats = {
            "total_operations": len(operation_files),
            "completed": 0,
            "failed": 0,
            "pending": 0
        }
        
        for op_file in operation_files:
            try:
                with open(op_file, 'r') as f:
                    operation = json.load(f)
                status = operation.get("status", "unknown")
                if status in stats:
                    stats[status] += 1
            except (json.JSONDecodeError, KeyError):
                continue
        
        return {
            "session": session,
            "statistics": stats,
            "expires_in": self._get_expires_in(session),
            "rollback_candidates": len(self.get_rollback_candidates())
        }
    
    # Private methods
    
    def _parse_duration(self, duration: str) -> int:
        """Парсит длительность в секунды"""
        duration = duration.lower().strip()
        
        if duration.endswith('h'):
            return int(duration[:-1]) * 3600
        elif duration.endswith('m'):
            return int(duration[:-1]) * 60
        elif duration.endswith('s'):
            return int(duration[:-1])
        else:
            # По умолчанию часы
            try:
                return int(duration) * 3600
            except ValueError:
                return 4 * 3600  # 4 часа по умолчанию
    
    def _is_session_expired(self, session: Dict[str, Any]) -> bool:
        """Проверяет, не истекла ли сессия"""
        expires_at = session.get("expires_at")
        if not expires_at:
            return False
        
        try:
            expires_datetime = datetime.fromisoformat(expires_at.replace('Z', '+00:00'))
            return datetime.utcnow().replace(tzinfo=expires_datetime.tzinfo) > expires_datetime
        except (ValueError, AttributeError):
            return True  # При ошибках парсинга считаем сессию истёкшей
    
    def _close_session(self, reason: str):
        """Закрывает сессию"""
        if self.session_file.exists():
            try:
                with open(self.session_file, 'r') as f:
                    session = json.load(f)
                
                session["status"] = "closed"
                session["closed_at"] = datetime.utcnow().isoformat() + "Z"
                session["close_reason"] = reason
                
                # Архивируем сессию
                archive_file = self.operations_dir / f"session-{session['session_id']}-archived.json"
                with open(archive_file, 'w') as f:
                    json.dump(session, f, indent=2)
                
                self.session_file.unlink()
                
                self._log_operation("session_closed", {
                    "session_id": session["session_id"],
                    "reason": reason
                })
            
            except (json.JSONDecodeError, KeyError) as e:
                self._log_operation("session_close_error", {"error": str(e)})
    
    def _update_session_stats(self, operation_id: str):
        """Обновляет статистику в сессии"""
        if not self.session_file.exists():
            return
        
        try:
            with open(self.session_file, 'r') as f:
                session = json.load(f)
            
            session["operations_count"] = session.get("operations_count", 0) + 1
            session["last_operation"] = operation_id
            session["last_operation_at"] = datetime.utcnow().isoformat() + "Z"
            
            with open(self.session_file, 'w') as f:
                json.dump(session, f, indent=2)
        
        except (json.JSONDecodeError, KeyError):
            pass  # Не критичная ошибка
    
    def _prepare_rollback_data(self, operation_data: Dict[str, Any]) -> Dict[str, Any]:
        """Подготавливает данные для rollback"""
        # Пока простое решение - сохраняем SHA файлов до изменения
        rollback_data = {
            "prepared_at": datetime.utcnow().isoformat() + "Z",
            "files_to_restore": [],
            "operations_to_reverse": []
        }
        
        # TODO: реальная логика rollback в следующей итерации
        
        return rollback_data
    
    def _get_expires_in(self, session: Dict[str, Any]) -> Optional[str]:
        """Возвращает человеко-читаемое время до истечения"""
        expires_at = session.get("expires_at")
        if not expires_at:
            return None
        
        try:
            expires_datetime = datetime.fromisoformat(expires_at.replace('Z', '+00:00'))
            now = datetime.utcnow().replace(tzinfo=expires_datetime.tzinfo)
            
            if expires_datetime <= now:
                return "expired"
            
            delta = expires_datetime - now
            hours, remainder = divmod(delta.total_seconds(), 3600)
            minutes, _ = divmod(remainder, 60)
            
            if hours >= 1:
                return f"{int(hours)}h {int(minutes)}m"
            else:
                return f"{int(minutes)}m"
        
        except (ValueError, AttributeError):
            return "unknown"
    
    def _log_operation(self, event_type: str, data: Dict[str, Any]):
        """Логирует событие в operations.log"""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "event": event_type,
            "data": data
        }
        
        with open(self.operations_log, 'a') as f:
            f.write(json.dumps(log_entry) + "\n")

def main():
    """CLI interface для тестирования session manager"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Bot Session Manager")
    parser.add_argument("--create", help="Create new session with goal")
    parser.add_argument("--duration", default="4h", help="Session duration")
    parser.add_argument("--status", action="store_true", help="Show session status")
    parser.add_argument("--close", help="Close session with reason")
    
    args = parser.parse_args()
    
    manager = BotSessionManager()
    
    if args.create:
        session = manager.create_session(goal=args.create, duration=args.duration)
        print(f"✅ Session created: {session['session_id']}")
        print(f"🎯 Goal: {session['goal']}")
        print(f"⏰ Duration: {session['duration']} (expires: {session['expires_at']})")
        print(f"🚀 Auto-approve: {', '.join(session['auto_approve'])}")
    
    elif args.status:
        summary = manager.get_session_summary()
        if summary["status"] == "no_active_session":
            print("❌ No active session")
        else:
            session = summary["session"]
            stats = summary["statistics"]
            print(f"✅ Active session: {session['session_id']}")
            print(f"🎯 Goal: {session['goal']}")
            print(f"⏰ Expires in: {summary['expires_in']}")
            print(f"📊 Operations: {stats['total_operations']} total, {stats['completed']} completed, {stats['failed']} failed")
            print(f"↩️ Rollback candidates: {summary['rollback_candidates']}")
    
    elif args.close:
        manager.close_session(reason=args.close)
        print(f"✅ Session closed: {args.close}")
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()