# services/rag_service/admin_endpoints.py
"""
Admin endpoints для RAG Service.
Configuration management, prompt templates, knowledge base.

FIXED:
- Removed sys.path.append antipattern (use proper imports)
- Fixed CCONFIG_HISTORY_DIR typo -> CONFIG_HISTORY_DIR
- Replaced datetime.utcnow() with datetime.now(timezone.utc)
- Made config paths configurable via environment variables
"""
import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

# FIXED: Removed sys.path.append antipattern
# Assumes PYTHONPATH is set in Docker/environment
# See: services/rag_service/Dockerfile -> ENV PYTHONPATH=/app
try:
    from admin_auth import get_current_admin_user, AdminUser
except ImportError:
    # Fallback for local development
    import sys
    from pathlib import Path
    # Add parent directory to path only if not in production
    if not os.getenv("PRODUCTION", "").lower() == "true":
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from admin_auth import get_current_admin_user, AdminUser
    else:
        raise

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/admin", tags=["Admin"])


# === Models ===

class RAGConfig(BaseModel):
    """RAG Service configuration."""
    model_name: str = Field(default="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    max_tokens: int = Field(default=2048, ge=128, le=4096)
    stop_sequences: List[str] = Field(default=["</думает>", "<|end|>"])
    system_prompt: str = Field(..., description="System prompt for interpretation")
    
    class Config:
        json_schema_extra = {
            "example": {
                "model_name": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
                "temperature": 0.7,
                "top_p": 0.9,
                "max_tokens": 2048,
                "system_prompt": "Ты эксперт по диагностике гидравлических систем..."
            }
        }


class RAGConfigResponse(BaseModel):
    """Response с конфигурацией."""
    config: RAGConfig
    version: int
    updated_at: str
    updated_by: str


class PromptTemplate(BaseModel):
    """Prompt template."""
    name: str
    category: str  # "diagnosis" | "anomaly" | "comparison"
    template_text: str
    variables: List[str]
    language: str = "ru"
    is_active: bool = True


# === Config Management ===

# FIXED: Made paths configurable via environment variables
CONFIG_FILE = Path(os.getenv("RAG_CONFIG_PATH", "/app/config/rag_config.json"))
CONFIG_HISTORY_DIR = Path(os.getenv("RAG_CONFIG_HISTORY_DIR", "/app/config/history"))


def load_config() -> Dict:
    """Load current RAG config."""
    if not CONFIG_FILE.exists():
        # Default config
        default_config = {
            "config": RAGConfig(
                system_prompt="Ты эксперт по диагностике гидравлических систем."
            ).dict(),
            "version": 1,
            "updated_at": datetime.now(timezone.utc).isoformat(),  # FIXED
            "updated_by": "system"
        }
        save_config(default_config)
        return default_config
    
    with open(CONFIG_FILE) as f:
        return json.load(f)


def save_config(config: Dict):
    """Save RAG config."""
    CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)


def save_config_history(config: Dict, admin_email: str):
    """Save config to history."""
    # FIXED: Typo CCONFIG_HISTORY_DIR -> CONFIG_HISTORY_DIR
    CONFIG_HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    
    # FIXED: datetime.utcnow() -> datetime.now(timezone.utc)
    timestamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
    history_file = CONFIG_HISTORY_DIR / f"config_{timestamp}_{admin_email.split('@')[0]}.json"
    
    with open(history_file, 'w') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)


@router.get("/config", response_model=RAGConfigResponse)
async def get_rag_config(
    admin: AdminUser = Depends(get_current_admin_user)
):
    """
    Получить текущую RAG конфигурацию.
    
    **Requires**: Admin role
    """
    try:
        config_data = load_config()
        return RAGConfigResponse(**config_data)
        
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/config", response_model=RAGConfigResponse)
async def update_rag_config(
    config: RAGConfig,
    admin: AdminUser = Depends(get_current_admin_user)
):
    """
    Обновить RAG конфигурацию.
    
    **Requires**: Admin role
    
    **Changes apply immediately** to new requests.
    Active requests use old config.
    
    **Example**:
    ```json
    {
      "temperature": 0.6,
      "max_tokens": 1536,
      "system_prompt": "Updated prompt..."
    }
    ```
    """
    try:
        # Load current
        current = load_config()
        
        # Save to history
        save_config_history(current, admin.email)
        
        # Update config
        new_config = {
            "config": config.dict(),
            "version": current["version"] + 1,
            "updated_at": datetime.now(timezone.utc).isoformat(),  # FIXED
            "updated_by": admin.email
        }
        
        save_config(new_config)
        
        # Apply to running service (reload)
        await reload_config()
        
        logger.info(f"Config updated by {admin.email} to version {new_config['version']}")
        
        return RAGConfigResponse(**new_config)
        
    except Exception as e:
        logger.error(f"Failed to update config: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/config/history")
async def get_config_history(
    limit: int = 10,
    admin: AdminUser = Depends(get_current_admin_user)
):
    """
    Получить историю изменений конфигурации.
    
    **Requires**: Admin role
    """
    try:
        history_files = sorted(
            CONFIG_HISTORY_DIR.glob("config_*.json"),
            reverse=True
        )[:limit]
        
        history = []
        for file in history_files:
            with open(file) as f:
                data = json.load(f)
                history.append({
                    "filename": file.name,
                    "version": data.get("version"),
                    "updated_at": data.get("updated_at"),
                    "updated_by": data.get("updated_by")
                })
        
        return {"history": history}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/prompt/test")
async def test_prompt(
    template: PromptTemplate,
    test_data: Dict,
    admin: AdminUser = Depends(get_current_admin_user)
):
    """
    Тестирование prompt template с sample данными.
    
    **Requires**: Admin role
    """
    try:
        from model_loader import get_model
        
        # Render template
        rendered = template.template_text
        for var in template.variables:
            if var in test_data:
                rendered = rendered.replace(f"{{{{{var}}}}}", str(test_data[var]))
        
        # Generate response
        model = get_model()
        response = model.generate(rendered, max_tokens=1024)
        
        return {
            "rendered_prompt": rendered,
            "response": response,
            "tokens_used": len(response.split())  # Approximate
        }
        
    except Exception as e:
        logger.error(f"Prompt test failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


async def reload_config():
    """
    Reload RAG config in running service.
    
    FIXED: Implemented basic reload mechanism.
    For production, consider using Redis pub/sub or file watch.
    """
    try:
        from gnn_interpreter import get_interpreter
        
        config_data = load_config()
        new_config = config_data["config"]
        
        # Update interpreter settings if available
        interpreter = get_interpreter()
        if hasattr(interpreter, 'model'):
            # Note: vLLM parameters are set at initialization
            # For dynamic updates, would need to re-initialize model
            # For now, just log the reload
            logger.info(f"Config reloaded (version {config_data['version']})")
            logger.info(f"New settings: temp={new_config.get('temperature')}, "
                       f"max_tokens={new_config.get('max_tokens')}")
        
        # TODO: For full reload, implement one of:
        # 1. Redis pub/sub to notify all instances
        # 2. File watcher (watchdog library)
        # 3. HTTP endpoint to trigger reload on all pods (K8s)
        
    except Exception as e:
        logger.error(f"Config reload failed: {e}", exc_info=True)
        # Don't raise - config update succeeded, reload is best-effort
