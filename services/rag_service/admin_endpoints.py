# services/rag_service/admin_endpoints.py
"""
Admin endpoints для RAG Service.
Configuration management, prompt templates, knowledge base.
"""
import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

import sys
sys.path.append('../shared')
from admin_auth import get_current_admin_user, AdminUser

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

CONFIG_FILE = Path("/app/config/rag_config.json")
CONFIG_HISTORY_DIR = Path("/app/config/history")


def load_config() -> Dict:
    """Load current RAG config."""
    if not CONFIG_FILE.exists():
        # Default config
        default_config = {
            "config": RAGConfig(
                system_prompt="Ты эксперт по диагностике гидравлических систем."
            ).dict(),
            "version": 1,
            "updated_at": datetime.utcnow().isoformat(),
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
    CCONFIG_HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
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
            "updated_at": datetime.utcnow().isoformat(),
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
        raise HTTPException(status_code=500, detail=str(e))


async def reload_config():
    """Reload RAG config in running service."""
    # Signal to reload (можно через Redis pub/sub или file watch)
    logger.info("Config reload triggered")
    # TODO: Implement actual reload mechanism
