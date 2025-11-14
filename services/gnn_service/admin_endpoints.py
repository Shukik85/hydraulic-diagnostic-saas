# services/gnn_service/admin_endpoints.py
"""
Admin endpoints для GNN Service.
Model deployment, configuration, training management.
"""
<<<<<<< HEAD
import os
import logging
import shutil
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

import sys
sys.path.append('../shared')
from admin_auth import get_current_admin_user, AdminUser
=======

import logging
import shutil
import sys
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

sys.path.append("../shared")
from admin_auth import AdminUser, get_current_admin_user
>>>>>>> da16424e28634340833b5d4c6ea96234fc52ac16

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/admin", tags=["Admin"])


# === Request/Response Models ===

<<<<<<< HEAD
class DeployModelRequest(BaseModel):
    """Request для deployment GNN модели."""
    model_path: str = Field(..., description="Path to model file (.onnx or .pt)")
    version: str = Field(..., description="Model version (e.g., '2.0.1')")
    description: Optional[str] = Field(None, description="Deployment description")
=======

class DeployModelRequest(BaseModel):
    """Request для deployment GNN модели."""

    model_path: str = Field(..., description="Path to model file (.onnx or .pt)")
    version: str = Field(..., description="Model version (e.g., '2.0.1')")
    description: str | None = Field(None, description="Deployment description")
>>>>>>> da16424e28634340833b5d4c6ea96234fc52ac16
    validate_first: bool = Field(True, description="Run validation before deploy")


class DeployModelResponse(BaseModel):
    """Response после deployment."""
<<<<<<< HEAD
=======

>>>>>>> da16424e28634340833b5d4c6ea96234fc52ac16
    status: str = Field(..., description="success | failed")
    deployment_id: str
    model_version: str
    deployed_at: str
<<<<<<< HEAD
    validation_results: Optional[Dict] = None
=======
    validation_results: dict | None = None
>>>>>>> da16424e28634340833b5d4c6ea96234fc52ac16
    message: str


class ModelInfo(BaseModel):
    """Информация о текущей модели."""
<<<<<<< HEAD
=======

>>>>>>> da16424e28634340833b5d4c6ea96234fc52ac16
    model_version: str
    model_path: str
    deployed_at: str
    model_size_mb: float
<<<<<<< HEAD
    input_shape: List[int]
    output_classes: int
    framework: str  # "onnx" or "pytorch"
=======
    input_shape: list[int]
    output_classes: int
    framework: str  # "pytorch"
>>>>>>> da16424e28634340833b5d4c6ea96234fc52ac16


class TrainingJobRequest(BaseModel):
    """Запуск обучения GNN."""
<<<<<<< HEAD
    dataset_path: str = Field(..., description="Path to training data")
    config: Dict = Field(..., description="Training hyperparameters")
=======

    dataset_path: str = Field(..., description="Path to training data")
    config: dict = Field(..., description="Training hyperparameters")
>>>>>>> da16424e28634340833b5d4c6ea96234fc52ac16
    experiment_name: str = Field(..., description="Name for MLflow tracking")


class TrainingJobResponse(BaseModel):
    """Response после старта обучения."""
<<<<<<< HEAD
    job_id: str
    status: str
    started_at: str
    tensorboard_url: Optional[str] = None
=======

    job_id: str
    status: str
    started_at: str
    tensorboard_url: str | None = None
>>>>>>> da16424e28634340833b5d4c6ea96234fc52ac16


# === Endpoints ===

<<<<<<< HEAD
@router.get("/model/info", response_model=ModelInfo)
async def get_current_model_info(
    admin: AdminUser = Depends(get_current_admin_user)
):
    """
    Получить информацию о текущей production модели.
    
=======

@router.get("/model/info", response_model=ModelInfo)
async def get_current_model_info(admin: AdminUser = Depends(get_current_admin_user)):
    """
    Получить информацию о текущей production модели.

>>>>>>> da16424e28634340833b5d4c6ea96234fc52ac16
    **Requires**: Admin role
    """
    try:
        # Read current model metadata
        model_info_path = Path("/app/models/current/model_info.json")
<<<<<<< HEAD
        
        if not model_info_path.exists():
            raise HTTPException(
                status_code=404,
                detail="Model info not found"
            )
        
        import json
        with open(model_info_path) as f:
            info = json.load(f)
        
        return ModelInfo(**info)
        
    except Exception as e:
        logger.error(f"Failed to get model info: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve model info: {str(e)}"
=======

        if not model_info_path.exists():
            raise HTTPException(status_code=404, detail="Model info not found")

        import json

        with open(model_info_path) as f:
            info = json.load(f)

        return ModelInfo(**info)

    except Exception as e:
        logger.error(f"Failed to get model info: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to retrieve model info: {str(e)}"
>>>>>>> da16424e28634340833b5d4c6ea96234fc52ac16
        )


@router.post("/model/deploy", response_model=DeployModelResponse)
async def deploy_model(
<<<<<<< HEAD
    request: DeployModelRequest,
    admin: AdminUser = Depends(get_current_admin_user)
):
    """
    Deploy новой GNN модели в production.
    
=======
    request: DeployModelRequest, admin: AdminUser = Depends(get_current_admin_user)
):
    """
    Deploy новой GNN модели в production.

>>>>>>> da16424e28634340833b5d4c6ea96234fc52ac16
    **Process**:
    1. Validate model file exists
    2. (Optional) Run validation tests
    3. Backup current model
    4. Copy new model to production path
    5. Update symlink
    6. Reload inference workers
<<<<<<< HEAD
    
    **Requires**: Admin role
    
=======

    **Requires**: Admin role

>>>>>>> da16424e28634340833b5d4c6ea96234fc52ac16
    **Example**:
    ```json
    {
      "model_path": "/models/universal_gnn_v2.onnx",
      "version": "2.0.1",
      "description": "Retrained with Nov-2025 data",
      "validate_first": true
    }
    ```
    """
    deployment_id = f"dpl_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
<<<<<<< HEAD
    
=======

>>>>>>> da16424e28634340833b5d4c6ea96234fc52ac16
    try:
        # 1. Validate model file exists
        model_path = Path(request.model_path)
        if not model_path.exists():
            raise HTTPException(
<<<<<<< HEAD
                status_code=404,
                detail=f"Model file not found: {request.model_path}"
            )
        
        logger.info(f"Deploying model: {request.model_path} (version {request.version})")
        
=======
                status_code=404, detail=f"Model file not found: {request.model_path}"
            )

        logger.info(
            f"Deploying model: {request.model_path} (version {request.version})"
        )

>>>>>>> da16424e28634340833b5d4c6ea96234fc52ac16
        # 2. Optional validation
        validation_results = None
        if request.validate_first:
            logger.info("Running model validation...")
            validation_results = await validate_model(model_path)
<<<<<<< HEAD
            
            if not validation_results["valid"]:
                raise HTTPException(
                    status_code=400,
                    detail=f"Model validation failed: {validation_results['errors']}"
                )
        
        # 3. Backup current model
        current_model = Path("/app/models/current/model.onnx")
        if current_model.exists():
            backup_path = Path(f"/app/models/backups/model_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.onnx")
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(current_model, backup_path)
            logger.info(f"Backed up current model to {backup_path}")
        
=======

            if not validation_results["valid"]:
                raise HTTPException(
                    status_code=400,
                    detail=f"Model validation failed: {validation_results['errors']}",
                )

        # 3. Backup current model
        current_model = Path("/app/models/current/model.onnx")
        if current_model.exists():
            backup_path = Path(
                f"/app/models/backups/model_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.onnx"
            )
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(current_model, backup_path)
            logger.info(f"Backed up current model to {backup_path}")

>>>>>>> da16424e28634340833b5d4c6ea96234fc52ac16
        # 4. Copy new model
        production_path = Path("/app/models/current/model.onnx")
        production_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(model_path, production_path)
<<<<<<< HEAD
        
=======

>>>>>>> da16424e28634340833b5d4c6ea96234fc52ac16
        # 5. Update metadata
        model_info = {
            "model_version": request.version,
            "model_path": str(production_path),
            "deployed_at": datetime.utcnow().isoformat(),
            "deployed_by": admin.email,
            "deployment_id": deployment_id,
<<<<<<< HEAD
            "model_size_mb": model_path.stat().st_size / (1024 * 1024)
        }
        
        import json
        with open("/app/models/current/model_info.json", "w") as f:
            json.dump(model_info, f, indent=2)
        
        # 6. TODO: Trigger reload of inference workers
        # For now, manual restart required
        # In production: send signal to workers or use K8s rolling update
        
        logger.info(f"Model deployed successfully: {deployment_id}")
        
=======
            "model_size_mb": model_path.stat().st_size / (1024 * 1024),
        }

        import json

        with open("/app/models/current/model_info.json", "w") as f:
            json.dump(model_info, f, indent=2)

        # 6. TODO: Trigger reload of inference workers
        # For now, manual restart required
        # In production: send signal to workers or use K8s rolling update

        logger.info(f"Model deployed successfully: {deployment_id}")

>>>>>>> da16424e28634340833b5d4c6ea96234fc52ac16
        return DeployModelResponse(
            status="success",
            deployment_id=deployment_id,
            model_version=request.version,
            deployed_at=datetime.utcnow().isoformat(),
            validation_results=validation_results,
<<<<<<< HEAD
            message=f"Model {request.version} deployed. Restart inference workers to apply."
        )
        
=======
            message=f"Model {request.version} deployed. Restart inference workers to apply.",
        )

>>>>>>> da16424e28634340833b5d4c6ea96234fc52ac16
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Deployment failed: {e}")
<<<<<<< HEAD
        raise HTTPException(
            status_code=500,
            detail=f"Deployment failed: {str(e)}"
        )
=======
        raise HTTPException(status_code=500, detail=f"Deployment failed: {str(e)}")
>>>>>>> da16424e28634340833b5d4c6ea96234fc52ac16


@router.post("/model/rollback")
async def rollback_model(
<<<<<<< HEAD
    backup_filename: str,
    admin: AdminUser = Depends(get_current_admin_user)
):
    """
    Rollback к предыдущей версии модели.
    
=======
    backup_filename: str, admin: AdminUser = Depends(get_current_admin_user)
):
    """
    Rollback к предыдущей версии модели.

>>>>>>> da16424e28634340833b5d4c6ea96234fc52ac16
    **Requires**: Admin role
    """
    try:
        backup_path = Path(f"/app/models/backups/{backup_filename}")
<<<<<<< HEAD
        
        if not backup_path.exists():
            raise HTTPException(status_code=404, detail="Backup not found")
        
        # Restore from backup
        production_path = Path("/app/models/current/model.onnx")
        shutil.copy(backup_path, production_path)
        
        logger.info(f"Rolled back to {backup_filename}")
        
        return {
            "status": "success",
            "message": f"Rolled back to {backup_filename}"
        }
        
=======

        if not backup_path.exists():
            raise HTTPException(status_code=404, detail="Backup not found")

        # Restore from backup
        production_path = Path("/app/models/current/model.onnx")
        shutil.copy(backup_path, production_path)

        logger.info(f"Rolled back to {backup_filename}")

        return {"status": "success", "message": f"Rolled back to {backup_filename}"}

>>>>>>> da16424e28634340833b5d4c6ea96234fc52ac16
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/training/start", response_model=TrainingJobResponse)
async def start_training(
<<<<<<< HEAD
    request: TrainingJobRequest,
    admin: AdminUser = Depends(get_current_admin_user)
):
    """
    Запустить обучение GNN модели.
    
    **Requires**: Admin role
    
=======
    request: TrainingJobRequest, admin: AdminUser = Depends(get_current_admin_user)
):
    """
    Запустить обучение GNN модели.

    **Requires**: Admin role

>>>>>>> da16424e28634340833b5d4c6ea96234fc52ac16
    **Note**: Training runs as Celery task.
    """
    try:
        from tasks.training import start_training_task
<<<<<<< HEAD
        
        # Generate job ID
        job_id = f"train_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        
=======

        # Generate job ID
        job_id = f"train_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

>>>>>>> da16424e28634340833b5d4c6ea96234fc52ac16
        # Start Celery task
        task = start_training_task.delay(
            job_id=job_id,
            dataset_path=request.dataset_path,
            config=request.config,
            experiment_name=request.experiment_name,
<<<<<<< HEAD
            started_by=admin.email
        )
        
        logger.info(f"Training job started: {job_id}")
        
=======
            started_by=admin.email,
        )

        logger.info(f"Training job started: {job_id}")

>>>>>>> da16424e28634340833b5d4c6ea96234fc52ac16
        return TrainingJobResponse(
            job_id=job_id,
            status="started",
            started_at=datetime.utcnow().isoformat(),
<<<<<<< HEAD
            tensorboard_url=f"http://tensorboard:6006/#scalars&run={job_id}"
        )
        
=======
            tensorboard_url=f"http://tensorboard:6006/#scalars&run={job_id}",
        )

>>>>>>> da16424e28634340833b5d4c6ea96234fc52ac16
    except Exception as e:
        logger.error(f"Failed to start training: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/training/{job_id}/status")
async def get_training_status(
<<<<<<< HEAD
    job_id: str,
    admin: AdminUser = Depends(get_current_admin_user)
):
    """
    Получить статус и метрики обучения.
    
=======
    job_id: str, admin: AdminUser = Depends(get_current_admin_user)
):
    """
    Получить статус и метрики обучения.

>>>>>>> da16424e28634340833b5d4c6ea96234fc52ac16
    **Requires**: Admin role
    """
    try:
        # Read from database or file system
        status_file = Path(f"/app/training_jobs/{job_id}/status.json")
<<<<<<< HEAD
        
        if not status_file.exists():
            raise HTTPException(status_code=404, detail="Training job not found")
        
        import json
        with open(status_file) as f:
            status_data = json.load(f)
        
        return status_data
        
=======

        if not status_file.exists():
            raise HTTPException(status_code=404, detail="Training job not found")

        import json

        with open(status_file) as f:
            status_data = json.load(f)

        return status_data

>>>>>>> da16424e28634340833b5d4c6ea96234fc52ac16
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


<<<<<<< HEAD
async def validate_model(model_path: Path) -> Dict:
    """
    Validate model before deployment.
    
    Returns:
        dict: Validation results
    """
    try:
        # Load model
        import onnx
        model = onnx.load(str(model_path))
        
        # Basic checks
        onnx.checker.check_model(model)
        
        # Check input/output shapes
        input_shape = [d.dim_value for d in model.graph.input[0].type.tensor_type.shape.dim]
        output_shape = [d.dim_value for d in model.graph.output[0].type.tensor_type.shape.dim]
        
        return {
            "valid": True,
            "input_shape": input_shape,
            "output_shape": output_shape,
            "opset_version": model.opset_import[0].version
        }
        
    except Exception as e:
        return {
            "valid": False,
            "errors": [str(e)]
        }
=======
# async def validate_model(model_path: Path) -> Dict:
#     """
#     Validate model before deployment.

#     Returns:
#         dict: Validation results
#     """
#     try:
#         # Load model
#         import onnx # НЕ ИСПОЛЬЗОВАТЬ!!!
#         model = onnx.load(str(model_path))

#         # Basic checks
#         onnx.checker.check_model(model)

#         # Check input/output shapes
#         input_shape = [d.dim_value for d in model.graph.input[0].type.tensor_type.shape.dim]
#         output_shape = [d.dim_value for d in model.graph.output[0].type.tensor_type.shape.dim]

#         return {
#             "valid": True,
#             "input_shape": input_shape,
#             "output_shape": output_shape,
#             "opset_version": model.opset_import[0].version
#         }

#     except Exception as e:
#         return {
#             "valid": False,
#             "errors": [str(e)]
#         }
>>>>>>> da16424e28634340833b5d4c6ea96234fc52ac16
