"""API client for GNN admin endpoints (FastAPI integration)"""
import requests
from typing import Any, Dict, Optional

class GNNAdminClient:
    def __init__(self, base_url: str, admin_token: Optional[str] = None):
        self.base_url = base_url.rstrip("/")
        self.token = admin_token
        self.headers = {"Authorization": f"Bearer {self.token}"} if self.token else {}
        self.timeout = 12  # seconds

    def deploy_model(self, model_path: str, version: str, description: str = "", validate_first=True) -> Dict[str, Any]:
        url = f"{self.base_url}/admin/model/deploy"
        data = {
            "model_path": model_path,
            "version": version,
            "description": description,
            "validate_first": validate_first
        }
        r = requests.post(url, json=data, headers=self.headers, timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def rollback_model(self, backup_filename: str) -> Dict[str, Any]:
        url = f"{self.base_url}/admin/model/rollback"
        data = {"backup_filename": backup_filename}
        r = requests.post(url, json=data, headers=self.headers, timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def get_model_info(self) -> Dict[str, Any]:
        url = f"{self.base_url}/admin/model/info"
        r = requests.get(url, headers=self.headers, timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def start_training(self, dataset_path: str, config: dict, experiment_name: str) -> Dict[str, Any]:
        url = f"{self.base_url}/admin/training/start"
        data = {
            "dataset_path": dataset_path,
            "config": config,
            "experiment_name": experiment_name
        }
        r = requests.post(url, json=data, headers=self.headers, timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def get_training_status(self, job_id: str) -> Dict[str, Any]:
        url = f"{self.base_url}/admin/training/{job_id}/status"
        r = requests.get(url, headers=self.headers, timeout=self.timeout)
        r.raise_for_status()
        return r.json()

# Usage example (to connect with settings/config):
# client = GNNAdminClient(base_url=settings.GNN_SERVICE_URL, admin_token=settings.GNN_ADMIN_TOKEN)
# client.deploy_model("/models/model.onnx", "2.1.0")
