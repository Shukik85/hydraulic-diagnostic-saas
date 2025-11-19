"""Update: метод test_inference для sandbox-тестирования моделей GNN."""

from typing import Any

import requests


class GNNAdminClient:
    def __init__(self, base_url: str, admin_token: str | None = None):
        self.base_url = base_url.rstrip("/")
        self.token = admin_token
        self.headers = {"Authorization": f"Bearer {self.token}"} if self.token else {}
        self.timeout = 12

    # ... существующие методы ...

    def test_inference(
        self, equipment_id: str, time_window: dict, model_path: str, sensor_data: dict | None = None
    ) -> dict[str, Any]:
        url = f"{self.base_url}/admin/model/test_inference"
        data = {
            "equipment_id": equipment_id,
            "time_window": time_window,
            "model_path": model_path,
            "sensor_data": sensor_data,
        }
        r = requests.post(url, json=data, headers=self.headers, timeout=self.timeout)
        r.raise_for_status()
        return r.json()
