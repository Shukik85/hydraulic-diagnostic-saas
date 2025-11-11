import yaml
from pathlib import Path

def aggregate_openapi_specs():
    specs = [
        Path("services/backend/openapi.yaml"),
        Path("services/gnn_service/openapi.yaml"),
    ]

    aggregated = {
        "openapi": "3.1.0",
        "info": {"title": "Hydraulic Diagnostics API", "version": "3.1.0"},
        "paths": {},
        "components": {"schemas": {}}
    }

    for spec_path in specs:
        if spec_path.exists():
            with open(spec_path) as f:
                spec = yaml.safe_load(f)
                aggregated["paths"].update(spec.get("paths", {}))
                aggregated["components"]["schemas"].update(
                    spec.get("components", {}).get("schemas", {})
                )
    with open("docs/api/openapi.yaml", "w") as f:
        yaml.dump(aggregated, f)

if __name__ == "__main__":
    aggregate_openapi_specs()