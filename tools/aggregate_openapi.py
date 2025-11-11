#!/usr/bin/env python3
"""
Aggregate OpenAPI specifications from all services
Supports: backend_fastapi, gnn_service, rag_service
"""
import yaml
import json
import requests
from pathlib import Path
from typing import Dict, Any

# Service endpoints (when running)
SERVICES = {
    "backend_fastapi": "http://localhost:8100/openapi.json",
    "gnn_service": "http://localhost:8001/openapi.json",
    "rag_service": "http://localhost:8002/openapi.json",
}

# Local spec paths (fallback if services not running)
LOCAL_SPECS = {
    "backend_fastapi": Path("services/backend_fastapi/openapi.yaml"),
    "gnn_service": Path("services/gnn_service/openapi.yaml"),
    "rag_service": Path("services/rag_service/openapi.yaml"),
}

OUTPUT_DIR = Path("docs/openapi")


def fetch_spec_from_service(service_name: str, url: str) -> Dict[Any, Any]:
    """Fetch OpenAPI spec from running service"""
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        spec = response.json()
        print(f"✅ Fetched {service_name} from {url}")
        return spec
    except Exception as e:
        print(f"⚠️  Failed to fetch {service_name}: {e}")
        return None


def load_spec_from_file(service_name: str, path: Path) -> Dict[Any, Any]:
    """Load OpenAPI spec from local file"""
    if not path.exists():
        print(f"❌ {service_name} spec not found: {path}")
        return None

    try:
        with open(path) as f:
            if path.suffix == '.yaml' or path.suffix == '.yml':
                spec = yaml.safe_load(f)
            else:
                spec = json.load(f)
        print(f"✅ Loaded {service_name} from {path}")
        return spec
    except Exception as e:
        print(f"❌ Failed to load {service_name}: {e}")
        return None


def aggregate_specs() -> Dict[Any, Any]:
    """Aggregate all service specs into one"""
    aggregated = {
        "openapi": "3.1.0",
        "info": {
            "title": "Hydraulic Diagnostics Platform API",
            "version": "1.0.0",
            "description": "Complete API for Hydraulic Diagnostics SaaS Platform",
            "contact": {
                "name": "API Support",
                "email": "support@hydraulic-diagnostics.com"
            }
        },
        "servers": [
            {
                "url": "https://api.hydraulic-diagnostics.com",
                "description": "Production"
            },
            {
                "url": "http://localhost:8100",
                "description": "Development (FastAPI)"
            }
        ],
        "paths": {},
        "components": {
            "schemas": {},
            "securitySchemes": {
                "ApiKeyAuth": {
                    "type": "apiKey",
                    "in": "header",
                    "name": "X-API-Key"
                },
                "BearerAuth": {
                    "type": "http",
                    "scheme": "bearer"
                }
            }
        },
        "security": [
            {"ApiKeyAuth": []},
            {"BearerAuth": []}
        ]
    }

    # Try fetching from running services first
    specs = {}
    for service_name, url in SERVICES.items():
        spec = fetch_spec_from_service(service_name, url)
        if spec:
            specs[service_name] = spec
        else:
            # Fallback to local file
            local_path = LOCAL_SPECS.get(service_name)
            if local_path:
                spec = load_spec_from_file(service_name, local_path)
                if spec:
                    specs[service_name] = spec

    # Merge specs
    for service_name, spec in specs.items():
        if not spec:
            continue

        # Merge paths (prefix with service name)
        for path, methods in spec.get("paths", {}).items():
            # Add service tag to all operations
            for method, operation in methods.items():
                if method in ["get", "post", "put", "patch", "delete"]:
                    if "tags" not in operation:
                        operation["tags"] = []
                    operation["tags"].insert(0, service_name)

            aggregated["paths"][path] = methods

        # Merge schemas
        schemas = spec.get("components", {}).get("schemas", {})
        for schema_name, schema_def in schemas.items():
            # Prefix schema names to avoid conflicts
            prefixed_name = f"{service_name}_{schema_name}"
            aggregated["components"]["schemas"][prefixed_name] = schema_def

    return aggregated


def save_specs(aggregated: Dict[Any, Any]):
    """Save aggregated and individual specs"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save aggregated spec
    aggregated_yaml = OUTPUT_DIR / "aggregated.yaml"
    aggregated_json = OUTPUT_DIR / "aggregated.json"

    with open(aggregated_yaml, 'w') as f:
        yaml.dump(aggregated, f, default_flow_style=False, sort_keys=False)
    print(f"✅ Saved aggregated spec: {aggregated_yaml}")

    with open(aggregated_json, 'w') as f:
        json.dump(aggregated, f, indent=2)
    print(f"✅ Saved aggregated spec: {aggregated_json}")

    # Save individual service specs
    for service_name, url in SERVICES.items():
        spec = fetch_spec_from_service(service_name, url)
        if spec:
            yaml_path = OUTPUT_DIR / f"{service_name}.yaml"
            json_path = OUTPUT_DIR / f"{service_name}.json"

            with open(yaml_path, 'w') as f:
                yaml.dump(spec, f, default_flow_style=False)

            with open(json_path, 'w') as f:
                json.dump(spec, f, indent=2)

            print(f"✅ Saved {service_name} spec: {yaml_path}")


def main():
    print("=" * 60)
    print("🔧 Aggregating OpenAPI Specifications")
    print("=" * 60)
    print()

    aggregated = aggregate_specs()

    print()
    print("📊 Summary:")
    print(f"   Paths: {len(aggregated['paths'])}")
    print(f"   Schemas: {len(aggregated['components']['schemas'])}")
    print()

    save_specs(aggregated)

    print()
    print("=" * 60)
    print("✅ OpenAPI aggregation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
