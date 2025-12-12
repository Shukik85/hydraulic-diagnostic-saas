"""FastAPI application for Hydraulic Diagnostics Service.

Modern REST API providing:
- Real-time equipment diagnostics
- Multi-component predictions
- Health monitoring
- System status

Features:
    - Async request handling
    - CORS middleware
    - Structured logging
    - Health checks
    - OpenAPI documentation

Python 3.14+ Features:
    - Deferred annotations
    - Pattern matching
    - Union types

Usage:
    from app.main import app
    import uvicorn
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
"""

__version__ = "0.1.0"
__author__ = "ML Engineering Team"
__all__ = ["app", "__version__"]
