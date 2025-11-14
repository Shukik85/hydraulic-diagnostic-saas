from setuptools import setup, find_packages

setup(
    name="hdx-shared",
    version="1.0.0",
    description="Shared utilities for Hydraulic Diagnostics SaaS",
    packages=find_packages(),
    python_requires=">=3.14",
    install_requires=[
        "pydantic>=2.12.0",
        "httpx>=0.25.0",
    ],
)
