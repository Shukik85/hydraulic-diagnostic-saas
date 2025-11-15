from setuptools import setup, find_packages

setup(
    name="hydraulic-shared",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "pydantic>=2.0.0",
        "pydantic-settings>=2.0.0",
    ],
    python_requires=">=3.11",
)
