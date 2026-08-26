"""Prism AI — 10B Parameter Foundation Model for Code."""

from setuptools import setup, find_packages

setup(
    name="prism-ai",
    version="0.1.0",
    description="Prism AI: A 10 Billion parameter foundation model for code generation",
    author="Prism AI Team",
    python_requires=">=3.10",
    packages=find_packages(),
    install_requires=[
        "torch>=2.1.0",
        "numpy>=1.24.0",
        "deepspeed>=0.12.0",
        "datasets>=2.15.0",
        "sentencepiece>=0.1.99",
        "tokenizers>=0.15.0",
        "wandb>=0.16.0",
        "tqdm>=4.65.0",
        "pyyaml>=6.0",
        "safetensors>=0.4.0",
    ],
    extras_require={
        "serve": ["fastapi>=0.104.0", "uvicorn>=0.24.0"],
        "eval": ["human-eval>=1.0.3"],
    },
)
