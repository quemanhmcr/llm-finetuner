# -*- coding: utf-8 -*-
"""
Setup script for the finetune package.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="finetune",
    version="1.0.0",
    author="ML Fine-tuning Team",
    author_email="team@example.com",
    description="Advanced Professional ML Fine-tuning System - 2025 Optimized Edition",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/example/finetune",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "finetune=finetune.src.__main__:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
