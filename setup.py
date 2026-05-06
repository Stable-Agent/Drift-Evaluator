"""Setup configuration for drift-evaluator package."""

from setuptools import setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="drift-evaluator",
    version="0.1.0",
    author="Stable-Agent Contributors",
    author_email="bradyt2215@gmail.com",
    description="Evaluation harness and synthetic datasets for LLM drift detection",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Stable-Agent/Drift-Evaluator",
    packages=["drift_evaluator"],
    package_dir={"drift_evaluator": "src"},
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Development Status :: 3 - Alpha",
    ],
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "requests>=2.31.0",
        "drift-detector>=0.1.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
        ],
    },
    keywords=[
        "llm", "drift", "evaluation", "benchmark", "synthetic-data",
    ],
)
