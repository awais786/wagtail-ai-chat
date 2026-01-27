"""
Setup configuration for Wagtail RAG package.
"""
from setuptools import find_packages, setup
import os

# Read README if it exists
readme_path = os.path.join(os.path.dirname(__file__), "README.md")
long_description = ""
if os.path.exists(readme_path):
    with open(readme_path, "r", encoding="utf-8") as fh:
        long_description = fh.read()

setup(
    name="wagtail-rag",
    version="0.1.0",
    author="Wagtail RAG Contributors",
    description="A plug-and-play RAG (Retrieval-Augmented Generation) chatbot for Wagtail CMS",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/awais786/wagtail-ai-chat",
    packages=find_packages(where=".", exclude=['tests', 'tests.*']),
    package_dir={"": "."},
    include_package_data=True,
    py_modules=[],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Framework :: Django",
        "Framework :: Wagtail",
    ],
    python_requires=">=3.8",
    install_requires=[
        "Django>=3.2",
        "wagtail>=4.0",
        "langchain>=0.1.0",
        "langchain-community>=0.0.20",
        "langchain-text-splitters>=0.0.1",
        "langchain-huggingface>=0.0.1",
        "chromadb>=0.4.0",
        "beautifulsoup4>=4.12.0",
    ],
    extras_require={
        "dev": [
            "black",
            "flake8",
            "pytest",
            "pytest-django",
        ],
    },
)

