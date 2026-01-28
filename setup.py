"""
Setup configuration for Wagtail RAG package.
"""
from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="wagtail-rag",
    version="0.1.0",
    author="Wagtail RAG Contributors",
    description="A plug-and-play RAG (Retrieval-Augmented Generation) chatbot for Wagtail CMS",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/awais786/wagtail-ai-chat",
    packages=find_packages(exclude=['tests', 'tests.*']),
    include_package_data=True,
    package_data={
        'wagtail_rag': ['templates/**/*.html'],
    },
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
        "langchain>=0.1.0",
        "langchain-community>=0.0.20",
        "langchain-text-splitters>=0.0.1",
        "chromadb>=0.4.0",
        "beautifulsoup4>=4.12.0",
        "tiktoken>=0.5.0",
    ],
    extras_require={
        # Embedding providers (install only what you need)
        "huggingface": [
            "langchain-huggingface>=0.0.1",
            "sentence-transformers>=2.2.0",
        ],
        "openai": [
            "langchain-openai>=0.0.1",
        ],
        "cohere": [
            "cohere>=4.0.0",
        ],
        "google": [
            "langchain-google-genai>=0.0.1",
        ],
        # LLM providers (install only what you need)
        "ollama": [
            "ollama>=0.1.0",
        ],
        "anthropic": [
            "langchain-anthropic>=0.0.1",
        ],
        # All providers (for convenience)
        "all": [
            "langchain-huggingface>=0.0.1",
            "sentence-transformers>=2.2.0",
            "langchain-openai>=0.0.1",
            "cohere>=4.0.0",
            "langchain-google-genai>=0.0.1",
            "ollama>=0.1.0",
            "langchain-anthropic>=0.0.1",
        ],
        # Development dependencies
        "dev": [
            "black",
            "flake8",
            "pytest",
            "pytest-django",
        ],
    },
)

