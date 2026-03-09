"""
Minimal Django settings for running tests.
"""

import os

# Build paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Security
SECRET_KEY = "test-secret-key-for-running-tests-only"
DEBUG = True
ALLOWED_HOSTS = ["*"]

# Application definition
INSTALLED_APPS = [
    "django.contrib.contenttypes",
    "django.contrib.auth",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "wagtail.contrib.forms",
    "wagtail.contrib.redirects",
    "wagtail.embeds",
    "wagtail.sites",
    "wagtail.users",
    "wagtail.snippets",
    "wagtail.documents",
    "wagtail.images",
    "wagtail.search",
    "wagtail.admin",
    "wagtail",
    "taggit",
    "modelcluster",
    "wagtail_rag",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "wagtail_rag.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.debug",
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

# Database
DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": ":memory:",
    }
}

# Static files
STATIC_URL = "/static/"
STATIC_ROOT = os.path.join(BASE_DIR, "staticfiles")

# Wagtail settings
WAGTAIL_SITE_NAME = "Test Site"

# Wagtail RAG settings
WAGTAIL_RAG = {
    "llm": {
        "provider": "ollama",
        "model": "llama2",
        "max_context_chars": 0,  # disabled in tests (covered by test_generation.py)
        "enable_history": True,
        "history_recent_messages": 6,
    },
    "embedding": {
        "provider": "sentence-transformers",
        "model": "all-MiniLM-L6-v2",
    },
    "vector_store": {
        "backend": "faiss",
        "path": os.path.join(BASE_DIR, "test_faiss_index"),
        "collection": "test_rag",
    },
    "indexing": {
        "chunk_size": 500,
        "chunk_overlap": 50,
    },
    "search": {
        "k": 8,
        "max_sources": 3,
    },
    "api": {
        "max_question_length": 150,
    },
}

# Additional test settings
USE_TZ = True
TIME_ZONE = "UTC"
LANGUAGE_CODE = "en-us"
