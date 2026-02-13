"""
Minimal Django settings for running tests.
"""
import os

# Build paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Security
SECRET_KEY = 'test-secret-key-for-running-tests-only'
DEBUG = True
ALLOWED_HOSTS = ['*']

# Application definition
INSTALLED_APPS = [
    'django.contrib.contenttypes',
    'django.contrib.auth',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',

    'wagtail.contrib.forms',
    'wagtail.contrib.redirects',
    'wagtail.embeds',
    'wagtail.sites',
    'wagtail.users',
    'wagtail.snippets',
    'wagtail.documents',
    'wagtail.images',
    'wagtail.search',
    'wagtail.admin',
    'wagtail',

    'taggit',
    'modelcluster',

    'wagtail_rag',
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

# No ROOT_URLCONF needed since we don't use URL routing in tests

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

# Database
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': ':memory:',
    }
}

# Static files
STATIC_URL = '/static/'
STATIC_ROOT = os.path.join(BASE_DIR, 'staticfiles')

# Wagtail settings
WAGTAIL_SITE_NAME = 'Test Site'

# Wagtail RAG settings (test configuration)
WAGTAIL_RAG_LLM_PROVIDER = 'ollama'
WAGTAIL_RAG_MODEL_NAME = 'llama2'
WAGTAIL_RAG_EMBEDDING_PROVIDER = 'sentence-transformers'
WAGTAIL_RAG_EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
WAGTAIL_RAG_VECTOR_STORE = 'faiss'
WAGTAIL_RAG_FAISS_INDEX_PATH = os.path.join(BASE_DIR, 'test_faiss_index')
WAGTAIL_RAG_CHUNK_SIZE = 500
WAGTAIL_RAG_CHUNK_OVERLAP = 50
WAGTAIL_RAG_MAX_RESULTS = 5
WAGTAIL_RAG_ENABLE_HYBRID_SEARCH = True
WAGTAIL_RAG_VECTOR_WEIGHT = 0.7
WAGTAIL_RAG_WAGTAIL_WEIGHT = 0.3
WAGTAIL_RAG_ENABLE_CHAT_HISTORY = True
WAGTAIL_RAG_MAX_CONTEXT_CHARS = 0
WAGTAIL_RAG_DEFAULT_FIELDS = ['introduction', 'body']

# Additional test settings
USE_TZ = True
TIME_ZONE = 'UTC'
LANGUAGE_CODE = 'en-us'
