from django.conf import settings
from django.core.management.base import BaseCommand

from wagtail_rag.indexing import build_rag_index


class Command(BaseCommand):
    help = 'Extract data from models, chunk it, and store in ChromaDB for RAG chatbot'

    def add_arguments(self, parser):
        parser.add_argument(
            '--reset-only',
            action='store_true',
            help='Only reset/clear the collection without indexing'
        )
        parser.add_argument(
            '--page-id',
            type=int,
            default=None,
            help='Re-index a specific page by ID (useful for updates)'
        )



    def handle(self, *args, **options):
        # Model configuration now comes entirely from settings (see CONFIGURATION_EXAMPLE.py)
        model_names = getattr(settings, 'WAGTAIL_RAG_MODELS', None)
        model_fields_arg = None

        # Read configuration that will be applied by the shared indexer
        chunk_size = getattr(settings, 'WAGTAIL_RAG_CHUNK_SIZE', 1000)
        chunk_overlap = getattr(settings, 'WAGTAIL_RAG_CHUNK_OVERLAP', 200)
        collection_name = getattr(settings, 'WAGTAIL_RAG_COLLECTION_NAME', 'wagtail_rag')
        reset_only = options['reset_only']
        page_id = options['page_id']


        # Display configuration being used (keep CLI UX unchanged)
        self.stdout.write('=' * 60)
        self.stdout.write(self.style.SUCCESS('Wagtail RAG Configuration:'))
        self.stdout.write('=' * 60)

        if model_names:
            self.stdout.write(f'Models to index (from settings): {", ".join(model_names)}')
        else:
            self.stdout.write('Models to index: ALL Page models (no WAGTAIL_RAG_MODELS set)')

        self.stdout.write(f'Chunk size: {chunk_size}')
        self.stdout.write(f'Chunk overlap: {chunk_overlap}')
        self.stdout.write(f'Collection name: {collection_name}')

        if model_fields_arg:
            self.stdout.write(f'Model-specific fields: {model_fields_arg}')

        embedding_provider = getattr(settings, 'WAGTAIL_RAG_EMBEDDING_PROVIDER', 'huggingface')
        embedding_model = getattr(settings, 'WAGTAIL_RAG_EMBEDDING_MODEL', None)
        self.stdout.write(f'Embedding provider: {embedding_provider} (used for vector search)')
        if embedding_model:
            self.stdout.write(f'Embedding model: {embedding_model}')
        else:
            self.stdout.write(f'Embedding model: default for {embedding_provider}')

        llm_provider = getattr(settings, 'WAGTAIL_RAG_LLM_PROVIDER', 'ollama')
        llm_model = getattr(settings, 'WAGTAIL_RAG_MODEL_NAME', None)
        self.stdout.write(f'LLM Provider: {llm_provider} (used when querying chatbot)')
        if llm_model:
            self.stdout.write(f'LLM Model: {llm_model}')
        else:
            provider_defaults = {
                'ollama': 'mistral',
                'openai': 'gpt-4',
                'anthropic': 'claude-3-sonnet-20240229',
            }
            default = provider_defaults.get(llm_provider, 'see settings')
            self.stdout.write(f'LLM Model: {default} (default for {llm_provider})')

        self.stdout.write('=' * 60)
        self.stdout.write('')

        # Delegate the heavy lifting to the shared indexing helper
        build_rag_index(
            model_names=model_names,
            reset_only=reset_only,
            page_id=page_id,
            stdout=self.stdout.write,
        )

