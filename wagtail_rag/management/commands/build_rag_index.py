from django.conf import settings
from django.core.management.base import BaseCommand

from wagtail_rag.content_extraction.index_builder import build_rag_index


class Command(BaseCommand):
    help = "Extract data from models, chunk it, and store in ChromaDB for RAG chatbot"

    def add_arguments(self, parser):
        parser.add_argument(
            "--reset-only",
            action="store_true",
            help="Only reset/clear the collection without indexing",
        )
        parser.add_argument(
            "--page-id",
            type=int,
            default=None,
            help="Re-index a specific page by ID (useful for updates)",
        )

    def handle(self, *args, **options):
        # Get configuration
        model_names = getattr(settings, "WAGTAIL_RAG_MODELS", None)
        chunk_size = getattr(settings, "WAGTAIL_RAG_CHUNK_SIZE", 1000)
        chunk_overlap = getattr(settings, "WAGTAIL_RAG_CHUNK_OVERLAP", 200)
        collection = getattr(settings, "WAGTAIL_RAG_COLLECTION_NAME", "wagtail_rag")

        # Display configuration
        self.stdout.write("=" * 60)
        self.stdout.write(self.style.SUCCESS("Wagtail RAG Configuration"))
        self.stdout.write("=" * 60)

        if model_names:
            self.stdout.write(f'Models: {", ".join(model_names)}')
        else:
            self.stdout.write("Models: ALL Page models")

        self.stdout.write(f"Chunk size: {chunk_size}")
        self.stdout.write(f"Chunk overlap: {chunk_overlap}")
        self.stdout.write(f"Collection: {collection}")

        # Embedding config
        emb_provider = getattr(
            settings, "WAGTAIL_RAG_EMBEDDING_PROVIDER", "huggingface"
        )
        emb_model = getattr(settings, "WAGTAIL_RAG_EMBEDDING_MODEL", None)
        self.stdout.write(f'Embeddings: {emb_provider}/{emb_model or "default"}')

        # LLM config
        llm_provider = getattr(settings, "WAGTAIL_RAG_LLM_PROVIDER", "ollama")
        llm_model = getattr(settings, "WAGTAIL_RAG_MODEL_NAME", None)
        self.stdout.write(f'LLM: {llm_provider}/{llm_model or "default"}')

        self.stdout.write("=" * 60)
        self.stdout.write("")

        # Run indexing
        build_rag_index(
            model_names=model_names,
            reset_only=options["reset_only"],
            page_id=options["page_id"],
            stdout=self.stdout.write,
        )
