import os

from django.apps import apps
from django.conf import settings
from django.core.management.base import BaseCommand
from wagtail.models import Page

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    try:
        from langchain.text_splitters import RecursiveCharacterTextSplitter
    except ImportError:
        try:
            from langchain.text_splitter import RecursiveCharacterTextSplitter
        except ImportError:
            raise ImportError(
                "Could not import RecursiveCharacterTextSplitter. "
                "Please install langchain-text-splitters: pip install langchain-text-splitters"
            )

try:
    from langchain_community.vectorstores import Chroma
except ImportError:
    try:
        from langchain.vectorstores import Chroma
    except ImportError:
        raise ImportError(
            "Could not import Chroma. "
            "Please install langchain-community: pip install langchain-community"
        )

# Import embedding factory
try:
    from wagtail_rag.embedding_providers import get_embeddings
except ImportError:
    # Fallback for older installations
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
        def get_embeddings(provider=None, model_name=None, **kwargs):
            if provider and provider.lower() not in ('huggingface', 'hf', None):
                raise ValueError(f"Only HuggingFace embeddings available. Requested: {provider}")
            return HuggingFaceEmbeddings(model_name=model_name or 'sentence-transformers/all-MiniLM-L6-v2', **kwargs)
    except ImportError:
        try:
            from langchain_community.embeddings import HuggingFaceEmbeddings
            def get_embeddings(provider=None, model_name=None, **kwargs):
                if provider and provider.lower() not in ('huggingface', 'hf', None):
                    raise ValueError(f"Only HuggingFace embeddings available. Requested: {provider}")
                return HuggingFaceEmbeddings(model_name=model_name or 'sentence-transformers/all-MiniLM-L6-v2', **kwargs)
        except ImportError:
            raise ImportError(
                "Could not import embeddings. "
                "Please install langchain-huggingface or langchain-community"
            )


class Command(BaseCommand):
    help = 'Extract data from models, chunk it, and store in ChromaDB for RAG chatbot'

    def add_arguments(self, parser):
        parser.add_argument(
            '--models',
            nargs='+',
            default=None,
            help='Specific page models to index (e.g., blog.BlogPage breads.BreadPage). If not specified, indexes all Page models.'
        )
        parser.add_argument(
            '--exclude-models',
            nargs='+',
            default=None,
            help='Page models to exclude from indexing (e.g., wagtailcore.Page)'
        )
        parser.add_argument(
            '--model-fields',
            nargs='+',
            default=None,
            help='Important fields to emphasize per model. Format: model_name:field1,field2 (e.g., breads.BreadPage:bread_type,origin)'
        )
        parser.add_argument(
            '--chunk-size',
            type=int,
            default=1000,
            help='Size of text chunks (default: 1000)'
        )
        parser.add_argument(
            '--chunk-overlap',
            type=int,
            default=200,
            help='Overlap between chunks (default: 200)'
        )
        parser.add_argument(
            '--collection-name',
            type=str,
            default=None,
            help='ChromaDB collection name (default: from WAGTAIL_RAG_COLLECTION_NAME setting or "wagtail_rag")'
        )
        parser.add_argument(
            '--reset',
            action='store_true',
            help='Reset/clear existing collection before indexing'
        )
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

    def _extract_streamfield_text(self, streamfield):
        """
        Generic method to extract text content from Wagtail StreamField.
        Handles common block types and falls back to string representation.
        """
        if not streamfield:
            return ''

        text_parts = []
        for block in streamfield:
            try:
                block_type = block.block_type.lower()
                block_value = block.value

                # Extract text based on block type
                if 'heading' in block_type:
                    text = self._extract_text_from_value(block_value, ['heading_text', 'text'])
                    if text:
                        text_parts.append(f"## {text}")

                elif 'paragraph' in block_type or 'richtext' in block_type:
                    if hasattr(block_value, 'source'):
                        # Strip HTML from RichText source
                        clean_text = self._extract_text_from_richtext(block_value.source)
                        if clean_text:
                            text_parts.append(clean_text)
                    else:
                        # Try to extract as string and strip HTML if it looks like HTML
                        raw_text = str(block_value)
                        if '<' in raw_text and '>' in raw_text:
                            clean_text = self._extract_text_from_richtext(raw_text)
                            if clean_text:
                                text_parts.append(clean_text)
                        else:
                            text_parts.append(raw_text)

                elif 'quote' in block_type:
                    quote_text = self._extract_text_from_value(block_value, ['text', 'quote'])
                    if quote_text:
                        # Strip HTML if present
                        if '<' in quote_text and '>' in quote_text:
                            quote_text = self._extract_text_from_richtext(quote_text)
                        if quote_text:
                            attribute = self._extract_text_from_value(block_value, ['attribute_name', 'attribute'])
                            text_parts.append(f'"{quote_text}"' + (f' - {attribute}' if attribute else ''))

                elif 'image' in block_type:
                    caption = self._extract_text_from_value(block_value, ['caption', 'alt'])
                    if caption:
                        text_parts.append(f"[Image: {caption}]")

                elif 'list' in block_type:
                    if isinstance(block_value, (list, tuple)):
                        for item in block_value:
                            item_text = self._extract_text_from_value(item, ['value', 'text']) or str(item)
                            text_parts.append(f"- {item_text}")
                    else:
                        text_parts.append(f"- {str(block_value)}")

                else:
                    # Generic fallback
                    text = self._extract_text_from_value(block_value,
                                                         ['text', 'content', 'value', 'body', 'description'])
                    if text:
                        # Strip HTML if it looks like HTML
                        if '<' in text and '>' in text:
                            clean_text = self._extract_text_from_richtext(text)
                            if clean_text:
                                text_parts.append(clean_text)
                        else:
                            text_parts.append(text)
                    else:
                        raw_text = str(block_value)
                        # Strip HTML if it looks like HTML
                        if '<' in raw_text and '>' in raw_text:
                            clean_text = self._extract_text_from_richtext(raw_text)
                            if clean_text:
                                text_parts.append(clean_text)
                        else:
                            text_parts.append(raw_text)

            except Exception:
                continue

        return '\n'.join(text_parts)

    def _extract_text_from_value(self, value, keys):
        """Helper to extract text from dict or return string value."""
        if isinstance(value, dict):
            for key in keys:
                if key in value and value[key]:
                    return str(value[key])
        elif isinstance(value, str) and value:
            return value
        return None

    def _extract_text_from_richtext(self, html_content):
        """
        Extract clean text from RichTextField HTML content.
        Strips HTML tags for better embeddings.
        """
        if not html_content:
            return ''

        try:
            # Try using BeautifulSoup if available
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html_content, "html.parser")
            return soup.get_text(separator=' ', strip=True)
        except ImportError:
            # Fallback: simple regex-based HTML tag removal
            import re
            # Remove HTML tags
            text = re.sub(r'<[^>]+>', '', html_content)
            # Decode HTML entities (basic)
            text = text.replace('&nbsp;', ' ')
            text = text.replace('&amp;', '&')
            text = text.replace('&lt;', '<')
            text = text.replace('&gt;', '>')
            text = text.replace('&quot;', '"')
            # Clean up whitespace
            text = ' '.join(text.split())
            return text

    def _get_page_models(self, model_names=None, exclude_models=None):
        """
        Get all Wagtail Page models dynamically.

        Args:
            model_names: List of specific model names to include (e.g., ['blog.BlogPage'])
            exclude_models: List of model names to exclude (e.g., ['wagtailcore.Page'])

        Returns:
            List of Page model classes
        """
        page_models = []

        # Get all models that inherit from Page
        for app_config in apps.get_app_configs():
            for model in app_config.get_models():
                if issubclass(model, Page) and model != Page:
                    model_name = f"{model._meta.app_label}.{model.__name__}"

                    # Check if model should be included
                    if model_names:
                        if model_name not in model_names:
                            continue

                    # Check if model should be excluded
                    if exclude_models:
                        if model_name in exclude_models:
                            continue

                    page_models.append(model)

        return page_models

    def _extract_page_text(self, page, important_fields=None):
        """
        Generic method to extract text content from any Wagtail Page.
        Works with common Wagtail fields and patterns.

        Args:
            page: The Wagtail Page instance
            important_fields: List of field names to emphasize (e.g., ['bread_type', 'origin'])
        """
        # Start with title
        text_parts = [f"Title: {page.title}"]

        # Track which fields we've already added to avoid duplication
        added_fields = set(['title'])

        # If important fields are specified, extract and add them first
        if important_fields:
            for field_name in important_fields:
                if field_name in added_fields:
                    continue  # Skip if already added
                if hasattr(page, field_name):
                    value = getattr(page, field_name, None)
                    if value:
                        # Handle ForeignKey relationships and strings safely
                        if hasattr(value, 'title') and not isinstance(value, str):
                            # e.g. related objects with a 'title' field
                            attr = getattr(value, 'title')
                            field_value = attr() if callable(attr) else attr
                        elif hasattr(value, 'name'):
                            attr = getattr(value, 'name')
                            field_value = attr() if callable(attr) else attr
                        elif hasattr(value, 'all'):  # ManyToMany
                            items = value.all()
                            if items:
                                field_value = ', '.join([
                                    getattr(item, 'name', getattr(item, 'title', str(item)))
                                    for item in items
                                ])
                            else:
                                continue
                        else:
                            # Fallback: just cast to string
                            field_value = str(value)

                        # Add important fields once (avoid noisy duplication)
                        label = field_name.replace('_', ' ').title()
                        text_parts.append(f"{label}: {field_value}")
                        added_fields.add(field_name)

        # Common Wagtail fields (only if not already in important_fields)
        common_fields = [
            'subtitle', 'introduction', 'description', 'summary',
            'date_published', 'first_published_at'
        ]

        for field_name in common_fields:
            if field_name in added_fields:
                continue  # Skip if already added via important_fields
            if hasattr(page, field_name):
                value = getattr(page, field_name, None)
                if value:
                    if field_name in ['date_published', 'first_published_at']:
                        text_parts.append(f"Published: {value}")
                    else:
                        # Capitalize field name for display
                        label = field_name.replace('_', ' ').title()
                        text_parts.append(f"{label}: {value}")
                    added_fields.add(field_name)

        # Extract StreamField content (common field name: 'body')
        streamfield_fields = ['body', 'content', 'backstory', 'instructions']
        for field_name in streamfield_fields:
            if hasattr(page, field_name):
                field_value = getattr(page, field_name, None)
                if field_value:
                    streamfield_text = self._extract_streamfield_text(field_value)
                    if streamfield_text:
                        label = field_name.replace('_', ' ').title()
                        text_parts.append(f"{label}:\n{streamfield_text}")

        # Extract RichTextField content
        if hasattr(page, '_meta'):
            for field in page._meta.get_fields():
                if hasattr(field, 'field') and hasattr(field.field, '__class__'):
                    field_class_name = field.field.__class__.__name__
                    if 'RichTextField' in field_class_name or 'RichText' in field_class_name:
                        value = getattr(page, field.name, None)
                        if value:
                            if hasattr(value, 'source'):
                                # Strip HTML tags for better embeddings
                                rich_text = self._extract_text_from_richtext(value.source)
                                if rich_text:
                                    text_parts.append(f"{field.name.title()}: {rich_text}")
                            else:
                                text_parts.append(f"{field.name.title()}: {str(value)}")

        # Extract tags if available
        if hasattr(page, 'get_tags'):
            try:
                tags = page.get_tags()
                if tags:
                    text_parts.append(f"Tags: {', '.join([str(tag) for tag in tags])}")
            except Exception:
                pass

        # Extract authors if available
        if hasattr(page, 'authors'):
            try:
                authors = page.authors()
                if authors:
                    author_names = []
                    for author in authors:
                        if hasattr(author, 'first_name') and hasattr(author, 'last_name'):
                            author_names.append(f"{author.first_name} {author.last_name}")
                        else:
                            author_names.append(str(author))
                    if author_names:
                        text_parts.append(f"Authors: {', '.join(author_names)}")
            except Exception:
                pass

        # Extract address if available
        if hasattr(page, 'address') and page.address:
            text_parts.append(f"Address: {page.address}")

        # Extract coordinates if available
        if hasattr(page, 'lat_long') and page.lat_long:
            text_parts.append(f"Coordinates: {page.lat_long}")

        # Extract operating hours if available
        if hasattr(page, 'operating_hours'):
            try:
                hours = page.operating_hours
                if hours:
                    hours_text = []
                    for hour in hours:
                        if hasattr(hour, 'day') and hasattr(hour, 'opening_time') and hasattr(hour, 'closing_time'):
                            hours_text.append(f"{hour.day}: {hour.opening_time} - {hour.closing_time}")
                    if hours_text:
                        text_parts.append(f"Operating Hours:\n" + '\n'.join(hours_text))
            except Exception:
                pass

        return '\n\n'.join(text_parts)

    def _parse_model_fields(self, model_fields_arg):
        """
        Parse --model-fields argument.

        Supports two formats:
        1. List format: ['model_name:field1,field2', 'model_name2:field3']
        2. Dictionary format: {'model_name': ['field1', 'field2'], 'model_name2': ['field3']}

        Returns:
            Dict mapping model names to list of important fields
        """
        if not model_fields_arg:
            return {}

        # If it's already a dictionary, return it as-is
        if isinstance(model_fields_arg, dict):
            return model_fields_arg

        # Otherwise, parse as list format
        model_fields_map = {}
        for item in model_fields_arg:
            if ':' in item:
                model_name, fields_str = item.split(':', 1)
                fields = [f.strip() for f in fields_str.split(',')]
                model_fields_map[model_name] = fields
        return model_fields_map

    def _get_all_content_field_names(self, model):
        """
        Return a filtered list of "content" field names for a model.

        Used when model fields contain a wildcard '*' for a model.
        Excludes reverse relations and obvious admin/internal fields so that
        the indexed text focuses on meaningful content, not Wagtail internals.
        """
        from django.db.models import Field

        exclude_names = {
            # Reverse relations / admin metadata
            'index_entries',
            'specific_workflow_states',
            'workflow_states',
            'revisions',
            'subscribers',
            'wagtail_admin_comments',
            'view_restrictions',
            'group_permissions',
            'aliases',
            'sites_rooted_here',
            # Low-level internals
            'content_type',
            'page_ptr',
            # Wagtail Page base class internals
            'path',
            'depth',
            'translation_key',
            'locale',
            'latest_revision',
            'live',
            'first_published_at',
            'last_published_at',
            'live_revision',
            'draft_title',
            'slug',
            'url_path',
            'owner',
            'latest_revision_created_at',
            'seo_title',
            'search_description',
            'show_in_menus',
            'has_unpublished_changes',
            'go_live_at',
            'expire_at',
            'expired',
            'locked',
            'locked_at',
            'locked_by',
        }

        field_names = []
        for f in model._meta.get_fields():
            # Skip auto-created or reverse relations
            if getattr(f, 'auto_created', False):
                continue
            # Only include concrete model fields
            if not isinstance(f, Field):
                continue
            if f.name in exclude_names:
                continue
            field_names.append(f.name)

        return field_names

    def _get_pages_to_index(self, model_names=None, exclude_models=None, model_fields_map=None, page_id=None):
        """
        Get pages to index based on model selection.

        Args:
            model_names: List of specific model names to include
            exclude_models: List of model names to exclude
            model_fields_map: Dict mapping model names to important fields
            page_id: Specific page ID to re-index (if provided, only this page is indexed)

        Returns:
            Tuple of (pages_text_list, metadata_list)
        """
        pages = []
        metadata = []

        # Get page models to index
        page_models = self._get_page_models(model_names=model_names, exclude_models=exclude_models)

        if not page_models:
            self.stdout.write(self.style.WARNING('No page models found to index.'))
            return pages, metadata

        self.stdout.write(f'Found {len(page_models)} page model(s) to index:')
        for model in page_models:
            self.stdout.write(f'  - {model._meta.app_label}.{model.__name__}')

        # Extract text from each page
        for model in page_models:
            model_name = f"{model._meta.app_label}.{model.__name__}"

            # Resolve important fields for this model.
            # Special case: ['*'] means "all content fields on this model".
            raw_model_fields = model_fields_map.get(model_name, [])
            if raw_model_fields == ['*']:
                try:
                    model_important_fields = self._get_all_content_field_names(model)
                except Exception:
                    model_important_fields = []
            else:
                model_important_fields = raw_model_fields

            # If page_id is specified, only get that specific page
            if page_id:
                try:
                    live_pages = model.objects.filter(id=page_id, live=True)
                    if not live_pages.exists():
                        continue  # Page not found in this model, try next
                    self.stdout.write(f'  Re-indexing page ID {page_id} from {model_name}...')
                except Exception as e:
                    self.stdout.write(self.style.WARNING(f'  Error finding page {page_id} in {model_name}: {e}'))
                    continue
            else:
                live_pages = model.objects.live()

            count = live_pages.count()

            if count == 0:
                if not page_id:  # Only warn if not looking for specific page
                    self.stdout.write(f'  No live pages found for {model_name}')
                continue

            if not page_id:
                self.stdout.write(f'  Indexing {count} pages from {model_name}...')

            for page in live_pages:
                try:
                    # Get important fields for this specific model
                    page_text = self._extract_page_text(
                        page,
                        important_fields=model_important_fields if model_important_fields else None,
                    )
                    if page_text.strip():  # Only add non-empty text
                        pages.append(page_text)

                        # Build metadata with page-specific fields
                        page_metadata = {
                            'source': model_name,
                            'model': model.__name__,
                            'app': model._meta.app_label,
                            'title': page.title,
                            'url': page.url if hasattr(page, 'url') else '',
                            'id': page.id,
                            'slug': page.slug if hasattr(page, 'slug') else '',
                        }

                        # Always print the extracted text for debugging/inspection
                        separator = '=' * 80
                        self.stdout.write(separator)
                        self.stdout.write(
                            f'Page: {page_metadata["model"]} (ID: {page_metadata["id"]}) '
                            f'- "{page_metadata["title"]}"'
                        )
                        self.stdout.write('-' * 80)
                        self.stdout.write(page_text)
                        self.stdout.write('\n')

                        # Add any important fields to metadata generically
                        if model_important_fields:
                            for field_name in model_important_fields:
                                if hasattr(page, field_name):
                                    value = getattr(page, field_name, None)
                                    if value is not None:
                                        # Handle ForeignKey and strings safely
                                        if hasattr(value, 'title') and not isinstance(value, str):
                                            attr = getattr(value, 'title')
                                            page_metadata[field_name] = attr() if callable(attr) else attr
                                        elif hasattr(value, 'name') and not isinstance(value, str):
                                            attr = getattr(value, 'name')
                                            page_metadata[field_name] = attr() if callable(attr) else attr
                                        # Handle ManyToMany
                                        elif hasattr(value, 'all'):
                                            items = value.all()
                                            if items:
                                                page_metadata[field_name] = ', '.join([
                                                    getattr(
                                                        item,
                                                        'name',
                                                        getattr(item, 'title', str(item)),
                                                    )
                                                    for item in items
                                                ])
                                        else:
                                            # Fallback: cast to string
                                            page_metadata[field_name] = str(value)

                        metadata.append(page_metadata)
                except Exception as e:
                    self.stdout.write(
                        self.style.WARNING(
                            f'  Error extracting text from {model_name} (ID: {page.id}): {e}'
                        )
                    )

        return pages, metadata

    def handle(self, *args, **options):
        # Get values from command line args, or fall back to settings, or use defaults
        model_names = options['models'] or getattr(settings, 'WAGTAIL_RAG_MODELS', None)
        exclude_models = options['exclude_models'] or []
        model_fields_arg = options['model_fields'] or []
        explicit_model_fields_provided = bool(options['model_fields'])

        # Support convenience syntax "app.Model:*" in WAGTAIL_RAG_MODELS / --models
        #
        # Example:
        #   WAGTAIL_RAG_MODELS = [
        #       "breads.BreadPage",
        #       "locations.LocationPage:*",
        #   ]
        #
        # becomes:
        #   model_names      = ["breads.BreadPage", "locations.LocationPage"]
        #   model_fields_arg = ["locations.LocationPage:*"]  (if no explicit model_fields provided)
        auto_model_fields = []
        if model_names:
            cleaned_model_names = []
            for name in model_names:
                if isinstance(name, str) and name.endswith(":*"):
                    base = name.split(":", 1)[0]
                    cleaned_model_names.append(base)
                    auto_model_fields.append(f"{base}:*")
                else:
                    cleaned_model_names.append(name)
            model_names = cleaned_model_names

            # Only apply auto-generated model fields if the user did NOT explicitly
            # pass --model-fields and if we don't already have model_fields_arg from CLI.
            if not explicit_model_fields_provided and not model_fields_arg and auto_model_fields:
                model_fields_arg = auto_model_fields

        chunk_size = options['chunk_size'] or getattr(settings, 'WAGTAIL_RAG_CHUNK_SIZE', 1000)
        chunk_overlap = options['chunk_overlap'] or getattr(settings, 'WAGTAIL_RAG_CHUNK_OVERLAP', 200)
        collection_name = options['collection_name']
        reset = options['reset']
        reset_only = options['reset_only']
        page_id = options['page_id']
        
        # Get default exclude models from settings
        default_excludes = getattr(
            settings,
            'WAGTAIL_RAG_EXCLUDE_MODELS',
            ['wagtailcore.Page', 'wagtailcore.Site']
        )
        exclude_models = list(set(exclude_models + default_excludes))
        
        # Parse model-specific fields
        model_fields_map = self._parse_model_fields(model_fields_arg)
        
        if page_id:
            self.stdout.write(f'Re-indexing specific page ID: {page_id}')
        
        if model_fields_map:
            self.stdout.write(f'Model-specific important fields: {model_fields_map}')

        # Get collection name from settings or use default
        if not collection_name:
            collection_name = getattr(
                settings,
                'WAGTAIL_RAG_COLLECTION_NAME',
                'wagtail_rag'
            )
        
        # Display configuration being used
        self.stdout.write('=' * 60)
        self.stdout.write(self.style.SUCCESS('Wagtail RAG Configuration:'))
        self.stdout.write('=' * 60)
        
        # Models configuration
        if model_names:
            self.stdout.write(f'Models to index: {", ".join(model_names)}')
        else:
            self.stdout.write('Models to index: ALL Page models')
        
        if exclude_models:
            self.stdout.write(f'Excluded models: {", ".join(exclude_models)}')
        
        # Chunking configuration
        self.stdout.write(f'Chunk size: {chunk_size}')
        self.stdout.write(f'Chunk overlap: {chunk_overlap}')
        
        # Collection configuration
        self.stdout.write(f'Collection name: {collection_name}')
        
        # Important fields
        if model_fields_map:
            self.stdout.write(f'Model-specific fields: {model_fields_map}')
        
        # Embedding configuration (used for indexing - converts text to vectors)
        embedding_provider = getattr(settings, 'WAGTAIL_RAG_EMBEDDING_PROVIDER', 'huggingface')
        embedding_model = getattr(settings, 'WAGTAIL_RAG_EMBEDDING_MODEL', None)
        self.stdout.write(f'Embedding provider: {embedding_provider} (used for vector search)')
        if embedding_model:
            self.stdout.write(f'Embedding model: {embedding_model}')
        else:
            self.stdout.write(f'Embedding model: default for {embedding_provider}')
        
        # LLM Provider configuration (used for querying - generates answers)
        llm_provider = getattr(settings, 'WAGTAIL_RAG_LLM_PROVIDER', 'ollama')
        llm_model = getattr(settings, 'WAGTAIL_RAG_MODEL_NAME', None)
        self.stdout.write(f'LLM Provider: {llm_provider} (used when querying chatbot)')
        if llm_model:
            self.stdout.write(f'LLM Model: {llm_model}')
        else:
            # Show default based on provider
            provider_defaults = {
                'ollama': 'mistral',
                'openai': 'gpt-4',
                'anthropic': 'claude-3-sonnet-20240229',
            }
            default = provider_defaults.get(llm_provider, 'see settings')
            self.stdout.write(f'LLM Model: {default} (default for {llm_provider})')
        
        self.stdout.write('=' * 60)
        self.stdout.write('')
        
        # If we only want to reset the collection, do it now and exit
        if reset_only:
            self.stdout.write('Resetting collection (reset-only mode)...')
            # Initialize embeddings and persistence directory just for Chroma connection
            embeddings = get_embeddings(
                provider=embedding_provider,
                model_name=embedding_model
            )
            persist_directory = getattr(
                settings,
                'WAGTAIL_RAG_CHROMA_PATH',
                os.path.join(settings.BASE_DIR, 'chroma_db')
            )
            os.makedirs(persist_directory, exist_ok=True)

            try:
                existing_db = Chroma(
                    persist_directory=persist_directory,
                    collection_name=collection_name,
                    embedding_function=embeddings
                )
                existing_db.delete_collection()
                self.stdout.write(self.style.SUCCESS(f'Collection "{collection_name}" cleared successfully'))
            except Exception as e:
                self.stdout.write(self.style.WARNING(f'Collection may not exist: {e}'))

            self.stdout.write(self.style.SUCCESS('Reset-only complete. No indexing performed.'))
            return

        self.stdout.write('Starting RAG index building...')
        
        # Get pages to index
        if model_names:
            self.stdout.write(f'Extracting data from specified models: {", ".join(model_names)}')
        else:
            self.stdout.write('Extracting data from all Page models...')
        
        if exclude_models:
            self.stdout.write(f'Excluding models: {", ".join(exclude_models)}')

        texts, metadatas = self._get_pages_to_index(
            model_names=model_names,
            exclude_models=exclude_models,
            model_fields_map=model_fields_map,
            page_id=page_id
        )

        if not texts:
            self.stdout.write(self.style.WARNING('No pages found to index.'))
            return

        self.stdout.write(f'Extracted {len(texts)} pages')

        # Split into chunks
        self.stdout.write(f'Chunking text (size: {chunk_size}, overlap: {chunk_overlap})...')
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )

        chunks = []
        chunk_metadatas = []
        chunk_ids = []
        for i, (text, metadata) in enumerate(zip(texts, metadatas)):
            text_chunks = text_splitter.split_text(text)

            # Build prefix with identifying information for each chunk
            # Include all available metadata for context
            prefix_parts = [f"Title: {metadata.get('title', 'Unknown')}"]

            # Add any additional metadata fields generically (exclude internal fields)
            exclude_from_prefix = {'source', 'model', 'app', 'id', 'slug', 'url', 'chunk_index', 'total_chunks'}
            for key, value in metadata.items():
                if key not in exclude_from_prefix and value:
                    prefix_parts.append(f"{key.replace('_', ' ').title()}: {value}")

            prefix = " | ".join(prefix_parts) + "\n\n"

            # Generate deterministic IDs based on page metadata
            page_id = metadata.get('id')
            app = metadata.get('app', 'unknown')
            model = metadata.get('model', 'unknown')

            for chunk_idx, chunk in enumerate(text_chunks):
                # Prepend identifying information to each chunk for better context
                chunk_with_context = prefix + chunk
                chunks.append(chunk_with_context)

                # Generate deterministic ID: app_model_pageid_chunkidx
                chunk_id = f"{app}_{model}_{page_id}_{chunk_idx}"
                chunk_ids.append(chunk_id)

                chunk_metadatas.append({
                    **metadata,
                    'chunk_index': chunk_idx,
                    'total_chunks': len(text_chunks)
                })

        self.stdout.write(f'Created {len(chunks)} chunks from {len(texts)} pages')

        # Initialize embeddings
        self.stdout.write('Initializing embeddings...')
        embeddings = get_embeddings(
            provider=embedding_provider,
            model_name=embedding_model
        )
        
        # Set up ChromaDB persistence directory
        persist_directory = getattr(
            settings,
            'WAGTAIL_RAG_CHROMA_PATH',
            os.path.join(settings.BASE_DIR, 'chroma_db')
        )
        os.makedirs(persist_directory, exist_ok=True)

        # Check if collection exists and warn about embedding dimension mismatch
        try:
            existing_db = Chroma(
                persist_directory=persist_directory,
                collection_name=collection_name,
                embedding_function=embeddings
            )
            # Try to get collection info to check dimensions
            try:
                # Get a sample to check dimensions
                sample = existing_db.get(limit=1)
                if sample and sample.get('ids'):
                    # Collection exists - check if we need to reset
                    if not reset and not reset_only:
                        self.stdout.write(
                            self.style.WARNING(
                                f'\n⚠️  WARNING: Collection "{collection_name}" already exists.\n'
                                f'   If you changed embedding providers/models, you MUST use --reset\n'
                                f'   to avoid dimension mismatch errors.\n'
                                f'   Current embedding: {embedding_provider} / {embedding_model or "default"}\n'
                            )
                        )
            except Exception:
                # Collection might be empty or have issues, continue
                pass
        except Exception:
            # Collection doesn't exist yet, that's fine
            pass

        # Reset collection if requested (need embeddings for Chroma connection)
        if reset:
            self.stdout.write(f'Resetting collection: {collection_name}')
            try:
                existing_db = Chroma(
                    persist_directory=persist_directory,
                    collection_name=collection_name,
                    embedding_function=embeddings
                )
                existing_db.delete_collection()
                self.stdout.write(self.style.SUCCESS(f'Collection "{collection_name}" cleared successfully'))
            except Exception as e:
                self.stdout.write(self.style.WARNING(f'Collection may not exist: {e}'))

        # Create/update vector store with deterministic IDs
        self.stdout.write(f'Storing chunks in ChromaDB (collection: {collection_name})...')
        
        # Check if collection exists to determine if we're updating or creating
        try:
            existing_db = Chroma(
                persist_directory=persist_directory,
                collection_name=collection_name,
                embedding_function=embeddings
            )
            # Collection exists - we'll update/add documents
            # Test if we can actually use it (check dimension compatibility)
            try:
                # Try to get collection count to verify compatibility
                existing_db._collection.count()
            except Exception as e:
                error_msg = str(e)
                if 'dimension' in error_msg.lower() or 'embedding' in error_msg.lower():
                    self.stdout.write(
                        self.style.ERROR(
                            f'\n❌ ERROR: Embedding dimension mismatch!\n'
                            f'   The existing collection was created with a different embedding model.\n'
                            f'   You MUST reset the collection first:\n'
                            f'   python manage.py build_rag_index --reset\n'
                            f'   Error: {error_msg}\n'
                        )
                    )
                    return
                else:
                    raise
            if page_id:
                # For single page updates, delete old chunks first
                self.stdout.write(f'Deleting old chunks for page ID {page_id}...')
                try:
                    # Get all existing chunks and filter by page ID in metadata
                    all_chunks = existing_db.get(include=['metadatas'])
                    if all_chunks and all_chunks.get('ids'):
                        # Find chunks that belong to this page
                        page_chunk_ids = []
                        for idx, chunk_id in enumerate(all_chunks['ids']):
                            if all_chunks.get('metadatas') and idx < len(all_chunks['metadatas']):
                                metadata = all_chunks['metadatas'][idx]
                                if metadata and metadata.get('id') == page_id:
                                    page_chunk_ids.append(chunk_id)

                        if page_chunk_ids:
                            existing_db.delete(ids=page_chunk_ids)
                            self.stdout.write(f'  Deleted {len(page_chunk_ids)} old chunks')
                except Exception as e:
                    self.stdout.write(self.style.WARNING(f'  Could not delete old chunks: {e}'))

            self.stdout.write('Updating/adding documents with deterministic IDs...')

            # Add or update documents with deterministic IDs
            try:
                existing_db.add_texts(
                    texts=chunks,
                    metadatas=chunk_metadatas,
                    ids=chunk_ids
                )
                vectorstore = existing_db
            except Exception as e:
                error_msg = str(e)
                if 'dimension' in error_msg.lower() or 'embedding' in error_msg.lower():
                    self.stdout.write(
                        self.style.ERROR(
                            f'\n❌ ERROR: Embedding dimension mismatch!\n'
                            f'   The collection was created with a different embedding model.\n'
                            f'   Current embedding: {embedding_provider} / {embedding_model or "default"}\n'
                            f'   You MUST reset the collection first:\n'
                            f'   python manage.py build_rag_index --reset\n'
                            f'   Error: {error_msg}\n'
                        )
                    )
                    return
                else:
                    raise
        except Exception as e:
            error_msg = str(e)
            # Check if it's a dimension mismatch error
            if 'dimension' in error_msg.lower() or 'embedding' in error_msg.lower():
                self.stdout.write(
                    self.style.ERROR(
                        f'\n❌ ERROR: Embedding dimension mismatch!\n'
                        f'   The collection was created with a different embedding model.\n'
                        f'   Current embedding: {embedding_provider} / {embedding_model or "default"}\n'
                        f'   You MUST reset the collection first:\n'
                        f'   python manage.py build_rag_index --reset\n'
                        f'   Error: {error_msg}\n'
                    )
                )
                return
            # Collection doesn't exist - create new one
            self.stdout.write('Creating new collection...')
            vectorstore = Chroma.from_texts(
                texts=chunks,
                metadatas=chunk_metadatas,
                ids=chunk_ids,
                embedding=embeddings,
                persist_directory=persist_directory,
                collection_name=collection_name
            )

        self.stdout.write(
            self.style.SUCCESS(
                f'Successfully indexed {len(chunks)} chunks in ChromaDB collection "{collection_name}"'
            )
        )
        self.stdout.write(f'Vector store saved to: {persist_directory}')

