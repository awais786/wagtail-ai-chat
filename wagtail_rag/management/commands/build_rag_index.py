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

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
    except ImportError:
        try:
            from langchain.embeddings import HuggingFaceEmbeddings
        except ImportError:
            raise ImportError(
                "Could not import HuggingFaceEmbeddings. "
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
            '--important-fields',
            nargs='+',
            default=None,
            help='Global important fields to emphasize in all models (e.g., bread_type origin category)'
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
                        text_parts.append(block_value.source)
                    else:
                        text_parts.append(str(block_value))
                
                elif 'quote' in block_type:
                    quote_text = self._extract_text_from_value(block_value, ['text', 'quote'])
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
                    text = self._extract_text_from_value(block_value, ['text', 'content', 'value', 'body', 'description'])
                    if text:
                        text_parts.append(text)
                    else:
                        text_parts.append(str(block_value))
            
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
        
        # If important fields are specified, extract and emphasize them first
        if important_fields:
            for field_name in important_fields:
                if hasattr(page, field_name):
                    value = getattr(page, field_name, None)
                    if value:
                        # Handle ForeignKey relationships
                        if hasattr(value, 'title'):
                            field_value = value.title
                        elif hasattr(value, 'name'):
                            field_value = value.name
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
                            field_value = str(value)
                        
                        # Emphasize important fields by adding them multiple times
                        label = field_name.replace('_', ' ').title()
                        text_parts.insert(0, f"{label}: {field_value}")  # Put at top
                        text_parts.append(f"{label}: {field_value}")  # Also at end
        
        # Common Wagtail fields
        common_fields = [
            'subtitle', 'introduction', 'description', 'summary',
            'date_published', 'first_published_at'
        ]
        
        for field_name in common_fields:
            if hasattr(page, field_name):
                value = getattr(page, field_name, None)
                if value:
                    if field_name in ['date_published', 'first_published_at']:
                        text_parts.append(f"Published: {value}")
                    else:
                        # Capitalize field name for display
                        label = field_name.replace('_', ' ').title()
                        text_parts.append(f"{label}: {value}")
        
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
        
        Format: model_name:field1,field2 model_name2:field3
        
        Returns:
            Dict mapping model names to list of important fields
        """
        if not model_fields_arg:
            return {}
        
        model_fields_map = {}
        for item in model_fields_arg:
            if ':' in item:
                model_name, fields_str = item.split(':', 1)
                fields = [f.strip() for f in fields_str.split(',')]
                model_fields_map[model_name] = fields
        return model_fields_map

    def _get_pages_to_index(self, model_names=None, exclude_models=None, model_fields_map=None, important_fields=None, page_id=None):
        """
        Get pages to index based on model selection.
        
        Args:
            model_names: List of specific model names to include
            exclude_models: List of model names to exclude
            model_fields_map: Dict mapping model names to important fields
            important_fields: Global list of important fields for all models
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
                    model_important_fields = model_fields_map.get(model_name, [])
                    # Combine with global important fields
                    all_important_fields = list(set((important_fields or []) + model_important_fields))
                    
                    page_text = self._extract_page_text(page, important_fields=all_important_fields if all_important_fields else None)
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
                        
                        # Add any important fields to metadata generically
                        if all_important_fields:
                            for field_name in all_important_fields:
                                if hasattr(page, field_name):
                                    value = getattr(page, field_name, None)
                                    if value:
                                        # Handle ForeignKey
                                        if hasattr(value, 'title'):
                                            page_metadata[field_name] = value.title
                                        elif hasattr(value, 'name'):
                                            page_metadata[field_name] = value.name
                                        # Handle ManyToMany
                                        elif hasattr(value, 'all'):
                                            items = value.all()
                                            if items:
                                                page_metadata[field_name] = ', '.join([
                                                    getattr(item, 'name', getattr(item, 'title', str(item)))
                                                    for item in items
                                                ])
                                        else:
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
        model_names = options['models']
        exclude_models = options['exclude_models'] or []
        model_fields_arg = options['model_fields'] or []
        important_fields = options['important_fields'] or []
        chunk_size = options['chunk_size']
        chunk_overlap = options['chunk_overlap']
        collection_name = options['collection_name']
        reset = options['reset']
        reset_only = options['reset_only']
        page_id = options['page_id']
        
        # Parse model-specific fields
        model_fields_map = self._parse_model_fields(model_fields_arg)
        
        if page_id:
            self.stdout.write(f'Re-indexing specific page ID: {page_id}')
        
        if model_fields_map:
            self.stdout.write(f'Model-specific important fields: {model_fields_map}')
        if important_fields:
            self.stdout.write(f'Global important fields: {important_fields}')

        # Get collection name from settings or use default
        if not collection_name:
            collection_name = getattr(
                settings,
                'WAGTAIL_RAG_COLLECTION_NAME',
                'wagtail_rag'
            )

        # Add default exclusions
        default_excludes = getattr(
            settings,
            'WAGTAIL_RAG_EXCLUDE_MODELS',
            ['wagtailcore.Page', 'wagtailcore.Site']
        )
        exclude_models = list(set(exclude_models + default_excludes))

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
            important_fields=important_fields,
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
        embedding_model = getattr(
            settings,
            'WAGTAIL_RAG_EMBEDDING_MODEL',
            'sentence-transformers/all-MiniLM-L6-v2'
        )
        self.stdout.write(f'Using embedding model: {embedding_model}')
        embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model
        )

        # Set up ChromaDB persistence directory
        persist_directory = getattr(
            settings,
            'WAGTAIL_RAG_CHROMA_PATH',
            os.path.join(settings.BASE_DIR, 'chroma_db')
        )
        os.makedirs(persist_directory, exist_ok=True)

        # Reset collection if requested (need embeddings for Chroma connection)
        if reset or reset_only:
            embedding_model = getattr(
                settings,
                'WAGTAIL_RAG_EMBEDDING_MODEL',
                'sentence-transformers/all-MiniLM-L6-v2'
            )
            temp_embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
            self.stdout.write(f'Resetting collection: {collection_name}')
            try:
                existing_db = Chroma(
                    persist_directory=persist_directory,
                    collection_name=collection_name,
                    embedding_function=temp_embeddings
                )
                existing_db.delete_collection()
                self.stdout.write(self.style.SUCCESS(f'Collection "{collection_name}" cleared successfully'))
            except Exception as e:
                self.stdout.write(self.style.WARNING(f'Collection may not exist: {e}'))
        
        # If reset-only, exit here without indexing
        if reset_only:
            self.stdout.write(self.style.SUCCESS('Reset complete. No indexing performed.'))
            return

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
            existing_db.add_texts(
                texts=chunks,
                metadatas=chunk_metadatas,
                ids=chunk_ids
            )
            vectorstore = existing_db
        except Exception:
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

