"""
Utility functions for content extraction and processing.

This module provides utilities for extracting text content from Wagtail Page models,
handling StreamFields, RichTextFields, and other common Wagtail field types.
"""

import html
import re
from typing import Optional

from wagtail.models import Page


def get_page_url(page: Page) -> str:
    """
    Get page URL safely, handling cases where it might not be available.
    
    Args:
        page: Wagtail Page instance
        
    Returns:
        Page URL or empty string if not available
    """
    try:
        if hasattr(page, "get_full_url"):
            url = page.get_full_url()
            if url:
                return url
        # Fallback: try to construct URL from url_path
        if hasattr(page, "url_path"):
            return page.url_path
        # Last resort: return empty string
        return ""
    except Exception:
        return ""


def extract_page_content(page: Page, important_fields=None) -> Optional[str]:
    """
    Extract comprehensive text content from a Wagtail page.
    
    This matches the extraction logic used during indexing in build_rag_index.py
    to ensure consistency between indexed content and search-time extraction.

    Args:
        page: Wagtail Page instance
        important_fields: Optional list of field names to emphasize (e.g., ['bread_type', 'origin'])

    Returns:
        Extracted text content or None
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
        'date_published', 'first_published_at', 'search_description'
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
                streamfield_text = extract_streamfield_text(field_value)
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
                            rich_text = clean_html(value.source)
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
                    text_parts.append("Operating Hours:\n" + '\n'.join(hours_text))
        except Exception:
            pass
    
    # Join all parts
    text = '\n\n'.join(text_parts)
    return text.strip() if text.strip() else None


def extract_streamfield_text(streamfield) -> str:
    """
    Extract text from a StreamField.
    
    This matches the extraction logic used during indexing in build_rag_index.py.
    """
    if not streamfield:
        return ""

    text_parts = []
    for block in streamfield:
        if hasattr(block, "value"):
            value = block.value
            if isinstance(value, str):
                text_parts.append(value)
            elif isinstance(value, dict):
                # Extract text from dict values
                for v in value.values():
                    if isinstance(v, str):
                        text_parts.append(v)
                    elif isinstance(v, dict):
                        # Handle nested dicts (e.g., in rich text blocks)
                        for nested_v in v.values():
                            if isinstance(nested_v, str):
                                text_parts.append(nested_v)
        elif isinstance(block, str):
            text_parts.append(block)
        elif hasattr(block, "block_type"):
            # Handle block types that might have different structures
            if hasattr(block, "value"):
                block_value = block.value
                if isinstance(block_value, str):
                    text_parts.append(block_value)
                elif isinstance(block_value, dict):
                    # Extract all string values from dict
                    for v in block_value.values():
                        if isinstance(v, str):
                            text_parts.append(v)

    return " ".join(text_parts)


def clean_html(text: str) -> str:
    """Remove HTML tags and decode entities."""
    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)
    # Decode HTML entities
    text = html.unescape(text)
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)
    return text.strip()

