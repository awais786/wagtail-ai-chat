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
        return page.get_full_url() or page.url_path or ""
    except Exception:
        return ""


def _get_field_value(value):
    """
    Extract string representation from various field types.

    Args:
        value: Field value (can be related object, string, or queryset)

    Returns:
        String representation or None
    """
    if not value:
        return None

    # Handle strings directly
    if isinstance(value, str):
        return value

    # Handle ManyToMany querysets
    if hasattr(value, 'all'):
        items = value.all()
        if items:
            return ', '.join([
                getattr(item, 'name', None) or getattr(item, 'title', None) or str(item)
                for item in items
            ])
        return None

    # Handle related objects with title/name attributes
    if hasattr(value, 'title'):
        attr = value.title
        return attr() if callable(attr) else attr

    if hasattr(value, 'name'):
        attr = value.name
        return attr() if callable(attr) else attr

    # Fallback to string conversion
    return str(value)


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
    added_fields = {'title'}

    # If important fields are specified, extract and add them first
    if important_fields:
        for field_name in important_fields:
            if field_name in added_fields:
                continue

            if hasattr(page, field_name):
                value = getattr(page, field_name, None)
                field_value = _get_field_value(value)

                if field_value:
                    label = field_name.replace('_', ' ').title()
                    text_parts.append(f"{label}: {field_value}")
                    added_fields.add(field_name)
    
    # Common Wagtail fields (only if not already in important_fields)
    common_fields = [
        'subtitle', 'introduction', 'description', 'summary',
        'date_published', 'first_published_at', 'search_description'
    ]
    
    for field_name in common_fields:
        if field_name in added_fields or not hasattr(page, field_name):
            continue

        value = getattr(page, field_name, None)
        if value:
            label = 'Published' if field_name in ['date_published', 'first_published_at'] else field_name.replace('_', ' ').title()
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
        tags = page.get_tags()
        if tags:
            text_parts.append(f"Tags: {', '.join(str(tag) for tag in tags)}")

    # Extract authors if available
    if hasattr(page, 'authors'):
        authors = page.authors() if callable(page.authors) else page.authors
        if authors:
            author_names = [
                f"{author.first_name} {author.last_name}".strip()
                if hasattr(author, 'first_name') and hasattr(author, 'last_name')
                else str(author)
                for author in authors
            ]
            if author_names:
                text_parts.append(f"Authors: {', '.join(author_names)}")

    # Extract address if available
    if hasattr(page, 'address') and page.address:
        text_parts.append(f"Address: {page.address}")
    
    # Extract coordinates if available
    if hasattr(page, 'lat_long') and page.lat_long:
        text_parts.append(f"Coordinates: {page.lat_long}")
    
    # Extract operating hours if available
    if hasattr(page, 'operating_hours') and page.operating_hours:
        hours_text = [
            f"{hour.day}: {hour.opening_time} - {hour.closing_time}"
            for hour in page.operating_hours
            if hasattr(hour, 'day') and hasattr(hour, 'opening_time') and hasattr(hour, 'closing_time')
        ]
        if hours_text:
            text_parts.append("Operating Hours:\n" + '\n'.join(hours_text))

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
                # Extract text from dict values recursively
                text_parts.extend(_extract_dict_strings(value))
        elif isinstance(block, str):
            text_parts.append(block)

    return " ".join(text_parts)


def _extract_dict_strings(data):
    """Recursively extract strings from a dictionary."""
    strings = []
    for value in data.values():
        if isinstance(value, str):
            strings.append(value)
        elif isinstance(value, dict):
            strings.extend(_extract_dict_strings(value))
    return strings


def clean_html(text: str) -> str:
    """Remove HTML tags and decode entities."""
    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)
    # Decode HTML entities
    text = html.unescape(text)
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)
    return text.strip()

