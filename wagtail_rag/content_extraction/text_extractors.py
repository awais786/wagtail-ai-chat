"""
Text extraction utilities for Wagtail pages.

This module provides utility functions for extracting plain text content from Wagtail Page models,
handling various field types including StreamFields, RichTextFields, and other common Wagtail fields.

Main Functions:
- get_page_url(): Safely get the URL of a Wagtail page
- extract_all_page_content_as_text(): Extract all text content from a page (title, fields, streamfields, richtext, tags, authors, etc.)
- extract_text_from_streamfield(): Extract text from a Wagtail StreamField block
- strip_html_tags_and_normalize_text(): Remove HTML tags, decode entities, and normalize whitespace

Helper Functions (internal):
- _convert_field_value_to_string(): Convert various Wagtail field types to plain strings
- _extract_all_strings_from_dict_recursively(): Recursively extract all strings from nested dictionaries
"""

import html
import re
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from wagtail.models import Page  # type: ignore
else:
    try:
        from wagtail.models import Page  # type: ignore
    except ImportError:
        # Fallback for environments without Wagtail
        Page = None  # type: ignore


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
    except (AttributeError, RuntimeError, Exception):
        return ""


def _convert_field_value_to_string(value):
    """
    Convert various Wagtail field types to a plain string representation.
    
    Handles:
    - Plain strings (returns as-is)
    - ManyToMany querysets (joins item names/titles)
    - Related objects with 'title' or 'name' attributes
    - Any other type (converts to string)

    Args:
        value: Field value (can be related object, string, queryset, or any type)

    Returns:
        String representation or None if value is empty/None
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


def extract_all_page_content_as_text(page: Page, important_fields=None) -> Optional[str]:
    """
    Extract all text content from a Wagtail page into a single formatted text string.
    
    Extracts and combines:
    - Page title
    - Important/custom fields (if specified)
    - Common fields (subtitle, introduction, description, summary, dates, search_description)
    - StreamField content (body, content, backstory, instructions)
    - RichTextField content (all RichText fields found on the page)
    - Tags (if available)
    - Authors (if available)
    - Address, coordinates, operating hours (if available)
    
    This matches the extraction logic used during indexing in build_rag_index.py
    to ensure consistency between indexed content and search-time extraction.

    Args:
        page: Wagtail Page instance
        important_fields: Optional list of field names to extract first/emphasize 
                         (e.g., ['bread_type', 'origin'])

    Returns:
        Formatted text string with all extracted content, or None if no content found
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
                field_value = _convert_field_value_to_string(value)

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
                streamfield_text = extract_text_from_streamfield(field_value)
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
                            rich_text = strip_html_tags_and_normalize_text(value.source)
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


def extract_text_from_streamfield(streamfield) -> str:
    """
    Extract all text content from a Wagtail StreamField block.
    
    Processes each block in the StreamField:
    - String blocks: added directly
    - Blocks with 'value' attribute: extracts string values or recursively processes dict values
    - Dict blocks: recursively extracts all string values from nested dictionaries
    
    This matches the extraction logic used during indexing in build_rag_index.py.
    
    Args:
        streamfield: Wagtail StreamField instance or iterable of blocks
        
    Returns:
        Single string containing all extracted text, space-separated
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
                text_parts.extend(_extract_all_strings_from_dict_recursively(value))
        elif isinstance(block, str):
            text_parts.append(block)

    return " ".join(text_parts)


def _extract_all_strings_from_dict_recursively(data):
    """
    Recursively extract all string values from a nested dictionary.
    
    Traverses the dictionary structure and collects all string values,
    handling nested dictionaries by recursively processing them.
    
    Args:
        data: Dictionary (may contain nested dictionaries)
        
    Returns:
        List of all string values found in the dictionary structure
    """
    strings = []
    for value in data.values():
        if isinstance(value, str):
            strings.append(value)
        elif isinstance(value, dict):
            strings.extend(_extract_all_strings_from_dict_recursively(value))
    return strings


def strip_html_tags_and_normalize_text(text: str) -> str:
    """
    Remove HTML tags, decode HTML entities, and normalize whitespace in text.
    
    Processing steps:
    1. Remove all HTML tags using regex
    2. Decode HTML entities (e.g., &amp; -> &, &lt; -> <)
    3. Normalize multiple whitespace characters to single spaces
    4. Strip leading/trailing whitespace
    
    Args:
        text: String that may contain HTML tags and entities
        
    Returns:
        Clean plain text with no HTML tags, decoded entities, and normalized whitespace
    """
    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)
    # Decode HTML entities
    text = html.unescape(text)
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# wagtail_page_to_documents has been moved to page_to_documents.py
# This function is now available via: from wagtail_rag.content_extraction.page_to_documents import wagtail_page_to_documents



