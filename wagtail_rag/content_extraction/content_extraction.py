"""Extract text from Wagtail pages (StreamField, RichTextField, etc.)."""

import html
import re
from typing import Optional

from wagtail.models import Page


def get_page_url(page: Page) -> str:
    try:
        return page.get_full_url() or page.url_path or ""
    except Exception:
        return ""


def _get_field_value(value):
    if not value:
        return None
    if isinstance(value, str):
        return value
    if hasattr(value, "all"):
        items = value.all()
        if items:
            return ", ".join(
                getattr(item, "name", None) or getattr(item, "title", None) or str(item)
                for item in items
            )
        return None
    if hasattr(value, "title"):
        attr = value.title
        return attr() if callable(attr) else attr
    if hasattr(value, "name"):
        attr = value.name
        return attr() if callable(attr) else attr
    return str(value)


def is_rich_text_field(field) -> bool:
    cls_name = getattr(field.__class__, "__name__", "") or ""
    return "RichTextField" in cls_name or "RichText" in cls_name


def extract_page_content(page: Page, important_fields=None) -> Optional[str]:
    text_parts = []
    added_fields = set()

    if page.title:
        text_parts.append(f"Title: {page.title}")
        added_fields.add("title")

    if important_fields:
        for field_name in important_fields:
            if field_name in added_fields or not hasattr(page, field_name):
                continue
            value = getattr(page, field_name, None)
            field_value = _get_field_value(value)
            if field_value:
                label = field_name.replace("_", " ").title()
                text_parts.append(f"{label}: {field_value}")
                added_fields.add(field_name)

    common_fields = [
        "subtitle",
        "introduction",
        "description",
        "summary",
        "date_published",
        "first_published_at",
        "search_description",
    ]
    for field_name in common_fields:
        if field_name in added_fields or not hasattr(page, field_name):
            continue
        value = getattr(page, field_name, None)
        if value:
            label = (
                "Published"
                if field_name in ("date_published", "first_published_at")
                else field_name.replace("_", " ").title()
            )
            text_parts.append(f"{label}: {value}")
            added_fields.add(field_name)

    streamfield_fields = ["body", "content", "backstory", "instructions"]
    for field_name in streamfield_fields:
        if field_name in added_fields or not hasattr(page, field_name):
            continue
        value = getattr(page, field_name, None)
        if value:
            streamfield_text = extract_streamfield_text(value)
            if streamfield_text:
                label = field_name.replace("_", " ").title()
                text_parts.append(f"{label}:\n{streamfield_text}")
                added_fields.add(field_name)

    if hasattr(page, "address") and page.address:
        text_parts.append(f"Address: {page.address}")

    if hasattr(page, "lat_long") and page.lat_long:
        text_parts.append(f"Coordinates: {page.lat_long}")

    text_parts.extend(extract_richtext_field_text(page))
    text_parts.extend(extract_authors_text(page))
    text_parts.extend(extract_operating_hours_text(page))
    text_parts.extend(extract_tags_text(page))

    text = "\n\n".join(text_parts)
    return text.strip() or None


def extract_tags_text(page) -> list:
    result = []
    if hasattr(page, "get_tags"):
        tags = page.get_tags()
        if tags:
            result.append(f"Tags: {', '.join(str(tag) for tag in tags)}")

    return result


def extract_streamfield_text(streamfield) -> str:
    if not streamfield:
        return ""
    parts = []
    for block in streamfield:
        if hasattr(block, "value"):
            value = block.value
            if isinstance(value, str):
                parts.append(value)

            elif isinstance(value, dict):
                parts.extend(_extract_dict_strings(value))

        elif isinstance(block, str):
            parts.append(block)

    return " ".join(parts)


def extract_richtext_field_text(page) -> list:
    result = []
    if hasattr(page, "_meta"):
        for field in page._meta.get_fields():
            if not hasattr(field, "field") or not is_rich_text_field(field):
                continue

            value = getattr(page, field.name, None)
            if not value:
                continue
            if hasattr(value, "source"):
                rich_text = clean_html(value.source)
                if rich_text:
                    result.append(f"{field.name.title()}: {rich_text}")
            else:
                result.append(f"{field.name.title()}: {str(value)}")

    return result


def extract_hours_text(page) -> list:
    result = []
    if not (hasattr(page, "operating_hours") and page.operating_hours):
        return result

    hours = [
        f"{h.day}: {h.opening_time} - {h.closing_time}"
        for h in page.operating_hours
        if hasattr(h, "day") and hasattr(h, "opening_time") and hasattr(h, "closing_time")
    ]
    if hours:
        result.append("Operating Hours:\n" + "\n".join(hours))

    return result

def extract_authors_text(page):
    result = []
    if not hasattr(page, "authors"):
        return result

    authors = page.authors() if callable(page.authors) else page.authors
    if authors:
        names = []
        for author in authors:
            if hasattr(author, "first_name") and hasattr(author, "last_name"):
                names.append(f"{author.first_name} {author.last_name}".strip())
            else:
                names.append(str(author))
        if names:
            result.append(f"Authors: {', '.join(names)}")

    return result

def _extract_dict_strings(data):
    out = []
    for value in data.values():
        if isinstance(value, str):
            out.append(value)
        elif isinstance(value, dict):
            out.extend(_extract_dict_strings(value))
    return out


def clean_html(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()
