#!/usr/bin/env python
"""
Quick test script for Wagtail RAG chatbot.

Usage:
    python manage.py shell < test_rag.py
    OR
    python test_rag.py  (if Django is configured)
"""
import os
import sys

# Try to setup Django automatically
try:
    import django
    from django.conf import settings
    
    # If settings not configured, try common patterns
    if not settings.configured:
        # Try to find settings module
        for path in sys.path:
            if os.path.exists(os.path.join(path, 'manage.py')):
                # Try to import settings from common locations
                try:
                    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'myproject.settings')
                    django.setup()
                    break
                except:
                    try:
                        os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'settings')
                        django.setup()
                        break
                    except:
                        pass
    else:
        django.setup()
except Exception as e:
    print("Warning: Could not auto-configure Django. Please run with:")
    print("  python manage.py shell < test_rag.py")
    print(f"Error: {e}\n")

try:
    from wagtail_rag.rag_chatbot import get_chatbot
except ImportError as e:
    print(f"Error: Could not import wagtail_rag. Make sure it's installed and in INSTALLED_APPS.")
    print(f"Import error: {e}")
    sys.exit(1)


def test_chatbot():
    """Test the Wagtail RAG chatbot."""
    print("=" * 60)
    print("Testing Wagtail RAG Chatbot")
    print("=" * 60)
    
    try:
        # Initialize chatbot
        print("\n1. Initializing chatbot...")
        chatbot = get_chatbot()
        print("   ✓ Chatbot initialized successfully")
        
        # Test search functionality
        print("\n2. Testing search functionality...")
        test_query = "test"
        results = chatbot.search_similar(test_query, k=3)
        print(f"   ✓ Found {len(results)} similar documents")
        if results:
            print(f"   Top result: {results[0]['metadata'].get('title', 'Unknown')}")
        
        # Test query functionality
        print("\n3. Testing query functionality...")
        question = "What content is available on this site?"
        print(f"   Question: {question}")
        print("   Processing...")
        
        result = chatbot.query(question)
        
        print(f"\n   Answer:")
        print(f"   {result['answer']}")
        print(f"\n   Sources ({len(result['sources'])}):")
        for i, source in enumerate(result['sources'][:5], 1):  # Show first 5
            title = source['metadata'].get('title', 'Unknown')
            url = source['metadata'].get('url', 'N/A')
            print(f"   {i}. {title}")
            if url:
                print(f"      URL: {url}")
        
        print("\n" + "=" * 60)
        print("✓ All tests completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print("\n" + "=" * 60)
        print("✗ Error during testing:")
        print(f"   {e}")
        print("=" * 60)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    test_chatbot()

