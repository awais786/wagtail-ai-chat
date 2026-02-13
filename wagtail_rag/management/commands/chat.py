"""
Django management command for interactive RAG chatbot testing from command line.
"""

import json
import traceback
import uuid
from typing import Any, Dict, Optional

from django.conf import settings
from django.core.management.base import BaseCommand

from wagtail_rag.rag_chatbot import get_chatbot

EXIT_COMMANDS = {"exit", "quit"}
SOURCES_USAGE = "Usage: sources on|off\n"


class Command(BaseCommand):
    help = "Interactive RAG chatbot for testing from command line"

    def add_arguments(self, parser):
        parser.add_argument(
            "-q",
            "--question",
            type=str,
            default=None,
            help="Ask a single question and exit (non-interactive mode)",
        )
        parser.add_argument(
            "--session-id",
            type=str,
            default=None,
            help="Session ID for chat history (auto-generated if not provided)",
        )
        parser.add_argument(
            "--no-history",
            action="store_true",
            help="Disable chat history (each question is independent)",
        )
        parser.add_argument(
            "--no-sources", action="store_true", help="Hide source documents in output"
        )
        parser.add_argument(
            "--filter",
            type=str,
            default=None,
            help='Metadata filter as JSON string, e.g. \'{"model": "BlogPage"}\'',
        )

    def handle(self, *args, **options):
        # Display configuration
        self._display_config(options)

        # Parse metadata filter if provided
        metadata_filter = self._parse_filter(options.get("filter"))

        # Initialize chatbot (uses Django settings)
        try:
            chatbot_kwargs: Dict[str, Any] = {}
            if metadata_filter:
                chatbot_kwargs["metadata_filter"] = metadata_filter

            chatbot = get_chatbot(**chatbot_kwargs)
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"Failed to initialize chatbot: {e}"))
            return

        # Single question mode
        if options["question"]:
            self._handle_single_question(
                chatbot,
                options["question"],
                options["session_id"],
                options["no_history"],
                options["no_sources"],
            )
            return

        # Interactive mode
        self._handle_interactive_mode(
            chatbot, options["session_id"], options["no_history"], options["no_sources"]
        )

    def _parse_filter(self, filter_value: Optional[str]) -> Optional[Dict[str, Any]]:
        """Parse --filter JSON value and validate expected shape."""
        if not filter_value:
            return None
        try:
            parsed = json.loads(filter_value)
        except json.JSONDecodeError as e:
            self.stdout.write(self.style.ERROR(f"Invalid JSON filter: {e}"))
            raise SystemExit(1)

        if not isinstance(parsed, dict):
            self.stdout.write(
                self.style.ERROR(
                    'Invalid filter: expected a JSON object (e.g. {"model": "BlogPage"})'
                )
            )
            raise SystemExit(1)
        return parsed

    def _display_config(self, options):
        """Display effective configuration from Django settings and CLI flags."""
        self.stdout.write("=" * 70)
        self.stdout.write(
            self.style.SUCCESS("Wagtail RAG Chatbot - Command Line Interface")
        )
        self.stdout.write("=" * 70)

        # LLM config (from Django settings)
        llm_provider = getattr(settings, "WAGTAIL_RAG_LLM_PROVIDER", "ollama")
        model_name = getattr(settings, "WAGTAIL_RAG_MODEL_NAME", "default")
        self.stdout.write(f"LLM: {llm_provider}/{model_name}")

        # Embedding config
        emb_provider = getattr(
            settings, "WAGTAIL_RAG_EMBEDDING_PROVIDER", "huggingface"
        )
        emb_model = getattr(settings, "WAGTAIL_RAG_EMBEDDING_MODEL", None)
        self.stdout.write(f'Embeddings: {emb_provider}/{emb_model or "default"}')

        # Vector store
        vector_store = getattr(settings, "WAGTAIL_RAG_VECTOR_STORE_BACKEND", "faiss")
        collection = getattr(settings, "WAGTAIL_RAG_COLLECTION_NAME", "wagtail_rag")
        self.stdout.write(f"Vector Store: {vector_store} (collection: {collection})")

        # Search settings
        use_hybrid = getattr(settings, "WAGTAIL_RAG_USE_HYBRID_SEARCH", True)
        use_multi_query = getattr(
            settings, "WAGTAIL_RAG_USE_LLM_QUERY_EXPANSION", False
        )
        retrieve_k = getattr(settings, "WAGTAIL_RAG_RETRIEVE_K", 8)
        self.stdout.write(
            f"Retrieval: k={retrieve_k}, hybrid={use_hybrid}, multi_query={use_multi_query}"
        )

        # Effective chat history mode
        history_enabled_in_settings = getattr(
            settings, "WAGTAIL_RAG_ENABLE_CHAT_HISTORY", True
        )
        effective_history = history_enabled_in_settings and not options.get(
            "no_history", False
        )
        if effective_history:
            self.stdout.write("Chat History: enabled")
        elif not history_enabled_in_settings:
            self.stdout.write("Chat History: disabled in settings")
        else:
            self.stdout.write("Chat History: disabled by --no-history")

        # Effective source display mode
        if options.get("no_sources", False):
            self.stdout.write("Sources Display: disabled by --no-sources")
        else:
            self.stdout.write("Sources Display: enabled")

        # Metadata filter
        if options.get("filter"):
            self.stdout.write(f'Filter: {options["filter"]}')

        self.stdout.write("=" * 70)
        self.stdout.write("")

    def _handle_single_question(
        self, chatbot, question, session_id, no_history, no_sources
    ):
        """Handle single question mode."""
        try:
            query_kwargs = self._build_query_kwargs(
                use_history=(not no_history),
                session_id=session_id,
            )

            # Query chatbot
            self.stdout.write(self.style.WARNING(f"\nQuestion: {question}\n"))
            result = chatbot.query(question, **query_kwargs)

            # Display answer
            self.stdout.write(self.style.SUCCESS("Answer:"))
            self.stdout.write(result.get("answer", "No answer generated"))
            self.stdout.write("")

            # Display sources
            if not no_sources:
                self._display_sources(result.get("sources", []))

        except Exception as e:
            self.stdout.write(self.style.ERROR(f"\nError: {e}"))
            traceback.print_exc()

    def _handle_interactive_mode(self, chatbot, session_id, no_history, no_sources):
        """Handle interactive chat mode."""
        # Generate session ID for history
        if not no_history and getattr(
            settings, "WAGTAIL_RAG_ENABLE_CHAT_HISTORY", True
        ):
            session_id = session_id or uuid.uuid4().hex[:8]
            use_history = True
            self.stdout.write(
                self.style.SUCCESS(f"Chat session started: {session_id}\n")
            )
        else:
            use_history = False
            self.stdout.write(self.style.SUCCESS("Chat started (no history)\n"))

        self.stdout.write("Type your questions below. Commands:")
        self.stdout.write('  - Type "exit", "quit", or press Ctrl+C to exit')
        self.stdout.write('  - Type "clear" to start a new session')
        self.stdout.write('  - Type "sources on/off" to toggle source display')
        self.stdout.write("=" * 70)
        self.stdout.write("")

        show_sources = not no_sources

        while True:
            try:
                # Get user input
                question = input(self.style.WARNING("You: ")).strip()

                if not question:
                    continue

                # Handle commands
                lower_question = question.lower()

                if lower_question in EXIT_COMMANDS:
                    self.stdout.write(self.style.SUCCESS("\nGoodbye!"))
                    break

                if lower_question == "clear":
                    if use_history:
                        session_id = uuid.uuid4().hex[:8]
                        self.stdout.write(
                            self.style.SUCCESS(f"New session started: {session_id}\n")
                        )
                    else:
                        self.stdout.write(
                            self.style.SUCCESS("Session cleared (history disabled)\n")
                        )
                    continue

                # Handle sources toggle command
                parts = lower_question.split()
                if parts and parts[0] == "sources":
                    if len(parts) != 2:
                        self.stdout.write(self.style.ERROR(SOURCES_USAGE))
                        continue
                    toggle = parts[1]
                    if toggle == "on":
                        show_sources = True
                        self.stdout.write(self.style.SUCCESS("Sources display: ON\n"))
                    elif toggle == "off":
                        show_sources = False
                        self.stdout.write(self.style.SUCCESS("Sources display: OFF\n"))
                    else:
                        self.stdout.write(self.style.ERROR(SOURCES_USAGE))
                    continue

                query_kwargs = self._build_query_kwargs(
                    use_history=use_history,
                    session_id=session_id,
                )
                result = chatbot.query(question, **query_kwargs)

                # Display answer
                self.stdout.write(
                    self.style.SUCCESS("Bot: ")
                    + result.get("answer", "No answer generated")
                )
                self.stdout.write("")

                # Display sources
                if show_sources:
                    self._display_sources(result.get("sources", []))

            except KeyboardInterrupt:
                self.stdout.write(self.style.SUCCESS("\n\nGoodbye!"))
                break
            except EOFError:
                self.stdout.write(self.style.SUCCESS("\n\nGoodbye!"))
                break
            except Exception as e:
                self.stdout.write(self.style.ERROR(f"\nError: {e}"))
                traceback.print_exc()
                self.stdout.write("")

    @staticmethod
    def _build_query_kwargs(
        use_history: bool, session_id: Optional[str]
    ) -> Dict[str, str]:
        """Build kwargs for chatbot.query based on history mode."""
        query_kwargs: Dict[str, str] = {}
        if use_history and getattr(settings, "WAGTAIL_RAG_ENABLE_CHAT_HISTORY", True):
            query_kwargs["session_id"] = session_id or uuid.uuid4().hex
        return query_kwargs

    def _display_sources(self, sources):
        """Display source documents."""
        if not sources:
            return

        self.stdout.write(self.style.HTTP_INFO(f"Sources ({len(sources)} documents):"))
        for i, source in enumerate(sources[:5], 1):  # Show top 5 sources
            metadata = source.get("metadata", {})
            title = metadata.get("title", "Unknown")
            url = metadata.get("url", "")
            model = metadata.get("model", "")
            content_preview = " ".join(source.get("content", "").split())[:150]

            self.stdout.write(f"  {i}. {title}")
            if model:
                self.stdout.write(f"     Model: {model}")
            if url:
                self.stdout.write(f"     URL: {url}")
            if content_preview:
                self.stdout.write(f"     Preview: {content_preview}...")
            self.stdout.write("")
