"""
Unified RAG management command.

Subcommands:
  index   -- Build / reset the vector-store index
  chat    -- Interactive or single-question chatbot
  test    -- Smoke-test the pipeline with predefined questions

Examples:
  python manage.py rag index
  python manage.py rag index --reset-only
  python manage.py rag index --page-id 42
  python manage.py rag chat
  python manage.py rag chat -q "What is sourdough bread?"
  python manage.py rag chat --search-only --no-sources
  python manage.py rag test
  python manage.py rag test --questions "What breads do you sell?" "Where are you located?"
"""

import json
import traceback
import uuid
from typing import Any, Optional

from django.conf import settings
from django.core.management.base import BaseCommand, CommandParser

from wagtail_rag.content_extraction.index_builder import build_rag_index
from wagtail_rag.chatbot import get_chatbot
from wagtail_rag.conf import conf

EXIT_COMMANDS = {"exit", "quit"}
SOURCES_USAGE = "Usage: sources on|off\n"

# Default smoke-test questions (override via --questions or WAGTAIL_RAG_TEST_QUESTIONS)
DEFAULT_TEST_QUESTIONS = [
    "What content is available on this site?",
    "Give me a summary of the main topics covered.",
    "What are the most recent pages or articles?",
]


class Command(BaseCommand):
    help = "Unified RAG pipeline: index | chat | test"

    def add_arguments(self, parser: CommandParser):
        subparsers = parser.add_subparsers(dest="subcommand", metavar="SUBCOMMAND")
        subparsers.required = True

        # ── index ──────────────────────────────────────────────────────────
        index_parser = subparsers.add_parser(
            "index", help="Build / reset the vector-store index"
        )
        index_parser.add_argument(
            "--reset-only",
            action="store_true",
            help="Clear the collection without re-indexing",
        )
        index_parser.add_argument(
            "--page-id",
            type=int,
            default=None,
            help="Re-index a single page by ID",
        )

        # ── chat ───────────────────────────────────────────────────────────
        chat_parser = subparsers.add_parser(
            "chat", help="Interactive or single-question chatbot"
        )
        chat_parser.add_argument(
            "-q",
            "--question",
            type=str,
            default=None,
            help="Ask a single question and exit",
        )
        chat_parser.add_argument(
            "--session-id",
            type=str,
            default=None,
            help="Session ID for chat history (auto-generated if omitted)",
        )
        chat_parser.add_argument(
            "--no-history",
            action="store_true",
            help="Disable chat history",
        )
        chat_parser.add_argument(
            "--no-sources",
            action="store_true",
            help="Hide source documents",
        )
        chat_parser.add_argument(
            "--filter",
            type=str,
            default=None,
            help='Metadata filter as JSON, e.g. \'{"model": "BlogPage"}\'',
        )
        chat_parser.add_argument(
            "--search-only",
            action="store_true",
            help="Return semantic search results only (skip LLM generation)",
        )

        # ── test ───────────────────────────────────────────────────────────
        test_parser = subparsers.add_parser(
            "test", help="Smoke-test the full RAG pipeline"
        )
        test_parser.add_argument(
            "--questions",
            nargs="+",
            metavar="Q",
            default=None,
            help="Questions to test (defaults to built-in set or WAGTAIL_RAG_TEST_QUESTIONS)",
        )
        test_parser.add_argument(
            "--search-only",
            action="store_true",
            help="Test retrieval only (skip LLM generation)",
        )
        test_parser.add_argument(
            "--filter",
            type=str,
            default=None,
            help='Metadata filter as JSON, e.g. \'{"model": "BlogPage"}\'',
        )

    # ── dispatch ──────────────────────────────────────────────────────────

    def handle(self, *args, **options):
        subcommand = options["subcommand"]
        if subcommand == "index":
            self._handle_index(options)
        elif subcommand == "chat":
            self._handle_chat(options)
        elif subcommand == "test":
            self._handle_test(options)

    # ══════════════════════════════════════════════════════════════════════
    # INDEX
    # ══════════════════════════════════════════════════════════════════════

    def _handle_index(self, options):
        model_names = getattr(settings, "WAGTAIL_RAG_MODELS", None)
        chunk_size = getattr(settings, "WAGTAIL_RAG_CHUNK_SIZE", 1000)
        chunk_overlap = getattr(settings, "WAGTAIL_RAG_CHUNK_OVERLAP", 200)
        collection = getattr(settings, "WAGTAIL_RAG_COLLECTION_NAME", "wagtail_rag")
        emb_provider = conf.embedding.provider
        emb_model = conf.embedding.model
        llm_provider = conf.llm.provider
        llm_model = conf.llm.model
        backend = getattr(settings, "WAGTAIL_RAG_VECTOR_STORE_BACKEND", "faiss")

        self._section("Wagtail RAG — Index")
        self.stdout.write(
            f"Models    : {', '.join(model_names) if model_names else 'ALL Page models'}"
        )
        self.stdout.write(f"Backend   : {backend}  collection={collection}")
        self.stdout.write(f"Chunking  : size={chunk_size}  overlap={chunk_overlap}")
        self.stdout.write(f"Embeddings: {emb_provider}/{emb_model or 'default'}")
        self.stdout.write(f"LLM       : {llm_provider}/{llm_model or 'default'}")
        self._divider()

        build_rag_index(
            model_names=model_names,
            reset_only=options["reset_only"],
            page_id=options["page_id"],
            stdout=self.stdout.write,
        )

    # ══════════════════════════════════════════════════════════════════════
    # CHAT
    # ══════════════════════════════════════════════════════════════════════

    def _handle_chat(self, options):
        self._print_chat_config(options)
        metadata_filter = self._parse_filter(options.get("filter"))

        try:
            chatbot = get_chatbot(
                **({"metadata_filter": metadata_filter} if metadata_filter else {})
            )
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"Failed to initialize chatbot: {e}"))
            self.stderr.write(traceback.format_exc())
            return

        if options.get("question"):
            self._single_question(
                chatbot,
                question=options["question"],
                session_id=options.get("session_id"),
                no_history=options["no_history"],
                no_sources=options["no_sources"],
                search_only=options["search_only"],
            )
        else:
            self._interactive_loop(
                chatbot,
                session_id=options.get("session_id"),
                no_history=options["no_history"],
                no_sources=options["no_sources"],
                search_only=options["search_only"],
            )

    def _print_chat_config(self, options):
        llm_provider = conf.llm.provider
        llm_model = conf.llm.model or "default"
        emb_provider = conf.embedding.provider
        emb_model = conf.embedding.model
        backend = getattr(settings, "WAGTAIL_RAG_VECTOR_STORE_BACKEND", "faiss")
        collection = getattr(settings, "WAGTAIL_RAG_COLLECTION_NAME", "wagtail_rag")
        use_hybrid = getattr(settings, "WAGTAIL_RAG_USE_HYBRID_SEARCH", True)
        retrieve_k = getattr(settings, "WAGTAIL_RAG_RETRIEVE_K", 8)
        history_in_settings = getattr(settings, "WAGTAIL_RAG_ENABLE_CHAT_HISTORY", True)
        effective_history = history_in_settings and not options.get("no_history", False)

        self._section("Wagtail RAG — Chat")
        self.stdout.write(f"LLM       : {llm_provider}/{llm_model}")
        self.stdout.write(f"Embeddings: {emb_provider}/{emb_model or 'default'}")
        self.stdout.write(f"Backend   : {backend}  collection={collection}")
        self.stdout.write(f"Retrieval : k={retrieve_k}  hybrid={use_hybrid}")
        self.stdout.write(f"History   : {'on' if effective_history else 'off'}")
        self.stdout.write(f"Sources   : {'off' if options.get('no_sources') else 'on'}")
        self.stdout.write(
            f"Mode      : {'SEARCH ONLY' if options.get('search_only') else 'Full RAG'}"
        )
        if options.get("filter"):
            self.stdout.write(f"Filter    : {options['filter']}")
        self._divider()

    def _single_question(
        self, chatbot, question, session_id, no_history, no_sources, search_only
    ):
        try:
            query_kwargs = self._build_query_kwargs(
                use_history=not no_history,
                session_id=session_id,
                search_only=search_only,
            )
            self.stdout.write(self.style.WARNING(f"\nQuestion: {question}\n"))
            result = chatbot.query(question, **query_kwargs)

            if not search_only:
                self.stdout.write(self.style.SUCCESS("Answer:"))
                self.stdout.write(result.get("answer", "No answer generated"))
                self.stdout.write("")
            else:
                self.stdout.write(self.style.SUCCESS("Search Results (LLM skipped):\n"))

            if not no_sources:
                self._display_sources(result.get("sources", []))

        except Exception as e:
            self.stdout.write(self.style.ERROR(f"\nError: {e}"))
            self.stderr.write(traceback.format_exc())

    def _interactive_loop(
        self, chatbot, session_id, no_history, no_sources, search_only
    ):
        history_enabled = getattr(settings, "WAGTAIL_RAG_ENABLE_CHAT_HISTORY", True)
        use_history = history_enabled and not no_history

        if use_history:
            session_id = session_id or uuid.uuid4().hex[:8]
            self.stdout.write(self.style.SUCCESS(f"Session: {session_id}\n"))
        else:
            self.stdout.write(self.style.SUCCESS("Chat started (no history)\n"))

        self.stdout.write(
            'Commands: "exit"/"quit" to stop  |  "clear" for new session  |  "sources on/off"'
        )
        self._divider()

        show_sources = not no_sources

        while True:
            try:
                question = input(self.style.WARNING("You: ")).strip()
                if not question:
                    continue

                lower = question.lower()

                if lower in EXIT_COMMANDS:
                    self.stdout.write(self.style.SUCCESS("\nGoodbye!"))
                    break

                if lower == "clear":
                    if use_history:
                        session_id = uuid.uuid4().hex[:8]
                        self.stdout.write(
                            self.style.SUCCESS(f"New session: {session_id}\n")
                        )
                    else:
                        self.stdout.write(self.style.SUCCESS("Cleared\n"))
                    continue

                parts = lower.split()
                if parts and parts[0] == "sources":
                    if len(parts) == 2 and parts[1] in ("on", "off"):
                        show_sources = parts[1] == "on"
                        self.stdout.write(
                            self.style.SUCCESS(f"Sources: {parts[1].upper()}\n")
                        )
                    else:
                        self.stdout.write(self.style.ERROR(SOURCES_USAGE))
                    continue

                query_kwargs = self._build_query_kwargs(
                    use_history=use_history,
                    session_id=session_id,
                    search_only=search_only,
                )
                result = chatbot.query(question, **query_kwargs)

                if not search_only:
                    self.stdout.write(
                        self.style.SUCCESS("Bot: ")
                        + result.get("answer", "No answer generated")
                    )
                    self.stdout.write("")
                else:
                    self.stdout.write(
                        self.style.SUCCESS("Search Results (LLM skipped):\n")
                    )

                if show_sources:
                    self._display_sources(result.get("sources", []))

            except (KeyboardInterrupt, EOFError):
                self.stdout.write(self.style.SUCCESS("\n\nGoodbye!"))
                break
            except Exception as e:
                self.stdout.write(self.style.ERROR(f"\nError: {e}"))
                self.stderr.write(traceback.format_exc())
                self.stdout.write("")

    # ══════════════════════════════════════════════════════════════════════
    # TEST
    # ══════════════════════════════════════════════════════════════════════

    def _handle_test(self, options):
        questions: list[str] = (
            options.get("questions")
            or getattr(settings, "WAGTAIL_RAG_TEST_QUESTIONS", None)
            or DEFAULT_TEST_QUESTIONS
        )
        search_only: bool = options.get("search_only", False)
        metadata_filter = self._parse_filter(options.get("filter"))

        self._section("Wagtail RAG — Smoke Test")
        self.stdout.write(f"Questions : {len(questions)}")
        self.stdout.write(
            f"Mode      : {'SEARCH ONLY' if search_only else 'Full RAG (search + LLM)'}"
        )
        if metadata_filter:
            self.stdout.write(f"Filter    : {json.dumps(metadata_filter)}")
        self._divider()

        # Initialise chatbot once
        self.stdout.write("Initializing chatbot...")
        try:
            chatbot = get_chatbot(
                **({"metadata_filter": metadata_filter} if metadata_filter else {})
            )
            self.stdout.write(self.style.SUCCESS("Chatbot ready.\n"))
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"FAILED to initialize chatbot: {e}"))
            self.stderr.write(traceback.format_exc())
            return

        passed = 0
        failed = 0

        for idx, question in enumerate(questions, 1):
            self.stdout.write(
                self.style.WARNING(f"[{idx}/{len(questions)}] {question}")
            )
            try:
                result = chatbot.query(question, search_only=search_only)
                sources = result.get("sources", [])
                answer = result.get("answer")

                # Pass criteria:
                #   search_only  -> at least 1 source retrieved
                #   full RAG     -> answer is a non-empty string AND at least 1 source
                if search_only:
                    ok = len(sources) > 0
                    verdict = "PASS" if ok else "FAIL (no sources retrieved)"
                else:
                    ok = bool(answer and answer.strip()) and len(sources) > 0
                    if not ok:
                        if not (answer and answer.strip()):
                            verdict = "FAIL (empty answer)"
                        else:
                            verdict = "FAIL (no sources retrieved)"
                    else:
                        verdict = "PASS"

                if ok:
                    passed += 1
                    self.stdout.write(self.style.SUCCESS(f"  Result : {verdict}"))
                else:
                    failed += 1
                    self.stdout.write(self.style.ERROR(f"  Result : {verdict}"))

                self.stdout.write(f"  Sources: {len(sources)}")

                if not search_only and answer:
                    preview = " ".join(answer.split())[:200]
                    self.stdout.write(
                        f"  Answer : {preview}{'...' if len(answer) > 200 else ''}"
                    )

                # Always show source titles
                for i, src in enumerate(sources[:3], 1):
                    meta = src.get("metadata", {}) or {}
                    title = meta.get("title") or meta.get("name") or "Unknown"
                    section = meta.get("section", "")
                    label = f"{title} [{section}]" if section else title
                    self.stdout.write(f"    src{i}: {label}")

            except Exception as e:
                failed += 1
                self.stdout.write(self.style.ERROR(f"  Result : ERROR — {e}"))
                self.stderr.write(traceback.format_exc())

            self.stdout.write("")

        # Summary
        self._divider()
        total = passed + failed
        summary = f"Results: {passed}/{total} passed"
        if failed == 0:
            self.stdout.write(self.style.SUCCESS(summary))
        else:
            self.stdout.write(self.style.ERROR(summary + f"  ({failed} failed)"))

    # ══════════════════════════════════════════════════════════════════════
    # SHARED HELPERS
    # ══════════════════════════════════════════════════════════════════════

    def _section(self, title: str):
        self.stdout.write("=" * 70)
        self.stdout.write(self.style.SUCCESS(title))
        self.stdout.write("=" * 70)

    def _divider(self):
        self.stdout.write("-" * 70)
        self.stdout.write("")

    def _parse_filter(self, filter_value: Optional[str]) -> Optional[dict[str, Any]]:
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
                    'Filter must be a JSON object, e.g. {"model": "BlogPage"}'
                )
            )
            raise SystemExit(1)
        return parsed

    @staticmethod
    def _build_query_kwargs(
        use_history: bool, session_id: Optional[str], search_only: bool = False
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"search_only": search_only}
        if use_history and getattr(settings, "WAGTAIL_RAG_ENABLE_CHAT_HISTORY", True):
            kwargs["session_id"] = session_id or uuid.uuid4().hex
        return kwargs

    def _display_sources(self, sources):
        if not sources:
            return
        self.stdout.write(self.style.WARNING(f"Sources ({len(sources)}):"))
        for i, source in enumerate(sources[:5], 1):
            meta = source.get("metadata", {}) or {}
            title = meta.get("title") or meta.get("name") or "Unknown"
            url = meta.get("url", "")
            model = meta.get("model", "")
            section = meta.get("section", "")
            content = source.get("content") or source.get("text") or ""
            preview = " ".join(content.split())[:150]

            line = f"  {i}. {title}"
            if section:
                line += f"  [{section}]"
            self.stdout.write(line)
            if model:
                self.stdout.write(f"     Model  : {model}")
            if url:
                self.stdout.write(f"     URL    : {url}")
            if preview:
                self.stdout.write(f"     Preview: {preview}...")
            self.stdout.write("")
