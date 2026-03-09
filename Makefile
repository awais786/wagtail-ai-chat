# Makefile for Wagtail RAG Chatbot

.PHONY: help install install-dev install-local install-openai install-all \
        index index-reset index-rebuild \
        chat test-rag test-rag-search \
        test test-cov lint format clean runserver

# Default target
help:
	@echo "Wagtail RAG — available targets:"
	@echo ""
	@echo "  Setup:"
	@echo "    install          Install package (no extras)"
	@echo "    install-dev      Install with test/dev dependencies"
	@echo "    install-local    Install with local stack (FAISS + Sentence Transformers + Ollama)"
	@echo "    install-openai   Install with OpenAI provider"
	@echo "    install-all      Install all providers"
	@echo ""
	@echo "  Index management:"
	@echo "    index            Build the RAG index"
	@echo "    index-reset      Clear the index without re-indexing"
	@echo "    index-rebuild    Clear then rebuild the index"
	@echo ""
	@echo "  Chat / pipeline testing:"
	@echo "    chat             Start interactive chat session"
	@echo "    test-rag         Smoke-test the full pipeline (search + LLM)"
	@echo "    test-rag-search  Smoke-test retrieval only (no LLM call)"
	@echo ""
	@echo "  Development:"
	@echo "    test             Run unit test suite"
	@echo "    test-cov         Run tests with coverage report"
	@echo "    lint             Check formatting and style"
	@echo "    format           Auto-format with black"
	@echo "    clean            Remove Python cache and build files"
	@echo "    runserver        Start Django development server"

# ── Setup ─────────────────────────────────────────────────────────────────────

install:
	pip install -e .

install-dev:
	pip install -e ".[test,dev]"

install-local:
	pip install -e ".[local,test,dev]"

install-openai:
	pip install -e ".[openai,test,dev]"

install-all:
	pip install -e ".[all,test,dev]"

# ── Index management ──────────────────────────────────────────────────────────

index:
	python manage.py rag index

index-reset:
	python manage.py rag index --clear

index-rebuild: index-reset index
	@echo "Index rebuilt."

# ── Chat / pipeline testing ───────────────────────────────────────────────────

chat:
	python manage.py rag chat

test-rag:
	python manage.py rag test

test-rag-search:
	python manage.py rag test --search-only

# ── Development ───────────────────────────────────────────────────────────────

test:
	pytest wagtail_rag/tests/ -v

test-cov:
	pytest wagtail_rag/tests/ --cov=wagtail_rag --cov-report=term-missing

lint:
	black --check wagtail_rag
	flake8 wagtail_rag --count --select=E9,F63,F7,F82 --show-source --statistics
	flake8 wagtail_rag --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics

format:
	black wagtail_rag

runserver:
	python manage.py runserver

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -f .coverage coverage.xml
