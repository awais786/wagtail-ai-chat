# Makefile for Wagtail RAG Chatbot

.PHONY: help index index-reset index-rebuild chat test-rag test-rag-search test test-cov lint format clean runserver

# Default target
help:
	@echo "Wagtail RAG — available targets:"
	@echo ""
	@echo "  Index management:"
	@echo "    index          Build the RAG index"
	@echo "    index-reset    Clear the index without re-indexing"
	@echo "    index-rebuild  Clear then rebuild the index"
	@echo ""
	@echo "  Chat / testing:"
	@echo "    chat           Start interactive chat session"
	@echo "    test-rag       Run smoke tests against the live pipeline"
	@echo ""
	@echo "  Development:"
	@echo "    test           Run unit test suite"
	@echo "    lint           Check formatting and style"
	@echo "    format         Auto-format with black"
	@echo "    clean          Remove Python cache files"
	@echo "    runserver      Start Django development server"

# ── Index management ─────────────────────────────────────────────────────────

index:
	python manage.py rag index

index-reset:
	python manage.py rag index --reset-only

index-rebuild: index-reset index
	@echo "Index rebuilt."

# ── Chat / pipeline testing ───────────────────────────────────────────────────

chat:
	python manage.py rag chat

# Smoke-test the full pipeline (retrieval + LLM) with built-in questions
test-rag:
	python manage.py rag test

# Smoke-test retrieval only (faster, no LLM call)
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
