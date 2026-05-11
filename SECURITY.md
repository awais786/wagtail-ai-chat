# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 0.1.x   | Yes       |

## Reporting a Vulnerability

Please **do not** open a public GitHub issue for security vulnerabilities.

Report privately via GitHub's security advisory form:
https://github.com/awais786/wagtail-ai-chat/security/advisories/new

Include:
- Description of the vulnerability and its impact
- Steps to reproduce
- Affected versions
- Any suggested fixes

**Response timeline:**
- Acknowledgement within 48 hours
- Assessment and severity triage within 7 days
- Patch and coordinated disclosure within 90 days for confirmed issues

## Dependency Security

### Known CVEs addressed (v0.1.2+)

The following minimum versions are enforced in `install_requires` to cover all
known CVEs across the LangChain ecosystem. `langchain-community>=0.3.27` is the
effective floor — it subsumes the SSRF (>=0.0.28) and pickle deserialization
(>=0.2.4) fixes as well as the XXE vulnerability:

| Package | Minimum | CVEs covered |
|---|---|---|
| `langchain` | `>=0.2.0` | Required by langchain-community ≥0.3.x |
| `langchain-community` | `>=0.3.27` | XXE attacks, SSRF in RequestsToolkit, pickle deserialization of untrusted data |
| `langchain-text-splitters` | `>=0.3.9` | XML injection |

### Keeping dependencies current

```bash
pip install --upgrade wagtail-rag
pip list --outdated | grep langchain
```

## Built-in Security Controls

### Input validation

| Control | Where | Default |
|---|---|---|
| Maximum question length | `conf.api.max_question_length` | 150 characters |
| Maximum POST body size | `conf.api.max_request_body_size` | 1 MB |
| GET `?q=` parameter length | `views.py` | Same as `max_question_length` |
| Session ID format | `views.py` (`_SESSION_ID_RE`) | `[a-zA-Z0-9_\-]{1,128}` |
| LLM kwargs whitelist | `views.py` (`_ALLOWED_LLM_KWARGS`) | temperature, max_tokens, top_p, top_k, timeout |

### Suspicious-pattern logging

`views.py` scans every question for common prompt-injection patterns
(override-instructions, jailbreak keywords, marker injection). Matches are
logged at `WARNING` level so operators can monitor them. The request is not
blocked at the view layer — use `prompt_guard.py` for blocking (see below).

### Prompt injection guard

`wagtail_rag/prompt_guard.py` provides a pre-LLM filter via `check_question()`.
When `result.blocked` is `True` the question is never forwarded to the LLM.

To plug in a custom backend:

```python
WAGTAIL_RAG = {
    "api": {
        "prompt_guard_backend": "myapp.guards.my_guard",
    },
}
```

The callable must have the signature `def my_guard(question: str) -> GuardResult`.

### Rate limiting

Built-in sliding-window per-IP rate limiter (in-memory):

```python
WAGTAIL_RAG = {
    "api": {
        "rate_limit_per_minute": 20,  # 0 = disabled (default)
    },
}
```

Note: the in-memory store is process-local. Use a shared store (Redis + django-ratelimit) for multi-process deployments.

### CSRF protection

The chat endpoint (`/api/rag/chat/`) enforces Django CSRF on POST requests via
`@ensure_csrf_cookie`. External clients must first GET any page (which sets the
`csrftoken` cookie) and mirror the token in subsequent POST requests as the
`X-CSRFToken` header.

### LLM-side injection defence

`LLMGenerator` uses a system prompt that explicitly instructs the model to:
- Answer only from the provided context
- Disregard any instructions embedded in the user question

This is complementary to the pre-filter guard — both layers should remain active.

## Recommended Production Settings

```python
# settings/production.py

# Django security headers
SECURE_HSTS_SECONDS = 31536000
SECURE_HSTS_INCLUDE_SUBDOMAINS = True
SECURE_HSTS_PRELOAD = True
SECURE_SSL_REDIRECT = True
SESSION_COOKIE_SECURE = True
CSRF_COOKIE_SECURE = True
CSRF_COOKIE_HTTPONLY = True
X_FRAME_OPTIONS = "DENY"
SECURE_CONTENT_TYPE_NOSNIFF = True

# Wagtail RAG
WAGTAIL_RAG = {
    "api": {
        "max_question_length": 150,
        "max_request_body_size": 1048576,
        "rate_limit_per_minute": 20,
    },
}
```

> **Note:** `SECURE_BROWSER_XSS_FILTER` was removed in Django 5.0 and should
> not be used. Modern browsers handle XSS protection natively.

## Scope

The following are **out of scope** for this security policy:

- Vulnerabilities in the host Wagtail project (report to the Wagtail project)
- Vulnerabilities in LLM providers (OpenAI, Anthropic, Ollama)
- Social engineering attacks against LLM output (hallucination, bias)
- Denial-of-service via expensive LLM queries (mitigate with rate limiting)
