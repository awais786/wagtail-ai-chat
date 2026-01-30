# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Security Considerations

### API Security

#### CSRF Exemption
The chat API endpoint (`/api/rag/chat/`) is **CSRF-exempt by design** to support:
- External clients and scripts
- Non-Django frontends
- API integrations

**Important**: If you expose this endpoint publicly, you **must** implement protection at the network or gateway level:

1. **Authentication**: Add authentication middleware
2. **Rate Limiting**: Prevent abuse with rate limiting (e.g., django-ratelimit)
3. **IP Allowlisting**: Restrict access to known IP addresses
4. **API Keys**: Require API key authentication
5. **Request Size Limits**: Already enforced (1MB default, configurable via `WAGTAIL_RAG_MAX_REQUEST_BODY_SIZE`)

Example using django-ratelimit:
```python
from django_ratelimit.decorators import ratelimit

@ratelimit(key='ip', rate='10/m', method='POST')
def rag_chat_api(request):
    # ... existing code
```

#### Input Validation
- User questions are validated for presence but not sanitized
- LLM providers handle prompt injection protection internally
- Consider adding additional input validation for your use case

### Data Security

#### FAISS Index Loading
The FAISS backend uses `allow_dangerous_deserialization=True` when loading indexes. This is required for FAISS to work but has security implications:

**Risks**:
- Pickle deserialization can execute arbitrary code
- Malicious index files could compromise your system

**Mitigations**:
1. Only load FAISS indexes from trusted sources
2. Restrict file system permissions on index directories
3. Do not allow user uploads of FAISS indexes
4. Consider using ChromaDB backend for untrusted environments

#### Secrets Management
API keys for LLM providers should be:
- Stored in environment variables or secure secret management systems
- Never committed to version control
- Rotated regularly
- Have minimal required permissions

Example using environment variables:
```python
import os
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
ANTHROPIC_API_KEY = os.environ.get('ANTHROPIC_API_KEY')
```

### Dependency Security

We regularly update dependencies to address security vulnerabilities. Current security updates:

- **langchain-community >= 0.3.27**: Fixes XXE (XML External Entity) attacks
- **langchain-community >= 0.0.28**: Fixes SSRF vulnerability in RequestsToolkit
- **langchain-community >= 0.2.4**: Fixes pickle deserialization of untrusted data
- **langchain-text-splitters >= 0.3.9**: Fixes XXE attacks via unsafe XSLT parsing

Keep your dependencies updated:
```bash
pip install --upgrade wagtail-rag
```

### Content Security

#### HTML Content
- Page content is extracted using BeautifulSoup
- HTML tags are stripped during indexing
- XSS risks are minimal as content is not rendered directly in responses

#### User Content
- User questions are logged for debugging (check your logging configuration)
- Ensure logs are properly secured and comply with privacy regulations
- Consider implementing PII (Personally Identifiable Information) filtering

### Network Security

#### HTTPS
Always use HTTPS in production to protect:
- API keys in transit
- User queries
- LLM responses

#### Firewall Rules
If running LLM providers locally (e.g., Ollama):
- Restrict network access to localhost
- Use firewall rules to prevent external access
- Monitor for unusual network activity

### Monitoring and Logging

#### Security Logging
The application logs:
- API requests and responses
- Errors and exceptions
- Provider initialization

**Recommendations**:
1. Review logs regularly for suspicious activity
2. Implement log rotation and retention policies
3. Use centralized logging for production systems
4. Monitor for:
   - Unusual query patterns
   - High error rates
   - Unauthorized access attempts

#### Error Messages
Error messages returned to users are intentionally generic to avoid information disclosure. Detailed errors are logged server-side.

## Reporting a Vulnerability

If you discover a security vulnerability, please report it by:

1. **Do NOT** open a public GitHub issue
2. Email the maintainers directly at [security contact - to be added]
3. Include:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if available)

We will:
- Acknowledge receipt within 48 hours
- Provide an initial assessment within 1 week
- Work on a fix and coordinate disclosure timeline
- Credit you in the security advisory (if desired)

## Security Best Practices for Deployment

### Production Checklist

- [ ] Update all dependencies to latest secure versions
- [ ] Enable HTTPS for all endpoints
- [ ] Implement authentication on API endpoints
- [ ] Add rate limiting
- [ ] Configure proper CORS headers
- [ ] Review and secure logging configuration
- [ ] Restrict FAISS index file permissions
- [ ] Use environment variables for API keys
- [ ] Enable Django security middleware
- [ ] Set `DEBUG = False`
- [ ] Configure `ALLOWED_HOSTS` properly
- [ ] Regular security audits and dependency updates

### Django Security Settings

```python
# Recommended Django security settings
SECURE_SSL_REDIRECT = True
SESSION_COOKIE_SECURE = True
CSRF_COOKIE_SECURE = True
SECURE_BROWSER_XSS_FILTER = True
SECURE_CONTENT_TYPE_NOSNIFF = True
X_FRAME_OPTIONS = 'DENY'
SECURE_HSTS_SECONDS = 31536000
SECURE_HSTS_INCLUDE_SUBDOMAINS = True
SECURE_HSTS_PRELOAD = True
```

## Security Updates

Subscribe to security advisories:
- GitHub Security Advisories for this repository
- LangChain security updates
- Django security releases
- Wagtail security releases

## Additional Resources

- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [Django Security](https://docs.djangoproject.com/en/stable/topics/security/)
- [Wagtail Security](https://docs.wagtail.org/en/stable/advanced_topics/security.html)
- [LangChain Security Best Practices](https://python.langchain.com/docs/security)
