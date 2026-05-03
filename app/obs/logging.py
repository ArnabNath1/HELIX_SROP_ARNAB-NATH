"""
Structured logging setup.

All log lines must include session_id, trace_id, user_id when available.
Use structlog's context vars for request-scoped fields.
"""
import logging
import sys

import structlog


import re

def redact_pii(_, __, event_dict):
    """Simple processor to redact common PII patterns."""
    # Redact email addresses
    email_re = r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"
    # Redact potential Google API keys (AIza...)
    api_key_re = r"AIza[0-9A-Za-z-_]{35}"
    
    for key, value in event_dict.items():
        if isinstance(value, str):
            value = re.sub(email_re, "[REDACTED_EMAIL]", value)
            value = re.sub(api_key_re, "[REDACTED_KEY]", value)
            event_dict[key] = value
    return event_dict


def configure_logging() -> None:
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            redact_pii,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(file=sys.stdout),
    )


# Usage in request handlers:
#   import structlog
#   log = structlog.get_logger()
#   structlog.contextvars.bind_contextvars(session_id=session_id, trace_id=trace_id)
#   log.info("pipeline_started", user_message_len=len(message))
