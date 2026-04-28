"""Task 15.3 — Request Logging Middleware."""
import time
import logging
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("api")

class LoggingMiddleware(BaseHTTPMiddleware):
    """Logs every request with method, path, status, duration."""
    
    async def dispatch(self, request: Request, call_next):
        start = time.time()
        method = request.method
        path = request.url.path
        client = request.client.host
        
        response = await call_next(request)
        duration = time.time() - start
        
        logger.info(f"{client} | {method} {path} | {response.status_code} | {duration:.3f}s")
        return response

# Usage in FastAPI:
# app.add_middleware(LoggingMiddleware)
