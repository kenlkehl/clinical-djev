import asyncio
import hmac
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from .backend import Backend
from .compiler import Compiler
from .config import Settings
from .errors import BackendUnavailable, DjevError
from .schemas import ExtractionRequest, ExtractionResponse
from .service import ExtractionService


class BodyLimitMiddleware:
    """Bound streamed bodies too, without echoing request data in errors."""

    def __init__(self, app, limit: int):
        self.app, self.limit = app, limit

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["method"] != "POST":
            return await self.app(scope, receive, send)
        parts, size = [], 0
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            size += len(message.get("body", b""))
            if size > self.limit:
                response = JSONResponse(
                    {"error": {"code": "body_too_large", "message": "Request body exceeds limit"}},
                    status_code=413,
                )
                return await response(scope, receive, send)
            parts.append(message.get("body", b""))
            if not message.get("more_body", False):
                break
        body = b"".join(parts)
        delivered = False

        async def bounded_receive():
            nonlocal delivered
            if not delivered:
                delivered = True
                return {"type": "http.request", "body": body, "more_body": False}
            return await receive()

        await self.app(scope, bounded_receive, send)


def create_app(settings: Settings | None = None, service: ExtractionService | None = None):
    settings = settings or Settings()
    active = 0

    @asynccontextmanager
    async def lifespan(app):
        if service is not None:
            app.state.service = service
            yield
            return
        os.environ["HF_HOME"] = settings.hf_home
        from transformers import AutoTokenizer

        backend = Backend(settings)
        try:
            limit = await backend.check()
            tokenizer = await asyncio.to_thread(
                AutoTokenizer.from_pretrained,
                settings.model,
                revision=settings.revision,
                cache_dir=str(Path(settings.hf_home) / "hub"),
                trust_remote_code=False,
            )
            compiler = Compiler(tokenizer, settings.canvas_length, limit)
            app.state.service = ExtractionService(settings, compiler, backend)
            # Probe real structured logprobs before announcing readiness.
            await app.state.service.extract(
                ExtractionRequest(
                    state="Synthetic setup check: the note explicitly says fever is present.",
                    questions={"check": {"type": "noul", "instructions": "Is fever documented?"}},
                    seed=0,
                )
            )
            yield
        finally:
            await backend.close()

    app = FastAPI(title="clinical-djev", version="0.1.0", lifespan=lifespan)
    app.add_middleware(BodyLimitMiddleware, limit=settings.max_body_bytes)
    bearer = HTTPBearer(auto_error=False)

    async def authenticate(
        credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer)],
    ):
        if settings.api_key and (
            credentials is None
            or not hmac.compare_digest(credentials.credentials, settings.api_key.get_secret_value())
        ):
            raise HTTPException(401, "Invalid API key", headers={"WWW-Authenticate": "Bearer"})

    @app.exception_handler(DjevError)
    async def known_error(request, exc):
        return JSONResponse(
            {"error": {"code": exc.code, "message": str(exc)}}, status_code=exc.status
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error(request, exc):
        # Pydantic normally includes the rejected value (possibly a whole patient
        # record). Report locations/types only, never input or exception context.
        return JSONResponse(
            {
                "error": {
                    "code": "invalid_request",
                    "message": "Request validation failed",
                    "fields": [{"loc": e["loc"], "type": e["type"]} for e in exc.errors()],
                }
            },
            status_code=422,
        )

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/ready", dependencies=[Depends(authenticate)])
    async def ready(request: Request):
        try:
            await request.app.state.service.backend.check()
        except BackendUnavailable:
            raise
        return {"status": "ready", "model": settings.served_model}

    @app.post(
        "/v1/systemone", response_model=ExtractionResponse, dependencies=[Depends(authenticate)]
    )
    async def extract(body: ExtractionRequest, request: Request):
        nonlocal active
        if active >= settings.max_active_requests:
            return JSONResponse(
                {"error": {"code": "busy", "message": "Request capacity reached; retry later"}},
                status_code=429,
                headers={"Retry-After": "1"},
            )
        active += 1
        try:
            return await request.app.state.service.extract(body)
        finally:
            active -= 1

    return app
