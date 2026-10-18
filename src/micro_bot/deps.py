import hmac
from typing import Annotated

from fastapi import Header, HTTPException, Request, status

from micro_bot.config import Settings, get_settings


def _matches(actual: str | None, expected: str | None) -> bool:
    return bool(actual and expected and hmac.compare_digest(actual, expected))


def _settings(request: Request) -> Settings:
    """The app's own settings (``create_app(settings=...)``), not the process-wide ones (I20).

    A bare FastAPI app without the application state falls back to the global settings.
    """
    state = getattr(request.app.state, "micro_bot_state", None)
    return state.settings if state is not None else get_settings()


def require_api_key(
    request: Request, authorization: Annotated[str | None, Header()] = None
) -> None:
    scheme, _, token = (authorization or "").partition(" ")
    if scheme != "Bearer" or not _matches(token, _settings(request).api_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="A valid API key is required: Authorization: Bearer <API_KEY>.",
        )


def require_admin(request: Request, x_admin_key: Annotated[str | None, Header()] = None) -> None:
    if not _matches(x_admin_key, _settings(request).admin_api_key):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access is required.",
        )
