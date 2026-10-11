import hmac
from typing import Annotated

from fastapi import Header, HTTPException, status

from mcclub_rag.config import Settings, get_settings
from mcclub_rag.schemas import Role


def _matches(actual: str | None, expected: str | None) -> bool:
    return bool(actual and expected and hmac.compare_digest(actual, expected))


def get_role(authorization: Annotated[str | None, Header()] = None) -> Role:
    settings = get_settings()
    if authorization is None:
        return "public"

    scheme, _, token = authorization.partition(" ")
    if scheme != "Bearer" or not _matches(token, settings.member_token):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid member token.",
        )
    return "member"


def require_admin(x_admin_key: Annotated[str | None, Header()] = None) -> None:
    settings: Settings = get_settings()
    if not _matches(x_admin_key, settings.admin_api_key):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access is required.",
        )
