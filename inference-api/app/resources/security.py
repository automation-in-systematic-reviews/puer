from fastapi import Depends, HTTPException, Request, status
from fastapi.security import APIKeyHeader

from app.resources import globals

header_scheme = APIKeyHeader(name="X-API-Key", auto_error=False)


def api_key_auth(
    request: Request,
    api_key: str | None = Depends(header_scheme),
) -> None:
    """Require a supplied valid API key with stable absent-key behavior."""
    if "X-API-Key" not in request.headers:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authenticated",
        )
    if api_key not in globals.api_keys:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="The API key is not correct",
        )
