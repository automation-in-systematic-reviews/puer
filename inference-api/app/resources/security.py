from fastapi import Depends, HTTPException, status
from fastapi.security import APIKeyHeader

from app.resources import globals

header_scheme = APIKeyHeader(name="X-API-Key")


def api_key_auth(api_key: str = Depends(header_scheme)):
    if api_key not in globals.api_keys:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="The API key is not correct",
        )
