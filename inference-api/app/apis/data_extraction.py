from fastapi import APIRouter, Depends, HTTPException

from app.funcs.title_abstract_screening import (
    ScreeningServiceError,
    TitleAbstractExtractionResponse,
    TitleAbstractRequest,
    extract_title_abstract,
)
from app.resources.security import api_key_auth

router = APIRouter()


@router.post(
    "/data_extraction/title_abstract/",
    dependencies=[Depends(api_key_auth)],
    response_model=TitleAbstractExtractionResponse,
)
async def post_title_abstract(
    request: TitleAbstractRequest,
) -> TitleAbstractExtractionResponse:
    """Extract strict study characteristics from a title and abstract."""
    try:
        res = await extract_title_abstract(request)
    except ScreeningServiceError as exc:
        raise HTTPException(
            status_code=503,
            detail=exc.detail.model_dump(mode="json"),
        ) from exc
    return res
