from typing import List, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from app.funcs.risk_of_bias import (
    RiskOfBiasAssessment,
    RiskOfBiasServiceError,
    UnknownRiskOfBiasDomainError,
    assess_pdf_document,
)
from app.resources.security import api_key_auth


router = APIRouter()


def _is_pdf_upload(pdf_bytes: bytes) -> bool:
    return pdf_bytes.startswith(b"%PDF-")


@router.post(
    "/risk_of_bias/assess",
    dependencies=[Depends(api_key_auth)],
    response_model=RiskOfBiasAssessment,
)
async def post_assess_pdf(
    study_id: str = Form(...),
    file: UploadFile = File(...),
    exposure_timepoint: Optional[str] = Form(None),
    outcome_timepoint: Optional[str] = Form(None),
    domains: Optional[List[str]] = Form(None),
) -> RiskOfBiasAssessment:
    pdf_bytes = await file.read()
    if not pdf_bytes:
        raise HTTPException(status_code=400, detail="Uploaded PDF is empty")
    if not _is_pdf_upload(pdf_bytes=pdf_bytes):
        raise HTTPException(
            status_code=400,
            detail="Only PDF uploads are supported",
        )
    try:
        res = await assess_pdf_document(
            filename=file.filename or "study.pdf",
            pdf_bytes=pdf_bytes,
            study_id=study_id,
            exposure_timepoint=exposure_timepoint,
            outcome_timepoint=outcome_timepoint,
            domains=domains,
        )
    except UnknownRiskOfBiasDomainError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RiskOfBiasServiceError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return res
