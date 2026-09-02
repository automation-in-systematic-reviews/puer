from typing import Optional

import numpy as np
from fastapi import APIRouter, Depends, HTTPException
from numpy.linalg import norm
from pydantic import BaseModel

from app.funcs.threshold import select_threshold, threshold_to_binary_labels
from app.funcs.title_abstract_screening import (
    ScreeningServiceError,
    StudyScreeningPredictionRequest,
    StudyScreeningPredictionResponse,
    predict_study_screening,
)
from app.resources import globals
from app.resources.security import api_key_auth


class PayloadModel(BaseModel):
    review_topic: str
    criteria: Optional[str] = None
    study_title: str
    study_abstract: str
    strategy: str


router = APIRouter()


@router.post(
    "/study_screening/predict/",
    dependencies=[Depends(api_key_auth)],
    response_model=StudyScreeningPredictionResponse,
)
async def post_predict(
    request: StudyScreeningPredictionRequest,
) -> StudyScreeningPredictionResponse:
    """Predict conservative screening from a title, abstract, and context."""
    try:
        res = await predict_study_screening(request)
    except ScreeningServiceError as exc:
        raise HTTPException(
            status_code=503,
            detail=exc.detail.model_dump(mode="json"),
        ) from exc
    return res


# @router.post("/study_screening/encode")
@router.post("/study_screening/encode", dependencies=[Depends(api_key_auth)])
async def post_encode(payload: PayloadModel):

    model = globals.models["study_screening"]["model"]

    study = "Title: " + payload.study_title + ". Abstract: " + payload.study_abstract

    if payload.criteria:
        query = "Query: " + payload.review_topic + ". Criteria: " + payload.criteria
    else:
        query = "Query: " + payload.review_topic + "."
    q_e = model.encode(query, convert_to_numpy=True)
    # q_e = q_e.reshape(1, 768)
    s_e = model.encode(study, convert_to_numpy=True)
    cosine = np.dot(q_e, s_e) / (norm(q_e) * norm(s_e))
    threshold = select_threshold(
        globals.thresholds, payload.review_topic, payload.strategy
    )
    decision = threshold_to_binary_labels([cosine], threshold)
    # return 1.0, 'included'
    return cosine.item(), decision[0]
