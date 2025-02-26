from typing import Optional

import numpy as np
from fastapi import APIRouter, Depends
from numpy.linalg import norm
from pydantic import BaseModel

from app.funcs.threshold import select_threshold, threshold_to_binary_labels
from app.resources import globals
from app.resources.security import api_key_auth


class PayloadModel(BaseModel):
    review_topic: str
    criteria: Optional[str] = None
    study_title: str
    study_abstract: str
    strategy: str


router = APIRouter()


# @router.post("/study_screening/encode")
@router.post("/study_screening/encode", dependencies=[Depends(api_key_auth)])
async def post_encode(payload: PayloadModel):

    model = globals.models["study_screening"]["model"]

    study = "Title: " + payload.study_title + ". Abstract: " + payload.study_abstract

    if payload.criteria:
        query = "Query: " + payload.review_topic + ". Criteria: " + payload.criteria
    else:
        query = "Query: " + payload.review_topic + "."
    print(query)
    print(study)

    q_e = model.encode(query, convert_to_numpy=True)
    # q_e = q_e.reshape(1, 768)
    s_e = model.encode(study, convert_to_numpy=True)
    cosine = np.dot(q_e, s_e) / (norm(q_e) * norm(s_e))
    print("Cosine Similarity:", cosine)
    threshold = select_threshold(
        globals.thresholds, payload.review_topic, payload.strategy
    )
    decision = threshold_to_binary_labels([cosine], threshold)
    print(decision)

    # return 1.0, 'included'
    return cosine.item(), decision[0]
