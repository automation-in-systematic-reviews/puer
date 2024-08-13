import torch
from fastapi import APIRouter
from pydantic import BaseModel
import numpy as np
from numpy.linalg import norm

from app.resources import globals


class PayloadModel(BaseModel):
    review_topic: str
    critiera: str
    study_title: str
    study_abstract: str


router = APIRouter()


@router.post("/study_screening/encode")
async def post_encode(payload: PayloadModel):
    # example taken from
    # https://huggingface.co/docs/transformers/model_doc/albert#transformers.AlbertForSequenceClassification
    model = globals.models["study_screening"]["model"]

    study = 'Title: ' + payload.study_title + '. Abstract: ' + payload.study_abstract

    if payload.critiera:
        query = 'Topic: ' + payload.review_topic + '. Criteria: ' + payload.critiera
    else:
        query = 'Topic: ' + payload.review_topic + '.'
    print(query)
    print(study)

    q_e = model.encode(query, convert_to_numpy=True)
    # q_e = q_e.reshape(1, 768)
    s_e = model.encode(study, convert_to_numpy=True)
    cosine = np.dot(q_e, s_e) / (norm(q_e) * norm(s_e))
    print("Cosine Similarity:\n", cosine)


    return cosine.item()
