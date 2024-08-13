from typing import List, Optional

import torch
from fastapi import APIRouter
from pydantic import BaseModel, field_validator

from app.resources import globals


class ExampleInputRecord(BaseModel):
    text: str


class ExampleInputBatch(BaseModel):
    example_record: List[ExampleInputRecord]
    param1: Optional[str]
    param2: Optional[str]

    @field_validator("example_record")
    def num_example_record_limit(cls, v):
        # https://docs.pydantic.dev/latest/concepts/validators/#field-validators
        num_limit = 2
        print(len(v))
        if len(v) > num_limit:
            raise ValueError(f"number of records must not exceed {num_limit}")
        return v


ExampleOutputBatch = List[str]


def example_encode_inference(text: str, tokenizer, model) -> str:
    inputs = tokenizer(text, return_tensors="pt")
    print(inputs)

    with torch.no_grad():
        logits = model(**inputs).logits
        predicted_class_id = logits.argmax().item()
        res = model.config.id2label[predicted_class_id]
        print(res)

    return res


router = APIRouter()


@router.post("/example/encode")
async def post_encode(payload: ExampleInputBatch) -> ExampleOutputBatch:
    # example taken from
    # https://huggingface.co/docs/transformers/model_doc/albert#transformers.AlbertForSequenceClassification
    tokenizer = globals.models["albert-imdb"]["tokenizer"]
    model = globals.models["albert-imdb"]["model"]
    print(payload)

    res = []
    for record in payload.example_record:
        text = record.text
        print(text)
        inference_res = example_encode_inference(
            text=text, tokenizer=tokenizer, model=model
        )
        res.append(inference_res)

    return res
