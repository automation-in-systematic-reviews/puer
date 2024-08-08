import torch
from fastapi import APIRouter
from pydantic import BaseModel

from app.resources import globals


class PayloadModel(BaseModel):
    text: str


router = APIRouter()


@router.post("/example/encode")
async def post_encode(payload: PayloadModel):
    # example taken from
    # https://huggingface.co/docs/transformers/model_doc/albert#transformers.AlbertForSequenceClassification
    tokenizer = globals.models["albert-imdb"]["tokenizer"]
    model = globals.models["albert-imdb"]["model"]

    text = payload.text
    print(text)

    inputs = tokenizer(text, return_tensors="pt")
    print(inputs)

    with torch.no_grad():
        logits = model(**inputs).logits
        predicted_class_id = logits.argmax().item()
        res = model.config.id2label[predicted_class_id]
        print(res)

    return res
