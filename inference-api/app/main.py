from contextlib import asynccontextmanager

import transformers
from fastapi import FastAPI

from app.resources import globals
from app.apis import example


@asynccontextmanager
async def lifespan(app: FastAPI):
    globals.models["albert-imdb"] = {
        "tokenizer": transformers.AutoTokenizer.from_pretrained(
            globals.path_to_albert_imdb
        ),
        "model": transformers.AlbertForSequenceClassification.from_pretrained(
            globals.path_to_albert_imdb
        ),
    }
    print("api initialization done.")
    yield
    globals.models.clear()


app = FastAPI(title="inference-api", lifespan=lifespan)


@app.get("/")
async def root():
    return "hello world"


@app.get("/check")
async def check() -> bool:
    res = globals.path_to_albert_imdb.exists()
    return res


app.include_router(example.router)
