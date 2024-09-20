from contextlib import asynccontextmanager

import transformers
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI

from app.funcs.threshold import read_thresholds
from app.resources import globals
from app.apis import debug, study_screening


@asynccontextmanager
async def lifespan(app: FastAPI):
    globals.models["albert-imdb"] = {
        "tokenizer": transformers.AutoTokenizer.from_pretrained(
            globals.paths["albert_imdb"]
        ),
        "model": transformers.AlbertForSequenceClassification.from_pretrained(
            globals.paths["albert_imdb"]
        ),
    }
    globals.models["study_screening"] = {
        "model": SentenceTransformer(str(globals.paths["study_screening"])),
    }
    globals.thresholds = read_thresholds(globals.paths["thresholds"])
    print("api initialization done.")
    yield
    globals.models.clear()


app = FastAPI(title="inference-api", lifespan=lifespan)


@app.get("/")
async def root():
    return "hello world"


@app.get("/check")
async def check() -> bool:
    path_exist_list = []
    for k, v in globals.paths.items():
        path_exist = v.exists()
        print(f"{k}, {path_exist}")
        path_exist_list.append(path_exist)
    res = sum(path_exist_list) == len(path_exist_list)

    return res


app.include_router(debug.router)
app.include_router(study_screening.router)
