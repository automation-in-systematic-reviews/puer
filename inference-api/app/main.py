from contextlib import asynccontextmanager

import transformers
from fastapi import FastAPI
from loguru import logger
from sentence_transformers import SentenceTransformer

from app.apis import data_extraction, debug, risk_of_bias, study_screening
from app.funcs.threshold import read_thresholds
from app.resources import globals


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("api config info")
    logger.info("api init: init models")
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
    logger.info("api initialization done.")
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
app.include_router(data_extraction.router)
app.include_router(risk_of_bias.router)
app.include_router(study_screening.router)
