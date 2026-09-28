from contextlib import asynccontextmanager

import transformers
from fastapi import Depends, FastAPI
from loguru import logger
from sentence_transformers import SentenceTransformer

from app.apis import data_extraction, debug, risk_of_bias, study_screening
from app.funcs import title_abstract_screening
from app.funcs.threshold import read_thresholds
from app.resources import globals
from app.resources.security import api_key_auth


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


@app.get("/check/auth", dependencies=[Depends(api_key_auth)])
async def check_auth() -> dict[str, bool]:
    """Check application credentials without invoking any model provider."""
    res = {"authenticated": True}
    return res


@app.get(
    "/check/extraction",
    response_model=title_abstract_screening.ExtractionCheckResponse,
    dependencies=[Depends(api_key_auth)],
)
async def check_extraction() -> title_abstract_screening.ExtractionCheckResponse:
    """Check local extraction-provider setup without invoking the provider."""
    res = title_abstract_screening.check_extraction_configuration()
    return res


app.include_router(debug.router)
app.include_router(data_extraction.router)
app.include_router(risk_of_bias.router)
app.include_router(study_screening.router)
