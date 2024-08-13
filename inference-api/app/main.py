from contextlib import asynccontextmanager

import transformers
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI

from app.resources import globals
from app.apis import study_screening


@asynccontextmanager
async def lifespan(app: FastAPI):
    globals.models["study_screening"] = {
        # "tokenizer": transformers.AutoTokenizer.from_pretrained(
        #     globals.path_to_study_screening
        # ),
        "model": SentenceTransformer(
            str(globals.path_to_study_screening)
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
<<<<<<< HEAD
async def check():
    res = globals.path_to_study_screening.exists()
=======
async def check() -> bool:
    res = globals.path_to_albert_imdb.exists()
>>>>>>> 4551c53a97bb514756bf44dfeda95a54359adde1
    return res


app.include_router(study_screening.router)
