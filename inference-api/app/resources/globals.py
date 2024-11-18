from pathlib import Path

from environs import Env

env = Env()
env.read_env()

api_keys = [env("WCRF_API_KEY")]

models = {}

paths = {
    "albert_imdb": Path("models") / "albert-base-v2-imdb",
    "study_screening": Path("models") / "cup_multi_gpu_24_05_30",
    "thresholds": Path("data") / "summary_24_08_08.csv",
}

threshold = None
