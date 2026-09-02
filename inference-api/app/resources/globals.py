from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from environs import Env

if TYPE_CHECKING:
    from pandas import DataFrame

env = Env()
env.read_env()

api_keys = [env("WCRF_API_KEY")]

openai_api_key = env("OPENAI_API_KEY", None)
openai_model = env("OPENAI_MODEL", "gpt-5.2")
openai_reasoning_effort = env("OPENAI_REASONING_EFFORT", "medium")
openai_study_screening_model = env("OPENAI_STUDY_SCREENING_MODEL", "gpt-5.6-terra")
openai_study_screening_reasoning_effort = env(
    "OPENAI_STUDY_SCREENING_REASONING_EFFORT", "medium"
)

models = {}

paths = {
    "albert_imdb": Path("models") / "albert-base-v2-imdb",
    "study_screening": Path("models") / "cup_multi_gpu_24_05_30",
    "thresholds": Path("data") / "summary_26_01_05.csv",
}

threshold = None
thresholds: DataFrame
