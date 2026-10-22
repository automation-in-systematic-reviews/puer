import sys
from collections.abc import Generator
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
from environs import Env
from fastapi.testclient import TestClient
from pandas import DataFrame


def pytest_configure(config: pytest.Config) -> None:
    """Isolate tests from local secret files and live provider configuration."""
    monkeypatch = pytest.MonkeyPatch()
    config.add_cleanup(monkeypatch.undo)
    monkeypatch.setattr(Env, "read_env", lambda *args, **kwargs: None)
    monkeypatch.setenv("WCRF_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_API_KEY", "")
    for name in (
        "PUER_LOAD_BERT_MODELS",
        "OPENAI_MODEL",
        "OPENAI_REASONING_EFFORT",
        "OPENAI_STUDY_SCREENING_MODEL",
        "OPENAI_STUDY_SCREENING_REASONING_EFFORT",
    ):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture(autouse=True)
def mock_bert_assets(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """Replace local ML libraries and asset reads without replacing lifespan."""
    from app import main

    tokenizer = MagicMock()
    classifier = MagicMock()
    classifier.return_value.logits.argmax.return_value.item.return_value = 0
    classifier.config.id2label = {0: "negative"}
    screening_model = MagicMock()
    screening_model.encode.return_value = np.array([0.5, 0.5, 0.5])
    transformers = MagicMock()
    transformers.AutoTokenizer.from_pretrained.return_value = tokenizer
    transformers.AlbertForSequenceClassification.from_pretrained.return_value = (
        classifier
    )
    sentence_transformers = MagicMock()
    sentence_transformers.SentenceTransformer.return_value = screening_model
    thresholds = DataFrame(
        {"topic": ["cancer"], "strategy": ["best_recall"], "threshold": [0.5]}
    )
    read_thresholds = MagicMock(return_value=thresholds)
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setitem(sys.modules, "sentence_transformers", sentence_transformers)
    monkeypatch.setitem(sys.modules, "torch", MagicMock())
    monkeypatch.setattr(main, "read_thresholds", read_thresholds)
    res = SimpleNamespace(
        transformers=transformers,
        sentence_transformers=sentence_transformers,
        read_thresholds=read_thresholds,
        tokenizer=tokenizer,
        classifier=classifier,
        screening_model=screening_model,
        thresholds=thresholds,
    )
    return res


@pytest.fixture
def no_bert_client(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mock_bert_assets: SimpleNamespace,
) -> Generator[TestClient, None, None]:
    """Run the production lifespan with local assets disabled and absent."""
    from app import main

    monkeypatch.setattr(main.globals, "load_bert_models", False)
    monkeypatch.setattr(
        main.globals,
        "paths",
        {name: tmp_path / name for name in main.globals.paths},
    )
    with TestClient(main.app) as client:
        assert main.globals.models == {}
        assert main.globals.thresholds is None
        yield client
    assets = mock_bert_assets
    assets.transformers.AutoTokenizer.from_pretrained.assert_not_called()
    assets.transformers.AlbertForSequenceClassification.from_pretrained.assert_not_called()
    assets.sentence_transformers.SentenceTransformer.assert_not_called()
    assets.read_thresholds.assert_not_called()


@pytest.fixture
def synthetic_extraction_provider_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Provide deterministic local provider settings for diagnostic tests."""
    from app.resources import globals

    monkeypatch.setattr(
        globals,
        "openai_api_key",
        "synthetic-provider-key-for-tests",
    )
    monkeypatch.setattr(
        globals,
        "openai_study_screening_model",
        "synthetic-extraction-model",
    )
    monkeypatch.setattr(
        globals,
        "openai_study_screening_reasoning_effort",
        "medium",
    )
