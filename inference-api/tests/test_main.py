import asyncio
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app import main

client = TestClient(main.app)


def test_ping():
    r = client.get("/")
    assert r.json() == "hello world"


def test_check():
    r = client.get("/check")
    assert r.json() is True


def test_lifespan_does_not_log_configured_api_keys(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Startup loads normal assets without emitting configured API-key values."""
    canary_key = "startup-api-key-canary"
    tokenizer = MagicMock()
    classifier = MagicMock()
    screening_model = MagicMock()
    thresholds = MagicMock()
    mock_logger = MagicMock()

    with (
        patch.object(main.globals, "api_keys", [canary_key]),
        patch.object(
            main.transformers.AutoTokenizer,
            "from_pretrained",
            return_value=tokenizer,
        ) as mock_tokenizer,
        patch.object(
            main.transformers.AlbertForSequenceClassification,
            "from_pretrained",
            return_value=classifier,
        ) as mock_classifier,
        patch.object(
            main,
            "SentenceTransformer",
            return_value=screening_model,
        ) as mock_screening_model,
        patch.object(main, "read_thresholds", return_value=thresholds) as mock_read,
        patch.object(main, "logger", mock_logger),
    ):
        asyncio.run(
            _run_lifespan(
                main.app,
                tokenizer=tokenizer,
                classifier=classifier,
                screening_model=screening_model,
                thresholds=thresholds,
            )
        )

    captured = capsys.readouterr()
    logged_values = " ".join(str(call) for call in mock_logger.method_calls)
    assert canary_key not in captured.out
    assert canary_key not in captured.err
    assert canary_key not in logged_values
    mock_tokenizer.assert_called_once_with(main.globals.paths["albert_imdb"])
    mock_classifier.assert_called_once_with(main.globals.paths["albert_imdb"])
    mock_screening_model.assert_called_once_with(
        str(main.globals.paths["study_screening"])
    )
    mock_read.assert_called_once_with(main.globals.paths["thresholds"])
    assert main.globals.models == {}


async def _run_lifespan(
    app: FastAPI,
    tokenizer: MagicMock,
    classifier: MagicMock,
    screening_model: MagicMock,
    thresholds: MagicMock,
) -> None:
    """Enter and exit the production lifespan for an isolated startup test."""
    async with main.lifespan(app):
        assert main.globals.models["albert-imdb"] == {
            "tokenizer": tokenizer,
            "model": classifier,
        }
        assert main.globals.models["study_screening"] == {
            "model": screening_model,
        }
        assert main.globals.thresholds is thresholds
