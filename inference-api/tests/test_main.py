import asyncio
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app import main

client = TestClient(main.app)
CHECK_AUTH_API_KEY = "check-auth-test-key"


def test_ping():
    r = client.get("/")
    assert r.json() == "hello world"


def test_check():
    r = client.get("/check")
    assert r.json() is True


def test_check_extraction_authenticated_success_has_local_diagnostics_only(
    synthetic_extraction_provider_config,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(main.globals, "api_keys", [CHECK_AUTH_API_KEY])
    fake_client = MagicMock()
    with patch.object(
        main.title_abstract_screening,
        "_get_openai_client",
        return_value=fake_client,
    ) as get_client:
        response = client.get(
            "/check/extraction",
            headers={"X-API-Key": CHECK_AUTH_API_KEY},
        )

    assert response.status_code == 200
    assert response.json() == {
        "ready": True,
        "provider_key_configured": True,
        "model_configured": True,
        "reasoning_effort_valid": True,
        "client_initialized": True,
        "responses_parse_available": True,
        "failure_reason": None,
        "error_type": None,
    }
    get_client.assert_called_once_with("extraction")
    fake_client.responses.parse.assert_not_called()
    fake_client.close.assert_called_once_with()


def test_check_extraction_invalid_configuration_returns_diagnostic_200(
    synthetic_extraction_provider_config,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(main.globals, "api_keys", [CHECK_AUTH_API_KEY])
    monkeypatch.setattr(main.globals, "openai_api_key", "  \t")
    with patch.object(
        main.title_abstract_screening,
        "_get_openai_client",
    ) as get_client:
        response = client.get(
            "/check/extraction",
            headers={"X-API-Key": CHECK_AUTH_API_KEY},
        )

    assert response.status_code == 200
    assert response.json() == {
        "ready": False,
        "provider_key_configured": False,
        "model_configured": True,
        "reasoning_effort_valid": True,
        "client_initialized": None,
        "responses_parse_available": None,
        "failure_reason": "missing_provider_key",
        "error_type": None,
    }
    get_client.assert_not_called()


def test_check_extraction_redacts_setup_error_from_body_and_output(
    synthetic_extraction_provider_config,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    secret = "route-constructor-secret-canary"
    monkeypatch.setattr(main.globals, "api_keys", [CHECK_AUTH_API_KEY])

    def fail_client(stage):
        try:
            raise RuntimeError(secret)
        except RuntimeError as cause:
            raise main.title_abstract_screening._service_error(
                "configuration_error", stage
            ) from cause

    with patch.object(
        main.title_abstract_screening,
        "_get_openai_client",
        side_effect=fail_client,
    ):
        response = client.get(
            "/check/extraction",
            headers={"X-API-Key": CHECK_AUTH_API_KEY},
        )

    captured = capsys.readouterr()
    assert response.status_code == 200
    assert response.json()["failure_reason"] == "client_initialization_failed"
    assert response.json()["error_type"] == "RuntimeError"
    assert secret not in response.text
    assert secret not in captured.out
    assert secret not in captured.err


@pytest.mark.parametrize(
    ("headers", "status"),
    [({}, 403), ({"X-API-Key": "wrong-extraction-check-key"}, 401)],
)
def test_check_extraction_rejects_missing_or_invalid_key(
    synthetic_extraction_provider_config,
    monkeypatch: pytest.MonkeyPatch,
    headers: dict[str, str],
    status: int,
) -> None:
    monkeypatch.setattr(main.globals, "api_keys", [CHECK_AUTH_API_KEY])
    with patch.object(
        main.title_abstract_screening,
        "_get_openai_client",
    ) as get_client:
        response = client.get("/check/extraction", headers=headers)

    assert response.status_code == status
    get_client.assert_not_called()


def test_check_auth_valid_key_returns_authenticated_without_provider_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(main.globals, "api_keys", [CHECK_AUTH_API_KEY])
    with patch("app.apis.data_extraction.extract_title_abstract") as mock_extract:
        response = client.get(
            "/check/auth",
            headers={"X-API-Key": CHECK_AUTH_API_KEY},
        )

    assert response.status_code == 200
    assert response.json() == {"authenticated": True}
    mock_extract.assert_not_called()


@pytest.mark.parametrize(
    ("headers", "status"),
    [({}, 403), ({"X-API-Key": "wrong-check-auth-key"}, 401)],
)
def test_check_auth_rejects_missing_or_invalid_key(
    monkeypatch: pytest.MonkeyPatch,
    headers: dict[str, str],
    status: int,
) -> None:
    monkeypatch.setattr(main.globals, "api_keys", [CHECK_AUTH_API_KEY])
    with patch("app.apis.data_extraction.extract_title_abstract") as mock_extract:
        response = client.get("/check/auth", headers=headers)

    assert response.status_code == status
    mock_extract.assert_not_called()


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
