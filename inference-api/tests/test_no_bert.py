import builtins
import runpy
import subprocess
import sys
from pathlib import Path
from textwrap import dedent
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app import main
from app.resources import globals

DEBUG_ENDPOINT = "/debug/encode"
SCREENING_ENDPOINT = "/study_screening/encode"
DEBUG_PAYLOAD = {
    "example_record": [{"text": "hello"}],
    "param1": None,
    "param2": None,
}
SCREENING_PAYLOAD = {
    "review_topic": "cancer",
    "study_title": "Study title",
    "study_abstract": "Study abstract",
    "strategy": "best_recall",
}


@pytest.mark.parametrize(
    ("value", "expected"),
    [(None, True), ("true", True), ("false", False), ("1", True), ("0", False)],
)
def test_load_bert_models_configuration(
    monkeypatch: pytest.MonkeyPatch, value: str | None, expected: bool
) -> None:
    """Local models default to enabled and accept boolean environment values."""
    if value is None:
        monkeypatch.delenv("PUER_LOAD_BERT_MODELS", raising=False)
    else:
        monkeypatch.setenv("PUER_LOAD_BERT_MODELS", value)
    config = runpy.run_path(str(globals.__file__))
    assert config["load_bert_models"] is expected


def test_disabled_startup_does_not_import_or_load_local_models(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mock_bert_assets: SimpleNamespace,
) -> None:
    """Disabled startup skips ML imports, asset reads, and stale model state."""
    monkeypatch.setattr(globals, "load_bert_models", False)
    monkeypatch.setattr(globals, "openai_api_key", None)
    monkeypatch.setattr(
        globals, "paths", {name: tmp_path / name for name in globals.paths}
    )
    monkeypatch.setattr(globals, "models", {"stale": object()})
    monkeypatch.setattr(globals, "thresholds", mock_bert_assets.thresholds)
    original_import = builtins.__import__

    def reject_ml_import(name: str, *args, **kwargs):
        """Fail if the disabled lifespan tries to import an ML library."""
        assert name.split(".")[0] not in {
            "transformers",
            "sentence_transformers",
            "torch",
        }
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", reject_ml_import)
    with TestClient(main.app) as client:
        assert client.get("/").json() == "hello world"
        assert globals.models == {}
        assert globals.thresholds is None
    assets = mock_bert_assets
    assets.transformers.AutoTokenizer.from_pretrained.assert_not_called()
    assets.transformers.AlbertForSequenceClassification.from_pretrained.assert_not_called()
    assets.sentence_transformers.SentenceTransformer.assert_not_called()
    assets.read_thresholds.assert_not_called()
    assert globals.models == {}
    assert globals.thresholds is None


def test_disabled_app_import_and_startup_without_ml_libraries() -> None:
    """A fresh API process needs no Transformers or PyTorch imports."""
    script = dedent(
        """
        import builtins
        from environs import Env
        from fastapi.testclient import TestClient

        Env.read_env = lambda *args, **kwargs: None
        original_import = builtins.__import__

        def reject_ml_import(name, *args, **kwargs):
            assert name.split(".")[0] not in {
                "transformers", "sentence_transformers", "torch"
            }
            return original_import(name, *args, **kwargs)

        builtins.__import__ = reject_ml_import
        from app.main import app

        with TestClient(app) as client:
            assert client.get("/").status_code == 200
            assert client.get("/check").json() is True
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[1],
        env={
            "WCRF_API_KEY": "fresh-process-test-key",
            "PUER_LOAD_BERT_MODELS": "false",
        },
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_disabled_check_ignores_missing_assets(
    no_bert_client: TestClient,
) -> None:
    """The boolean asset check excludes intentionally disabled local assets."""
    assert not any(path.exists() for path in globals.paths.values())
    response = no_bert_client.get("/check")
    assert response.status_code == 200
    assert response.json() is True


@pytest.mark.parametrize(
    ("endpoint", "payload"),
    [(DEBUG_ENDPOINT, DEBUG_PAYLOAD), (SCREENING_ENDPOINT, SCREENING_PAYLOAD)],
)
def test_disabled_legacy_routes_return_503(
    no_bert_client: TestClient, endpoint: str, payload: dict
) -> None:
    """Valid requests get an explicit 503 rather than missing-model errors."""
    response = no_bert_client.post(
        endpoint, json=payload, headers={"X-API-Key": "test-key"}
    )
    assert response.status_code == 503
    assert response.json() == {"detail": "Local BERT models are disabled."}


@pytest.mark.parametrize(
    ("headers", "status"),
    [({}, 403), ({"X-API-Key": "wrong-key"}, 401)],
)
def test_disabled_screening_preserves_authentication(
    no_bert_client: TestClient, headers: dict[str, str], status: int
) -> None:
    """Disabling local models does not bypass screening authentication."""
    response = no_bert_client.post(
        SCREENING_ENDPOINT, json=SCREENING_PAYLOAD, headers=headers
    )
    assert response.status_code == status


def test_disabled_debug_preserves_unauthenticated_access(
    no_bert_client: TestClient,
) -> None:
    """Debug requests still require no key, but cannot run disabled models."""
    response = no_bert_client.post(DEBUG_ENDPOINT, json=DEBUG_PAYLOAD)
    assert response.status_code == 503


@pytest.mark.parametrize("endpoint", [DEBUG_ENDPOINT, SCREENING_ENDPOINT])
def test_disabled_legacy_routes_preserve_request_validation(
    no_bert_client: TestClient, endpoint: str
) -> None:
    """Invalid payloads are rejected before entering the disabled handler."""
    response = no_bert_client.post(endpoint, json={}, headers={"X-API-Key": "test-key"})
    assert response.status_code == 422


def test_disabled_mode_preserves_auth_check(no_bert_client: TestClient) -> None:
    """Local authentication checks do not depend on BERT assets."""
    response = no_bert_client.get("/check/auth", headers={"X-API-Key": "test-key"})
    assert response.status_code == 200
    assert response.json() == {"authenticated": True}
