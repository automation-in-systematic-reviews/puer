import pytest

from app.resources import globals


@pytest.fixture
def synthetic_extraction_provider_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Provide deterministic local provider settings for diagnostic tests."""
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
