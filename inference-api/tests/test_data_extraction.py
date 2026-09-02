from collections.abc import Generator
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.apis import data_extraction
from app.funcs import title_abstract_screening as screening
from app.resources import globals

API_ENDPOINT = "/data_extraction/title_abstract/"
API_KEY = "test-key"


def _characteristics() -> dict[str, Any]:
    """Return a complete characteristics payload for route contracts."""
    res = {
        "human_study": "yes",
        "publication_type": "primary_research",
        "study_design": "prospective_cohort",
        "population": "Adults",
        "setting": "United Kingdom",
        "sample_size": 12500,
        "exposures": ["physical activity"],
        "comparators": ["lower physical activity"],
        "outcomes": ["colorectal cancer"],
        "outcome_types": ["incidence"],
        "follow_up": "10 years",
        "effect_measures": ["hazard ratio"],
        "confidence_intervals_reported": "yes",
        "analysis_methods": ["Cox regression"],
        "evidence": [
            {
                "characteristic": "study_design",
                "source": "abstract",
                "quote": "prospective cohort",
            }
        ],
        "limitations": [],
    }
    return res


def _extraction_response() -> screening.TitleAbstractExtractionResponse:
    """Return a strict extraction response without provider work."""
    res = screening.TitleAbstractExtractionResponse(
        characteristics=screening.StudyCharacteristics(**_characteristics()),
        provenance=screening.ScreeningProvenance(
            model="gpt-5.6-terra",
            reasoning_effort="medium",
            prompt_version=screening.EXTRACTION_PROMPT_VERSION,
            schema_version=screening.SCREENING_SCHEMA_VERSION,
        ),
    )
    return res


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> Generator[TestClient, None, None]:
    """Build an extraction-only application without the production lifespan."""
    monkeypatch.setattr(globals, "api_keys", [API_KEY])
    app = FastAPI()
    app.include_router(data_extraction.router)
    with TestClient(app) as test_client:
        yield test_client


def _headers() -> dict[str, str]:
    """Return valid authentication headers for isolated route tests."""
    res = {"X-API-Key": API_KEY}
    return res


def _payload(**overrides: Any) -> dict[str, Any]:
    """Return a valid title-and-abstract request payload."""
    res = {
        "title": "Physical activity and colorectal cancer incidence",
        "abstract": "A prospective cohort examined colorectal cancer.",
    }
    res.update(overrides)
    return res


def test_extraction_delegates_once_and_serializes_response(client: TestClient) -> None:
    """The strict extraction route delegates exactly once to its async service."""
    expected = _extraction_response()
    with patch(
        "app.apis.data_extraction.extract_title_abstract",
        new=AsyncMock(return_value=expected),
    ) as mock_extract:
        response = client.post(API_ENDPOINT, json=_payload(), headers=_headers())

    assert response.status_code == 200
    assert response.json() == expected.model_dump(mode="json")
    mock_extract.assert_awaited_once_with(screening.TitleAbstractRequest(**_payload()))


@pytest.mark.parametrize("headers, status", [({}, 403), ({"X-API-Key": "bad"}, 401)])
def test_extraction_authentication_contract(
    client: TestClient, headers: dict[str, str], status: int
) -> None:
    """Absent keys are forbidden while supplied invalid keys are unauthorized."""
    response = client.post(API_ENDPOINT, json=_payload(), headers=headers)
    assert response.status_code == status


@pytest.mark.parametrize("abstract", [None, "", "   \t\n"])
def test_extraction_accepts_title_only_abstract_boundaries(
    client: TestClient, abstract: str | None
) -> None:
    """Null and blank abstracts reach the service as title-only input."""
    expected = _extraction_response()
    with patch(
        "app.apis.data_extraction.extract_title_abstract",
        new=AsyncMock(return_value=expected),
    ) as mock_extract:
        payload = _payload()
        if abstract is not None:
            payload["abstract"] = abstract
        else:
            payload["abstract"] = None
        response = client.post(API_ENDPOINT, json=payload, headers=_headers())

    assert response.status_code == 200
    assert mock_extract.await_args is not None
    assert mock_extract.await_args.args[0].abstract is None


def test_extraction_accepts_omitted_abstract(client: TestClient) -> None:
    """An omitted abstract reaches the service as a title-only request."""
    expected = _extraction_response()
    with patch(
        "app.apis.data_extraction.extract_title_abstract",
        new=AsyncMock(return_value=expected),
    ) as mock_extract:
        response = client.post(
            API_ENDPOINT,
            json={"title": "Physical activity and colorectal cancer incidence"},
            headers=_headers(),
        )

    assert response.status_code == 200
    assert mock_extract.await_args is not None
    assert mock_extract.await_args.args[0].abstract is None


def test_extraction_rejects_strict_invalid_input_before_service(
    client: TestClient,
) -> None:
    """Extra fields and blank titles are rejected without provider delegation."""
    with patch(
        "app.apis.data_extraction.extract_title_abstract",
        new=AsyncMock(),
    ) as mock_extract:
        response = client.post(
            API_ENDPOINT,
            json=_payload(title="   ", undeclared="value"),
            headers=_headers(),
        )

    assert response.status_code == 422
    mock_extract.assert_not_awaited()


def test_extraction_maps_service_error_to_stable_503(client: TestClient) -> None:
    """Service errors preserve their public code, stage, and message exactly."""
    detail = screening.ScreeningErrorDetail(
        code="provider_transient",
        stage="extraction",
        message="The extraction provider is temporarily unavailable.",
    )
    with patch(
        "app.apis.data_extraction.extract_title_abstract",
        new=AsyncMock(side_effect=screening.ScreeningServiceError(detail)),
    ):
        response = client.post(API_ENDPOINT, json=_payload(), headers=_headers())

    assert response.status_code == 503
    assert response.json() == {"detail": detail.model_dump(mode="json")}


def test_extraction_openapi_declares_exact_path_and_contract_models(
    client: TestClient,
) -> None:
    """OpenAPI exposes the trailing-slash extraction path and shared schemas."""
    schema = client.get("/openapi.json").json()
    operation = schema["paths"][API_ENDPOINT]["post"]

    assert operation["requestBody"]["content"]["application/json"]["schema"] == {
        "$ref": "#/components/schemas/TitleAbstractRequest"
    }
    assert operation["responses"]["200"]["content"]["application/json"]["schema"] == {
        "$ref": "#/components/schemas/TitleAbstractExtractionResponse"
    }
    assert "TitleAbstractRequest" in schema["components"]["schemas"]
    assert "TitleAbstractExtractionResponse" in schema["components"]["schemas"]
