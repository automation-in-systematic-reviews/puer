from collections.abc import Generator
from io import StringIO
from typing import Any
from unittest.mock import AsyncMock, MagicMock, call, patch

import numpy as np
import pytest
from environs import Env
from fastapi import FastAPI
from fastapi.testclient import TestClient
from loguru import logger

from app.apis import study_screening
from app.funcs import title_abstract_screening as screening
from app.main import app
from app.resources import globals

env = Env()
env.read_env()

WCRF_API_KEY = env("WCRF_API_KEY")
API_ENDPOINT = "/study_screening/encode"
PREDICT_ENDPOINT = "/study_screening/predict/"
TEST_API_KEY = "test-key"


def _prediction_characteristics() -> dict[str, Any]:
    """Return a complete strict characteristics payload for prediction tests."""
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
        "evidence": [],
        "limitations": [],
    }
    return res


def _prediction_payload(**overrides: Any) -> dict[str, Any]:
    """Return a valid strict prediction request payload."""
    res = {
        "title": "Physical activity and colorectal cancer incidence",
        "abstract": "A prospective cohort examined colorectal cancer.",
        "review_topic": "physical activity",
        "criteria": "Include cohort studies reporting colorectal cancer.",
        "characteristics": _prediction_characteristics(),
    }
    res.update(overrides)
    return res


def _prediction_response() -> screening.StudyScreeningPredictionResponse:
    """Return a strict prediction response without provider work."""
    res = screening.StudyScreeningPredictionResponse(
        decision="included",
        confidence="low",
        rationale="The criterion is unclear from the available evidence.",
        characteristic_conflicts=[],
        criterion_assessments=[
            screening.CriterionAssessment(
                criterion="Eligible cancer outcome",
                status="unclear",
                rationale="The abstract is incomplete.",
                evidence=[],
            )
        ],
        provenance=screening.ScreeningProvenance(
            model="gpt-5.6-terra",
            reasoning_effort="medium",
            prompt_version=screening.PREDICTION_PROMPT_VERSION,
            schema_version=screening.SCREENING_SCHEMA_VERSION,
        ),
    )
    return res


@pytest.fixture
def prediction_client(
    monkeypatch: pytest.MonkeyPatch,
) -> Generator[TestClient, None, None]:
    """Build a screening-only app without production lifespan initialization."""
    monkeypatch.setattr(globals, "api_keys", [TEST_API_KEY])
    isolated_app = FastAPI()
    isolated_app.include_router(study_screening.router)
    with TestClient(isolated_app) as client:
        yield client


def _prediction_headers() -> dict[str, str]:
    """Return valid authentication headers for isolated prediction tests."""
    res = {"X-API-Key": TEST_API_KEY}
    return res


def test_basic():
    with TestClient(app) as client:
        input_payload = {
            "strategy": "best_recall",
            "review_topic": "cancer",
            "criteria": None,
            "study_title": "study title",
            "study_abstract": "study abstract",
        }
        headers = {
            "Accept": "application/json",
            "x-api-key": WCRF_API_KEY,
        }
        r = client.post(API_ENDPOINT, json=input_payload, headers=headers)
        assert r.status_code == 200
        assert r.json()


def test_post_encode():
    with TestClient(app) as client:
        input_payload = {
            "review_topic": "topic",
            "criteria": "criteria",
            "study_title": "study title",
            "study_abstract": "study abstract.",
            "strategy": "best_recall",
        }

        headers = {
            "Accept": "application/json",
            "x-api-key": WCRF_API_KEY,
        }

        mock_model = MagicMock()
        mock_model.encode.side_effect = [
            np.array([0.5, 0.5, 0.5]),  # Mocked query encoding
            np.array([0.5, 0.5, 0.5]),  # Mocked study encoding
        ]

        with (
            patch(
                "app.apis.study_screening.globals.models",
                {"study_screening": {"model": mock_model}},
            ),
            patch("app.apis.study_screening.select_threshold") as mock_select_threshold,
            patch(
                "app.apis.study_screening.threshold_to_binary_labels"
            ) as mock_threshold_to_binary_labels,
        ):
            mock_select_threshold.return_value = 0.8
            mock_threshold_to_binary_labels.return_value = ["included"]

            # Send POST request to the /study_screening/encode endpoint
            response = client.post(
                API_ENDPOINT,
                json=input_payload,
                headers=headers,
            )

            # Assertions for response status and response body
            assert response.status_code == 200

            expected_cosine = 1.0000000000000002  # Floating-Point Arithmetic
            expected_decision = "included"

            assert response.json() == [expected_cosine, expected_decision]
            assert mock_model.encode.call_count == 2
            mock_model.encode.assert_has_calls(
                [
                    call(
                        "Query: topic. Criteria: criteria",
                        convert_to_numpy=True,
                    ),
                    call(
                        "Title: study title. Abstract: study abstract.",
                        convert_to_numpy=True,
                    ),
                ]
            )
            mock_select_threshold.assert_called_once()
            assert mock_select_threshold.call_args is not None
            assert mock_select_threshold.call_args.args[1:] == (
                "topic",
                "best_recall",
            )
            mock_threshold_to_binary_labels.assert_called_once_with(
                [expected_cosine],
                0.8,
            )


def test_post_encode_does_not_emit_payload_or_result_diagnostics(
    prediction_client: TestClient,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Legacy inference preserves results without logging its sensitive inputs."""
    criteria = "criteria-canary"
    title = "title-canary"
    abstract = "abstract-canary"
    decision = "included-canary"
    expected_cosine = 1.0000000000000002
    input_payload = {
        "review_topic": "topic-canary",
        "criteria": criteria,
        "study_title": title,
        "study_abstract": abstract,
        "strategy": "best_recall",
    }
    mock_model = MagicMock()
    mock_model.encode.side_effect = [
        np.array([0.5, 0.5, 0.5]),
        np.array([0.5, 0.5, 0.5]),
    ]
    application_logs = StringIO()
    sink_id = logger.add(application_logs)

    try:
        with (
            patch(
                "app.apis.study_screening.globals.models",
                {"study_screening": {"model": mock_model}},
            ),
            patch(
                "app.apis.study_screening.select_threshold",
                return_value=0.8,
            ) as mock_select_threshold,
            patch(
                "app.apis.study_screening.threshold_to_binary_labels",
                return_value=[decision],
            ) as mock_threshold,
        ):
            response = prediction_client.post(
                API_ENDPOINT,
                json=input_payload,
                headers=_prediction_headers(),
            )
    finally:
        logger.remove(sink_id)

    captured = capsys.readouterr()
    diagnostic_values = [
        criteria,
        title,
        abstract,
        str(expected_cosine),
        decision,
    ]
    assert response.status_code == 200
    assert response.json() == [expected_cosine, decision]
    assert mock_model.encode.call_args_list == [
        call(
            "Query: topic-canary. Criteria: criteria-canary",
            convert_to_numpy=True,
        ),
        call(
            "Title: title-canary. Abstract: abstract-canary",
            convert_to_numpy=True,
        ),
    ]
    mock_select_threshold.assert_called_once_with(
        study_screening.globals.thresholds,
        "topic-canary",
        "best_recall",
    )
    mock_threshold.assert_called_once_with([expected_cosine], 0.8)
    for value in diagnostic_values:
        assert value not in captured.out
        assert value not in captured.err
        assert value not in application_logs.getvalue()


def test_no_api_key():
    with TestClient(app) as client:
        input_payload = {
            "strategy": "best_recall",
            "review_topic": "cancer",
            "criteria": None,
            "study_title": "study title",
            "study_abstract": "study abstract",
        }
        r = client.post(API_ENDPOINT, json=input_payload)
        assert r.status_code == 403


def test_wrong_api_key():
    with TestClient(app) as client:
        input_payload = {
            "strategy": "best_recall",
            "review_topic": "cancer",
            "criteria": None,
            "study_title": "study title",
            "study_abstract": "study abstract",
        }
        headers = {"Accept": "application/json", "x-api-key": "foobar"}
        r = client.post(API_ENDPOINT, json=input_payload, headers=headers)
        assert r.status_code == 401


def test_prediction_delegates_once_and_serializes_response(
    prediction_client: TestClient,
) -> None:
    """The prediction route delegates exactly once to its async service."""
    expected = _prediction_response()
    with patch(
        "app.apis.study_screening.predict_study_screening",
        new=AsyncMock(return_value=expected),
    ) as mock_predict:
        response = prediction_client.post(
            PREDICT_ENDPOINT,
            json=_prediction_payload(),
            headers=_prediction_headers(),
        )

    assert response.status_code == 200
    assert response.json() == expected.model_dump(mode="json")
    mock_predict.assert_awaited_once_with(
        screening.StudyScreeningPredictionRequest(**_prediction_payload())
    )


@pytest.mark.parametrize("headers, status", [({}, 403), ({"X-API-Key": "bad"}, 401)])
def test_prediction_authentication_contract(
    prediction_client: TestClient, headers: dict[str, str], status: int
) -> None:
    """Prediction shares the explicit absent and invalid key behavior."""
    response = prediction_client.post(
        PREDICT_ENDPOINT,
        json=_prediction_payload(),
        headers=headers,
    )
    assert response.status_code == status


@pytest.mark.parametrize("abstract", [None, "", "   \t\n"])
def test_prediction_accepts_title_only_abstract_boundaries(
    prediction_client: TestClient, abstract: str | None
) -> None:
    """Null and blank title-only payloads are delegated unchanged."""
    expected = _prediction_response()
    with patch(
        "app.apis.study_screening.predict_study_screening",
        new=AsyncMock(return_value=expected),
    ) as mock_predict:
        response = prediction_client.post(
            PREDICT_ENDPOINT,
            json=_prediction_payload(abstract=abstract),
            headers=_prediction_headers(),
        )

    assert response.status_code == 200
    assert mock_predict.await_args is not None
    assert mock_predict.await_args.args[0].abstract is None


def test_prediction_accepts_omitted_abstract(
    prediction_client: TestClient,
) -> None:
    """An omitted abstract reaches prediction as a title-only request."""
    expected = _prediction_response()
    payload = _prediction_payload()
    del payload["abstract"]
    with patch(
        "app.apis.study_screening.predict_study_screening",
        new=AsyncMock(return_value=expected),
    ) as mock_predict:
        response = prediction_client.post(
            PREDICT_ENDPOINT,
            json=payload,
            headers=_prediction_headers(),
        )

    assert response.status_code == 200
    assert mock_predict.await_args is not None
    assert mock_predict.await_args.args[0].abstract is None


def test_prediction_rejects_strict_invalid_input_before_service(
    prediction_client: TestClient,
) -> None:
    """Blank required context and undeclared fields return FastAPI 422 errors."""
    with patch(
        "app.apis.study_screening.predict_study_screening",
        new=AsyncMock(),
    ) as mock_predict:
        response = prediction_client.post(
            PREDICT_ENDPOINT,
            json=_prediction_payload(
                title=" ",
                review_topic=" ",
                criteria=" ",
                undeclared="value",
            ),
            headers=_prediction_headers(),
        )

    assert response.status_code == 422
    mock_predict.assert_not_awaited()


def test_prediction_rejects_unresolved_criteria_before_provider_call(
    prediction_client: TestClient,
) -> None:
    """Unresolved criteria fail through the service before client construction."""
    with patch(
        "app.funcs.title_abstract_screening._get_openai_client",
    ) as mock_client:
        response = prediction_client.post(
            PREDICT_ENDPOINT,
            json=_prediction_payload(
                criteria="Include [INSERT CANCER OUTCOME] studies."
            ),
            headers=_prediction_headers(),
        )

    assert response.status_code == 503
    assert response.json() == {
        "detail": {
            "code": "provider_contract_error",
            "stage": "prediction",
            "message": (
                "The prediction provider response did not satisfy the response "
                "contract."
            ),
        }
    }
    mock_client.assert_not_called()


def test_prediction_maps_service_error_to_stable_503(
    prediction_client: TestClient,
) -> None:
    """Prediction service errors retain their public detail object exactly."""
    detail = screening.ScreeningErrorDetail(
        code="provider_request_error",
        stage="prediction",
        message="The prediction provider rejected the request.",
    )
    with patch(
        "app.apis.study_screening.predict_study_screening",
        new=AsyncMock(side_effect=screening.ScreeningServiceError(detail)),
    ):
        response = prediction_client.post(
            PREDICT_ENDPOINT,
            json=_prediction_payload(),
            headers=_prediction_headers(),
        )

    assert response.status_code == 503
    assert response.json() == {"detail": detail.model_dump(mode="json")}


def test_prediction_openapi_declares_exact_path_and_contract_models(
    prediction_client: TestClient,
) -> None:
    """OpenAPI exposes the trailing-slash prediction path and shared schemas."""
    schema = prediction_client.get("/openapi.json").json()
    operation = schema["paths"][PREDICT_ENDPOINT]["post"]

    assert operation["requestBody"]["content"]["application/json"]["schema"] == {
        "$ref": "#/components/schemas/StudyScreeningPredictionRequest"
    }
    assert operation["responses"]["200"]["content"]["application/json"]["schema"] == {
        "$ref": "#/components/schemas/StudyScreeningPredictionResponse"
    }
    assert "StudyScreeningPredictionRequest" in schema["components"]["schemas"]
    assert "StudyScreeningPredictionResponse" in schema["components"]["schemas"]
