from unittest.mock import MagicMock, patch

import numpy as np
from environs import Env
from fastapi.testclient import TestClient

from app.main import app

env = Env()
env.read_env()

WCRF_API_KEY = env("WCRF_API_KEY")
API_ENDPOINT = "/study_screening/encode"


def test_basic():
    with TestClient(app) as client:
        input_payload = {
            "strategy": "best_recall",
            "review_topic": "cancer",
            "crtieria": None,
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

        with patch(
            "app.apis.study_screening.globals.models",
            {"study_screening": {"model": mock_model}},
        ), patch(
            "app.apis.study_screening.select_threshold"
        ) as mock_select_threshold, patch(
            "app.funcs.threshold.threshold_to_binary_labels"
        ) as mock_threshold_to_binary_labels:

            mock_select_threshold.return_value = 0.8
            mock_threshold_to_binary_labels.return_value = "included"

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


def test_no_api_key():
    with TestClient(app) as client:
        input_payload = {
            "strategy": "best_recall",
            "review_topic": "cancer",
            "crtieria": None,
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
            "crtieria": None,
            "study_title": "study title",
            "study_abstract": "study abstract",
        }
        headers = {"Accept": "application/json", "x-api-key": "foobar"}
        r = client.post(API_ENDPOINT, json=input_payload, headers=headers)
        assert r.status_code == 401
