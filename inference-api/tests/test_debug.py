from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from app.main import app
import numpy as np


def test_debug_encode():
    with TestClient(app) as client:
        input_payload = {
            "example_record": [
                {"text": "hello"},
                {"text": "world"},
            ],
            "param1": None,
            "param2": None,
        }
        r = client.post("/debug/encode", json=input_payload)
        assert r.status_code == 200
        assert r.json()


def test_debug_encode_limit():
    with TestClient(app) as client:
        input_payload = {
            "example_record": [
                {"text": "hello"},
                {"text": "world"},
                {"text": "hello world"},
            ],
            "param1": None,
            "param2": None,
        }
        r = client.post("/debug/encode", json=input_payload)
        assert r.status_code == 422  # Unprocessable Entity


def test_post_encode():
    with TestClient(app) as client:
        input_payload = {
            "review_topic": "topic",
            "criteria": "criteria",
            "study_title": "study title",
            "study_abstract": "study abstract.",
            "strategy": "best_recall",
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
            response = client.post("/study_screening/encode", json=input_payload)

            # Assertions for response status and response body
            assert response.status_code == 200

            expected_cosine = 1.0000000000000002  # Floating-Point Arithmetic
            expected_decision = "included"

            assert response.json() == [expected_cosine, expected_decision]
