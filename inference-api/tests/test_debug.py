from fastapi.testclient import TestClient

from app.main import app


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
