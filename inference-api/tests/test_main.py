from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_ping():
    r = client.get("/")
    assert r.json() == "hello world"


def test_check():
    r = client.get("/check")
    assert r.json() is True
