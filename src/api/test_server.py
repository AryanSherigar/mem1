import pytest
from fastapi.testclient import TestClient
from src.api.server import app

client = TestClient(app)

# make a test for the /health endpoint
def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


