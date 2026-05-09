import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

mock_predictor = MagicMock()
mock_predictor.classes = ["Tomato_healthy", "Tomato_Early_blight"]
mock_predictor.predict.return_value = {
    "prediction": "Tomato healthy",
    "confidence": 95.0,
    "top_k": [{"species": "Tomato healthy", "confidence": 95.0}]
}

with patch("src.api.predictor.get_predictor", return_value=mock_predictor):
    from src.api.app import app

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "LeafLens" in response.json()["message"]