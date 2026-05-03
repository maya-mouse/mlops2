import pytest
from fastapi.testclient import TestClient
from app.main import app
from ml.train import train_and_save

client = TestClient(app)

@pytest.fixture(scope="module", autouse=True)
def prepare_artifacts():
    train_and_save()

def test_metrics_endpoint_status():
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "ml_predictions_total" in response.text

def test_prediction_increments_counter():
    initial_metrics = client.get("/metrics").text
    
    payload = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }
    client.post("/predict", json=payload)
    

    updated_metrics = client.get("/metrics").text
    assert 'status="success"' in updated_metrics
    assert initial_metrics != updated_metrics