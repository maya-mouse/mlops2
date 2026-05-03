import numpy as np
import pytest
from app.drift import DriftDetector

FEATURE_NAMES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

def test_no_drift_on_same_distribution():
    rng = np.random.default_rng(42)
    ref = rng.normal(loc=5.0, scale=1.0, size=(100, 4))
    cur = rng.normal(loc=5.0, scale=1.0, size=(100, 4))
    
    detector = DriftDetector(ref, FEATURE_NAMES)
    result = detector.detect(cur, alpha=0.05)
    
    assert result["drift_detected"] is False
    assert result["n_drifted_features"] == 0

def test_drift_on_shifted_distribution():
    rng = np.random.default_rng(42)
    ref = rng.normal(loc=5.0, scale=1.0, size=(100, 4))
    cur = rng.normal(loc=10.0, scale=1.0, size=(100, 4))
    
    detector = DriftDetector(ref, FEATURE_NAMES)
    result = detector.detect(cur, alpha=0.05)
    
    assert result["drift_detected"] is True
    assert result["n_drifted_features"] == 4

def test_drift_api_response_structure():
    ref = np.zeros((10, 4))
    cur = np.ones((10, 4))
    detector = DriftDetector(ref, FEATURE_NAMES)
    result = detector.detect(cur)
    
    assert "per_feature" in result
    assert "p_value" in result["per_feature"]["sepal_length"]