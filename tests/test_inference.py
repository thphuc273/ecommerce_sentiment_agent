import pytest
import requests
import time
from typing import Dict, Any, List
import json

# Constants
INFERENCE_URL = "http://localhost:8004"
ANALYZE_ENDPOINT = f"{INFERENCE_URL}/analyze"
HEALTH_ENDPOINT = f"{INFERENCE_URL}/health"
TIMEOUT = 10  # seconds

# Test sample reviews with expected sentiment
TEST_SAMPLES = [
    {
        "text": "This product exceeded all my expectations! The quality is outstanding.",
        "expected_sentiment": "positive"
    },
    {
        "text": "I absolutely love this item. Best purchase I've made all year!",
        "expected_sentiment": "positive"
    },
    {
        "text": "Works as advertised. Nothing special but does the job.",
        "expected_sentiment": "neutral"
    },
    {
        "text": "The product is okay I guess. Shipping was fast at least.",
        "expected_sentiment": "neutral"
    },
    {
        "text": "Not worth the price. I'm a bit disappointed with the purchase.",
        "expected_sentiment": "negative"
    },
    {
        "text": "Terrible quality. It broke after just two uses. Would not recommend.",
        "expected_sentiment": "negative"
    },
    {
        "text": "This could have been better, but it's not the worst I've seen.",
        "expected_sentiment": "neutral"
    },
    {
        "text": "Fantastic product, I've already ordered three more for my family!",
        "expected_sentiment": "positive"
    },
    {
        "text": "Complete waste of money. Customer service refused to help with return.",
        "expected_sentiment": "negative"
    },
    {
        "text": "The design is beautiful but functionality is lacking.",
        "expected_sentiment": "neutral"
    },
]

@pytest.fixture(scope="module")
def ensure_service_running():
    """Ensure the inference service is running before tests."""
    try:
        response = requests.get(HEALTH_ENDPOINT, timeout=TIMEOUT)
        assert response.status_code == 200, "Inference service health check failed"
        service_status = response.json()
        assert service_status.get("status") == "ok", "Inference service is not healthy"
        return True
    except (requests.ConnectionError, requests.Timeout) as e:
        pytest.skip(f"Inference service not available: {str(e)}")
    except Exception as e:
        pytest.fail(f"Failed to check if inference service is running: {str(e)}")

def test_analyze_endpoint_structure(ensure_service_running):
    """Test that the /analyze endpoint returns the expected JSON structure."""
    payload = {"query_text": "This is a test review"}
    
    response = requests.post(ANALYZE_ENDPOINT, json=payload, timeout=TIMEOUT)
    
    # Assert response status code
    assert response.status_code == 200, f"Expected status code 200, got {response.status_code}"
    
    # Parse response
    result = response.json()
    
    # Check required fields
    assert "sentiment" in result, "Response missing 'sentiment' field"
    assert "confidence" in result, "Response missing 'confidence' field"
    assert "summary" in result, "Response missing 'summary' field"
    assert "similar_reviews" in result, "Response missing 'similar_reviews' field"
    
    # Check types
    assert isinstance(result["sentiment"], str), "sentiment should be a string"
    assert isinstance(result["confidence"], (int, float)), "confidence should be a number"
    assert isinstance(result["summary"], str), "summary should be a string"
    assert isinstance(result["similar_reviews"], list), "similar_reviews should be a list"
    
    # Check confidence range
    assert 0 <= result["confidence"] <= 1, "confidence should be between 0 and 1"

def test_analyze_endpoint_latency(ensure_service_running):
    """Test that the /analyze endpoint responds within 2 seconds."""
    payload = {"query_text": "This is a test review for latency measurement"}
    
    # Record start time
    start_time = time.time()
    
    # Make request
    response = requests.post(ANALYZE_ENDPOINT, json=payload, timeout=TIMEOUT)
    
    # Calculate latency
    latency = time.time() - start_time
    
    # Assert response status code
    assert response.status_code == 200, f"Expected status code 200, got {response.status_code}"
    
    # Check latency
    assert latency < 2.0, f"Response time exceeded 2 seconds: {latency:.2f}s"
    
    print(f"Analyze endpoint latency: {latency:.2f} seconds")

def test_sentiment_accuracy_on_samples(ensure_service_running):
    """Test sentiment accuracy on predefined samples with expected labels."""
    results = []
    
    for i, sample in enumerate(TEST_SAMPLES):
        # Record start time for individual sample latency
        start_time = time.time()
        
        # Make request
        payload = {"query_text": sample["text"]}
        response = requests.post(ANALYZE_ENDPOINT, json=payload, timeout=TIMEOUT)
        
        # Calculate latency
        latency = time.time() - start_time
        
        # Check response status
        assert response.status_code == 200, f"Sample {i+1} failed with status {response.status_code}"
        
        # Get result
        result = response.json()
        
        # Extract actual sentiment (normalize to lowercase)
        actual_sentiment = result["sentiment"].lower()
        if actual_sentiment.startswith("label_"):
            # Handle numeric labels: LABEL_0 => negative, LABEL_1 => neutral, LABEL_2 => positive
            label_map = {"label_0": "negative", "label_1": "neutral", "label_2": "positive"}
            actual_sentiment = label_map.get(actual_sentiment, actual_sentiment)
        
        # Save result for report
        results.append({
            "sample": i + 1,
            "text": sample["text"],
            "expected": sample["expected_sentiment"],
            "actual": actual_sentiment,
            "confidence": result["confidence"],
            "correct": actual_sentiment == sample["expected_sentiment"],
            "latency": latency
        })
    
    # Calculate accuracy
    correct_count = sum(1 for r in results if r["correct"])
    accuracy = correct_count / len(results)
    avg_latency = sum(r["latency"] for r in results) / len(results)
    
    # Log results
    print(f"\nSentiment Analysis Results:")
    print(f"Accuracy: {accuracy:.2%} ({correct_count}/{len(results)})")
    print(f"Average Latency: {avg_latency:.3f} seconds\n")
    
    for r in results:
        status = "✓" if r["correct"] else "✗"
        print(f"{status} Sample {r['sample']}: Expected '{r['expected']}', got '{r['actual']}' "
              f"(confidence: {r['confidence']:.2f}, latency: {r['latency']:.3f}s)")
    
    # Assert minimum accuracy threshold (can be adjusted based on model expectations)
    assert accuracy >= 0.7, f"Accuracy below threshold: {accuracy:.2%}"
    
    # Assert all samples processed within latency threshold
    slow_samples = [r for r in results if r["latency"] >= 2.0]
    assert len(slow_samples) == 0, f"{len(slow_samples)} samples exceeded 2s latency threshold"

def test_empty_request_handling(ensure_service_running):
    """Test that the /analyze endpoint properly handles empty requests."""
    # Empty text
    payload = {"query_text": ""}
    response = requests.post(ANALYZE_ENDPOINT, json=payload, timeout=TIMEOUT)
    
    # We expect a 400 Bad Request or similar error code
    assert response.status_code >= 400, "Empty text should result in an error"
    
    # Missing text field
    payload = {}
    response = requests.post(ANALYZE_ENDPOINT, json=payload, timeout=TIMEOUT)
    
    # We expect a 422 Unprocessable Entity or similar error code
    assert response.status_code >= 400, "Missing query_text field should result in an error"

if __name__ == "__main__":
    # This allows running the tests directly with python test_inference.py
    pytest.main(["-v", __file__])
