"""
Integration tests for FastAPI endpoints
Run with: pytest test_api_integration.py -v
Note: Uses FastAPI TestClient (no server needed)
"""

import pytest
from fastapi.testclient import TestClient
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from app.main import app

# Create test client (will be initialized in fixture)
@pytest.fixture(scope="module")
def client():
    """Create test client"""
    return TestClient(app)


class TestHealthEndpoints:
    """Test health and info endpoints"""
    
    def test_root(self, client):
        """Test root endpoint"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert "version" in data
        assert "endpoints" in data
    
    def test_health_check(self, client):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "modules" in data
        assert data["modules"]["preprocessing"] is True
        assert data["modules"]["language_detection"] is True
        assert data["modules"]["sentiment_analysis"] is True


class TestPreprocessingEndpoint:
    """Test preprocessing endpoint"""
    
    def test_basic_preprocessing(self, client):
        """Test basic text preprocessing"""
        response = client.post(
            "/api/v1/preprocess",
            json={"text": "Hello world! This is a test."}
        )
        assert response.status_code == 200
        data = response.json()
        assert "original" in data
        assert "cleaned" in data
        assert "tokens" in data
        assert "token_count" in data
        assert len(data["tokens"]) == data["token_count"]
    
    def test_hinglish_preprocessing(self, client):
        """Test Hinglish text preprocessing"""
        response = client.post(
            "/api/v1/preprocess",
            json={"text": "Yeh bahut accha hai yaar! 😊"}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["token_count"] > 0
    
    def test_empty_text_error(self, client):
        """Test error handling for empty text"""
        response = client.post(
            "/api/v1/preprocess",
            json={"text": ""}
        )
        assert response.status_code == 422  # Validation error
    
    def test_whitespace_only_error(self, client):
        """Test error handling for whitespace only"""
        response = client.post(
            "/api/v1/preprocess",
            json={"text": "   "}
        )
        assert response.status_code == 422


class TestLanguageDetectionEndpoint:
    """Test language detection endpoint"""
    
    def test_english_detection(self, client):
        """Test English text detection"""
        response = client.post(
            "/api/v1/detect-language",
            json={"text": "Hello world this is English"}
        )
        assert response.status_code == 200
        data = response.json()
        assert "tokens" in data
        assert "labels" in data
        assert "statistics" in data
        assert "is_code_mixed" in data
        assert "dominant_language" in data
    
    def test_hindi_detection(self, client):
        """Test Hindi text detection"""
        response = client.post(
            "/api/v1/detect-language",
            json={"text": "यह बहुत अच्छा है"}  # "This is very good" in Devanagari
        )
        assert response.status_code == 200
        data = response.json()
        assert data["dominant_language"] == "lang2"
    
    def test_code_mixed_detection(self, client):
        """Test code-mixed text detection"""
        response = client.post(
            "/api/v1/detect-language",
            json={"text": "Main bahut happy hoon today"}
        )
        assert response.status_code == 200
        data = response.json()
        # Should detect both languages
        assert len(data["labels"]) > 0


class TestSentimentEndpoint:
    """Test sentiment analysis endpoint"""
    
    def test_positive_sentiment(self, client):
        """Test positive sentiment detection"""
        response = client.post(
            "/api/v1/analyze-sentiment",
            json={"text": "This is amazing and wonderful!"}
        )
        assert response.status_code == 200
        data = response.json()
        assert "label" in data
        assert "confidence" in data
        assert "scores" in data
        assert data["label"] == "positive"
        assert 0 <= data["confidence"] <= 1
    
    def test_negative_sentiment(self, client):
        """Test negative sentiment detection"""
        response = client.post(
            "/api/v1/analyze-sentiment",
            json={"text": "This is terrible and awful"}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["label"] == "negative"
    
    def test_hinglish_sentiment(self, client):
        """Test Hinglish sentiment"""
        response = client.post(
            "/api/v1/analyze-sentiment",
            json={"text": "Yeh bahut accha hai yaar!"}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["label"] in ["positive", "negative", "neutral"]


class TestFullAnalysisEndpoint:
    """Test complete analysis endpoint"""
    
    def test_full_analysis(self, client):
        """Test complete NLP analysis"""
        response = client.post(
            "/api/v1/analyze",
            json={"text": "Yeh movie bahut accha hai! I loved it!"}
        )
        assert response.status_code == 200
        data = response.json()
        
        # Check all components present
        assert "original_text" in data
        assert "cleaned_text" in data
        assert "tokens" in data
        assert "token_count" in data
        assert "language_detection" in data
        assert "sentiment" in data
        
        # Check language detection structure
        lang = data["language_detection"]
        assert "language" in lang
        assert "confidence" in lang
        assert "is_hinglish" in lang
        assert "is_indian_language" in lang
        
        # Check sentiment structure
        sent = data["sentiment"]
        assert "label" in sent
        assert "score" in sent
    
    def test_full_analysis_english(self, client):
        """Test analysis on pure English"""
        response = client.post(
            "/api/v1/analyze",
            json={"text": "This is an amazing product!"}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["sentiment"]["label"] == "positive"
    
    def test_full_analysis_with_entities(self, client):
        """Test analysis with named entities"""
        response = client.post(
            "/api/v1/analyze",
            json={"text": "Virat Kohli played amazingly well"}
        )
        assert response.status_code == 200
        data = response.json()
        # Check that language detection structure exists
        assert "language" in data["language_detection"]
        assert "confidence" in data["language_detection"]


class TestBatchEndpoint:
    """Test batch analysis endpoint"""
    
    def test_batch_analysis(self, client):
        """Test batch processing"""
        response = client.post(
            "/api/v1/analyze/batch",
            json={
                "texts": [
                    "This is amazing!",
                    "This is terrible",
                    "Yeh accha hai"
                ]
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert "count" in data
        assert "results" in data
        assert data["count"] == 3
        assert len(data["results"]) == 3
    
    def test_batch_single_item(self, client):
        """Test batch with single item"""
        response = client.post(
            "/api/v1/analyze/batch",
            json={"texts": ["Hello world"]}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 1
    
    def test_batch_empty_error(self, client):
        """Test batch with empty list"""
        response = client.post(
            "/api/v1/analyze/batch",
            json={"texts": []}
        )
        assert response.status_code == 422  # Validation error


class TestErrorHandling:
    """Test error handling"""
    
    def test_missing_field(self, client):
        """Test missing required field"""
        response = client.post("/api/v1/analyze", json={})
        assert response.status_code == 422
    
    def test_invalid_json(self, client):
        """Test invalid JSON"""
        response = client.post(
            "/api/v1/analyze",
            content=b"invalid json",
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422
    
    def test_text_too_long(self, client):
        """Test text exceeding max length"""
        long_text = "word " * 2000  # Should exceed 5000 chars
        response = client.post(
            "/api/v1/analyze",
            json={"text": long_text}
        )
        assert response.status_code == 422


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
