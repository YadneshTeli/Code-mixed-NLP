"""
Tests for new V2 API endpoints
Tests the enhanced V2 endpoints with multilingual support
"""

import pytest
from fastapi.testclient import TestClient
from app.main import app


@pytest.fixture
def client():
    """Create a test client"""
    return TestClient(app)


class TestV2PreprocessingEndpoint:
    """Test V2 preprocessing endpoint"""
    
    def test_basic_preprocessing(self, client):
        """Test basic text preprocessing"""
        response = client.post(
            "/api/v2/preprocess",
            json={"text": "Hello, World! This is a test."}
        )
        assert response.status_code == 200
        data = response.json()
        
        # Check response structure (V2 uses shorter field names)
        assert "original" in data
        assert "processed" in data
        assert "tokens" in data
        assert "tokens_count" in data
        assert "sentence_count" in data
        assert "preprocessing_method" in data
        
        # Verify data
        assert data["original"] == "Hello, World! This is a test."
        assert isinstance(data["tokens"], list)
        assert data["tokens_count"] > 0
        assert data["sentence_count"] >= 1
    
    def test_hinglish_preprocessing(self, client):
        """Test preprocessing with Hinglish text"""
        response = client.post(
            "/api/v2/preprocess",
            json={"text": "Yaar, kal meeting bahut tight thi! 😅"}
        )
        assert response.status_code == 200
        data = response.json()
        
        assert data["original"] == "Yaar, kal meeting bahut tight thi! 😅"
        assert len(data["tokens"]) > 0
        assert data["tokens_count"] > 0
    
    def test_devanagari_preprocessing(self, client):
        """Test preprocessing with Devanagari script"""
        response = client.post(
            "/api/v2/preprocess",
            json={"text": "यह एक परीक्षण है"}
        )
        assert response.status_code == 200
        data = response.json()
        
        assert data["original"] == "यह एक परीक्षण है"
        assert len(data["tokens"]) > 0
    
    def test_empty_text_error(self, client):
        """Test error handling for empty text"""
        response = client.post(
            "/api/v2/preprocess",
            json={"text": ""}
        )
        assert response.status_code == 422
    
    def test_whitespace_only_error(self, client):
        """Test error handling for whitespace-only text"""
        response = client.post(
            "/api/v2/preprocess",
            json={"text": "   "}
        )
        assert response.status_code == 422


class TestV2LanguageDetectionEndpoint:
    """Test V2 language detection endpoint"""
    
    def test_english_detection(self, client):
        """Test English language detection"""
        response = client.post(
            "/api/v2/detect-language",
            json={"text": "This is a simple English sentence."}
        )
        assert response.status_code == 200
        data = response.json()
        
        # Check response structure
        assert "detected_language" in data
        assert "language_name" in data
        assert "confidence" in data
        assert "is_hinglish" in data
        assert "is_reliable" in data
        assert "token_level_detection" in data
        
        # Check detection
        assert data["detected_language"] == "en"
        assert data["language_name"] == "English"
        assert data["confidence"] > 0.8
        assert data["is_reliable"] is True
        
        # Check token-level data
        token_data = data["token_level_detection"]
        assert "tokens" in token_data
        assert "labels" in token_data
        assert "statistics" in token_data
        assert isinstance(token_data["tokens"], list)
        assert len(token_data["tokens"]) > 0
    
    def test_hindi_detection(self, client):
        """Test Hindi language detection"""
        response = client.post(
            "/api/v2/detect-language",
            json={"text": "यह एक हिंदी वाक्य है"}
        )
        assert response.status_code == 200
        data = response.json()
        
        assert data["detected_language"] == "hi"
        assert data["language_name"] == "Hindi"
        assert data["confidence"] > 0.8
    
    def test_hinglish_detection(self, client):
        """Test Hinglish code-mixed detection"""
        response = client.post(
            "/api/v2/detect-language",
            json={"text": "Yaar kal meeting bahut important thi"}
        )
        assert response.status_code == 200
        data = response.json()
        
        # Should detect as Hinglish (code-mixed)
        assert "is_hinglish" in data
        # Token-level should show mixed languages
        token_data = data["token_level_detection"]
        assert len(token_data["labels"]) > 0
    
    def test_multilingual_detection(self, client):
        """Test detection of other languages"""
        test_cases = [
            ("Bonjour le monde", "fr"),  # French
            ("Hola mundo", "es"),  # Spanish
            ("Hallo Welt", "de"),  # German
        ]
        
        for text, expected_lang in test_cases:
            response = client.post(
                "/api/v2/detect-language",
                json={"text": text}
            )
            assert response.status_code == 200
            data = response.json()
            assert data["detected_language"] == expected_lang
    
    def test_empty_text_error(self, client):
        """Test error handling for empty text"""
        response = client.post(
            "/api/v2/detect-language",
            json={"text": ""}
        )
        assert response.status_code == 422


class TestV2SentimentAnalysisEndpoint:
    """Test V2 sentiment analysis endpoint"""
    
    def test_positive_sentiment_english(self, client):
        """Test positive sentiment in English"""
        response = client.post(
            "/api/v2/analyze-sentiment",
            json={"text": "This product is absolutely amazing! I love it!"}
        )
        assert response.status_code == 200
        data = response.json()
        
        # Check response structure
        assert "sentiment" in data
        assert "confidence" in data
        assert "confidence_level" in data
        assert "scores" in data
        assert "model_used" in data
        assert "route" in data
        
        # Check sentiment
        assert data["sentiment"] == "positive"
        assert data["confidence"] > 0.7
        assert data["confidence_level"] in ["low", "medium", "high"]
        
        # Check scores
        scores = data["scores"]
        assert "positive" in scores
        assert "negative" in scores
        assert scores["positive"] > scores["negative"]
    
    def test_negative_sentiment_english(self, client):
        """Test negative sentiment in English"""
        response = client.post(
            "/api/v2/analyze-sentiment",
            json={"text": "This is terrible and disappointing."}
        )
        assert response.status_code == 200
        data = response.json()
        
        assert data["sentiment"] == "negative"
        assert data["scores"]["negative"] > data["scores"]["positive"]
    
    def test_neutral_sentiment(self, client):
        """Test neutral sentiment"""
        response = client.post(
            "/api/v2/analyze-sentiment",
            json={"text": "The sky is blue."}
        )
        assert response.status_code == 200
        data = response.json()
        
        # Should be neutral or have similar positive/negative scores
        assert data["sentiment"] in ["neutral", "positive", "negative"]
    
    def test_hinglish_sentiment(self, client):
        """Test sentiment analysis for Hinglish text"""
        response = client.post(
            "/api/v2/analyze-sentiment",
            json={"text": "Yaar yeh movie bahut acchi hai! Must watch!"}
        )
        assert response.status_code == 200
        data = response.json()
        
        # Should use CM-BERT for Hinglish
        assert data["sentiment"] in ["positive", "negative", "neutral"]
        assert "model_used" in data
        assert data["confidence"] > 0.0
    
    def test_hindi_sentiment(self, client):
        """Test sentiment analysis for Hindi text"""
        response = client.post(
            "/api/v2/analyze-sentiment",
            json={"text": "यह बहुत अच्छा है"}
        )
        assert response.status_code == 200
        data = response.json()
        
        assert data["sentiment"] in ["positive", "negative", "neutral"]
    
    def test_multilingual_sentiment(self, client):
        """Test sentiment for other languages"""
        response = client.post(
            "/api/v2/analyze-sentiment",
            json={"text": "C'est magnifique!"}  # French: "It's magnificent!"
        )
        assert response.status_code == 200
        data = response.json()
        
        # Should use XLM-RoBERTa for non-Hindi/English
        assert data["sentiment"] in ["positive", "negative", "neutral"]
        assert "model_used" in data
    
    def test_route_information(self, client):
        """Test that route information is provided"""
        response = client.post(
            "/api/v2/analyze-sentiment",
            json={"text": "This is great!"}
        )
        assert response.status_code == 200
        data = response.json()
        
        assert "route" in data
        assert data["route"] in ["hinglish", "multilingual", "english"]
    
    def test_confidence_levels(self, client):
        """Test confidence level categorization"""
        response = client.post(
            "/api/v2/analyze-sentiment",
            json={"text": "This is excellent!"}
        )
        assert response.status_code == 200
        data = response.json()
        
        assert data["confidence_level"] in ["low", "medium", "high"]
        
        # High confidence should have confidence > 0.8
        if data["confidence"] > 0.8:
            assert data["confidence_level"] == "high"
    
    def test_empty_text_error(self, client):
        """Test error handling for empty text"""
        response = client.post(
            "/api/v2/analyze-sentiment",
            json={"text": ""}
        )
        assert response.status_code == 422


class TestV2ErrorHandling:
    """Test V2 error handling"""
    
    def test_missing_text_field(self, client):
        """Test error when text field is missing"""
        response = client.post(
            "/api/v2/preprocess",
            json={}
        )
        assert response.status_code == 422
    
    def test_invalid_json(self, client):
        """Test error with invalid JSON"""
        response = client.post(
            "/api/v2/preprocess",
            data="not json",
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422
    
    def test_text_too_long(self, client):
        """Test error handling for very long text"""
        # Create a very long text (over 10000 chars)
        long_text = "word " * 3000
        
        response = client.post(
            "/api/v2/preprocess",
            json={"text": long_text}
        )
        assert response.status_code == 422


class TestV2Compatibility:
    """Test V2 endpoints return expected formats"""
    
    def test_preprocessing_response_format(self, client):
        """Verify V2 preprocessing response format"""
        response = client.post(
            "/api/v2/preprocess",
            json={"text": "Test text"}
        )
        data = response.json()
        
        # V2 uses shorter field names for efficiency
        required_fields = [
            "original",
            "processed",
            "tokens",
            "tokens_count",
            "sentence_count",
            "preprocessing_method"
        ]
        
        for field in required_fields:
            assert field in data, f"Missing field: {field}"
    
    def test_language_detection_response_format(self, client):
        """Verify V2 language detection response format"""
        response = client.post(
            "/api/v2/detect-language",
            json={"text": "Test text"}
        )
        data = response.json()
        
        required_fields = [
            "detected_language",
            "language_name",
            "confidence",
            "is_hinglish",
            "is_reliable",
            "token_level_detection"
        ]
        
        for field in required_fields:
            assert field in data, f"Missing field: {field}"
        
        # Check token-level structure
        token_data = data["token_level_detection"]
        assert "tokens" in token_data
        assert "labels" in token_data
        assert "statistics" in token_data
    
    def test_sentiment_response_format(self, client):
        """Verify V2 sentiment response format"""
        response = client.post(
            "/api/v2/analyze-sentiment",
            json={"text": "Test text"}
        )
        data = response.json()
        
        required_fields = [
            "sentiment",
            "confidence",
            "confidence_level",
            "scores",
            "model_used",
            "route"
        ]
        
        for field in required_fields:
            assert field in data, f"Missing field: {field}"
        
        # Check scores structure
        scores = data["scores"]
        assert "positive" in scores
        assert "negative" in scores
