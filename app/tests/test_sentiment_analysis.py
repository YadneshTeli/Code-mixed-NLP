"""
Unit tests for sentiment analysis module
"""

import pytest
from app.sentiment_analysis.analyzer import SentimentAnalyzer, analyze_sentiment


class TestSentimentAnalyzer:
    """Test suite for SentimentAnalyzer class"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.analyzer = SentimentAnalyzer()
    
    def test_positive_sentiment(self):
        """Test positive sentiment detection"""
        result = self.analyzer.analyze("This is amazing and wonderful!")
        assert result['sentiment'] == 'positive'
        assert result['confidence'] > 0.5
    
    def test_negative_sentiment(self):
        """Test negative sentiment detection"""
        result = self.analyzer.analyze("This is terrible and awful")
        assert result['sentiment'] == 'negative'
        assert result['confidence'] > 0.5
    
    def test_neutral_sentiment(self):
        """Test neutral sentiment detection"""
        result = self.analyzer.analyze("This is a sentence")
        # Should return some sentiment
        assert result['sentiment'] in ['positive', 'negative', 'neutral']
    
    def test_hinglish_positive(self):
        """Test positive Hinglish text"""
        result = self.analyzer.analyze("Yeh bahut accha hai yaar!")
        assert result['sentiment'] == 'positive'
    
    def test_hinglish_negative(self):
        """Test negative Hinglish text"""
        result = self.analyzer.analyze("Bahut bura tha")
        assert result['sentiment'] == 'negative'
    
    def test_result_structure(self):
        """Test result dictionary structure"""
        result = self.analyzer.analyze("Test text")
        
        assert 'text' in result
        assert 'sentiment' in result
        assert 'scores' in result
        assert 'confidence' in result
        
        assert result['text'] == "Test text"
        assert isinstance(result['scores'], dict)
        assert isinstance(result['confidence'], (int, float))
    
    def test_scores_structure(self):
        """Test scores dictionary"""
        result = self.analyzer.analyze("Test text")
        scores = result['scores']
        
        # Should have sentiment scores
        assert len(scores) >= 2  # At least positive and negative
        
        # All scores should be between 0 and 1
        for score in scores.values():
            assert 0 <= score <= 1
    
    def test_empty_text(self):
        """Test with empty input"""
        result = self.analyzer.analyze("")
        assert result['sentiment'] == 'neutral'
    
    def test_whitespace_only(self):
        """Test with whitespace only"""
        result = self.analyzer.analyze("   ")
        assert result['sentiment'] == 'neutral'
    
    def test_get_sentiment_label(self):
        """Test convenience method for getting just the label"""
        label = self.analyzer.get_sentiment_label("This is amazing!")
        assert label in ['positive', 'negative', 'neutral']
        assert isinstance(label, str)
    
    def test_batch_analysis(self):
        """Test batch sentiment analysis"""
        texts = [
            "This is great!",
            "This is terrible",
            "This is okay"
        ]
        results = self.analyzer.analyze_batch(texts)
        
        assert len(results) == 3
        assert all('sentiment' in r for r in results)
        assert results[0]['sentiment'] == 'positive'
        assert results[1]['sentiment'] == 'negative'
    
    def test_long_text(self):
        """Test with long text"""
        long_text = "This is amazing! " * 50
        result = self.analyzer.analyze(long_text)
        assert result['sentiment'] == 'positive'
    
    def test_mixed_sentiment(self):
        """Test text with mixed sentiments"""
        result = self.analyzer.analyze("This is great but also terrible")
        # Should return some sentiment
        assert result['sentiment'] in ['positive', 'negative', 'neutral']
    
    def test_special_characters(self):
        """Test with special characters"""
        result = self.analyzer.analyze("Amazing!!! 😊 👍")
        assert result['sentiment'] == 'positive'
    
    def test_numbers(self):
        """Test with numbers"""
        result = self.analyzer.analyze("This is 100% amazing")
        assert result['sentiment'] == 'positive'


class TestConvenienceFunction:
    """Test convenience function"""
    
    def test_analyze_sentiment_function(self):
        """Test standalone analyze_sentiment function"""
        result = analyze_sentiment("This is wonderful!")
        
        assert 'text' in result
        assert 'sentiment' in result
        assert 'scores' in result
        assert 'confidence' in result
    
    def test_custom_model(self):
        """Test with custom model name"""
        # Should not crash (may fall back to rule-based)
        result = analyze_sentiment("Test", model_name="distilbert-base-uncased-finetuned-sst-2-english")
        assert 'sentiment' in result


class TestEdgeCases:
    """Test edge cases"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.analyzer = SentimentAnalyzer()
    
    def test_single_word_positive(self):
        """Test single positive word"""
        result = self.analyzer.analyze("amazing")
        assert result['sentiment'] == 'positive'
    
    def test_single_word_negative(self):
        """Test single negative word"""
        result = self.analyzer.analyze("terrible")
        assert result['sentiment'] == 'negative'
    
    def test_punctuation_only(self):
        """Test with only punctuation"""
        result = self.analyzer.analyze("!!! ... ???")
        assert result['sentiment'] in ['positive', 'negative', 'neutral']
    
    def test_uppercase_text(self):
        """Test with all uppercase"""
        result = self.analyzer.analyze("THIS IS AMAZING!")
        assert result['sentiment'] == 'positive'
    
    def test_lowercase_text(self):
        """Test with all lowercase"""
        result = self.analyzer.analyze("this is amazing")
        assert result['sentiment'] == 'positive'
    
    def test_repeated_words(self):
        """Test with repeated words"""
        result = self.analyzer.analyze("amazing amazing amazing")
        assert result['sentiment'] == 'positive'
    
    def test_negation(self):
        """Test negation handling"""
        result = self.analyzer.analyze("not bad")
        # Behavior may vary based on model
        assert result['sentiment'] in ['positive', 'negative', 'neutral']
    
    def test_sarcasm(self):
        """Test sarcasm (difficult case)"""
        result = self.analyzer.analyze("Oh great, just what I needed")
        # Sarcasm detection is hard, just check it doesn't crash
        assert result['sentiment'] in ['positive', 'negative', 'neutral']
    
    def test_multiple_sentences(self):
        """Test with multiple sentences"""
        result = self.analyzer.analyze("This is great. I love it. Amazing stuff.")
        assert result['sentiment'] == 'positive'
    
    def test_code_mixed_intensity(self):
        """Test code-mixed text with intensity"""
        result = self.analyzer.analyze("Ekdum zabardast movie hai!")
        # Should detect positive sentiment
        assert result['sentiment'] in ['positive', 'neutral']


class TestConfidence:
    """Test confidence scores"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.analyzer = SentimentAnalyzer()
    
    def test_confidence_range(self):
        """Test confidence is in valid range"""
        result = self.analyzer.analyze("This is amazing")
        assert 0 <= result['confidence'] <= 1
    
    def test_high_confidence_positive(self):
        """Test high confidence for clear positive"""
        result = self.analyzer.analyze("Absolutely amazing and wonderful!")
        # Strong positive should have higher confidence
        assert result['sentiment'] == 'positive'
        assert result['confidence'] > 0.6
    
    def test_high_confidence_negative(self):
        """Test high confidence for clear negative"""
        result = self.analyzer.analyze("Absolutely terrible and awful!")
        # Strong negative should have higher confidence
        assert result['sentiment'] == 'negative'
        assert result['confidence'] > 0.6


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
