"""
Unit tests for language detection module
"""

import pytest
from app.language_detection.detector import LanguageDetector, detect_language


class TestLanguageDetector:
    """Test suite for LanguageDetector class"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.detector = LanguageDetector()
    
    def test_english_detection(self):
        """Test pure English text"""
        result = self.detector.detect_text("Hello world how are you")
        assert 'lang1' in [l for l in result['labels']]
        assert result['is_code_mixed'] == False
    
    def test_hindi_detection(self):
        """Test pure Hindi (romanized) text"""
        result = self.detector.detect_text("Yeh bahut accha hai")
        assert all(l == 'lang2' for l in result['labels'])
        assert result['is_code_mixed'] == False
    
    def test_code_mixed_detection(self):
        """Test code-mixed Hinglish text"""
        result = self.detector.detect_text("Main bahut happy hoon")
        assert result['is_code_mixed'] == True
        assert 'lang1' in result['labels']
        assert 'lang2' in result['labels']
    
    def test_named_entity_detection(self):
        """Test named entity recognition"""
        result = self.detector.detect_text("Virat Kohli is playing")
        assert 'ne' in result['labels']
    
    def test_punctuation_detection(self):
        """Test punctuation classification"""
        result = self.detector.detect_text("Hello , world !")
        labels = result['labels']
        # Comma and exclamation should be 'other' (when separated by spaces)
        assert any(l == 'other' for l in labels)
    
    def test_token_language_detection(self):
        """Test single token detection"""
        assert self.detector.detect_token_language("hai") == 'lang2'
        assert self.detector.detect_token_language("the") == 'lang1'
        assert self.detector.detect_token_language("!") == 'other'
    
    def test_empty_text(self):
        """Test with empty input"""
        result = self.detector.detect_text("")
        assert result['tokens'] == []
        assert result['labels'] == []
    
    def test_sentence_detection(self):
        """Test sentence-level detection"""
        tokens = ["Hello", "yaar", "how", "are", "you"]
        labels = self.detector.detect_sentence(tokens)
        assert len(labels) == len(tokens)
        assert labels[1] == 'lang2'  # 'yaar' is Hindi
    
    def test_dominant_language_english(self):
        """Test dominant language detection for English"""
        dominant = self.detector.get_dominant_language("Hello world this is English")
        assert dominant == 'lang1'
    
    def test_dominant_language_hindi(self):
        """Test dominant language detection for Hindi"""
        dominant = self.detector.get_dominant_language("Yeh bahut accha hai boss")
        assert dominant == 'lang2'
    
    def test_statistics(self):
        """Test statistics calculation"""
        result = self.detector.detect_text("Hello world yaar kaise ho")
        stats = result['statistics']
        assert 'lang1' in stats or 'ne' in stats  # Hello/world could be NE
        assert 'lang2' in stats  # yaar, kaise, ho are Hindi
        assert 'count' in stats[list(stats.keys())[0]]
        assert 'percentage' in stats[list(stats.keys())[0]]
    
    def test_hindi_word_list(self):
        """Test common Hindi words are detected"""
        hindi_words = ["hai", "hain", "kya", "kaise", "yaar", "bhai"]
        for word in hindi_words:
            assert self.detector.detect_token_language(word) == 'lang2'
    
    def test_english_stopwords(self):
        """Test English stopwords are detected"""
        english_words = ["the", "is", "are", "was", "were", "this", "that"]
        for word in english_words:
            assert self.detector.detect_token_language(word) == 'lang1'
    
    def test_devanagari_script(self):
        """Test Devanagari characters"""
        result = self.detector.detect_token_language("नमस्ते")
        assert result == 'lang2'
    
    def test_numbers(self):
        """Test number classification"""
        result = self.detector.detect_token_language("123")
        assert result == 'other'
    
    def test_capitalized_words(self):
        """Test capitalized word detection"""
        # Common names should be detected as named entities
        assert self.detector.detect_token_language("Modi") == 'ne'
        assert self.detector.detect_token_language("India") == 'ne'


class TestConvenienceFunction:
    """Test convenience function"""
    
    def test_detect_language_function(self):
        """Test standalone detect_language function"""
        result = detect_language("Hello world")
        assert 'tokens' in result
        assert 'labels' in result
        assert 'statistics' in result


class TestEdgeCases:
    """Test edge cases"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.detector = LanguageDetector()
    
    def test_single_word(self):
        """Test with single word"""
        result = self.detector.detect_text("hello")
        assert len(result['tokens']) == 1
        assert len(result['labels']) == 1
    
    def test_mixed_case(self):
        """Test mixed case handling"""
        result1 = self.detector.detect_text("Hello")
        result2 = self.detector.detect_text("HELLO")
        result3 = self.detector.detect_text("hello")
        # All should detect as some form of English
        assert all('lang1' in r['labels'] or 'ne' in r['labels'] for r in [result1, result2, result3])
    
    def test_special_characters(self):
        """Test special characters"""
        result = self.detector.detect_text("@user #hashtag")
        assert len(result['tokens']) == 2
    
    def test_multiple_spaces(self):
        """Test text with multiple spaces"""
        result = self.detector.detect_text("hello    world")
        # Should handle multiple spaces gracefully
        assert len(result['tokens']) == 2
    
    def test_only_punctuation(self):
        """Test text with only punctuation"""
        result = self.detector.detect_text("... !!! ???")
        assert all(l == 'other' for l in result['labels'])
    
    def test_mixed_script(self):
        """Test mixed script (Devanagari + Roman)"""
        result = self.detector.detect_text("Hello नमस्ते world")
        assert len(result['tokens']) == 3
        assert 'lang2' in result['labels']  # Devanagari
        assert 'lang1' in result['labels'] or 'ne' in result['labels']  # English


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
