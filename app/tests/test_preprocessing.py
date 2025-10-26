"""
Unit tests for the preprocessing module
"""

import pytest
from app.preprocessing.cleaner import HinglishCleaner, clean_text, tokenize


class TestHinglishCleaner:
    """Test suite for HinglishCleaner class"""
    
    def test_url_removal(self):
        """Test URL removal"""
        cleaner = HinglishCleaner(remove_urls=True)
        text = "Check this out https://example.com and http://test.com"
        result = cleaner.clean_text(text)
        assert "https://example.com" not in result
        assert "http://test.com" not in result
        assert "Check this out" in result
    
    def test_mention_removal(self):
        """Test @mention removal"""
        cleaner = HinglishCleaner(remove_mentions=True)
        text = "Hey @user1 and @user2 check this"
        result = cleaner.clean_text(text)
        assert "@user1" not in result
        assert "@user2" not in result
        assert "Hey" in result
        assert "check this" in result
    
    def test_hashtag_processing(self):
        """Test hashtag # removal while keeping word"""
        cleaner = HinglishCleaner(remove_hashtags=True)
        text = "This is #awesome and #cool"
        result = cleaner.clean_text(text)
        assert "#" not in result
        assert "awesome" in result
        assert "cool" in result
    
    def test_lowercase_conversion(self):
        """Test lowercase conversion"""
        cleaner = HinglishCleaner(lowercase=True)
        text = "Hello World TESTING"
        result = cleaner.clean_text(text)
        assert result == "hello world testing"
    
    def test_emoji_preservation(self):
        """Test emoji preservation"""
        cleaner = HinglishCleaner(preserve_emojis=True)
        text = "Happy day 😊 👍"
        result = cleaner.clean_text(text)
        assert "😊" in result
        assert "👍" in result
    
    def test_punctuation_removal(self):
        """Test punctuation removal"""
        cleaner = HinglishCleaner(remove_punctuation=True)
        text = "Hello! How are you? I'm fine."
        result = cleaner.clean_text(text)
        assert "!" not in result
        assert "?" not in result
        # Apostrophe might be removed
    
    def test_number_removal(self):
        """Test number removal"""
        cleaner = HinglishCleaner(remove_numbers=True)
        text = "I have 123 apples and 456 oranges"
        result = cleaner.clean_text(text)
        assert "123" not in result
        assert "456" not in result
        assert "apples" in result
    
    def test_empty_text(self):
        """Test with empty input"""
        cleaner = HinglishCleaner()
        assert cleaner.clean_text("") == ""
        assert cleaner.clean_text(None) == ""
    
    def test_tokenization(self):
        """Test basic tokenization"""
        cleaner = HinglishCleaner()
        text = "This is a test"
        tokens = cleaner.tokenize(text)
        assert len(tokens) == 4
        assert tokens == ["This", "is", "a", "test"]
    
    def test_hinglish_text(self):
        """Test with actual Hinglish text"""
        cleaner = HinglishCleaner(
            remove_urls=True,
            remove_mentions=True,
            preserve_emojis=True
        )
        text = "Yeh bahut accha hai @user! Really nice 😊 https://test.com"
        result = cleaner.clean_text(text)
        
        assert "Yeh bahut accha hai" in result
        assert "Really nice" in result
        assert "😊" in result
        assert "@user" not in result
        assert "https://test.com" not in result
    
    def test_process_method(self):
        """Test complete process method"""
        cleaner = HinglishCleaner()
        text = "Test text here"
        result = cleaner.process(text)
        
        assert 'original' in result
        assert 'cleaned' in result
        assert 'tokens' in result
        assert 'token_count' in result
        assert result['original'] == text
        assert isinstance(result['tokens'], list)
        assert result['token_count'] == len(result['tokens'])


class TestConvenienceFunctions:
    """Test convenience functions"""
    
    def test_clean_text_function(self):
        """Test clean_text convenience function"""
        text = "Test @user https://test.com"
        result = clean_text(text, remove_urls=True, remove_mentions=True)
        assert "@user" not in result
        assert "https://test.com" not in result
        assert "Test" in result
    
    def test_tokenize_function(self):
        """Test tokenize convenience function"""
        text = "This is a test"
        tokens = tokenize(text)
        assert isinstance(tokens, list)
        assert len(tokens) > 0
    
    def test_tokenize_without_cleaning(self):
        """Test tokenization without cleaning"""
        text = "Test @user"
        tokens = tokenize(text, clean_first=False)
        # Should have @ symbol when not cleaned (tokenizes to separate '@' and 'user')
        assert any("@" in token for token in tokens)


class TestEdgeCases:
    """Test edge cases and special scenarios"""
    
    def test_very_long_text(self):
        """Test with very long text"""
        cleaner = HinglishCleaner()
        text = "word " * 1000  # 1000 words
        result = cleaner.clean_text(text)
        assert len(result) > 0
    
    def test_special_characters(self):
        """Test with special characters"""
        cleaner = HinglishCleaner()
        text = "Test with symbols: $ % ^ & * ( )"
        result = cleaner.clean_text(text)
        assert "Test with symbols" in result
    
    def test_mixed_languages(self):
        """Test with Hindi and English mixed"""
        cleaner = HinglishCleaner()
        text = "Main bahut happy hoon aaj"
        result = cleaner.clean_text(text)
        assert "Main bahut happy hoon aaj" in result
    
    def test_only_emojis(self):
        """Test with only emojis"""
        cleaner = HinglishCleaner(preserve_emojis=True)
        text = "😊 👍 ❤️"
        result = cleaner.clean_text(text)
        assert len(result) > 0
    
    def test_whitespace_handling(self):
        """Test extra whitespace removal"""
        cleaner = HinglishCleaner()
        text = "Too    many     spaces"
        result = cleaner.clean_text(text)
        assert "Too many spaces" == result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
