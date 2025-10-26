"""
Text Cleaning and Preprocessing Module
Handles Hinglish (Hindi-English code-mixed) text processing
"""

import re
import string
from typing import List, Dict, Tuple
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Download required NLTK data (will skip if already downloaded)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)


class HinglishCleaner:
    """
    Text cleaner optimized for Hinglish (Hindi-English code-mixed) text
    """
    
    def __init__(
        self,
        remove_urls: bool = True,
        remove_mentions: bool = False,
        remove_hashtags: bool = False,
        remove_punctuation: bool = False,
        lowercase: bool = False,
        remove_numbers: bool = False,
        preserve_emojis: bool = True
    ):
        """
        Initialize the cleaner with processing options
        
        Args:
            remove_urls: Remove URLs from text
            remove_mentions: Remove @mentions
            remove_hashtags: Remove #hashtags (keeps the word)
            remove_punctuation: Remove punctuation marks
            lowercase: Convert to lowercase
            remove_numbers: Remove numeric characters
            preserve_emojis: Keep emojis in text
        """
        self.remove_urls = remove_urls
        self.remove_mentions = remove_mentions
        self.remove_hashtags = remove_hashtags
        self.remove_punctuation = remove_punctuation
        self.lowercase = lowercase
        self.remove_numbers = remove_numbers
        self.preserve_emojis = preserve_emojis
        
        # Regex patterns
        self.url_pattern = re.compile(
            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        )
        self.mention_pattern = re.compile(r'@\w+')
        self.hashtag_pattern = re.compile(r'#(\w+)')
        
        # Emoji pattern (Unicode ranges)
        self.emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F680-\U0001F6FF"  # transport & map symbols
            "\U0001F1E0-\U0001F1FF"  # flags
            "\U00002702-\U000027B0"
            "\U000024C2-\U0001F251"
            "]+",
            flags=re.UNICODE
        )
    
    def clean_text(self, text: str) -> str:
        """
        Clean the input text based on initialization parameters
        
        Args:
            text: Input text to clean
            
        Returns:
            Cleaned text string
        """
        if not text or not isinstance(text, str):
            return ""
        
        # Store emojis if preserving
        emojis = []
        if self.preserve_emojis:
            emojis = self.emoji_pattern.findall(text)
        
        # Remove URLs
        if self.remove_urls:
            text = self.url_pattern.sub(' ', text)
        
        # Remove mentions
        if self.remove_mentions:
            text = self.mention_pattern.sub(' ', text)
        
        # Remove hashtag symbols but keep words
        if self.remove_hashtags:
            text = self.hashtag_pattern.sub(r'\1', text)
        
        # Remove numbers
        if self.remove_numbers:
            text = re.sub(r'\d+', ' ', text)
        
        # Remove punctuation
        if self.remove_punctuation:
            if not self.preserve_emojis:
                text = text.translate(str.maketrans('', '', string.punctuation))
            else:
                # Remove punctuation but not emojis
                text = re.sub(f'[{re.escape(string.punctuation)}]', ' ', text)
        
        # Convert to lowercase
        if self.lowercase:
            text = text.lower()
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Re-append emojis if preserved
        if self.preserve_emojis and emojis:
            text = text + ' ' + ' '.join(emojis)
        
        return text.strip()
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into words
        
        Args:
            text: Input text to tokenize
            
        Returns:
            List of tokens
        """
        if not text:
            return []
        
        try:
            tokens = word_tokenize(text)
            return [token for token in tokens if token.strip()]
        except Exception:
            # Fallback to simple split if NLTK fails
            return text.split()
    
    def process(self, text: str) -> Dict[str, any]:
        """
        Complete processing pipeline: clean and tokenize
        
        Args:
            text: Input text to process
            
        Returns:
            Dictionary with original, cleaned text, and tokens
        """
        cleaned = self.clean_text(text)
        tokens = self.tokenize(cleaned)
        
        return {
            'original': text,
            'cleaned': cleaned,
            'tokens': tokens,
            'token_count': len(tokens)
        }


def clean_text(
    text: str,
    remove_urls: bool = True,
    remove_mentions: bool = False,
    remove_hashtags: bool = False,
    remove_punctuation: bool = False,
    lowercase: bool = False
) -> str:
    """
    Convenience function for quick text cleaning
    
    Args:
        text: Text to clean
        remove_urls: Remove URLs
        remove_mentions: Remove @mentions
        remove_hashtags: Remove # from hashtags
        remove_punctuation: Remove punctuation
        lowercase: Convert to lowercase
        
    Returns:
        Cleaned text
    """
    cleaner = HinglishCleaner(
        remove_urls=remove_urls,
        remove_mentions=remove_mentions,
        remove_hashtags=remove_hashtags,
        remove_punctuation=remove_punctuation,
        lowercase=lowercase
    )
    return cleaner.clean_text(text)


def tokenize(text: str, clean_first: bool = True) -> List[str]:
    """
    Convenience function for quick tokenization
    
    Args:
        text: Text to tokenize
        clean_first: Apply basic cleaning before tokenizing
        
    Returns:
        List of tokens
    """
    cleaner = HinglishCleaner()
    if clean_first:
        text = cleaner.clean_text(text)
    return cleaner.tokenize(text)


if __name__ == "__main__":
    # Test the cleaner
    test_texts = [
        "Hey @user check this out! https://example.com #awesome 😊",
        "Yeh bahut accha hai! Really nice vibe 👍",
        "RT @someone: This is a test tweet with URLs http://test.com"
    ]
    
    cleaner = HinglishCleaner(
        remove_urls=True,
        remove_mentions=True,
        remove_hashtags=True,
        preserve_emojis=True
    )
    
    print("=" * 60)
    print("HINGLISH TEXT CLEANER - TEST")
    print("=" * 60)
    
    for i, text in enumerate(test_texts, 1):
        result = cleaner.process(text)
        print(f"\n[Test {i}]")
        print(f"Original:  {result['original']}")
        print(f"Cleaned:   {result['cleaned']}")
        print(f"Tokens:    {result['tokens']}")
        print(f"Count:     {result['token_count']}")
