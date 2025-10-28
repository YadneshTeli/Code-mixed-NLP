"""
Hybrid Text Preprocessor combining NLTK and spaCy
Optimized for code-mixed Hinglish text processing
"""

import re
from typing import List, Dict, Optional
import nltk
from nltk.corpus import stopwords

# Lazy import spaCy (will be loaded on first use)
_spacy_nlp = None


def get_spacy_nlp():
    """Lazy load spaCy model"""
    global _spacy_nlp
    if _spacy_nlp is None:
        try:
            import spacy
            # Load small English model with minimal components for speed
            _spacy_nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
            # Add sentencizer for sentence boundary detection
            if 'sentencizer' not in _spacy_nlp.pipe_names:
                _spacy_nlp.add_pipe('sentencizer')
            print("✅ spaCy model loaded successfully")
        except OSError:
            print("⚠️  spaCy model not found. Install with: python -m spacy download en_core_web_sm")
            raise
    return _spacy_nlp


class HybridPreprocessor:
    """
    Hybrid preprocessor combining NLTK and spaCy for optimal performance
    - spaCy: Fast tokenization, sentence splitting
    - NLTK: Stopwords, text cleaning utilities
    - Custom: Code-mixed text handling
    """
    
    def __init__(
        self,
        remove_urls: bool = True,
        remove_mentions: bool = False,
        remove_hashtags: bool = False,
        remove_punctuation: bool = False,
        lowercase: bool = True,
        remove_stopwords: bool = False,
        preserve_emojis: bool = True
    ):
        """
        Initialize hybrid preprocessor
        
        Args:
            remove_urls: Remove URLs from text
            remove_mentions: Remove @mentions
            remove_hashtags: Remove #hashtags
            remove_punctuation: Remove punctuation marks
            lowercase: Convert to lowercase
            remove_stopwords: Remove English and Hindi stopwords
            preserve_emojis: Keep emojis in text
        """
        self.remove_urls = remove_urls
        self.remove_mentions = remove_mentions
        self.remove_hashtags = remove_hashtags
        self.remove_punctuation = remove_punctuation
        self.lowercase = lowercase
        self.remove_stopwords = remove_stopwords
        self.preserve_emojis = preserve_emojis
        
        # Download NLTK data if needed
        self._ensure_nltk_data()
        
        # English stopwords from NLTK
        self.stop_words_en = set(stopwords.words('english'))
        
        # Hindi stopwords (custom list - common words)
        self.stop_words_hi = {
            # Devanagari
            'का', 'के', 'की', 'को', 'में', 'से', 'पर', 'और', 'है', 'हैं',
            'था', 'थी', 'थे', 'हो', 'ही', 'भी', 'कि', 'जो', 'यह', 'वह',
            'ने', 'तो', 'एक', 'दो', 'तीन', 'चार', 'पांच',
            # Romanized Hindi
            'ka', 'ke', 'ki', 'ko', 'mein', 'main', 'se', 'par', 'aur',
            'hai', 'hain', 'tha', 'thi', 'the', 'ho', 'hi', 'bhi',
            'ki', 'jo', 'yeh', 'ye', 'woh', 'wo', 'ne', 'toh', 'to',
            'ek', 'do', 'teen', 'char', 'paanch'
        }
        
        # Combined stopwords
        self.all_stopwords = self.stop_words_en | self.stop_words_hi
        
        # Regex patterns
        self.url_pattern = re.compile(
            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        )
        self.mention_pattern = re.compile(r'@\w+')
        self.hashtag_pattern = re.compile(r'#\w+')
        self.emoji_pattern = re.compile(
            "["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags
            u"\U00002702-\U000027B0"
            u"\U000024C2-\U0001F251"
            "]+",
            flags=re.UNICODE
        )
        self.devanagari_pattern = re.compile(r'[\u0900-\u097F]+')
        
        print("🔧 Hybrid Preprocessor initialized")
    
    def _ensure_nltk_data(self):
        """Download required NLTK data"""
        try:
            stopwords.words('english')
        except LookupError:
            nltk.download('stopwords', quiet=True)
    
    def is_devanagari(self, text: str) -> bool:
        """Check if text contains Devanagari script"""
        return bool(self.devanagari_pattern.search(text))
    
    def is_hinglish(self, text: str) -> bool:
        """Check if text is code-mixed Hinglish"""
        has_hindi = self.is_devanagari(text)
        has_english = bool(re.search(r'[a-zA-Z]', text))
        return has_hindi and has_english
    
    def clean_text(self, text: str) -> str:
        """
        Clean text using custom rules
        
        Args:
            text: Input text to clean
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        cleaned = text
        
        # Remove URLs
        if self.remove_urls:
            cleaned = self.url_pattern.sub('', cleaned)
        
        # Remove mentions
        if self.remove_mentions:
            cleaned = self.mention_pattern.sub('', cleaned)
        
        # Remove hashtag symbol but keep the word
        if self.remove_hashtags:
            cleaned = self.hashtag_pattern.sub(lambda m: m.group(0)[1:], cleaned)
        
        # Extract and preserve emojis
        emojis = []
        if self.preserve_emojis:
            emojis = self.emoji_pattern.findall(cleaned)
            cleaned = self.emoji_pattern.sub(' EMOJI ', cleaned)
        
        # Remove extra whitespace
        cleaned = ' '.join(cleaned.split())
        
        # Restore emojis
        if self.preserve_emojis and emojis:
            for emoji in emojis:
                cleaned = cleaned.replace('EMOJI', emoji, 1)
        
        return cleaned.strip()
    
    def tokenize_spacy(self, text: str) -> List[str]:
        """
        Fast tokenization using spaCy
        
        Args:
            text: Text to tokenize
            
        Returns:
            List of tokens
        """
        nlp = get_spacy_nlp()
        doc = nlp(text)
        return [token.text for token in doc]
    
    def sentence_split(self, text: str) -> List[str]:
        """
        Split text into sentences using spaCy
        
        Args:
            text: Text to split
            
        Returns:
            List of sentences
        """
        nlp = get_spacy_nlp()
        doc = nlp(text)
        return [sent.text.strip() for sent in doc.sents]
    
    def remove_stopwords_from_tokens(self, tokens: List[str]) -> List[str]:
        """
        Remove stopwords from token list
        
        Args:
            tokens: List of tokens
            
        Returns:
            Filtered tokens
        """
        if not self.remove_stopwords:
            return tokens
        
        return [
            token for token in tokens
            if token.lower() not in self.all_stopwords
        ]
    
    def preprocess(self, text: str) -> Dict:
        """
        Complete preprocessing pipeline
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with preprocessing results
        """
        if not text or not text.strip():
            return {
                'original_text': text,
                'cleaned_text': '',
                'tokens': [],
                'filtered_tokens': [],
                'sentence_count': 0,
                'language': 'unknown',
                'is_hinglish': False,
                'has_devanagari': False
            }
        
        # Detect language
        from app.language_detection.fasttext_detector import FastTextDetector
        detector = FastTextDetector()
        lang_result = detector.detect(text)
        
        # Step 1: Clean text
        cleaned = self.clean_text(text)
        
        # Step 2: Lowercase if needed
        if self.lowercase:
            # Preserve Devanagari script (don't lowercase)
            if not self.is_devanagari(cleaned):
                cleaned = cleaned.lower()
        
        # Step 3: Tokenize using spaCy (fast)
        tokens = self.tokenize_spacy(cleaned)
        
        # Step 4: Remove punctuation if needed
        if self.remove_punctuation:
            tokens = [
                token for token in tokens
                if not all(c in '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~' for c in token)
            ]
        
        # Step 5: Remove stopwords if needed
        filtered_tokens = self.remove_stopwords_from_tokens(tokens)
        
        # Step 6: Get sentence count
        sentence_count = len(self.sentence_split(text))
        
        return {
            'original_text': text,
            'cleaned_text': cleaned,
            'tokens': tokens,
            'filtered_tokens': filtered_tokens,
            'token_count': len(tokens),
            'filtered_token_count': len(filtered_tokens),
            'sentence_count': sentence_count,
            'language': lang_result.get('language', 'unknown'),
            'language_confidence': lang_result.get('confidence', 0.0),
            'is_hinglish': lang_result.get('is_hinglish', False),
            'has_devanagari': self.is_devanagari(text)
        }
    
    def preprocess_batch(self, texts: List[str]) -> List[Dict]:
        """
        Preprocess multiple texts
        
        Args:
            texts: List of texts to preprocess
            
        Returns:
            List of preprocessing results
        """
        return [self.preprocess(text) for text in texts]


# Convenience function
def preprocess_text(
    text: str,
    remove_urls: bool = True,
    lowercase: bool = True,
    preserve_emojis: bool = True
) -> Dict:
    """
    Quick preprocessing function
    
    Args:
        text: Text to preprocess
        remove_urls: Remove URLs
        lowercase: Convert to lowercase
        preserve_emojis: Keep emojis
        
    Returns:
        Preprocessing results
    """
    preprocessor = HybridPreprocessor(
        remove_urls=remove_urls,
        lowercase=lowercase,
        preserve_emojis=preserve_emojis
    )
    return preprocessor.preprocess(text)


if __name__ == "__main__":
    # Test the hybrid preprocessor
    print("=" * 70)
    print("HYBRID PREPROCESSOR TEST (NLTK + spaCy)")
    print("=" * 70)
    
    test_texts = [
        "Yaar ye movie toh ekdum mast thi! 😊 #Bollywood",
        "यह restaurant का food बहुत अच्छा है! Must try! 🍕",
        "Check out this link: https://example.com @user #awesome",
        "Aaj ka match ekdum zabardast tha! India ne jeet liya! 🏏"
    ]
    
    preprocessor = HybridPreprocessor(
        remove_urls=True,
        remove_hashtags=True,
        lowercase=True,
        preserve_emojis=True
    )
    
    for i, text in enumerate(test_texts, 1):
        print(f"\n[Test {i}]")
        print(f"Original: {text}")
        
        result = preprocessor.preprocess(text)
        
        print(f"Cleaned: {result['cleaned_text']}")
        print(f"Tokens: {result['tokens']}")
        print(f"Token count: {result['token_count']}")
        print(f"Sentences: {result['sentence_count']}")
        print(f"Hinglish: {result['is_hinglish']}")
        print(f"Devanagari: {result['has_devanagari']}")
    
    print("\n" + "=" * 70)
    print("✅ Hybrid Preprocessor test complete!")
