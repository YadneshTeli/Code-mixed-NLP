"""
Language Detection Module for Hinglish (Hindi-English Code-Mixed) Text
Uses rule-based approach optimized for code-mixed content
"""

import re
from typing import List, Dict, Tuple
from collections import defaultdict


class LanguageDetector:
    """
    Detects language at token level for Hinglish text
    Labels: lang1 (English), lang2 (Hindi/Romanized Hindi), ne (Named Entity), other (punctuation/special)
    """
    
    def __init__(self):
        """Initialize the language detector with Hindi indicators"""
        
        print("🔧 Initializing Language Detector (Rule-based)")
        
        # Common Hindi words in Roman script (Hinglish) - EXCLUDING English conflicts
        self.hindi_words = {
            # Common verbs
            'hai', 'hain', 'tha', 'ho', 'hoga', 'hogi', 'kar', 'karo', 'kiya', 'kiye',
            'chahiye', 'milega', 'aaya', 'gaya', 'diya', 'liya', 'dekho', 'dekha', 'suno', 'bolo',
            'karo', 'kro', 'laga', 'lage', 'lagi', 'raha', 'rahe', 'rahi', 'hu', 'hoon', 'hein',
            # Common nouns/adjectives
            'bahut', 'accha', 'acha', 'bura', 'sahi', 'galat', 'kya', 'kaise', 'kaun', 'kab', 
            'kahan', 'kyun', 'yeh', 'woh', 'yaar', 'bhai', 'dost', 'log', 'sab', 'koi', 'kuch',
            'baat', 'baar', 'din', 'raat', 'jagah', 'waqt', 'ek', 'do', 'teen', 'char',
            # Pronouns
            'main', 'mujhe', 'mera', 'mere', 'meri', 'hum', 'humara', 'humare', 'humari',
            'tum', 'tumhara', 'tumhare', 'tumhari', 'aap', 'aapka', 'aapke', 'aapki',
            'mai', 'mein', 'apne', 'apna', 'apni', 'uska', 'uske', 'uski', 'ye', 'wo',
            # Common words - REMOVED: 'the', 'to', 'or', 'he', 'me', 'k', 'h' (English conflicts)
            'nahi', 'nahin', 'nhi', 'bhi', 'hi', 'toh', 'se', 'ka', 'ke', 'ki', 'ko',
            'ne', 'par', 'pe', 'aur', 'ya', 'lekin', 'agar', 'tab', 'abhi', 'ab',
            'phir', 'kabhi', 'liye',
            # Expressions
            'ji', 'sirji', 'yarr', 'yaar', 'chal', 'chalo', 'are', 'arre',
            'haan', 'han', 'naa', 'na', 'matlab', 'kyunki', 'isliye', 'waise', 'fir',
            'jai', 'hind', 'bharat', 'desh', 'har', 'mat'
        }
        
        # Hindi character patterns (Devanagari)
        self.devanagari_pattern = re.compile(r'[\u0900-\u097F]+')
        
        # Punctuation and special characters
        self.punctuation_pattern = re.compile(r'^[^\w\s]+$|^[\.\,\!\?\;\:\-\–\—\(\)\[\]\{\}\"\'\/\\]+$')
        
        # Common English stopwords (subset)
        self.english_stopwords = {
            'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
            'do', 'does', 'did', 'will', 'would', 'should', 'could', 'may', 'might', 'must',
            'can', 'this', 'that', 'these', 'those', 'a', 'an', 'and', 'or', 'but', 'if',
            'for', 'with', 'about', 'as', 'at', 'by', 'from', 'in', 'of', 'on', 'to'
        }
        
        print("✅ Detector ready!")
        
    def detect_token_language(self, token: str) -> str:
        """
        Detect language of a single token
        
        Args:
            token: Single word/token to analyze
            
        Returns:
            Language label: 'lang1' (English), 'lang2' (Hindi), 'ne' (Named Entity), 'other'
        """
        if not token or not token.strip():
            return 'other'
        
        token_clean = token.strip()
        token_lower = token_clean.lower()
        
        # Check for punctuation/special characters
        if self.punctuation_pattern.match(token_clean):
            return 'other'
        
        # Check for Devanagari script (definitely Hindi)
        if self.devanagari_pattern.search(token_clean):
            return 'lang2'
        
        # Check for Named Entities (capitalized words)
        if token_clean[0].isupper() and len(token_clean) > 1:
            # Could be a proper noun/named entity
            if token_lower not in self.hindi_words and token_lower not in self.english_stopwords:
                # Likely a named entity
                return 'ne'
        
        # Check for English stopwords FIRST (they're more specific)
        if token_lower in self.english_stopwords:
            return 'lang1'
        
        # Check for Hindi words
        if token_lower in self.hindi_words:
            return 'lang2'
        
        # Check for numbers
        if token_clean.isdigit():
            return 'other'
        
        # Check for mixed alphanumeric (URLs, handles, etc.)
        if any(c.isdigit() for c in token_clean) and any(c.isalpha() for c in token_clean):
            return 'other'
        
        # Check for common English patterns (words with common English endings)
        english_suffixes = ['ing', 'ed', 'ly', 'tion', 'ment', 'ness', 'able', 'ible']
        for suffix in english_suffixes:
            if token_lower.endswith(suffix) and len(token_clean) > len(suffix) + 2:
                return 'lang1'
        
        # Default: if mostly ASCII and longer, likely English
        ascii_ratio = sum(1 for c in token_clean if ord(c) < 128) / len(token_clean)
        if ascii_ratio > 0.9:
            return 'lang1'
        
        # Default fallback
        return 'lang1'
    
    def detect_sentence(self, tokens: List[str]) -> List[str]:
        """
        Detect language for each token in a sentence
        
        Args:
            tokens: List of tokens from a sentence
            
        Returns:
            List of language labels corresponding to each token
        """
        return [self.detect_token_language(token) for token in tokens]
    
    def detect_text(self, text: str, tokenize: bool = True) -> Dict:
        """
        Detect languages in text
        
        Args:
            text: Input text to analyze
            tokenize: Whether to tokenize the text (default: True)
            
        Returns:
            Dictionary with tokens, labels, and statistics
        """
        if tokenize:
            # Simple whitespace tokenization
            tokens = text.split()
        else:
            tokens = [text]
        
        labels = self.detect_sentence(tokens)
        
        # Calculate statistics
        label_counts = defaultdict(int)
        for label in labels:
            label_counts[label] += 1
        
        total = len(labels)
        stats = {
            label: {
                'count': count,
                'percentage': round(count / total * 100, 2) if total > 0 else 0
            }
            for label, count in label_counts.items()
        }
        
        return {
            'tokens': tokens,
            'labels': labels,
            'statistics': stats,
            'is_code_mixed': len(label_counts) > 1 and ('lang1' in label_counts and 'lang2' in label_counts)
        }
    
    def get_dominant_language(self, text: str) -> str:
        """
        Get the dominant language in the text
        
        Args:
            text: Input text
            
        Returns:
            Dominant language label
        """
        result = self.detect_text(text)
        stats = result['statistics']
        
        # Filter out 'other' and 'ne' for dominant language
        lang_stats = {k: v['count'] for k, v in stats.items() if k in ['lang1', 'lang2']}
        
        if not lang_stats:
            return 'unknown'
        
        return max(lang_stats.items(), key=lambda x: x[1])[0]


def detect_language(text: str) -> Dict:
    """
    Convenience function for quick language detection
    
    Args:
        text: Input text to analyze
        
    Returns:
        Detection results dictionary
    """
    detector = LanguageDetector()
    return detector.detect_text(text)


if __name__ == "__main__":
    # Test the detector
    test_sentences = [
        "Hello world this is a test",  # English
        "Yeh bahut accha hai",  # Hindi
        "Main aaj bahut happy hoon",  # Code-mixed
        "Virat Kohli is playing cricket",  # English with NE
        "Modi ji ne accha kaam kiya",  # Hindi with NE
        "@user check this out! #awesome"  # Social media
    ]
    
    detector = LanguageDetector()
    
    print("\n" + "=" * 70)
    print("LANGUAGE DETECTOR - TEST")
    print("=" * 70)
    
    for i, text in enumerate(test_sentences, 1):
        result = detector.detect_text(text)
        
        print(f"\n[Test {i}] {text}")
        print(f"Tokens: {result['tokens']}")
        print(f"Labels: {result['labels']}")
        print(f"Code-mixed: {result['is_code_mixed']}")
        print(f"Stats: {result['statistics']}")
