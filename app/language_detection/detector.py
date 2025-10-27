"""
Language Detection Module for Hinglish (Hindi-English Code-Mixed) Text
Uses rule-based approach optimized for code-mixed content
"""

import re
from typing import List, Dict, Tuple
from collections import defaultdict

# Map for user-friendly output
LANG_LABEL_MAP = {
    'lang1': 'English',
    'lang2': 'Hindi',
    'ne': 'Named Entity',
    'other': 'Other'
}

class LanguageDetector:
    """
    Detects language at token level for Hinglish text
    Labels: lang1 (English), lang2 (Hindi/Romanized Hindi), ne (Named Entity), other (punctuation/special)
    """
    
    def __init__(self):
        """Initialize the language detector with Hindi indicators"""
        
        print("🔧 Initializing Language Detector (Rule-based)")
        
        # Common romanized Hindi/Hindustani words
        self.hindi_words = {
            'hai', 'hain', 'tha', 'thi', 'the', 'hoon', 'ho', 'kya', 'kyun', 'kaise', 
            'kaun', 'kab', 'kaha', 'kahaan', 'yeh', 'ye', 'woh', 'wo', 'yahan', 'wahan',
            'main', 'mein', 'tu', 'tum', 'aap', 'hum', 'unka', 'uska', 'iska', 'mera',
            'tera', 'tumhara', 'hamara', 'apna', 'kuch', 'koi', 'sab', 'sabhi',
            'bahut', 'bohot', 'thoda', 'zyada', 'kam', 'aur', 'ya', 'lekin', 'par',
            'ki', 'ka', 'ke', 'ko', 'se', 'mein', 'pe', 'par', 'tak', 'liye',
            'accha', 'acha', 'achha', 'bura', 'khrab', 'kharab', 'theek', 'thik',
            'nahi', 'nahin', 'na', 'haan', 'ha', 'bilkul', 'zaroor', 'shayad',
            'abhi', 'ab', 'phir', 'fir', 'kabhi', 'jab', 'tab', 'kal', 'aaj',
            'karna', 'karo', 'karke', 'kiya', 'kiye', 'kar', 'karega', 'karenge',
            'hona', 'ho', 'hua', 'hue', 'hogi', 'hoga', 'honge', 'rahe', 'raha', 'rahi',
            'jana', 'jao', 'gaya', 'gayi', 'gaye', 'jaa', 'jaana', 'jaayenge',
            'aana', 'aao', 'aaya', 'aayi', 'aaye', 'aa', 'aaenge', 'aata', 'aati',
            'dekho', 'dekha', 'dekhi', 'dekhe', 'dekhna', 'dekhenge', 'dekh',
            'bolo', 'bola', 'boli', 'bole', 'bolna', 'bolenge', 'bol',
            'suno', 'suna', 'suni', 'sune', 'sunna', 'sunenge', 'sun',
            'samjho', 'samjha', 'samjhi', 'samjhe', 'samajh', 'samjhna',
            'chahiye', 'chahte', 'chahta', 'chahti', 'chah', 'chahe',
            'laga', 'lagi', 'lage', 'lagta', 'lagti', 'lagte', 'lag',
            'diya', 'diyo', 'diye', 'deta', 'deti', 'dete', 'de', 'do',
            'liya', 'liyo', 'liye', 'leta', 'leti', 'lete', 'le', 'lo',
            'pata', 'malum', 'matlab', 'yaani', 'yani', 'bhi', 'baat', 'cheez',
            'kaam', 'kam', 'din', 'raat', 'subah', 'shaam', 'sham', 'time',
            'log', 'logo', 'logon', 'aadmi', 'admi', 'ladka', 'ladki', 'baccha', 'bachcha',
            'ghar', 'school', 'office', 'dost', 'friend', 'dosti', 'pyar', 'pyaar',
            'khana', 'pani', 'paani', 'chai', 'coffee', 'khao', 'piyo', 'pio',
            'sahi', 'galat', 'sacchi', 'saccha', 'jhooth', 'jhuth', 'sach',
            'chalo', 'chal', 'chala', 'chali', 'chale', 'ruko', 'ruk', 'ruka', 'ruki',
            'baith', 'baitho', 'baitha', 'baithi', 'khada', 'khadi', 'khade', 'khado',
            'bas', 'thoda', 'jyada', 'zyada', 'kafi', 'poora', 'pura', 'adha',
            'ek', 'do', 'teen', 'char', 'paanch', 'panch', 'chhe', 'saat', 'aath', 'nau', 'das'
        }
        
        self.devanagari_pattern = re.compile(r'[\u0900-\u097F]+')
        self.punctuation_pattern = re.compile(r'^[^\w\s]+$|^[\.\,\!\?\;\:\-\–\—\(\)\[\]\{\}\"\'\/\\]+$')
        
        # Common English stopwords
        self.english_stopwords = {
            'the', 'is', 'are', 'was', 'were', 'am', 'be', 'been', 'being',
            'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing',
            'a', 'an', 'and', 'or', 'but', 'if', 'then', 'so', 'than',
            'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it',
            'we', 'they', 'them', 'their', 'what', 'which', 'who', 'when',
            'where', 'why', 'how', 'all', 'each', 'every', 'both', 'few',
            'more', 'most', 'other', 'some', 'such', 'no', 'not', 'only',
            'own', 'same', 'too', 'very', 'can', 'will', 'just', 'should',
            'now', 'of', 'at', 'by', 'for', 'with', 'about', 'against',
            'between', 'into', 'through', 'during', 'before', 'after',
            'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off'
        }
        
        print("✅ Detector ready!")
        
    def detect_token_language(self, token: str) -> str:
        """
        Detect language of a single token
        
        Args:
            token: Single word/token to analyze
            
        Returns:
            Internal label: 'lang1', 'lang2', 'ne', or 'other'
        """
        if not token or not token.strip():
            return 'other'
        
        token_clean = token.strip()
        token_lower = token_clean.lower()
        
        # Check for punctuation
        if self.punctuation_pattern.match(token_clean):
            return 'other'
        
        # Check for Devanagari script
        if self.devanagari_pattern.search(token_clean):
            return 'lang2'
        
        # Check for Named Entities (capitalized words)
        if token_clean[0].isupper() and len(token_clean) > 1:
            if token_lower not in self.hindi_words and token_lower not in self.english_stopwords:
                return 'ne'
        
        # Check English stopwords
        if token_lower in self.english_stopwords:
            return 'lang1'
        
        # Check Hindi words
        if token_lower in self.hindi_words:
            return 'lang2'
        
        # Check for numbers
        if token_clean.isdigit():
            return 'other'
        
        # Check for mixed alphanumeric
        if any(c.isdigit() for c in token_clean) and any(c.isalpha() for c in token_clean):
            return 'other'
        
        # Check English suffixes
        english_suffixes = ['ing', 'ed', 'ly', 'tion', 'ment', 'ness', 'able', 'ible']
        for suffix in english_suffixes:
            if token_lower.endswith(suffix) and len(token_clean) > len(suffix) + 2:
                return 'lang1'
        
        # Check ASCII ratio
        ascii_ratio = sum(1 for c in token_clean if ord(c) < 128) / len(token_clean)
        if ascii_ratio > 0.9:
            return 'lang1'
        
        # Default to English
        return 'lang1'
    
    def detect_sentence(self, tokens: List[str]) -> List[str]:
        """Detect language for each token in a list"""
        return [self.detect_token_language(token) for token in tokens]
    
    def detect_text(self, text: str, tokenize: bool = True, readable: bool = True) -> Dict:
        """
        Detect languages in text
        
        Args:
            text: Input text to analyze
            tokenize: Whether to tokenize the text (default: True)
            readable: If True, outputs user-friendly labels (default: True)
            
        Returns:
            Dictionary with tokens, labels, and statistics
        """
        if tokenize:
            tokens = text.split()
        else:
            tokens = [text]
        
        # Get internal labels
        labels_internal = self.detect_sentence(tokens)
        
        # Convert to readable if needed
        labels = [LANG_LABEL_MAP.get(x, x) if readable else x for x in labels_internal]
        
        # Calculate statistics using internal labels
        label_counts = defaultdict(int)
        for label in labels_internal:
            label_counts[label] += 1
        
        total = len(labels_internal)
        
        # Build statistics dictionary
        stats = {}
        for label, count in label_counts.items():
            output_label = LANG_LABEL_MAP.get(label, label) if readable else label
            stats[output_label] = {
                'count': count,
                'percentage': round((count / total) * 100, 2) if total > 0 else 0
            }
        
        # Get dominant language (internal)
        dominant_internal = self.get_dominant_language(text, labels_internal)
        dominant = LANG_LABEL_MAP.get(dominant_internal, dominant_internal) if readable else dominant_internal

        return {
            'tokens': tokens,
            'labels': labels,
            'statistics': stats,
            'is_code_mixed': len([k for k in label_counts if k in ['lang1', 'lang2']]) > 1,
            'dominant_language': dominant
        }
        
    def get_dominant_language(self, text: str, cached_labels=None) -> str:
        """
        Get the dominant language in the text
        
        Args:
            text: Input text
            cached_labels: Precomputed internal labels if available
            
        Returns:
            Dominant language label (internal: 'lang1' or 'lang2')
        """
        if cached_labels is None:
            # Recalculate labels
            result = self.detect_text(text, readable=False)
            labels_internal = result['labels']
        else:
            labels_internal = cached_labels
        
        # Count only language labels (exclude 'ne' and 'other')
        label_counts = defaultdict(int)
        for label in labels_internal:
            if label in ['lang1', 'lang2']:
                label_counts[label] += 1
        
        if not label_counts:
            return 'unknown'
        
        # Return the most common language
        return max(label_counts.items(), key=lambda x: x[1])[0]

def detect_language(text: str) -> Dict:
    """Convenience function to detect language in text"""
    detector = LanguageDetector()
    return detector.detect_text(text)

if __name__ == "__main__":
    # Test the detector
    test_sentences = [
        "Hello world this is a test",
        "Yeh bahut accha hai",
        "Main aaj bahut happy hoon",
        "Virat Kohli is playing cricket",
        "Modi ji ne accha kaam kiya",
        "@user check this out! #awesome"
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
        print(f"Dominant: {result['dominant_language']}")
        print(f"Stats: {result['statistics']}")