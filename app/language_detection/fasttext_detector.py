"""
Multilingual Language Detector using FastText

NOTE: FastText (fasttext-wheel) requires NumPy 1.x (incompatible with NumPy 2.x).
Current environment uses NumPy 1.26.4 for FastText compatibility.

PRIORITY:
1. FastText (primary - 176 languages, high accuracy with NumPy 1.x)
2. langdetect (fallback - 55+ languages, works with any NumPy version)
"""

from typing import Dict, List, Optional
import re
import numpy as np

# Lazy import language detectors
_fasttext_model = None
_langdetect_available = False
_use_langdetect_primary = False  # Use FastText as primary (NumPy 1.26.4 compatible)


def get_fasttext_detector():
    """
    Lazy load language detector
    
    Priority with NumPy 1.26.4:
    1. FastText (primary - 176 languages, requires NumPy 1.x)
    2. langdetect (fallback - 55+ languages, NumPy-independent)
    """
    global _fasttext_model, _langdetect_available, _use_langdetect_primary
    
    if _fasttext_model is None:
        # Try FastText first (works with NumPy 1.26.4)
        try:
            import fasttext
            import os
            import urllib.request
            
            # Download pretrained language identification model if not exists
            model_path = os.path.join(os.path.dirname(__file__), 'lid.176.ftz')
            if not os.path.exists(model_path):
                print("[LOAD] Downloading FastText model (176 languages)...")
                url = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz"
                urllib.request.urlretrieve(url, model_path)
                print("[OK] Model downloaded")
            
            _fasttext_model = fasttext.load_model(model_path)
            _langdetect_available = False
            _use_langdetect_primary = False
            print("[OK] FastText loaded (176 languages, NumPy 1.26.4)")
            return _fasttext_model
        except Exception as e:
            print(f"[WARN] FastText loading failed: {e}")
            print("[INFO] Trying langdetect fallback...")
        
        # Fallback to langdetect
        try:
            from langdetect import detect_langs
            _fasttext_model = detect_langs
            _langdetect_available = True
            _use_langdetect_primary = True
            print("[OK] langdetect loaded (fallback, 55+ languages)")
            return _fasttext_model
        except ImportError:
            print("[ERROR] No language detector available")
            print("[INSTALL] Run: pip install fasttext-wheel or pip install langdetect")
            raise
    
    return _fasttext_model


class FastTextDetector:
    """
    Fast language detector supporting 176 languages
    Uses FastText for quick and accurate language identification
    """
    
    def __init__(self):
        """Initialize the FastText language detector"""
        
        # Language code mappings
        self.indian_languages = {
            'hi', 'bn', 'ta', 'te', 'mr', 'gu', 'kn', 'ml',
            'pa', 'or', 'as', 'ur', 'sa', 'ks', 'sd', 'ne',
            'mai', 'mni', 'sat', 'kok', 'doi', 'brx'
        }
        
        # Extended language name mappings (176 languages supported by FastText)
        self.language_names = {
            'en': 'English', 'hi': 'Hindi', 'es': 'Spanish', 'fr': 'French', 
            'de': 'German', 'zh': 'Chinese', 'ja': 'Japanese', 'ko': 'Korean',
            'ar': 'Arabic', 'ru': 'Russian', 'pt': 'Portuguese', 'it': 'Italian',
            'nl': 'Dutch', 'pl': 'Polish', 'tr': 'Turkish', 'vi': 'Vietnamese',
            'th': 'Thai', 'id': 'Indonesian', 'ms': 'Malay', 'fa': 'Persian',
            'he': 'Hebrew', 'sv': 'Swedish', 'da': 'Danish', 'no': 'Norwegian',
            'fi': 'Finnish', 'cs': 'Czech', 'el': 'Greek', 'hu': 'Hungarian',
            'ro': 'Romanian', 'uk': 'Ukrainian', 'bn': 'Bengali', 'ta': 'Tamil',
            'te': 'Telugu', 'mr': 'Marathi', 'gu': 'Gujarati', 'kn': 'Kannada',
            'ml': 'Malayalam', 'pa': 'Punjabi', 'ur': 'Urdu', 'sw': 'Swahili',
            'am': 'Amharic', 'ne': 'Nepali', 'si': 'Sinhala', 'km': 'Khmer',
            'my': 'Burmese', 'lo': 'Lao', 'ka': 'Georgian', 'hy': 'Armenian',
            'az': 'Azerbaijani', 'eu': 'Basque', 'be': 'Belarusian', 'bg': 'Bulgarian',
            'ca': 'Catalan', 'hr': 'Croatian', 'et': 'Estonian', 'gl': 'Galician',
            'is': 'Icelandic', 'lv': 'Latvian', 'lt': 'Lithuanian', 'mk': 'Macedonian',
            'mt': 'Maltese', 'sk': 'Slovak', 'sl': 'Slovenian', 'sq': 'Albanian',
            'sr': 'Serbian', 'af': 'Afrikaans', 'cy': 'Welsh', 'ga': 'Irish',
            'lb': 'Luxembourgish', 'yi': 'Yiddish', 'zu': 'Zulu', 'xh': 'Xhosa',
            'sn': 'Shona', 'st': 'Sesotho', 'tn': 'Setswana', 'yo': 'Yoruba',
            'ig': 'Igbo', 'ha': 'Hausa', 'so': 'Somali', 'mg': 'Malagasy',
            'ceb': 'Cebuano', 'tl': 'Tagalog', 'jv': 'Javanese', 'su': 'Sundanese',
            'ht': 'Haitian Creole', 'la': 'Latin', 'eo': 'Esperanto', 'ku': 'Kurdish',
            'bo': 'Tibetan', 'mn': 'Mongolian', 'kk': 'Kazakh', 'uz': 'Uzbek',
            'ky': 'Kyrgyz', 'tg': 'Tajik', 'ps': 'Pashto', 'sd': 'Sindhi',
            'ckb': 'Central Kurdish', 'dv': 'Dhivehi', 'or': 'Odia', 'as': 'Assamese',
            'sa': 'Sanskrit', 'ks': 'Kashmiri', 'gom': 'Konkani', 'mai': 'Maithili',
            'br': 'Breton', 'oc': 'Occitan', 'co': 'Corsican', 'gd': 'Scottish Gaelic',
            'fy': 'Western Frisian', 'fo': 'Faroese', 'kw': 'Cornish', 'gv': 'Manx'
        }
        
        self.major_languages = {
            'en': 'English',
            'hi': 'Hindi',
            'es': 'Spanish',
            'fr': 'French',
            'de': 'German',
            'zh': 'Chinese',
            'ja': 'Japanese',
            'ko': 'Korean',
            'ar': 'Arabic',
            'ru': 'Russian',
            'pt': 'Portuguese',
            'it': 'Italian'
        }
        
        # Devanagari script pattern
        self.devanagari_pattern = re.compile(r'[\u0900-\u097F]+')
        
        print("[*] Multilingual Language Detector initialized (176 languages)")
    
    def is_devanagari(self, text: str) -> bool:
        """Check if text contains Devanagari script"""
        return bool(self.devanagari_pattern.search(text))
    
    def is_hinglish(self, text: str) -> bool:
        """
        Check if text is likely Hinglish (code-mixed Hindi-English)
        
        Args:
            text: Input text
            
        Returns:
            True if text appears to be code-mixed
        """
        has_hindi = self.is_devanagari(text)
        has_english = bool(re.search(r'[a-zA-Z]', text))
        
        # Also check for common Romanized Hindi words
        hindi_roman_indicators = {
            'yaar', 'aaj', 'kal', 'hai', 'hain', 'kya', 'kaise',
            'accha', 'acha', 'mast', 'ekdum', 'bahut', 'bohot',
            'toh', 'matlab', 'bhai', 'dost', 'arre', 'arrey',
            'maine', 'kaha', 'tha', 'nahi', 'nahin', 'hoon', 'hoon',
            'karo', 'karna', 'kiya', 'gaya', 'jaa', 'raha', 'rahe',
            'hota', 'hoti', 'kar', 'karega', 'karenge', 'wala', 'wale',
            'bola', 'boli', 'sab', 'koi', 'kuch', 'kyun', 'kyunki',
            'lekin', 'par', 'aur', 'woh', 'ye', 'isko', 'usko'
        }
        
        text_lower = text.lower()
        has_roman_hindi = any(word in text_lower for word in hindi_roman_indicators)
        
        return (has_hindi and has_english) or (has_roman_hindi and has_english)
    
    def detect(self, text: str, threshold: float = 0.5) -> Dict:
        """
        Detect language of input text
        
        Args:
            text: Input text to analyze
            threshold: Minimum confidence threshold (0-1)
            
        Returns:
            Dictionary with detection results
        """
        if not text or not text.strip():
            return {
                'language': 'unknown',
                'lang_name': 'Unknown',
                'score': 0.0,
                'confidence': 0.0,  # Alias for compatibility
                'is_reliable': False,
                'is_hinglish': False,
                'is_indian': False,
                'supported_languages': 176
            }
        
        # Check for Hinglish first (code-mixed text)
        if self.is_hinglish(text):
            return {
                'language': 'hinglish',
                'lang_name': 'Hinglish (Code-mixed)',
                'score': 0.95,  # High confidence for detected code-mixing
                'confidence': 0.95,  # Alias for compatibility
                'is_reliable': True,
                'is_hinglish': True,
                'is_indian': True,
                'supported_languages': 176
            }
        
        # Use language detector
        try:
            detector = get_fasttext_detector()
            
            # Check which detector we're using
            if _langdetect_available:
                # langdetect returns list of Language objects
                results = detector(text)
                if results:
                    lang_code = results[0].lang
                    confidence = results[0].prob
                else:
                    raise Exception("No language detected")
            else:
                # FastText model returns predictions (now patched for NumPy 2.x)
                predictions = detector.predict(text.replace('\n', ' '), k=1)
                lang_code = predictions[0][0].replace('__label__', '')
                confidence = float(predictions[1][0])
            
            # Normalize language codes (e.g., zh-cn -> zh, zh-tw -> zh)
            if '-' in lang_code:
                base_lang = lang_code.split('-')[0]
                # Keep the full code for variants, but use base for name lookup
                lang_name = self.language_names.get(base_lang, base_lang.upper())
                # Normalize common variants to base language
                if base_lang in ['zh', 'en', 'es', 'pt', 'ar']:
                    lang_code = base_lang
            else:
                # Get language name from extended mapping, fallback to code.upper()
                lang_name = self.language_names.get(lang_code, lang_code.upper())
            
            # Check if Indian language
            is_indian = lang_code in self.indian_languages or lang_code.split('-')[0] in self.indian_languages
            
            # Check reliability
            is_reliable = confidence >= threshold
            
            return {
                'language': lang_code,
                'lang_name': lang_name,
                'score': round(confidence, 4),
                'confidence': round(confidence, 4),  # Alias for compatibility
                'is_reliable': is_reliable,
                'is_hinglish': False,
                'is_indian': is_indian,
                'supported_languages': 176
            }
            
        except Exception as e:
            print(f"⚠️  Language detection error: {e}")
            # Fallback to simple detection
            if self.is_devanagari(text):
                return {
                    'language': 'hi',
                    'lang_name': 'Hindi',
                    'score': 0.85,
                    'confidence': 0.85,  # Alias for compatibility
                    'is_reliable': True,
                    'is_hinglish': False,
                    'is_indian': True,
                    'supported_languages': 176
                }
            else:
                return {
                    'language': 'en',
                    'lang_name': 'English',
                    'score': 0.70,
                    'confidence': 0.70,  # Alias for compatibility
                    'is_reliable': False,
                    'is_hinglish': False,
                    'is_indian': False,
                    'supported_languages': 176
                }
    
    def detect_batch(self, texts: List[str]) -> List[Dict]:
        """
        Detect language for multiple texts
        
        Args:
            texts: List of texts to analyze
            
        Returns:
            List of detection results
        """
        return [self.detect(text) for text in texts]
    
    def get_language_distribution(self, text: str) -> Dict:
        """
        Analyze language distribution in text
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with language statistics
        """
        if not text or not text.strip():
            return {
                'total_chars': 0,
                'languages': {}
            }
        
        # Count character types
        latin_chars = len(re.findall(r'[a-zA-Z]', text))
        devanagari_chars = len(self.devanagari_pattern.findall(text))
        total_alpha = latin_chars + devanagari_chars
        
        if total_alpha == 0:
            return {
                'total_chars': len(text),
                'languages': {}
            }
        
        distribution = {}
        
        if latin_chars > 0:
            distribution['English/Latin'] = round(latin_chars / total_alpha * 100, 2)
        
        if devanagari_chars > 0:
            distribution['Hindi/Devanagari'] = round(devanagari_chars / total_alpha * 100, 2)
        
        return {
            'total_chars': len(text),
            'alphabetic_chars': total_alpha,
            'languages': distribution,
            'is_code_mixed': len(distribution) > 1
        }


# Convenience function
def detect_language(text: str) -> Dict:
    """
    Quick language detection
    
    Args:
        text: Text to analyze
        
    Returns:
        Detection result dictionary
    """
    detector = FastTextDetector()
    return detector.detect(text)


if __name__ == "__main__":
    # Test the language detector
    print("=" * 70)
    print("MULTILINGUAL LANGUAGE DETECTOR TEST (FastText)")
    print("=" * 70)
    
    test_texts = [
        "Hello, this is a test in English.",
        "यह हिंदी में एक परीक्षण है।",
        "Yaar ye movie toh ekdum mast thi! 😊",
        "यह restaurant का food बहुत अच्छा है!",
        "Esto es una prueba en español.",
        "これは日本語のテストです。",
        "Ceci est un test en français.",
        "Aaj main bahut happy hoon yaar!",
        "Das ist ein Test auf Deutsch."
    ]
    
    detector = FastTextDetector()
    
    for i, text in enumerate(test_texts, 1):
        print(f"\n[Test {i}] {text}")
        
        result = detector.detect(text)
        
        print(f"  → Language: {result['lang_name']} ({result['lang']})")
        print(f"  → Confidence: {result['score']:.2%}")
        print(f"  → Reliable: {result['is_reliable']}")
        print(f"  → Hinglish: {result['is_hinglish']}")
        print(f"  → Indian: {result['is_indian']}")
        
        # Distribution analysis
        dist = detector.get_language_distribution(text)
        if dist['languages']:
            print(f"  → Distribution: {dist['languages']}")
    
    print("\n" + "=" * 70)
    print(f"✅ Language detector test complete! (Supports {result['supported_languages']} languages)")
