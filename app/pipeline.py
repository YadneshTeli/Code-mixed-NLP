"""
Integrated NLP Pipeline for Hinglish Text
Combines preprocessing, language detection, and sentiment analysis
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from typing import Dict, List, Optional
from app.preprocessing.cleaner import HinglishCleaner
from app.language_detection.detector import LanguageDetector
from app.sentiment_analysis.analyzer import SentimentAnalyzer


class HinglishNLPPipeline:
    """
    Complete NLP pipeline for Hinglish (code-mixed Hindi-English) text
    """
    
    def __init__(
        self,
        clean_text: bool = True,
        detect_language: bool = True,
        analyze_sentiment: bool = True,
        sentiment_model: str = None
    ):
        """
        Initialize the NLP pipeline
        
        Args:
            clean_text: Enable text preprocessing
            detect_language: Enable language detection
            analyze_sentiment: Enable sentiment analysis
            sentiment_model: Optional sentiment model name
        """
        print("🚀 Initializing Hinglish NLP Pipeline")
        print("=" * 70)
        
        self.clean_text_enabled = clean_text
        self.detect_language_enabled = detect_language
        self.analyze_sentiment_enabled = analyze_sentiment
        
        # Initialize components
        if clean_text:
            print("\n📝 Module 1: Text Preprocessing")
            self.cleaner = HinglishCleaner(
                remove_urls=True,
                remove_mentions=False,
                remove_hashtags=False,
                remove_punctuation=False,
                lowercase=False,
                preserve_emojis=True
            )
            print("   ✅ Preprocessing ready")
        
        if detect_language:
            print("\n🌍 Module 2: Language Detection")
            self.language_detector = LanguageDetector()
        
        if analyze_sentiment:
            print("\n😊 Module 3: Sentiment Analysis")
            self.sentiment_analyzer = SentimentAnalyzer(model_name=sentiment_model) if sentiment_model else SentimentAnalyzer()
        
        print("\n" + "=" * 70)
        print("✅ Pipeline Ready!\n")
    
    def process(self, text: str) -> Dict:
        """
        Process text through the complete pipeline
        
        Args:
            text: Input text to process
            
        Returns:
            Dictionary with all analysis results
        """
        result = {
            'original_text': text
        }
        
        # Step 1: Clean text
        if self.clean_text_enabled:
            cleaned = self.cleaner.clean_text(text)
            tokens = self.cleaner.tokenize(cleaned)
            result['cleaned_text'] = cleaned
            result['tokens'] = tokens
            result['token_count'] = len(tokens)
        else:
            cleaned = text
            tokens = text.split()
            result['cleaned_text'] = text
            result['tokens'] = tokens
            result['token_count'] = len(tokens)
        
        # Step 2: Detect language
        if self.detect_language_enabled:
            lang_result = self.language_detector.detect_text(cleaned, tokenize=False)
            lang_result['tokens'] = tokens  # Use our tokens
            lang_result['labels'] = self.language_detector.detect_sentence(tokens)
            
            result['language_detection'] = {
                'labels': lang_result['labels'],
                'statistics': lang_result['statistics'],
                'is_code_mixed': lang_result['is_code_mixed'],
                'dominant_language': self.language_detector.get_dominant_language(cleaned)
            }
        
        # Step 3: Analyze sentiment
        if self.analyze_sentiment_enabled:
            sentiment_result = self.sentiment_analyzer.analyze(cleaned)
            result['sentiment'] = {
                'label': sentiment_result['sentiment'],
                'confidence': sentiment_result['confidence'],
                'scores': sentiment_result['scores']
            }
        
        return result
    
    def process_batch(self, texts: List[str]) -> List[Dict]:
        """
        Process multiple texts
        
        Args:
            texts: List of texts to process
            
        Returns:
            List of analysis results
        """
        return [self.process(text) for text in texts]
    
    def analyze_text(self, text: str, verbose: bool = True) -> Dict:
        """
        Analyze text and optionally print results
        
        Args:
            text: Input text
            verbose: Print formatted output
            
        Returns:
            Analysis results
        """
        result = self.process(text)
        
        if verbose:
            self._print_results(result)
        
        return result
    
    def _print_results(self, result: Dict):
        """
        Print formatted analysis results
        
        Args:
            result: Analysis results dictionary
        """
        print("\n" + "=" * 70)
        print("HINGLISH TEXT ANALYSIS")
        print("=" * 70)
        
        print(f"\n📄 Original Text:")
        print(f"   {result['original_text']}")
        
        if 'cleaned_text' in result and result['cleaned_text'] != result['original_text']:
            print(f"\n🧹 Cleaned Text:")
            print(f"   {result['cleaned_text']}")
        
        print(f"\n🔤 Tokens ({result['token_count']}):")
        print(f"   {result['tokens']}")
        
        if 'language_detection' in result:
            lang = result['language_detection']
            print(f"\n🌍 Language Detection:")
            print(f"   Labels: {lang['labels']}")
            print(f"   Dominant: {lang['dominant_language']}")
            print(f"   Code-mixed: {lang['is_code_mixed']}")
            print(f"   Statistics:")
            for label, stats in lang['statistics'].items():
                print(f"      {label}: {stats['count']} tokens ({stats['percentage']:.1f}%)")
        
        if 'sentiment' in result:
            sent = result['sentiment']
            print(f"\n😊 Sentiment Analysis:")
            print(f"   Label: {sent['label'].upper()}")
            print(f"   Confidence: {sent['confidence']:.2%}")
            print(f"   Scores:")
            for label, score in sent['scores'].items():
                bar = "█" * int(score * 30)
                print(f"      {label:10}: {bar} {score:.2%}")
        
        print("\n" + "=" * 70)


def analyze_hinglish_text(text: str) -> Dict:
    """
    Convenience function for quick analysis
    
    Args:
        text: Hinglish text to analyze
        
    Returns:
        Complete analysis results
    """
    pipeline = HinglishNLPPipeline()
    return pipeline.process(text)


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("HINGLISH NLP PIPELINE - DEMO")
    print("=" * 70)
    
    # Create pipeline
    pipeline = HinglishNLPPipeline()
    
    # Test samples
    test_texts = [
        "Yeh movie bahut accha hai yaar! I loved it!",
        "Terrible experience tha. Bilkul bekaar!",
        "Virat Kohli is playing amazingly well today",
    ]
    
    # Analyze each text
    for i, text in enumerate(test_texts, 1):
        print(f"\n{'='*70}")
        print(f"SAMPLE {i}")
        pipeline.analyze_text(text, verbose=True)
