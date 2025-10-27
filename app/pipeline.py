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
            clean_text: Whether to perform text cleaning
            detect_language: Whether to perform language detection
            analyze_sentiment: Whether to perform sentiment analysis
            sentiment_model: Specific sentiment model to use (optional)
        """
        print("🔧 Initializing Hinglish NLP Pipeline...")
        
        self.clean_text = clean_text
        self.detect_language = detect_language
        self.analyze_sentiment = analyze_sentiment
        
        # Initialize components
        if self.clean_text:
            self.cleaner = HinglishCleaner()
            
        if self.detect_language:
            self.language_detector = LanguageDetector()
            
        if self.analyze_sentiment:
            self.sentiment_analyzer = SentimentAnalyzer(model_name=sentiment_model)
            
        print("✅ Pipeline ready!\n")
    
    def process(self, text: str) -> Dict:
        """
        Process a single text through the complete pipeline
        
        Args:
            text: Input text to process
            
        Returns:
            Dictionary with all analysis results
        """
        result = {
            'original_text': text
        }
        
        # Step 1: Preprocessing
        if self.clean_text:
            preprocessing_result = self.cleaner.process(text)
            result['cleaned_text'] = preprocessing_result['cleaned']
            result['tokens'] = preprocessing_result['tokens']
            result['token_count'] = preprocessing_result['token_count']
            processed_text = preprocessing_result['cleaned']
        else:
            result['cleaned_text'] = text
            result['tokens'] = text.split()
            result['token_count'] = len(text.split())
            processed_text = text
        
        # Step 2: Language Detection
        if self.detect_language:
            # ⚠️ KEY FIX: Ensure readable=True for human-readable labels
            lang_result = self.language_detector.detect_text(processed_text, readable=True)
            result['language_detection'] = {
                'labels': lang_result['labels'],
                'statistics': lang_result['statistics'],
                'is_code_mixed': lang_result['is_code_mixed'],
                'dominant_language': lang_result['dominant_language']
            }
        
        # Step 3: Sentiment Analysis
        if self.analyze_sentiment:
            sentiment_result = self.sentiment_analyzer.analyze(processed_text)
            result['sentiment'] = {
                'label': sentiment_result['sentiment'],
                'confidence': sentiment_result['confidence'],
                'scores': sentiment_result['scores']
            }
        
        return result
    
    def process_batch(self, texts: List[str]) -> List[Dict]:
        """
        Process multiple texts through the pipeline
        
        Args:
            texts: List of texts to process
            
        Returns:
            List of analysis results
        """
        results = []
        for text in texts:
            result = self.process(text)
            results.append(result)
        return results
    
    def analyze_text(self, text: str, verbose: bool = True) -> Dict:
        """
        Analyze text with optional verbose output
        
        Args:
            text: Input text to analyze
            verbose: Whether to print results
            
        Returns:
            Complete analysis results
        """
        result = self.process(text)
        
        if verbose:
            self._print_results(result)
        
        return result
    
    def _print_results(self, result: Dict):
        """
        Pretty print analysis results
        
        Args:
            result: Analysis results dictionary
        """
        print("\n" + "=" * 70)
        print("ANALYSIS RESULTS")
        print("=" * 70)
        
        print(f"\n📝 Original Text:\n   {result['original_text']}")
        
        if 'cleaned_text' in result:
            print(f"\n🧹 Cleaned Text:\n   {result['cleaned_text']}")
            print(f"\n🔢 Token Count: {result['token_count']}")
            print(f"   Tokens: {result['tokens'][:10]}..." if len(result['tokens']) > 10 else f"   Tokens: {result['tokens']}")
        
        if 'language_detection' in result:
            lang = result['language_detection']
            print(f"\n🌐 Language Detection:")
            print(f"   Code-mixed: {lang['is_code_mixed']}")
            print(f"   Dominant: {lang['dominant_language']}")
            print(f"   Statistics: {lang['statistics']}")
            print(f"   Labels: {lang['labels'][:10]}..." if len(lang['labels']) > 10 else f"   Labels: {lang['labels']}")
        
        if 'sentiment' in result:
            sent = result['sentiment']
            print(f"\n😊 Sentiment Analysis:")
            print(f"   Label: {sent['label']}")
            print(f"   Confidence: {sent['confidence']:.2%}")
            print(f"   Scores: {sent['scores']}")
        
        print("\n" + "=" * 70 + "\n")


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
        print(f"TEST {i}/{len(test_texts)}")
        print(f"{'='*70}")
        result = pipeline.analyze_text(text, verbose=True)