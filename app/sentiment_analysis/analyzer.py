"""
Sentiment Analysis Module for Hinglish Text
Uses Hugging Face transformers for multilingual sentiment detection
"""

from typing import Dict, List, Optional
from transformers import pipeline
import warnings

warnings.filterwarnings('ignore')


class SentimentAnalyzer:
    """
    Sentiment analyzer for code-mixed Hinglish text
    Supports: positive, negative, neutral sentiment detection
    """
    
    def __init__(self, model_name: str = "distilbert-base-uncased-finetuned-sst-2-english"):
        """
        Initialize sentiment analyzer
        
        Args:
            model_name: Hugging Face model for sentiment (default: distilbert for speed)
        """
        print(f"🔧 Initializing Sentiment Analyzer")
        print(f"📦 Loading model: {model_name}")
        
        try:
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model=model_name,
                top_k=None  # Return all scores
            )
            print("✅ Model loaded successfully!")
            
        except Exception as e:
            print(f"⚠️  Model loading failed: {e}")
            print("📌 Falling back to rule-based sentiment")
            self.sentiment_pipeline = None
            self._init_rule_based()
    
    def _init_rule_based(self):
        """Initialize rule-based sentiment detection as fallback"""
        
        # Positive words (English + Hindi)
        self.positive_words = {
            'good', 'great', 'awesome', 'excellent', 'amazing', 'wonderful', 'fantastic',
            'love', 'loved', 'loving', 'like', 'liked', 'best', 'better', 'happy', 'nice',
            'perfect', 'beautiful', 'brilliant', 'superb', 'outstanding', 'incredible',
            # Hindi positive
            'accha', 'acha', 'badhiya', 'mast', 'zabardast', 'kamaal', 'sahi', 'badiya',
            'khushi', 'pyar', 'pyara', 'sundar', 'shandar', 'ekdum', 'bahut'
        }
        
        # Negative words (English + Hindi)
        self.negative_words = {
            'bad', 'worst', 'terrible', 'awful', 'horrible', 'poor', 'disappointing',
            'hate', 'hated', 'dislike', 'sad', 'angry', 'annoying', 'boring', 'ugly',
            'stupid', 'worse', 'useless', 'pathetic', 'disgusting', 'fail', 'failed',
            # Hindi negative
            'bura', 'bekaar', 'ghatiya', 'kharab', 'galat', 'bekar', 'faltu', 'bakwas',
            'nautanki', 'bewakoof', 'pagal', 'nonsense'
        }
    
    def analyze(self, text: str) -> Dict:
        """
        Analyze sentiment of text
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary with sentiment label and scores
        """
        if not text or not text.strip():
            return {
                'text': text,
                'sentiment': 'neutral',
                'scores': {'positive': 0.33, 'neutral': 0.34, 'negative': 0.33},
                'confidence': 0.34
            }
        
        # Use transformer model if available
        if self.sentiment_pipeline:
            return self._transformer_analysis(text)
        else:
            return self._rule_based_analysis(text)
    
    def _transformer_analysis(self, text: str) -> Dict:
        """
        Use transformer model for sentiment analysis
        
        Args:
            text: Input text
            
        Returns:
            Sentiment analysis result
        """
        try:
            # Get predictions
            results = self.sentiment_pipeline(text)[0]
            
            # Convert to our format
            scores = {}
            for item in results:
                label = item['label'].lower()
                score = item['score']
                
                # Map labels
                if 'pos' in label or label == 'label_2':
                    scores['positive'] = score
                elif 'neg' in label or label == 'label_0':
                    scores['negative'] = score
                elif 'neu' in label or label == 'label_1':
                    scores['neutral'] = score
            
            # Determine dominant sentiment
            sentiment = max(scores.items(), key=lambda x: x[1])[0]
            confidence = scores[sentiment]
            
            return {
                'text': text,
                'sentiment': sentiment,
                'scores': scores,
                'confidence': round(confidence, 4)
            }
            
        except Exception as e:
            print(f"⚠️  Analysis error: {e}")
            return self._rule_based_analysis(text)
    
    def _rule_based_analysis(self, text: str) -> Dict:
        """
        Rule-based sentiment analysis (fallback)
        
        Args:
            text: Input text
            
        Returns:
            Sentiment analysis result
        """
        words = text.lower().split()
        
        pos_count = sum(1 for w in words if w in self.positive_words)
        neg_count = sum(1 for w in words if w in self.negative_words)
        total_count = len(words)
        
        # Calculate scores
        if total_count == 0:
            scores = {'positive': 0.33, 'neutral': 0.34, 'negative': 0.33}
        else:
            pos_score = pos_count / total_count
            neg_score = neg_count / total_count
            neu_score = 1 - (pos_score + neg_score)
            
            # Normalize
            total = pos_score + neg_score + neu_score
            scores = {
                'positive': round(pos_score / total, 4) if total > 0 else 0.33,
                'negative': round(neg_score / total, 4) if total > 0 else 0.33,
                'neutral': round(neu_score / total, 4) if total > 0 else 0.34
            }
        
        # Determine sentiment
        sentiment = max(scores.items(), key=lambda x: x[1])[0]
        confidence = scores[sentiment]
        
        return {
            'text': text,
            'sentiment': sentiment,
            'scores': scores,
            'confidence': confidence
        }
    
    def analyze_batch(self, texts: List[str]) -> List[Dict]:
        """
        Analyze sentiment for multiple texts
        
        Args:
            texts: List of text strings
            
        Returns:
            List of sentiment analysis results
        """
        return [self.analyze(text) for text in texts]
    
    def get_sentiment_label(self, text: str) -> str:
        """
        Get just the sentiment label
        
        Args:
            text: Input text
            
        Returns:
            Sentiment label: 'positive', 'negative', or 'neutral'
        """
        result = self.analyze(text)
        return result['sentiment']


def analyze_sentiment(text: str, model_name: str = None) -> Dict:
    """
    Convenience function for quick sentiment analysis
    
    Args:
        text: Text to analyze
        model_name: Optional model name
        
    Returns:
        Sentiment analysis result
    """
    if model_name:
        analyzer = SentimentAnalyzer(model_name=model_name)
    else:
        analyzer = SentimentAnalyzer()
    
    return analyzer.analyze(text)


if __name__ == "__main__":
    print("=" * 70)
    print("SENTIMENT ANALYZER - TEST")
    print("=" * 70)
    
    # Create analyzer
    analyzer = SentimentAnalyzer()
    
    # Test samples
    test_texts = [
        "This is amazing! I love it!",
        "This is terrible and disappointing",
        "It's okay, nothing special",
        "Yeh bahut accha hai yaar!",
        "Bahut bura experience tha",
        "Mast movie hai boss, ekdum zabardast!"
    ]
    
    print("\n📝 Testing sentiment analysis:\n")
    
    for text in test_texts:
        result = analyzer.analyze(text)
        
        print(f"Text: {text}")
        print(f"  → Sentiment: {result['sentiment'].upper()}")
        print(f"  → Confidence: {result['confidence']:.2%}")
        print(f"  → Scores: ", end="")
        for label, score in result['scores'].items():
            print(f"{label}: {score:.2%}  ", end="")
        print("\n")
    
    print("=" * 70)
