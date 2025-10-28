"""
Hybrid NLP Pipeline with Smart Routing

This module implements the main HybridNLPPipeline that intelligently routes text 
through specialized models based on language detection:
- Hinglish/Hindi/English → HingBERT + CM-BERT (92-96% accuracy)
- Other languages → XLM-RoBERTa (87% accuracy)

Components:
1. HybridPreprocessor - NLTK + spaCy preprocessing
2. FastTextDetector - Quick 176-language detection (10-20ms)
3. HingBERTDetector - Token-level Hinglish detection (96% accuracy)
4. CMBERTAnalyzer - Hinglish/English sentiment (92-94% accuracy)
5. XLMRoBERTaAnalyzer - Multilingual sentiment (87% accuracy)

Author: Yadnesh Teli
Date: January 2025
"""

import logging
from typing import Dict, List, Optional, Tuple
from enum import Enum

from app.preprocessing.hybrid_preprocessor import HybridPreprocessor
from app.language_detection.fasttext_detector import FastTextDetector
from app.language_detection.hingbert_detector import HingBERTDetector
from app.sentiment_analysis.cmbert_analyzer import CMBERTAnalyzer
from app.sentiment_analysis.xlm_roberta_analyzer import XLMRoBERTaAnalyzer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PipelineRoute(str, Enum):
    """Pipeline routing strategies"""
    HINGLISH = "hinglish"  # HingBERT + CM-BERT
    MULTILINGUAL = "multilingual"  # XLM-RoBERTa
    FALLBACK = "fallback"  # Default when detection fails


class HybridNLPPipeline:
    """
    Hybrid NLP Pipeline with intelligent model routing.
    
    This pipeline combines multiple state-of-the-art models:
    - FastText for quick language detection (176 languages)
    - HingBERT for token-level Hinglish detection (96% accuracy)
    - CM-BERT for Hinglish/English sentiment (92-94% accuracy)
    - XLM-RoBERTa for multilingual sentiment (87% accuracy)
    
    All models use lazy loading to minimize memory footprint.
    
    Example:
        >>> pipeline = HybridNLPPipeline()
        >>> result = pipeline.analyze("Yaar this movie is amazing! 🎬")
        >>> print(result['sentiment'], result['route'])
        'positive', 'hinglish'
    """
    
    def __init__(self):
        """Initialize pipeline with lazy-loaded components"""
        logger.info("Initializing HybridNLPPipeline...")
        
        # Preprocessing (always loaded, lightweight)
        self.preprocessor = HybridPreprocessor()
        
        # Language detection (lazy loaded)
        self._fasttext = None
        self._hingbert = None
        
        # Sentiment analyzers (lazy loaded)
        self._cmbert = None
        self._xlm_roberta = None
        
        # Routing configuration
        self.hinglish_languages = {'hi', 'en'}  # Hindi and English
        self.confidence_threshold = 0.6  # Minimum confidence for routing
        
        logger.info("HybridNLPPipeline initialized (lazy loading enabled)")
    
    @property
    def fasttext(self) -> FastTextDetector:
        """Lazy load FastText detector"""
        if self._fasttext is None:
            logger.info("Loading FastText detector...")
            self._fasttext = FastTextDetector()
        return self._fasttext
    
    @property
    def hingbert(self) -> HingBERTDetector:
        """Lazy load HingBERT detector"""
        if self._hingbert is None:
            logger.info("Loading HingBERT detector...")
            self._hingbert = HingBERTDetector()
        return self._hingbert
    
    @property
    def cmbert(self) -> CMBERTAnalyzer:
        """Lazy load CM-BERT analyzer"""
        if self._cmbert is None:
            logger.info("Loading CM-BERT analyzer...")
            self._cmbert = CMBERTAnalyzer()
        return self._cmbert
    
    @property
    def xlm_roberta(self) -> XLMRoBERTaAnalyzer:
        """Lazy load XLM-RoBERTa analyzer"""
        if self._xlm_roberta is None:
            logger.info("Loading XLM-RoBERTa analyzer...")
            self._xlm_roberta = XLMRoBERTaAnalyzer()
        return self._xlm_roberta
    
    def determine_route(self, text: str) -> Tuple[PipelineRoute, Dict]:
        """
        Determine optimal routing strategy based on language detection.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Tuple of (route, detection_info)
            
        Logic:
            1. Quick FastText detection
            2. Check if Hinglish (Hindi/English mix or Romanized Hindi)
            3. Route to specialized model accordingly
        """
        try:
            # Quick language detection with FastText
            fasttext_result = self.fasttext.detect(text)
            detected_lang = fasttext_result['language']
            confidence = fasttext_result['confidence']
            is_hinglish = fasttext_result['is_hinglish']
            
            detection_info = {
                'detected_language': detected_lang,
                'confidence': confidence,
                'is_hinglish': is_hinglish,
                'method': 'fasttext'
            }
            
            # Route decision
            if is_hinglish or detected_lang in self.hinglish_languages:
                # Use specialized Hinglish/English models
                logger.info(f"Routing to HINGLISH pipeline (detected: {detected_lang}, hinglish={is_hinglish})")
                return PipelineRoute.HINGLISH, detection_info
            else:
                # Use multilingual model
                logger.info(f"Routing to MULTILINGUAL pipeline (detected: {detected_lang})")
                return PipelineRoute.MULTILINGUAL, detection_info
                
        except Exception as e:
            logger.error(f"Error in route determination: {e}")
            # Fallback to multilingual (safer default)
            return PipelineRoute.FALLBACK, {
                'detected_language': 'unknown',
                'confidence': 0.0,
                'is_hinglish': False,
                'method': 'fallback',
                'error': str(e)
            }
    
    def analyze(
        self,
        text: str,
        include_preprocessing: bool = True,
        include_language_details: bool = True
    ) -> Dict:
        """
        Perform complete analysis with intelligent routing.
        
        Args:
            text: Input text to analyze
            include_preprocessing: Include preprocessing details in output
            include_language_details: Include detailed language detection info
            
        Returns:
            Dict with keys:
                - sentiment: 'positive', 'negative', or 'neutral'
                - confidence: Confidence score (0-1)
                - scores: Individual class scores
                - route: Pipeline route used
                - language_detection: Language detection details
                - preprocessing: Preprocessing details (optional)
                - model_used: Model name used for sentiment
                - processing_time_ms: Total processing time
        """
        import time
        start_time = time.time()
        
        try:
            # 1. Preprocessing
            logger.info("Step 1: Preprocessing...")
            preprocess_result = self.preprocessor.preprocess(text)
            preprocessed_text = ' '.join(preprocess_result['filtered_tokens'])
            
            # 2. Route determination
            logger.info("Step 2: Determining route...")
            route, detection_info = self.determine_route(text)
            
            # 3. Sentiment analysis based on route
            logger.info(f"Step 3: Analyzing sentiment via {route} route...")
            
            if route == PipelineRoute.HINGLISH:
                # Use HingBERT + CM-BERT for Hinglish/English
                sentiment_result = self.cmbert.analyze(preprocessed_text)
                
                # Optional: Get detailed language breakdown with HingBERT
                if include_language_details:
                    try:
                        hingbert_result = self.hingbert.detect_sentence(text)
                        detection_info['token_level_detection'] = {
                            'hindi_percentage': hingbert_result['statistics']['hindi_percentage'],
                            'english_percentage': hingbert_result['statistics']['english_percentage'],
                            'mixed_percentage': hingbert_result['statistics']['mixed_percentage'],
                            'is_code_mixed': hingbert_result['is_code_mixed']
                        }
                    except Exception as e:
                        logger.warning(f"HingBERT detailed detection failed: {e}")
                
                model_used = "CM-BERT (l3cube-pune/hing-sentiment-roberta)"
                
            elif route == PipelineRoute.MULTILINGUAL:
                # Use XLM-RoBERTa for other languages
                sentiment_result = self.xlm_roberta.analyze(
                    preprocessed_text,
                    language=detection_info.get('detected_language')
                )
                model_used = "XLM-RoBERTa (cardiffnlp/twitter-xlm-roberta-base-sentiment)"
                
            else:
                # Fallback to XLM-RoBERTa (most general)
                logger.warning("Using fallback route")
                sentiment_result = self.xlm_roberta.analyze(preprocessed_text)
                model_used = "XLM-RoBERTa (fallback)"
            
            # Calculate processing time
            processing_time = (time.time() - start_time) * 1000  # Convert to ms
            
            # 4. Build comprehensive result
            result = {
                'sentiment': sentiment_result['sentiment'],
                'confidence': sentiment_result['confidence'],
                'confidence_level': sentiment_result.get('confidence_level', 'medium'),
                'scores': sentiment_result['scores'],
                'route': route.value,
                'model_used': model_used,
                'language_detection': detection_info,
                'processing_time_ms': round(processing_time, 2)
            }
            
            # Optional fields
            if include_preprocessing:
                result['preprocessing'] = {
                    'original_length': len(text),
                    'preprocessed_length': len(preprocessed_text),
                    'tokens_count': preprocess_result['token_count'],
                    'filtered_tokens_count': preprocess_result['filtered_token_count']
                }
            
            logger.info(f"Analysis complete: {result['sentiment']} ({result['confidence']:.2f}) via {route}")
            return result
            
        except Exception as e:
            logger.error(f"Error in pipeline analysis: {e}")
            return {
                'sentiment': 'neutral',
                'confidence': 0.0,
                'confidence_level': 'low',
                'scores': {'positive': 0.33, 'negative': 0.33, 'neutral': 0.34},
                'route': 'error',
                'model_used': 'none',
                'language_detection': {'error': str(e)},
                'processing_time_ms': 0.0,
                'error': str(e)
            }
    
    def analyze_batch(
        self,
        texts: List[str],
        include_preprocessing: bool = False,
        include_language_details: bool = False
    ) -> List[Dict]:
        """
        Analyze multiple texts with batching optimizations.
        
        Args:
            texts: List of texts to analyze
            include_preprocessing: Include preprocessing details
            include_language_details: Include language detection details
            
        Returns:
            List of analysis results
            
        Note:
            This implementation routes all texts individually for now.
            Future optimization: Batch texts by detected route for efficiency.
        """
        logger.info(f"Batch analyzing {len(texts)} texts...")
        results = []
        
        for i, text in enumerate(texts):
            logger.info(f"Processing text {i+1}/{len(texts)}")
            result = self.analyze(
                text,
                include_preprocessing=include_preprocessing,
                include_language_details=include_language_details
            )
            results.append(result)
        
        return results
    
    def get_supported_languages(self) -> Dict:
        """
        Get information about supported languages.
        
        Returns:
            Dict with language support information
        """
        xlm_languages = self.xlm_roberta.get_supported_languages()
        
        return {
            'total_languages': 176,  # FastText detection
            'hinglish_optimized': {
                'languages': ['hi', 'en'],
                'model': 'CM-BERT',
                'accuracy': '92-94%',
                'description': 'State-of-the-art Hinglish and English sentiment analysis'
            },
            'multilingual_support': {
                'direct_support': xlm_languages,
                'total_via_transfer': '100+',
                'model': 'XLM-RoBERTa',
                'accuracy': '85-90%',
                'description': 'Multilingual sentiment via transfer learning'
            },
            'language_detection': {
                'method': 'FastText',
                'languages': 176,
                'accuracy': '95% (single language), 90% (code-mixed)'
            }
        }
    
    def health_check(self) -> Dict:
        """
        Perform health check on all pipeline components.
        
        Returns:
            Dict with health status of each component
        """
        health = {
            'pipeline': 'healthy',
            'components': {}
        }
        
        try:
            # Test preprocessor
            test_text = "Test text for health check"
            self.preprocessor.preprocess(test_text)
            health['components']['preprocessor'] = 'healthy'
        except Exception as e:
            health['components']['preprocessor'] = f'unhealthy: {e}'
            health['pipeline'] = 'degraded'
        
        try:
            # Test FastText (lazy load)
            self.fasttext.detect(test_text)
            health['components']['fasttext'] = 'healthy'
        except Exception as e:
            health['components']['fasttext'] = f'unhealthy: {e}'
            health['pipeline'] = 'degraded'
        
        # Optional: Test other models (only if already loaded to avoid cold start)
        if self._hingbert is not None:
            try:
                self.hingbert.detect_sentence(test_text)
                health['components']['hingbert'] = 'healthy'
            except Exception as e:
                health['components']['hingbert'] = f'unhealthy: {e}'
        
        if self._cmbert is not None:
            try:
                self.cmbert.analyze(test_text)
                health['components']['cmbert'] = 'healthy'
            except Exception as e:
                health['components']['cmbert'] = f'unhealthy: {e}'
        
        if self._xlm_roberta is not None:
            try:
                self.xlm_roberta.analyze(test_text)
                health['components']['xlm_roberta'] = 'healthy'
            except Exception as e:
                health['components']['xlm_roberta'] = f'unhealthy: {e}'
        
        return health


# Global singleton instance (lazy loaded)
_pipeline_instance: Optional[HybridNLPPipeline] = None


def get_pipeline() -> HybridNLPPipeline:
    """
    Get or create global pipeline instance (singleton pattern).
    
    Returns:
        HybridNLPPipeline instance
    """
    global _pipeline_instance
    if _pipeline_instance is None:
        logger.info("Creating global HybridNLPPipeline instance...")
        _pipeline_instance = HybridNLPPipeline()
    return _pipeline_instance
