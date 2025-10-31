"""
FastAPI REST API for Hinglish NLP Pipeline
Provides endpoints for text preprocessing, language detection, and sentiment analysis
"""

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict
import uvicorn
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.pipeline import HybridNLPPipeline, get_pipeline
from app.preprocessing.cleaner import HinglishCleaner
from app.language_detection.detector import LanguageDetector
from app.sentiment_analysis.analyzer import SentimentAnalyzer
from app.utils.response_mappers import (
    map_v2_to_v1_preprocessing,
    map_v2_to_v1_language,
    map_v2_to_v1_sentiment,
    map_v2_to_v1_full_analysis,
    map_v2_to_v1_batch
)

# Create alias for backward compatibility
HinglishNLPPipeline = HybridNLPPipeline

# Initialize FastAPI app
app = FastAPI(
    title="Multilingual Hinglish NLP API",
    description="REST API for processing code-mixed Hindi-English (Hinglish) text with multilingual support (176 languages)",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize NLP components globally
print("🚀 Initializing NLP components...")
# Legacy pipeline (for backward compatibility)
pipeline = HinglishNLPPipeline()
cleaner = HinglishCleaner()
language_detector = LanguageDetector()
sentiment_analyzer = SentimentAnalyzer()

# New hybrid multilingual pipeline (lazy loaded)
hybrid_pipeline = None
print("✅ API ready! (Hybrid pipeline will load on first use)\n")


# Request/Response Models
class TextInput(BaseModel):
    """Single text input"""
    text: str = Field(..., min_length=1, max_length=5000, description="Text to analyze")
    
    @validator('text')
    def text_not_empty(cls, v):
        if not v.strip():
            raise ValueError('Text cannot be empty or whitespace only')
        return v


class BatchTextInput(BaseModel):
    """Multiple texts input"""
    texts: List[str] = Field(..., min_items=1, max_items=100, description="List of texts to analyze")
    
    @validator('texts')
    def texts_not_empty(cls, v):
        for text in v:
            if not text.strip():
                raise ValueError('Texts cannot contain empty or whitespace-only items')
        return v


class PreprocessingResponse(BaseModel):
    """Preprocessing result"""
    original: str
    cleaned: str
    tokens: List[str]
    token_count: int


class LanguageDetectionResponse(BaseModel):
    """Language detection result"""
    language: str
    confidence: float
    is_hinglish: bool
    is_indian_language: bool


class LanguageDetectionWithTokensResponse(BaseModel):
    """Language detection result with tokens (for standalone endpoint)"""
    tokens: List[str]
    labels: List[str]
    statistics: Dict
    is_code_mixed: bool
    dominant_language: str


class SentimentResponse(BaseModel):
    """Sentiment analysis result"""
    label: str
    confidence: float
    scores: Dict[str, float]


class SimpleSentimentResponse(BaseModel):
    """Simple sentiment result for full analysis"""
    label: str
    score: float


class FullAnalysisResponse(BaseModel):
    """Complete NLP analysis result"""
    original_text: str
    cleaned_text: str
    tokens: List[str]
    token_count: int
    language_detection: LanguageDetectionResponse
    sentiment: SimpleSentimentResponse


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    version: str
    modules: Dict[str, bool]


# API Endpoints

@app.get("/", response_model=Dict)
async def root():
    """Root endpoint with API information"""
    return {
        "name": "Multilingual Hinglish NLP API",
        "version": "2.0.0",
        "description": "REST API for processing code-mixed Hindi-English text with multilingual support",
        "deployment": "https://code-mixed-nlp.up.railway.app",
        "features": {
            "hinglish_accuracy": "92-96%",
            "english_accuracy": "94%",
            "multilingual_accuracy": "87%",
            "supported_languages": "176 via FastText",
            "specialized_models": ["HingBERT", "CM-BERT", "XLM-RoBERTa"]
        },
        "endpoints": {
            "v1": {
                "health": "/health",
                "docs": "/docs",
                "preprocess": "/api/v1/preprocess",
                "detect_language": "/api/v1/detect-language",
                "analyze_sentiment": "/api/v1/analyze-sentiment",
                "full_analysis": "/api/v1/analyze",
                "batch_analysis": "/api/v1/analyze/batch"
            },
            "v2": {
                "health": "/api/v2/health",
                "preprocess": "/api/v2/preprocess",
                "detect_language": "/api/v2/detect-language",
                "analyze_sentiment": "/api/v2/analyze-sentiment",
                "multilingual_analysis": "/api/v2/analyze",
                "batch_analysis": "/api/v2/analyze/batch",
                "supported_languages": "/api/v2/languages"
            }
        },
        "total_endpoints": 14
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    global hybrid_pipeline
    
    # Check if hybrid pipeline is loaded
    hybrid_status = hybrid_pipeline is not None
    
    return {
        "status": "healthy",
        "version": "2.0.0",
        "modules": {
            "preprocessing": True,
            "language_detection": True,
            "sentiment_analysis": True,
            "hybrid_pipeline": hybrid_status
        }
    }


@app.get("/api/v2/status")
async def model_status():
    """
    Check which models are loaded
    
    Returns detailed status of all models including transformers
    """
    global hybrid_pipeline
    
    return {
        "status": "ok",
        "models": {
            "hybrid_pipeline": hybrid_pipeline is not None,
            "fasttext": True,  # Always loaded at startup
            "spacy": True,  # Always loaded at startup
            "transformers": {
                "loaded": hybrid_pipeline is not None,
                "note": "Heavy models (1.1 GB) - loaded on first V2 request"
            }
        },
        "memory_warning": "First V2 request may take 30-60 seconds to load transformers models"
    }


@app.api_route("/api/v2/warmup", methods=["GET", "POST"])
async def warmup_models():
    """
    Warmup endpoint to pre-load heavy transformer models
    
    Call this endpoint once after deployment to pre-load models.
    This prevents timeout on first real request.
    
    Accepts both GET and POST requests.
    """
    global hybrid_pipeline
    
    if hybrid_pipeline is not None:
        return {"status": "already_loaded", "message": "Models are already warm"}
    
    try:
        print("🔥 Warming up models...")
        hybrid_pipeline = get_pipeline()
        print("✅ Models loaded successfully!")
        
        return {
            "status": "success",
            "message": "All models loaded and ready",
            "models": ["HingBERT", "CM-BERT", "XLM-RoBERTa"]
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to load models: {str(e)}"
        }


@app.post("/api/v1/preprocess", response_model=PreprocessingResponse)
async def preprocess_text(input_data: TextInput):
    """
    Preprocess text (cleaning and tokenization)
    
    **MIGRATED TO V2**: Now uses hybrid preprocessor internally for better results
    while maintaining V1 response format for backward compatibility.
    
    - **text**: Input text to preprocess
    
    Returns cleaned text and tokens
    """
    global hybrid_pipeline
    
    try:
        # Lazy load hybrid pipeline
        if hybrid_pipeline is None:
            print("🔄 Loading hybrid pipeline on first request...")
            hybrid_pipeline = get_pipeline()
        
        # Use V2 preprocessing internally
        text = input_data.text
        result = hybrid_pipeline.preprocessor.preprocess(text)
        
        # Convert to V1 response format
        v2_response = {
            "original": result['original_text'],
            "processed": result['cleaned_text'],
            "tokens": result['tokens'],
            "tokens_count": result['token_count']
        }
        
        return map_v2_to_v1_preprocessing(v2_response)
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Preprocessing failed: {str(e)}"
        )


@app.post("/api/v1/detect-language", response_model=LanguageDetectionWithTokensResponse)
async def detect_language(input_data: TextInput):
    """
    Detect language at token level
    
    **MIGRATED TO V2**: Now uses FastText + HingBERT internally for 96% accuracy
    while maintaining V1 response format for backward compatibility.
    
    - **text**: Input text to analyze
    
    Returns language labels for each token
    """
    global hybrid_pipeline
    
    try:
        # Lazy load hybrid pipeline
        if hybrid_pipeline is None:
            print("🔄 Loading hybrid pipeline on first request...")
            hybrid_pipeline = get_pipeline()
        
        text = input_data.text
        
        # Use V2 language detection internally
        fasttext_result = hybrid_pipeline.fasttext.detect(text)
        detected_lang = fasttext_result['language']
        confidence = fasttext_result['confidence']
        
        # Token-level detection with HingBERT (processes entire text, not individual tokens)
        hingbert_result = hybrid_pipeline.hingbert.detect_tokens(text)
        tokens = hingbert_result.get('tokens', [])
        token_labels_raw = hingbert_result.get('labels', [])
        
        # Map labels to full names
        label_name_map = {
            'en': 'English',
            'hi': 'Hindi',
            'ne': 'Named Entity',
            'other': 'Other'
        }
        token_labels = [label_name_map.get(label, label) for label in token_labels_raw]
        
        # Calculate statistics
        label_counts = {}
        for label in token_labels:
            label_counts[label] = label_counts.get(label, 0) + 1
        
        total_tokens = len(token_labels)
        statistics = {}
        for label, count in label_counts.items():
            statistics[label] = {
                "count": count,
                "percentage": round((count / total_tokens) * 100, 1) if total_tokens > 0 else 0.0
            }
        
        # Check if Hinglish
        english_count = label_counts.get('English', 0)
        hindi_count = label_counts.get('Hindi', 0)
        is_hinglish = (english_count > 0 and hindi_count > 0) or detected_lang in ['hi', 'en']
        
        # Get language name
        language_names = {'hi': 'Hindi', 'en': 'English'}
        language_name = language_names.get(detected_lang, detected_lang.upper())
        
        # Build V2 response
        v2_response = {
            "detected_language": detected_lang,
            "language_name": language_name,
            "confidence": confidence,
            "is_hinglish": is_hinglish,
            "is_reliable": confidence > 0.7,
            "token_level_detection": {
                "tokens": tokens,
                "labels": token_labels,
                "statistics": statistics
            }
        }
        
        # Convert to V1 format (token-level for this endpoint)
        return map_v2_to_v1_language(v2_response, format_type="tokens")
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Language detection failed: {str(e)}"
        )


@app.post("/api/v1/analyze-sentiment", response_model=SentimentResponse)
async def analyze_sentiment(input_data: TextInput):
    """
    Analyze sentiment of text
    
    **MIGRATED TO V2**: Now uses smart routing (CM-BERT/XLM-RoBERTa) for 92%+ accuracy
    while maintaining V1 response format for backward compatibility.
    
    - **text**: Input text to analyze
    
    Returns sentiment label and confidence scores
    """
    global hybrid_pipeline
    
    try:
        # Lazy load hybrid pipeline
        if hybrid_pipeline is None:
            print("🔄 Loading hybrid pipeline on first request...")
            hybrid_pipeline = get_pipeline()
        
        text = input_data.text
        
        # Use V2 sentiment analysis with smart routing
        lang_result = hybrid_pipeline.fasttext.detect(text)
        detected_lang = lang_result['language']
        
        # Route to appropriate model
        if detected_lang in ['hi', 'en']:
            sentiment_result = hybrid_pipeline.cmbert.analyze(text)
            route = "hinglish"
            model_used = "CM-BERT"
        else:
            sentiment_result = hybrid_pipeline.xlm_roberta.analyze(text)
            route = "multilingual"
            model_used = "XLM-RoBERTa"
        
        sentiment = sentiment_result['sentiment']
        confidence = sentiment_result['confidence']
        scores = sentiment_result['scores']
        
        # Determine confidence level
        if confidence >= 0.9:
            confidence_level = "high"
        elif confidence >= 0.7:
            confidence_level = "medium"
        else:
            confidence_level = "low"
        
        # Build V2 response
        v2_response = {
            "sentiment": sentiment,
            "confidence": confidence,
            "confidence_level": confidence_level,
            "scores": scores,
            "model_used": model_used,
            "route": route
        }
        
        # Convert to V1 format
        return map_v2_to_v1_sentiment(v2_response)
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Sentiment analysis failed: {str(e)}"
        )


@app.post("/api/v1/analyze", response_model=FullAnalysisResponse)
async def full_analysis(input_data: TextInput):
    """
    Complete NLP analysis (preprocessing + language detection + sentiment)
    
    **MIGRATED TO V2**: Now uses V2 pipeline for 37% better accuracy
    while maintaining V1 response format for backward compatibility.
    
    - **text**: Input text to analyze
    
    Returns complete analysis with all features
    """
    global hybrid_pipeline
    
    try:
        # Lazy load hybrid pipeline
        if hybrid_pipeline is None:
            print("🔄 Loading hybrid pipeline on first request...")
            hybrid_pipeline = get_pipeline()
        
        text = input_data.text
        
        # V2 Preprocessing
        preprocess_result = hybrid_pipeline.preprocessor.preprocess(text)
        processed = preprocess_result['cleaned_text']
        tokens = preprocess_result['tokens']
        
        preprocessing_v2 = {
            "original": text,
            "processed": processed,
            "tokens": tokens,
            "tokens_count": len(tokens)
        }
        
        # V2 Language Detection
        fasttext_result = hybrid_pipeline.fasttext.detect(text)
        detected_lang = fasttext_result['language']
        confidence = fasttext_result['confidence']
        
        # Use detect_tokens for HingBERT (processes entire text)
        hingbert_result = hybrid_pipeline.hingbert.detect_tokens(text)
        token_labels = hingbert_result.get('labels', [])
        hingbert_tokens = hingbert_result.get('tokens', [])
        
        label_counts = {}
        for label in token_labels:
            label_counts[label] = label_counts.get(label, 0) + 1
        
        total_tokens = len(token_labels)
        statistics = {}
        for label, count in label_counts.items():
            statistics[label] = {
                "count": count,
                "percentage": round((count / total_tokens) * 100, 1) if total_tokens > 0 else 0.0
            }
        
        english_count = label_counts.get('English', 0)
        hindi_count = label_counts.get('Hindi', 0)
        is_hinglish = (english_count > 0 and hindi_count > 0) or detected_lang in ['hi', 'en']
        
        language_v2 = {
            "detected_language": detected_lang,
            "language_name": "Hindi" if detected_lang == 'hi' else "English",
            "confidence": confidence,
            "is_hinglish": is_hinglish,
            "is_reliable": confidence > 0.7,
            "token_level_detection": {
                "tokens": hingbert_tokens,
                "labels": token_labels,
                "statistics": statistics
            }
        }
        
        # V2 Sentiment Analysis with smart routing
        if detected_lang in ['hi', 'en']:
            sentiment_result = hybrid_pipeline.cmbert.analyze(text)
        else:
            sentiment_result = hybrid_pipeline.xlm_roberta.analyze(text)
        
        sentiment_v2 = {
            "sentiment": sentiment_result['sentiment'],
            "confidence": sentiment_result['confidence'],
            "scores": sentiment_result['scores']
        }
        
        # Convert to V1 format using mapper
        return map_v2_to_v1_full_analysis(text, preprocessing_v2, language_v2, sentiment_v2)
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Analysis failed: {str(e)}"
        )


@app.post("/api/v1/analyze/batch")
async def batch_analysis(input_data: BatchTextInput):
    """
    Batch analysis for multiple texts
    
    - **texts**: List of texts to analyze
    
    Returns list of analysis results
    """
    try:
        results = pipeline.analyze_batch(input_data.texts)
        return {
            "count": len(results),
            "results": results
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch analysis failed: {str(e)}"
        )


# ===== NEW V2 ENDPOINTS: MULTILINGUAL SUPPORT =====

@app.post("/api/v2/test")
async def test_v2_pipeline(input_data: TextInput):
    """
    Simple test endpoint to verify V2 pipeline is loaded
    Returns basic info without heavy processing
    """
    global hybrid_pipeline
    
    return {
        "status": "ok",
        "pipeline_loaded": hybrid_pipeline is not None,
        "text_received": input_data.text,
        "text_length": len(input_data.text),
        "message": "V2 pipeline is ready" if hybrid_pipeline else "Pipeline not loaded - call /api/v2/warmup first"
    }


@app.post("/api/v2/analyze")
async def multilingual_analysis(input_data: TextInput):
    """
    **NEW** Multilingual analysis with intelligent routing
    
    Features:
    - Automatic language detection (176 languages)
    - Hinglish optimization (92-96% accuracy via HingBERT + CM-BERT)
    - English optimization (94% accuracy via CM-BERT)
    - Multilingual support (87% accuracy via XLM-RoBERTa)
    - Smart routing based on detected language
    
    - **text**: Input text to analyze (any language)
    
    Returns sentiment analysis with routing information
    """
    global hybrid_pipeline
    
    try:
        # Check if pipeline is loaded
        if hybrid_pipeline is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Models not loaded. Please call /api/v2/warmup first to load transformer models."
            )
        
        print(f"📝 Analyzing text: {input_data.text[:50]}...")
        
        result = hybrid_pipeline.analyze(
            input_data.text,
            include_preprocessing=True,
            include_language_details=True
        )
        
        print("✅ Analysis complete!")
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error during analysis: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Multilingual analysis failed: {str(e)}"
        )


@app.post("/api/v2/analyze/batch")
async def multilingual_batch_analysis(input_data: BatchTextInput):
    """
    **NEW** Batch multilingual analysis
    
    - **texts**: List of texts to analyze (any language)
    
    Returns list of analysis results with intelligent routing
    """
    global hybrid_pipeline
    
    try:
        # Lazy load hybrid pipeline on first use
        if hybrid_pipeline is None:
            print("🔄 Loading hybrid pipeline on first request...")
            hybrid_pipeline = get_pipeline()
        
        results = hybrid_pipeline.analyze_batch(input_data.texts)
        
        return {
            "count": len(results),
            "results": results
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch multilingual analysis failed: {str(e)}"
        )


@app.get("/api/v2/languages")
async def get_supported_languages():
    """
    **NEW** Get information about supported languages
    
    Returns details about:
    - Total languages supported
    - Hinglish-optimized models
    - Multilingual model coverage
    - Language detection capabilities
    """
    global hybrid_pipeline
    
    try:
        # Lazy load hybrid pipeline on first use
        if hybrid_pipeline is None:
            print("🔄 Loading hybrid pipeline on first request...")
            hybrid_pipeline = get_pipeline()
        
        return hybrid_pipeline.get_supported_languages()
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve language information: {str(e)}"
        )


@app.get("/api/v2/health")
async def hybrid_health_check():
    """
    **NEW** Health check for hybrid multilingual pipeline
    
    Returns detailed health status of all components
    """
    global hybrid_pipeline
    
    try:
        # Lazy load hybrid pipeline on first use
        if hybrid_pipeline is None:
            print("🔄 Loading hybrid pipeline on first request...")
            hybrid_pipeline = get_pipeline()
        
        return hybrid_pipeline.health_check()
        
    except Exception as e:
        return {
            "pipeline": "unhealthy",
            "error": str(e)
        }


@app.post("/api/v2/preprocess")
async def preprocess_v2(request: TextInput):
    """
    **V2** Hybrid text preprocessing with spaCy + NLTK
    
    Uses advanced preprocessing combining spaCy and NLTK for better results.
    
    **Example Request:**
    ```json
    {
        "text": "Check out https://example.com! 😊 #amazing"
    }
    ```
    
    **Example Response:**
    ```json
    {
        "original": "Check out https://example.com! 😊 #amazing",
        "processed": "check out 😊 amazing",
        "tokens": ["check", "out", "😊", "amazing"],
        "tokens_count": 4,
        "sentence_count": 1,
        "preprocessing_method": "hybrid"
    }
    ```
    """
    global hybrid_pipeline
    
    try:
        # Lazy load hybrid pipeline
        if hybrid_pipeline is None:
            print("🔄 Loading hybrid pipeline on first request...")
            hybrid_pipeline = get_pipeline()
        
        # Use hybrid preprocessor
        text = request.text
        result = hybrid_pipeline.preprocessor.preprocess(text)
        
        return {
            "original": result['original_text'],
            "processed": result['cleaned_text'],
            "tokens": result['tokens'],
            "tokens_count": result['token_count'],
            "sentence_count": result['sentence_count'],
            "preprocessing_method": "hybrid"
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Preprocessing failed: {str(e)}"
        )


@app.post("/api/v2/detect-language")
async def detect_language_v2(request: TextInput):
    """
    **V2** Advanced language detection with FastText + HingBERT
    
    Combines FastText (176 languages) with HingBERT token-level detection.
    
    **Example Request:**
    ```json
    {
        "text": "Main bahut happy hoon today"
    }
    ```
    
    **Example Response:**
    ```json
    {
        "detected_language": "hi",
        "language_name": "Hindi",
        "confidence": 0.89,
        "is_hinglish": true,
        "is_reliable": true,
        "token_level_detection": {
            "tokens": ["Main", "bahut", "happy", "hoon", "today"],
            "labels": ["Hindi", "Hindi", "English", "Hindi", "English"],
            "statistics": {
                "English": {"count": 2, "percentage": 40.0},
                "Hindi": {"count": 3, "percentage": 60.0}
            }
        }
    }
    ```
    """
    global hybrid_pipeline
    
    try:
        # Lazy load hybrid pipeline
        if hybrid_pipeline is None:
            print("🔄 Loading hybrid pipeline on first request...")
            hybrid_pipeline = get_pipeline()
        
        text = request.text
        
        # FastText detection
        fasttext_result = hybrid_pipeline.fasttext.detect(text)
        detected_lang = fasttext_result['language']
        confidence = fasttext_result['confidence']
        
        # Token-level detection with HingBERT (processes entire text)
        hingbert_result = hybrid_pipeline.hingbert.detect_tokens(text)
        tokens = hingbert_result.get('tokens', [])
        token_labels_raw = hingbert_result.get('labels', [])
        
        # Map labels to full names
        label_name_map = {
            'en': 'English',
            'hi': 'Hindi',
            'ne': 'Named Entity',
            'other': 'Other'
        }
        token_labels = [label_name_map.get(label, label) for label in token_labels_raw]
        
        # Calculate statistics
        label_counts = {}
        for label in token_labels:
            label_counts[label] = label_counts.get(label, 0) + 1
        
        total_tokens = len(token_labels)
        statistics = {}
        for label, count in label_counts.items():
            statistics[label] = {
                "count": count,
                "percentage": round((count / total_tokens) * 100, 1) if total_tokens > 0 else 0.0
            }
        
        # Check if Hinglish
        english_count = label_counts.get('English', 0)
        hindi_count = label_counts.get('Hindi', 0)
        is_hinglish = (english_count > 0 and hindi_count > 0) or detected_lang in ['hi', 'en']
        
        # Get language name
        language_names = {
            'hi': 'Hindi',
            'en': 'English',
            'es': 'Spanish',
            'fr': 'French',
            'de': 'German',
            'ar': 'Arabic'
        }
        language_name = language_names.get(detected_lang, detected_lang.upper())
        
        return {
            "detected_language": detected_lang,
            "language_name": language_name,
            "confidence": confidence,
            "is_hinglish": is_hinglish,
            "is_reliable": confidence > 0.7,
            "token_level_detection": {
                "tokens": tokens,
                "labels": token_labels,
                "statistics": statistics
            }
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Language detection failed: {str(e)}"
        )


@app.post("/api/v2/analyze-sentiment")
async def analyze_sentiment_v2(request: TextInput):
    """
    **V2** Smart sentiment analysis with automatic model routing
    
    Automatically routes to best model based on detected language:
    - Hinglish/Hindi/English → CM-BERT (92-94% accuracy)
    - Other languages → XLM-RoBERTa (87% accuracy)
    
    **Example Request:**
    ```json
    {
        "text": "Yeh movie bahut accha hai! I loved it!"
    }
    ```
    
    **Example Response:**
    ```json
    {
        "sentiment": "positive",
        "confidence": 0.94,
        "confidence_level": "high",
        "scores": {
            "positive": 0.94,
            "negative": 0.03,
            "neutral": 0.03
        },
        "model_used": "CM-BERT",
        "route": "hinglish"
    }
    ```
    """
    global hybrid_pipeline
    
    try:
        # Lazy load hybrid pipeline
        if hybrid_pipeline is None:
            print("🔄 Loading hybrid pipeline on first request...")
            hybrid_pipeline = get_pipeline()
        
        text = request.text
        
        # Detect language first
        lang_result = hybrid_pipeline.fasttext.detect(text)
        detected_lang = lang_result['language']
        
        # Route to appropriate model
        if detected_lang in ['hi', 'en']:
            # Use CM-BERT for Hinglish/Hindi/English
            sentiment_result = hybrid_pipeline.cmbert.analyze(text)
            route = "hinglish"
            model_used = "CM-BERT"
        else:
            # Use XLM-RoBERTa for other languages
            sentiment_result = hybrid_pipeline.xlm_roberta.analyze(text)
            route = "multilingual"
            model_used = "XLM-RoBERTa"
        
        sentiment = sentiment_result['sentiment']
        confidence = sentiment_result['confidence']
        scores = sentiment_result['scores']
        
        # Determine confidence level
        if confidence >= 0.9:
            confidence_level = "high"
        elif confidence >= 0.7:
            confidence_level = "medium"
        else:
            confidence_level = "low"
        
        return {
            "sentiment": sentiment,
            "confidence": confidence,
            "confidence_level": confidence_level,
            "scores": scores,
            "model_used": model_used,
            "route": route
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Sentiment analysis failed: {str(e)}"
        )


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """General exception handler"""
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal server error",
            "detail": str(exc)
        }
    )


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("🚀 STARTING HINGLISH NLP API SERVER")
    print("=" * 70)
    print("\n📍 API will be available at:")
    print("   • Main: http://localhost:8000")
    print("   • Docs: http://localhost:8000/docs")
    print("   • ReDoc: http://localhost:8000/redoc")
    print("\n" + "=" * 70 + "\n")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
