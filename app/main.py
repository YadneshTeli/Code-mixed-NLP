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

from app.pipeline import HinglishNLPPipeline
from app.preprocessing.cleaner import HinglishCleaner
from app.language_detection.detector import LanguageDetector
from app.sentiment_analysis.analyzer import SentimentAnalyzer

# Initialize FastAPI app
app = FastAPI(
    title="Hinglish NLP API",
    description="REST API for processing code-mixed Hindi-English (Hinglish) text",
    version="1.0.0",
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
pipeline = HinglishNLPPipeline()
cleaner = HinglishCleaner()
language_detector = LanguageDetector()
sentiment_analyzer = SentimentAnalyzer()
print("✅ API ready!\n")


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
    labels: List[str]
    statistics: Dict
    is_code_mixed: bool
    dominant_language: str


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


class FullAnalysisResponse(BaseModel):
    """Complete NLP analysis result"""
    original_text: str
    cleaned_text: str
    tokens: List[str]
    token_count: int
    language_detection: LanguageDetectionResponse
    sentiment: SentimentResponse


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
        "name": "Hinglish NLP API",
        "version": "1.0.0",
        "description": "REST API for processing code-mixed Hindi-English text",
        "endpoints": {
            "health": "/health",
            "docs": "/docs",
            "preprocessing": "/api/v1/preprocess",
            "language_detection": "/api/v1/detect-language",
            "sentiment_analysis": "/api/v1/analyze-sentiment",
            "full_analysis": "/api/v1/analyze"
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "version": "1.0.0",
        "modules": {
            "preprocessing": True,
            "language_detection": True,
            "sentiment_analysis": True
        }
    }


@app.post("/api/v1/preprocess", response_model=PreprocessingResponse)
async def preprocess_text(input_data: TextInput):
    """
    Preprocess text (cleaning and tokenization)
    
    - **text**: Input text to preprocess
    
    Returns cleaned text and tokens
    """
    try:
        result = cleaner.process(input_data.text)
        return {
            "original": result['original'],
            "cleaned": result['cleaned'],
            "tokens": result['tokens'],
            "token_count": result['token_count']
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Preprocessing failed: {str(e)}"
        )


@app.post("/api/v1/detect-language", response_model=LanguageDetectionWithTokensResponse)
async def detect_language(input_data: TextInput):
    """
    Detect language at token level
    
    - **text**: Input text to analyze
    
    Returns language labels for each token
    """
    try:
        result = language_detector.detect_text(input_data.text)
        dominant = language_detector.get_dominant_language(input_data.text)
        
        return {
            "tokens": result['tokens'],
            "labels": result['labels'],
            "statistics": result['statistics'],
            "is_code_mixed": result['is_code_mixed'],
            "dominant_language": dominant
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Language detection failed: {str(e)}"
        )


@app.post("/api/v1/analyze-sentiment", response_model=SentimentResponse)
async def analyze_sentiment(input_data: TextInput):
    """
    Analyze sentiment of text
    
    - **text**: Input text to analyze
    
    Returns sentiment label and confidence scores
    """
    try:
        result = sentiment_analyzer.analyze(input_data.text)
        
        return {
            "label": result['sentiment'],
            "confidence": result['confidence'],
            "scores": result['scores']
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Sentiment analysis failed: {str(e)}"
        )


@app.post("/api/v1/analyze", response_model=FullAnalysisResponse)
async def full_analysis(input_data: TextInput):
    """
    Complete NLP analysis (preprocessing + language detection + sentiment)
    
    - **text**: Input text to analyze
    
    Returns complete analysis with all features
    """
    try:
        result = pipeline.process(input_data.text)
        
        return {
            "original_text": result['original_text'],
            "cleaned_text": result['cleaned_text'],
            "tokens": result['tokens'],
            "token_count": result['token_count'],
            "language_detection": result['language_detection'],
            "sentiment": result['sentiment']
        }
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
        results = pipeline.process_batch(input_data.texts)
        return {
            "count": len(results),
            "results": results
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch analysis failed: {str(e)}"
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
