"""
Pydantic models for request/response schemas
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional, List, Dict, Any
from datetime import datetime


class InputText(BaseModel):
    """Simple text input model for basic endpoints"""
    text: str = Field(
        ..., 
        min_length=1,
        max_length=1000,
        description="Input text to process",
        examples=["Yaar kal meeting bahut tight thi"]
    )

    @field_validator('text')
    @classmethod
    def text_not_empty(cls, v: str) -> str:
        """Validate that text is not just whitespace"""
        if not v or not v.strip():
            raise ValueError('Text cannot be empty or just whitespace')
        return v.strip()


class TextInput(BaseModel):
    """Extended input model for full pipeline processing"""
    text: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="Input text to process",
        examples=["Yaar kal meeting bahut tight thi"]
    )
    include_translation: bool = Field(
        default=False,
        description="Whether to include translation in response"
    )
    target_language: Optional[str] = Field(
        default="en",
        description="Target language for translation (en or hi)",
        pattern="^(en|hi)$"
    )

    @field_validator('text')
    @classmethod
    def text_not_empty(cls, v: str) -> str:
        """Validate that text is not just whitespace"""
        if not v or not v.strip():
            raise ValueError('Text cannot be empty or just whitespace')
        return v.strip()

    @field_validator('target_language')
    @classmethod
    def validate_language(cls, v: Optional[str]) -> str:
        """Validate target language code"""
        if v and v not in ['en', 'hi']:
            raise ValueError('Target language must be either "en" or "hi"')
        return v or "en"


class SentimentResult(BaseModel):
    """Sentiment analysis result"""
    label: str = Field(
        ...,
        description="Sentiment label (positive, negative, neutral)",
        examples=["negative"]
    )
    score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score for the sentiment",
        examples=[0.78]
    )


class AnalysisResult(BaseModel):
    """Complete analysis result from the pipeline"""
    original_text: str = Field(
        ...,
        description="Original input text",
        examples=["Yaar kal meeting bahut tight thi"]
    )
    tokens: List[str] = Field(
        ...,
        description="List of tokens from text",
        examples=[["Yaar", "kal", "meeting", "bahut", "tight", "thi"]]
    )
    languages: List[str] = Field(
        ...,
        description="Language code for each token (en, hi)",
        examples=[["hi", "hi", "en", "hi", "en", "hi"]]
    )
    sentiment: SentimentResult = Field(
        ...,
        description="Sentiment analysis result"
    )
    translation: Optional[str] = Field(
        default=None,
        description="Translated text (if requested)",
        examples=["Friend, yesterday's meeting was very tiring."]
    )
    processing_time: float = Field(
        ...,
        ge=0.0,
        description="Time taken to process request (in seconds)",
        examples=[1.23]
    )
    timestamp: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat(),
        description="Timestamp of processing"
    )


class LanguageDetectionResult(BaseModel):
    """Language detection only result"""
    tokens: List[str] = Field(
        ...,
        description="List of tokens",
        examples=[["Yaar", "kal", "meeting", "bahut", "tight", "thi"]]
    )
    languages: List[str] = Field(
        ...,
        description="Language codes for each token",
        examples=[["hi", "hi", "en", "hi", "en", "hi"]]
    )
    language_distribution: Dict[str, int] = Field(
        default_factory=dict,
        description="Count of tokens per language",
        examples=[{"hi": 4, "en": 2}]
    )


class SentimentAnalysisResult(BaseModel):
    """Sentiment analysis only result"""
    text: str = Field(
        ...,
        description="Input text",
        examples=["Yaar kal meeting bahut tight thi"]
    )
    sentiment: SentimentResult = Field(
        ...,
        description="Sentiment result"
    )


class TranslationResult(BaseModel):
    """Translation only result"""
    original: str = Field(
        ...,
        description="Original text",
        examples=["Yaar kal meeting bahut tight thi"]
    )
    translated: str = Field(
        ...,
        description="Translated text",
        examples=["Friend, yesterday's meeting was very tiring."]
    )
    source_language: str = Field(
        default="mixed",
        description="Source language (mixed for code-mixed text)"
    )
    target_language: str = Field(
        ...,
        description="Target language code",
        examples=["en"]
    )


class ErrorResponse(BaseModel):
    """Error response model"""
    error: str = Field(
        ...,
        description="Error type",
        examples=["ValidationError"]
    )
    detail: str = Field(
        ...,
        description="Detailed error message",
        examples=["Text cannot be empty"]
    )
    timestamp: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat(),
        description="Timestamp of error"
    )


class HealthCheckResponse(BaseModel):
    """Health check response"""
    status: str = Field(
        default="healthy",
        description="Service health status"
    )
    version: str = Field(
        default="1.0.0",
        description="API version"
    )
    timestamp: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat(),
        description="Current timestamp"
    )
    models_loaded: Dict[str, bool] = Field(
        default_factory=dict,
        description="Status of loaded ML models"
    )


# Example usage and validation
if __name__ == "__main__":
    # Test InputText
    input_text = InputText(text="Yaar kal meeting bahut tight thi")
    print(f"InputText: {input_text.model_dump_json(indent=2)}")
    
    # Test TextInput
    text_input = TextInput(
        text="Yaar kal meeting bahut tight thi",
        include_translation=True,
        target_language="en"
    )
    print(f"\nTextInput: {text_input.model_dump_json(indent=2)}")
    
    # Test AnalysisResult
    sentiment = SentimentResult(label="negative", score=0.78)
    analysis = AnalysisResult(
        original_text="Yaar kal meeting bahut tight thi",
        tokens=["Yaar", "kal", "meeting", "bahut", "tight", "thi"],
        languages=["hi", "hi", "en", "hi", "en", "hi"],
        sentiment=sentiment,
        translation="Friend, yesterday's meeting was very tiring.",
        processing_time=1.23
    )
    print(f"\nAnalysisResult: {analysis.model_dump_json(indent=2)}")
