"""
Response Mappers for V1 to V2 Migration

This module provides utility functions to convert V2 API responses
to V1 format for backward compatibility.
"""

from typing import Dict, List, Any


def map_v2_to_v1_preprocessing(v2_response: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert V2 preprocessing response to V1 format.
    
    V2 Response:
        {
            "original": str,
            "processed": str,
            "tokens": List[str],
            "tokens_count": int,
            "sentence_count": int,
            "preprocessing_method": str
        }
    
    V1 Response:
        {
            "original": str,
            "cleaned": str,
            "tokens": List[str],
            "token_count": int
        }
    
    Args:
        v2_response: V2 preprocessing response
        
    Returns:
        V1 formatted response
    """
    return {
        "original": v2_response.get("original", ""),
        "cleaned": v2_response.get("processed", ""),
        "tokens": v2_response.get("tokens", []),
        "token_count": v2_response.get("tokens_count", 0)
    }


def map_v2_to_v1_language(v2_response: Dict[str, Any], format_type: str = "full") -> Dict[str, Any]:
    """
    Convert V2 language detection response to V1 format.
    
    V2 Response:
        {
            "detected_language": str,
            "language_name": str,
            "confidence": float,
            "is_hinglish": bool,
            "is_reliable": bool,
            "token_level_detection": {
                "tokens": List[str],
                "labels": List[str],
                "statistics": Dict
            }
        }
    
    V1 Response:
        {
            "tokens": List[str],
            "labels": List[str],
            "statistics": {
                "lang1": {"count": int, "percentage": float},
                "lang2": {"count": int, "percentage": float}
            },
            "is_code_mixed": bool,
            "dominant_language": str
        }
    
    Args:
        v2_response: V2 language detection response
        format_type: "full" for simple format (full analysis), "tokens" for token-level format (language detection endpoint)
        
    Returns:
        V1 formatted response
    """
    # Determine which format to return based on format_type parameter
    if format_type == "full":
        # Full analysis format - return basic language info for V1 LanguageDetectionResponse
        detected_lang = v2_response.get("detected_language", "en")
        
        # Map language code to V1 format: "lang1" (English) or "lang2" (Hindi)
        v1_language = "lang1" if detected_lang == "en" else "lang2"
        
        # Check if it's an Indian language
        indian_languages = ["hi", "mr", "ta", "te", "kn", "ml", "bn", "gu", "pa", "ur"]
        is_indian = detected_lang in indian_languages
        
        return {
            "language": v1_language,
            "confidence": v2_response.get("confidence", 0.0),
            "is_hinglish": v2_response.get("is_hinglish", False),
            "is_indian_language": is_indian
        }
    else:
        # Token-level format for language detection endpoint
        token_level = v2_response.get("token_level_detection", {})
        tokens = token_level.get("tokens", [])
        labels = token_level.get("labels", [])
        
        # Map V2 labels to V1 format
        # V2: "English", "Hindi", "Named Entity", "Other"
        # V1: "lang1" (English), "lang2" (Hindi), "ne", "other"
        label_mapping = {
            "English": "lang1",
            "Hindi": "lang2",
            "Named Entity": "ne",
            "Other": "other"
        }
        
        v1_labels = [label_mapping.get(label, label.lower()) for label in labels]
        
        # Calculate statistics in V1 format
        lang1_count = v1_labels.count("lang1")
        lang2_count = v1_labels.count("lang2")
        total = len(v1_labels) if v1_labels else 1
        
        statistics = {
            "lang1": {
                "count": lang1_count,
                "percentage": round((lang1_count / total) * 100, 1)
            },
            "lang2": {
                "count": lang2_count,
                "percentage": round((lang2_count / total) * 100, 1)
            }
        }
        
        # Determine dominant language
        dominant_language = "lang1" if lang1_count >= lang2_count else "lang2"
        
        # Determine if code-mixed
        is_code_mixed = v2_response.get("is_hinglish", False) or (lang1_count > 0 and lang2_count > 0)
        
        return {
            "tokens": tokens,
            "labels": v1_labels,
            "statistics": statistics,
            "is_code_mixed": is_code_mixed,
            "dominant_language": dominant_language
        }


def map_v2_to_v1_sentiment(v2_response: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert V2 sentiment response to V1 format.
    
    V2 Response:
        {
            "sentiment": str,
            "confidence": float,
            "confidence_level": str,
            "scores": {
                "positive": float,
                "negative": float,
                "neutral": float
            },
            "model_used": str,
            "route": str
        }
    
    V1 Response:
        {
            "label": str,
            "confidence": float,
            "scores": {
                "positive": float,
                "negative": float
            }
        }
    
    Args:
        v2_response: V2 sentiment response
        
    Returns:
        V1 formatted response (simple for full analysis, detailed for sentiment-only)
    """
    scores = v2_response.get("scores", {})
    confidence = v2_response.get("confidence", 0.0)
    
    # V1 only had positive/negative (no neutral)
    v1_scores = {
        "positive": scores.get("positive", 0.0),
        "negative": scores.get("negative", 0.0)
    }
    
    # For full analysis, return simple format with just label and score
    # For sentiment-only endpoint, return detailed format with confidence and scores
    return {
        "label": v2_response.get("sentiment", "neutral"),
        "score": confidence,  # SimpleSentimentResponse uses 'score'
        "confidence": confidence,  # SentimentResponse uses 'confidence'  
        "scores": v1_scores
    }


def map_v2_to_v1_full_analysis(
    original_text: str,
    preprocessing: Dict[str, Any],
    language: Dict[str, Any],
    sentiment: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Combine V2 responses into V1 full analysis format.
    
    Args:
        original_text: The original input text
        preprocessing: V2 preprocessing response
        language: V2 language detection response
        sentiment: V2 sentiment response
        
    Returns:
        V1 formatted full analysis response
    """
    return {
        "original_text": original_text,
        "cleaned_text": preprocessing.get("processed", ""),
        "tokens": preprocessing.get("tokens", []),
        "token_count": preprocessing.get("tokens_count", 0),
        "language_detection": map_v2_to_v1_language(language),
        "sentiment": map_v2_to_v1_sentiment(sentiment)
    }


def map_v2_to_v1_batch(v2_batch_response: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert V2 batch response to V1 format.
    
    Args:
        v2_batch_response: V2 batch analysis response
        
    Returns:
        V1 formatted batch response
    """
    results = v2_batch_response.get("results", [])
    
    v1_results = []
    for result in results:
        # Each V2 result contains all analysis data
        v1_result = {
            "original_text": result.get("text", ""),
            "cleaned_text": result.get("preprocessing", {}).get("processed_text", ""),
            "tokens": result.get("preprocessing", {}).get("processed_text", "").split(),
            "token_count": result.get("preprocessing", {}).get("tokens_count", 0),
            "language_detection": {
                "labels": [],
                "statistics": {
                    "lang1": {"count": 0, "percentage": 0.0},
                    "lang2": {"count": 0, "percentage": 0.0}
                },
                "is_code_mixed": result.get("language_detection", {}).get("is_hinglish", False),
                "dominant_language": "lang1"
            },
            "sentiment": {
                "label": result.get("sentiment", "neutral"),
                "confidence": result.get("confidence", 0.0),
                "scores": {
                    "positive": result.get("scores", {}).get("positive", 0.0),
                    "negative": result.get("scores", {}).get("negative", 0.0)
                }
            }
        }
        v1_results.append(v1_result)
    
    return {
        "count": len(v1_results),
        "results": v1_results
    }
