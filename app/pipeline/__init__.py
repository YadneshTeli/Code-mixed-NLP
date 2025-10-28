"""
Pipeline Module

Contains both legacy and new multilingual NLP pipelines:
- HinglishNLPPipeline: Original pipeline (backward compatibility)
- HybridNLPPipeline: New multilingual pipeline with smart routing
"""

from app.pipeline.hybrid_nlp_pipeline import HybridNLPPipeline, get_pipeline

__all__ = ['HybridNLPPipeline', 'get_pipeline']
