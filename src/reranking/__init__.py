"""
Reranking Module for RAG Applications
Provides advanced re-ranking capabilities for improving retrieval accuracy
"""

from .reranker import (
    # Main class
    AdvancedRAGReranker,
    
    # Individual reranker implementations
    BaseReranker,
    CrossEncoderReranker,
    ColBERTReranker,
    CohereReranker,
    HybridReranker,
    
    # Data structure
    Document,
)

# Module metadata
__version__ = "1.0.0"
__author__ = "Your Name"
__description__ = "Advanced multi-stage reranker for RAG applications"

# Default configuration
DEFAULT_CROSS_ENCODER_MODEL = 'BAAI/bge-reranker-large'
DEFAULT_STRATEGY = 'hybrid'
DEFAULT_TOP_K = 10
DEFAULT_MIN_SCORE_THRESHOLD = 0.3

# Available models for cross-encoder
AVAILABLE_MODELS = {
    'minilm': 'cross-encoder/ms-marco-MiniLM-L-12-v2',
    'multibert': 'cross-encoder/ms-marco-MultiBERT-L-12',
    'bge-large': 'BAAI/bge-reranker-large',
    'bge-base': 'BAAI/bge-reranker-base',
    'bge-v2-m3': 'BAAI/bge-reranker-v2-m3',
}

# Available strategies
STRATEGIES = ['cross_encoder', 'colbert', 'cohere', 'hybrid']

# Public API
__all__ = [
    # Main interface
    'AdvancedRAGReranker',
    
    # Individual rerankers (for advanced users)
    'BaseReranker',
    'CrossEncoderReranker', 
    'ColBERTReranker',
    'CohereReranker',
    'HybridReranker',
    
    # Data structures
    'Document',
    
    # Helper functions
    'create_reranker',
    'get_available_models',
    'validate_strategy',
    
    # Constants
    'AVAILABLE_MODELS',
    'STRATEGIES',
    'DEFAULT_STRATEGY',
    'DEFAULT_TOP_K',
]


def create_reranker(strategy: str = DEFAULT_STRATEGY, 
                   model_name: str = None,
                   cohere_api_key: str = None,
                   **kwargs) -> AdvancedRAGReranker:
    """
    Factory function to create a reranker with specified configuration
    
    Args:
        strategy: One of 'cross_encoder', 'colbert', 'cohere', or 'hybrid'
        model_name: Specific model name for cross-encoder (or key from AVAILABLE_MODELS)
        cohere_api_key: Optional API key for Cohere reranker
        **kwargs: Additional arguments passed to AdvancedRAGReranker
    
    Returns:
        Configured AdvancedRAGReranker instance
    
    Example:
        >>> reranker = create_reranker('hybrid')
        >>> reranker = create_reranker('cross_encoder', model_name='bge-large')
    """
    # Validate strategy
    if strategy not in STRATEGIES:
        raise ValueError(f"Strategy must be one of {STRATEGIES}, got {strategy}")
    
    # Handle model name shortcuts
    if model_name and model_name in AVAILABLE_MODELS:
        model_name = AVAILABLE_MODELS[model_name]
    
    # Create and return reranker
    return AdvancedRAGReranker(
        strategy=strategy,
        cohere_api_key=cohere_api_key,
        **kwargs
    )


def get_available_models() -> dict:
    """
    Get dictionary of available pre-trained models
    
    Returns:
        Dictionary mapping model shortcuts to full model names
    """
    return AVAILABLE_MODELS.copy()


def validate_strategy(strategy: str) -> bool:
    """
    Validate if a strategy is supported
    
    Args:
        strategy: Strategy name to validate
    
    Returns:
        True if strategy is valid, False otherwise
    """
    return strategy in STRATEGIES


def get_recommended_model(use_case: str = 'general') -> str:
    """
    Get recommended model based on use case
    
    Args:
        use_case: One of 'general', 'speed', 'accuracy', 'multilingual'
    
    Returns:
        Model name string
    """
    recommendations = {
        'general': 'BAAI/bge-reranker-base',
        'speed': 'cross-encoder/ms-marco-MiniLM-L-12-v2',
        'accuracy': 'BAAI/bge-reranker-large',
        'multilingual': 'BAAI/bge-reranker-v2-m3',
    }
    return recommendations.get(use_case, DEFAULT_CROSS_ENCODER_MODEL)


# Optional: Initialize logging for the module
import logging

# Create module logger
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())  # Prevent logging unless configured by user


# Optional: Quick validation on import
def _validate_imports():
    """Validate that required packages are installed"""
    required_packages = []
    
    try:
        import sentence_transformers
    except ImportError:
        required_packages.append('sentence-transformers')
    
    try:
        import transformers
    except ImportError:
        required_packages.append('transformers')
    
    try:
        import torch
    except ImportError:
        required_packages.append('torch')
    
    if required_packages:
        logger.warning(
            f"Missing required packages: {', '.join(required_packages)}. "
            f"Install with: uv add {' '.join(required_packages)}"
        )
        return False
    return True


# Validate on import (optional - can be commented out if you prefer)
_imports_valid = _validate_imports()

# Convenience function for quick setup
def quick_setup(chroma_collection=None) -> AdvancedRAGReranker:
    """
    Quick setup with optimal defaults for immediate use
    
    Args:
        chroma_collection: Optional ChromaDB collection for testing
    
    Returns:
        Configured reranker ready to use
    
    Example:
        >>> from src.reranking import quick_setup
        >>> reranker = quick_setup()
        >>> results = reranker.rerank_from_chroma(query, chroma_results)
    """
    import os
    
    # Check for Cohere API key in environment
    cohere_key = os.getenv('COHERE_API_KEY', None)
    
    # Use hybrid if Cohere is available, otherwise use best free model
    strategy = 'hybrid' if cohere_key else 'cross_encoder'
    
    logger.info(f"Quick setup using {strategy} strategy")
    
    return create_reranker(
        strategy=strategy,
        cohere_api_key=cohere_key,
        enable_feedback_loop=True
    )


# Module initialization message (optional)
if __name__ == "__main__":
    print(f"Reranking module v{__version__}")
    print(f"Available strategies: {', '.join(STRATEGIES)}")
    print(f"Available models: {', '.join(AVAILABLE_MODELS.keys())}")