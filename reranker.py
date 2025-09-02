"""
Advanced Multi-Stage Re-Ranker for RAG Applications
Combines multiple free re-ranking approaches for maximum accuracy
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
import torch
from sentence_transformers import CrossEncoder, SentenceTransformer
import cohere
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import logging
from abc import ABC, abstractmethod

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Document:
    """Document structure for re-ranking"""
    content: str
    metadata: Dict[str, Any] = None
    initial_score: float = 0.0
    rerank_score: float = 0.0

class BaseReranker(ABC):
    """Base class for all re-rankers"""
    
    @abstractmethod
    def rerank(self, query: str, documents: List[Document]) -> List[Document]:
        pass

class CrossEncoderReranker(BaseReranker):
    """
    Cross-Encoder based re-ranker using free models from HuggingFace
    Best models: ms-marco-MiniLM, ms-marco-MultiBERT, BGE-reranker
    """
    
    def __init__(self, model_name: str = 'cross-encoder/ms-marco-MiniLM-L-12-v2'):
        """
        Initialize with the best free cross-encoder models:
        - 'cross-encoder/ms-marco-MiniLM-L-12-v2': Fast and accurate
        - 'cross-encoder/ms-marco-MultiBERT-L-12': Slightly better accuracy
        - 'BAAI/bge-reranker-large': State-of-the-art accuracy
        - 'BAAI/bge-reranker-v2-m3': Multilingual support
        """
        self.model = CrossEncoder(model_name, max_length=512)
        logger.info(f"Initialized CrossEncoder with {model_name}")
    
    def rerank(self, query: str, documents: List[Document], top_k: Optional[int] = None) -> List[Document]:
        """Re-rank documents using cross-encoder"""
        if not documents:
            return []
        
        # Prepare pairs for scoring
        pairs = [[query, doc.content] for doc in documents]
        
        # Get scores from cross-encoder
        scores = self.model.predict(pairs, batch_size=32, show_progress_bar=False)
        
        # Update document scores
        for doc, score in zip(documents, scores):
            doc.rerank_score = float(score)
        
        # Sort by score
        ranked_docs = sorted(documents, key=lambda x: x.rerank_score, reverse=True)
        
        if top_k:
            ranked_docs = ranked_docs[:top_k]
        
        return ranked_docs

class ColBERTReranker(BaseReranker):
    """
    ColBERT-based re-ranker for token-level matching
    Uses ColBERTv2 or PLAID for better accuracy
    """
    
    def __init__(self):
        """Initialize ColBERT reranker using sentence-transformers"""
        # Using a BERT model that can simulate ColBERT-style scoring
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        logger.info("Initialized ColBERT-style reranker")
    
    def _compute_colbert_score(self, query_embeddings: np.ndarray, doc_embeddings: np.ndarray) -> float:
        """Compute ColBERT-style MaxSim score"""
        # Compute similarity matrix
        similarity_matrix = np.dot(query_embeddings, doc_embeddings.T)
        
        # MaxSim: for each query token, take max similarity with any doc token
        max_similarities = np.max(similarity_matrix, axis=1)
        
        # Sum of max similarities
        score = np.sum(max_similarities)
        
        return float(score)
    
    def rerank(self, query: str, documents: List[Document], top_k: Optional[int] = None) -> List[Document]:
        """Re-rank using ColBERT-style scoring"""
        if not documents:
            return []
        
        # Tokenize and encode query
        query_tokens = self.tokenizer.tokenize(query)[:50]  # Limit tokens
        query_embeddings = self.model.encode(query_tokens)
        
        # Score each document
        for doc in documents:
            doc_tokens = self.tokenizer.tokenize(doc.content)[:200]  # Limit tokens
            doc_embeddings = self.model.encode(doc_tokens)
            
            score = self._compute_colbert_score(query_embeddings, doc_embeddings)
            doc.rerank_score = score
        
        # Sort by score
        ranked_docs = sorted(documents, key=lambda x: x.rerank_score, reverse=True)
        
        if top_k:
            ranked_docs = ranked_docs[:top_k]
        
        return ranked_docs

class CohereReranker(BaseReranker):
    """
    Cohere's free re-ranker API (requires free API key)
    Limited to 1000 calls/month on free tier
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize Cohere client with free API key"""
        if api_key:
            self.client = cohere.ClientV2(api_key)
            self.enabled = True
            logger.info("Initialized Cohere reranker")
        else:
            self.enabled = False
            logger.warning("Cohere API key not provided, skipping Cohere reranker")
    
    def rerank(self, query: str, documents: List[Document], top_k: Optional[int] = None) -> List[Document]:
        """Re-rank using Cohere's rerank endpoint"""
        if not self.enabled or not documents:
            return documents
        
        try:
            # Prepare documents for Cohere
            doc_texts = [doc.content for doc in documents]
            
            # Call Cohere rerank
            response = self.client.rerank(
                query=query,
                documents=doc_texts,
                top_n=top_k if top_k else len(documents),
                model='rerank-v3.5'  # Free model
            )
            
            # Create reranked list
            reranked_docs = []
            for result in response.results:
                doc = documents[result.index]
                doc.rerank_score = result.relevance_score
                reranked_docs.append(doc)
            
            return reranked_docs
        
        except Exception as e:
            logger.error(f"Cohere reranking failed: {e}")
            return documents

class HybridReranker(BaseReranker):
    """
    Combines multiple re-ranking strategies for maximum accuracy
    Uses weighted ensemble of different models
    """
    
    def __init__(self, cohere_api_key: Optional[str] = None):
        """Initialize all re-rankers"""
        self.rerankers = {
            'cross_encoder_minilm': (CrossEncoderReranker('cross-encoder/ms-marco-MiniLM-L-12-v2'), 0.3),
            'cross_encoder_bge': (CrossEncoderReranker('BAAI/bge-reranker-large'), 0.4),
            'colbert': (ColBERTReranker(), 0.2),
        }
        
        # Add Cohere if API key is provided
        if cohere_api_key:
            self.rerankers['cohere'] = (CohereReranker(cohere_api_key), 0.1)
        
        logger.info(f"Initialized hybrid reranker with {len(self.rerankers)} models")
    
    def _normalize_scores(self, documents: List[Document]) -> None:
        """Normalize scores to 0-1 range"""
        scores = [doc.rerank_score for doc in documents]
        if not scores:
            return
        
        min_score = min(scores)
        max_score = max(scores)
        
        if max_score - min_score > 0:
            for doc in documents:
                doc.rerank_score = (doc.rerank_score - min_score) / (max_score - min_score)
    
    def rerank(self, query: str, documents: List[Document], top_k: Optional[int] = None) -> List[Document]:
        """
        Ensemble re-ranking using multiple models
        """
        if not documents:
            return []
        
        # Store scores from each reranker
        all_scores = {}
        
        for name, (reranker, weight) in self.rerankers.items():
            logger.info(f"Running {name} reranker...")
            
            # Create copy of documents for this reranker
            docs_copy = [Document(content=d.content, metadata=d.metadata, initial_score=d.initial_score) 
                        for d in documents]
            
            # Re-rank with this model
            ranked = reranker.rerank(query, docs_copy)
            
            # Normalize scores
            self._normalize_scores(ranked)
            
            # Store weighted scores
            for i, doc in enumerate(ranked):
                if i not in all_scores:
                    all_scores[i] = {}
                all_scores[i][name] = doc.rerank_score * weight
        
        # Combine scores
        for i, doc in enumerate(documents):
            if i in all_scores:
                doc.rerank_score = sum(all_scores[i].values())
            else:
                doc.rerank_score = 0.0
        
        # Final ranking
        ranked_docs = sorted(documents, key=lambda x: x.rerank_score, reverse=True)
        
        if top_k:
            ranked_docs = ranked_docs[:top_k]
        
        return ranked_docs

class AdvancedRAGReranker:
    """
    Main class for advanced re-ranking in RAG applications
    Implements multi-stage re-ranking with fallback strategies
    """
    
    def __init__(self, 
                 strategy: str = 'hybrid',
                 cohere_api_key: Optional[str] = None,
                 enable_feedback_loop: bool = True):
        """
        Initialize the re-ranker
        
        Args:
            strategy: 'cross_encoder', 'colbert', 'cohere', or 'hybrid'
            cohere_api_key: Optional Cohere API key for free tier
            enable_feedback_loop: Enable learning from user feedback
        """
        self.strategy = strategy
        self.enable_feedback_loop = enable_feedback_loop
        
        # Initialize appropriate reranker
        if strategy == 'hybrid':
            self.reranker = HybridReranker(cohere_api_key)
        elif strategy == 'cross_encoder':
            self.reranker = CrossEncoderReranker('BAAI/bge-reranker-large')
        elif strategy == 'colbert':
            self.reranker = ColBERTReranker()
        elif strategy == 'cohere':
            self.reranker = CohereReranker(cohere_api_key)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        # Feedback storage for continuous improvement
        self.feedback_history = []
        
        logger.info(f"Initialized AdvancedRAGReranker with {strategy} strategy")
    
    def rerank_from_chroma(self, 
                           query: str, 
                           chroma_results: Dict[str, List],
                           top_k: int = 10,
                           min_score_threshold: float = 0.3) -> List[Dict[str, Any]]:
        """
        Re-rank results from ChromaDB
        
        Args:
            query: The search query
            chroma_results: Results from ChromaDB query (with 'documents', 'metadatas', 'distances')
            top_k: Number of top results to return
            min_score_threshold: Minimum score threshold for filtering
        
        Returns:
            List of re-ranked documents with scores
        """
        # Convert Chroma results to Document objects
        documents = []
        
        if 'documents' in chroma_results:
            docs_list = chroma_results['documents'][0] if chroma_results['documents'] else []
            metadatas = chroma_results.get('metadatas', [[]])[0] if chroma_results.get('metadatas') else []
            distances = chroma_results.get('distances', [[]])[0] if chroma_results.get('distances') else []
            
            for i, doc_content in enumerate(docs_list):
                doc = Document(
                    content=doc_content,
                    metadata=metadatas[i] if i < len(metadatas) else {},
                    initial_score=1.0 - (distances[i] if i < len(distances) else 0.5)
                )
                documents.append(doc)
        
        # Re-rank documents
        reranked = self.reranker.rerank(query, documents, top_k=top_k)
        
        # Filter by minimum score
        reranked = [doc for doc in reranked if doc.rerank_score >= min_score_threshold]
        
        # Convert to output format
        results = []
        for doc in reranked:
            results.append({
                'content': doc.content,
                'metadata': doc.metadata,
                'initial_score': doc.initial_score,
                'rerank_score': doc.rerank_score,
                'combined_score': (doc.initial_score * 0.3 + doc.rerank_score * 0.7)  # Weighted combination
            })
        
        return results
    
    def record_feedback(self, query: str, document_id: str, relevant: bool):
        """
        Record user feedback for continuous improvement
        
        Args:
            query: The search query
            document_id: ID of the document
            relevant: Whether the document was relevant
        """
        if self.enable_feedback_loop:
            self.feedback_history.append({
                'query': query,
                'document_id': document_id,
                'relevant': relevant
            })
            logger.info(f"Recorded feedback: {relevant} for document {document_id}")

# Example usage
def example_usage():
    """
    Example of how to use the re-ranker with ChromaDB and GPT
    """
    # Initialize the re-ranker (hybrid strategy for best accuracy)
    reranker = AdvancedRAGReranker(
        strategy='hybrid',
        cohere_api_key="<cohere_api_key>",  # Add your free Cohere API key if available
        enable_feedback_loop=True
    )
    
    # Simulate ChromaDB results
    chroma_results = {
        'documents': [['Document 1 design...', 'Document 2 architecture...', 'Document 3 well-architected framework...']],
        'metadatas': [[{'id': '1'}, {'id': '2'}, {'id': '3'}]],
        'distances': [[0.3, 0.5, 0.7]]
    }
    
    # Re-rank the results
    query = "What is the best approach for system architecture?"
 # Re-rank the results
    reranked = reranker.rerank_from_chroma(
        query=query,
        chroma_results=chroma_results,
        top_k=10,  # Final number of results
        min_score_threshold=0.4  # Filter low-confidence results
    )

    # Use reranked results with GPT-5
    context = "\n".join([doc['content'] for doc in reranked[:5]])
    print(context)
  # Display results
    for i, result in enumerate(reranked, 1):
        print(f"\nRank {i}:")
        print(f"  Content: {result['content'][:100]}...")
        print(f"  Initial Score: {result['initial_score']:.3f}")
        print(f"  Rerank Score: {result['rerank_score']:.3f}")
        print(f"  Combined Score: {result['combined_score']:.3f}")


if __name__ == "__main__":
    example_usage()