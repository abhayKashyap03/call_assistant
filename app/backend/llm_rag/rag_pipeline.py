"""RAG (Retrieval Augmented Generation) module with advanced search capabilities.

This module implements a comprehensive RAG system with:
1. Document chunking with token-based overlap
2. Sentence-transformers embeddings (all-MiniLM-L6-v2)
3. FAISS vector storage with flat L2 distance
4. Search-R1 algorithm for enhanced retrieval
5. Multi-stage retrieval and reranking

Math & Theory:
- Vector embeddings map text to high-dimensional space where semantic similarity
  correlates with geometric distance (cosine similarity or L2 distance)
- FAISS uses optimized nearest neighbor search algorithms
- Search-R1 implements retrieval-focused reasoning with multi-hop search
"""


import logging
import os
import re
import pickle
from typing import List, Dict, Tuple, Union, Optional, Any, TYPE_CHECKING
from dataclasses import dataclass
import numpy as np

from sentence_transformers import SentenceTransformer
import faiss

from app.backend.llm_rag.extraction.base_extractor import Document
from app.backend.llm_rag.extraction.file_extractor import FileExtractor
from app.backend.llm_rag.extraction.web_extractor import WebExtractor


# Setup logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class ContentExtractorFactory:
    """
    Factory to select the appropriate extractor based on the source type.
    """
    def __init__(self):
        self.file_extractor = FileExtractor()
        self.web_extractor = WebExtractor()

    def _get_extractor(self, src: str):
        if os.path.isdir(src) or os.path.isfile(src):
            return self.file_extractor
        return self.web_extractor

    def extract(self, src: str) -> Document:
        return self._get_extractor(src).extract(src)


@dataclass
class DocumentChunk:
    """Represents a chunk of text with metadata."""
    text: str
    source: str
    chunk_id: int
    start_char: int
    end_char: int


@dataclass
class SearchResult:
    """Represents a search result with relevance scores."""
    chunk: DocumentChunk
    similarity_score: float
    relevance_score: float  # After reranking
    reasoning_path: List[str]  # For Search-R1


class TextChunker:
    """
    Advanced text chunking with token-based overlap.
    
    Theory:
    Token-based chunking preserves semantic coherence better than character-based.
    Overlap ensures important information isn't lost at chunk boundaries.
    
    Algorithm:
    1. Tokenize text using approximate token counting (words ≈ tokens * 0.75)
    2. Create sliding windows with specified overlap
    3. Preserve sentence boundaries when possible
    """
    
    def __init__(self, chunk_size: int = 400, overlap: int = 50):
        print('!!! --- TextChunker initialized --- !!!')
        self.chunk_size = chunk_size
        self.overlap = overlap
        
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count (1 token ≈ 0.75 words for English)."""
        words = len(text.split())
        return int(words / 0.75)
    
    def _split_by_sentences(self, text: str) -> List[str]:
        """Split text into sentences using regex."""
        # Handle common sentence endings
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def chunk_text(self, text: str, source: str = "") -> List[DocumentChunk]:
        """
        Chunk text with token-based overlap.
        Returns a list of DocumentChunk objects.
        """
        sentences = self._split_by_sentences(text)
        chunks = []
        chunk_id = 0
        current_chunk = ""
        current_sentences = []
        start_char = 0
        for sentence in sentences:
            test_chunk = current_chunk + " " + sentence if current_chunk else sentence
            if self._estimate_tokens(test_chunk) <= self.chunk_size:
                current_chunk = test_chunk
                current_sentences.append(sentence)
            else:
                if current_chunk:
                    end_char = start_char + len(current_chunk)
                    # All extra info in metadata
                    chunks.append(DocumentChunk(
                        text=current_chunk.strip(),
                        source=source,
                        chunk_id=chunk_id,
                        start_char=start_char,
                        end_char=end_char
                        ))
                    chunk_id += 1
                if self.overlap > 0 and current_sentences:
                    overlap_text = ""
                    overlap_sentences = []
                    for i in range(len(current_sentences) - 1, -1, -1):
                        test_overlap = current_sentences[i] + " " + overlap_text if overlap_text else current_sentences[i]
                        if self._estimate_tokens(test_overlap) <= self.overlap:
                            overlap_text = test_overlap
                            overlap_sentences.insert(0, current_sentences[i])
                        else:
                            break
                    current_chunk = overlap_text + " " + sentence if overlap_text else sentence
                    current_sentences = overlap_sentences + [sentence]
                    start_char = end_char - len(overlap_text) if overlap_text else end_char
                else:
                    current_chunk = sentence
                    current_sentences = [sentence]
                    start_char = end_char if chunks else 0
        if current_chunk:
            end_char = start_char + len(current_chunk)
            chunks.append(DocumentChunk(
                text=current_chunk.strip(),
                source=source,
                chunk_id=chunk_id,
                start_char=start_char,
                end_char=end_char
            ))
        return chunks


class EmbeddingEngine:
    """
    Handles text embedding using sentence-transformers.
    
    Theory:
    Sentence-transformers create dense vector representations where:
    - Similar texts have similar vectors (high cosine similarity)
    - The model is trained on semantic textual similarity tasks
    - all-MiniLM-L6-v2 provides good balance of speed vs quality (384 dimensions)
    
    Mathematical Foundation:
    Given text T, embedding E = f(T) where f is the transformer model.
    Similarity between texts A and B: sim(A,B) = cos(E_A, E_B) = (E_A · E_B) / (||E_A|| ||E_B||)
    """
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        print('!!! --- EmbeddingEngine initialized --- !!!')
        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        logger.info(f"Embedding dimension: {self.dimension}")
    
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings
            
        Returns:
            NumPy array of embeddings (shape: [n_texts, embedding_dim])
        """
        if not texts:
            return np.empty((0, self.dimension))
        
        logger.info(f"Generating embeddings for {len(texts)} texts")
        embeddings = self.model.encode(texts, show_progress_bar=True)
        return embeddings
    
    def embed_single(self, text: str) -> np.ndarray:
        """Generate embedding for a single text."""
        return self.model.encode([text])[0]


class VectorStore:
    """
    FAISS-based vector storage with flat L2 index.
    
    Theory:
    FAISS (Facebook AI Similarity Search) implements efficient nearest neighbor search.
    Flat L2 index performs exhaustive search using L2 (Euclidean) distance:
    
    L2 distance: d(x,y) = sqrt(sum((x_i - y_i)^2))
    
    For normalized vectors, L2 distance relates to cosine similarity:
    cos_sim(x,y) = 1 - (L2_dist(x,y)^2 / 2)
    
    Advantages of Flat L2:
    - Exact search (no approximation)
    - Simple and reliable
    - Good for smaller datasets (<1M vectors)
    """
    
    def __init__(self, dimension: int):
        print('!!! --- VectorStore initialized --- !!!')
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)  # L2 distance index
        self.chunks: List[DocumentChunk] = []
        self.embeddings: Optional[np.ndarray] = None
    
    def add_chunks(self, chunks: List[DocumentChunk], embeddings: np.ndarray):
        """
        Add chunks and their embeddings to the vector store.
        
        Args:
            chunks: List of DocumentChunk objects
            embeddings: Corresponding embeddings array
        """
        if len(chunks) != len(embeddings):
            raise ValueError("Number of chunks must match number of embeddings")
        
        # Normalize embeddings for better similarity computation
        normalized_embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        self.index.add(normalized_embeddings.astype('float32'))
        self.chunks.extend(chunks)
        
        if self.embeddings is None:
            self.embeddings = normalized_embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, normalized_embeddings])
        
        logger.info(f"Added {len(chunks)} chunks to vector store. Total: {len(self.chunks)}")
    
    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Tuple[DocumentChunk, float]]:
        """
        Search for similar chunks.
        
        Args:
            query_embedding: Query vector
            k: Number of results to return
            
        Returns:
            List of (chunk, distance) tuples
        """
        if len(self.chunks) == 0:
            return []
        
        # Normalize query embedding
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        query_embedding = query_embedding.reshape(1, -1).astype('float32')
        
        # Search using FAISS
        distances, indices = self.index.search(query_embedding, min(k, len(self.chunks)))
        
        results = []
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < len(self.chunks):  # Valid index
                results.append((self.chunks[idx], float(dist)))
        
        return results
    
    def save(self, filepath: str):
        """Save the vector store to disk."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, f"{filepath}.faiss")
        
        # Save chunks and metadata
        store_data = {
            'chunks': self.chunks,
            'dimension': self.dimension,
            'embeddings': self.embeddings
        }
        
        with open(f"{filepath}.pkl", 'wb') as f:
            pickle.dump(store_data, f)
        
        logger.info(f"Vector store saved to {filepath}")
    
    def load(self, filepath: str):
        """Load the vector store from disk."""
        # Load FAISS index
        self.index = faiss.read_index(f"{filepath}.faiss")
        
        # Load chunks and metadata
        with open(f"{filepath}.pkl", 'rb') as f:
            store_data = pickle.load(f)
        
        self.chunks = store_data['chunks']
        self.dimension = store_data['dimension']
        self.embeddings = store_data.get('embeddings')
        
        logger.info(f"Vector store loaded from {filepath}. Contains {len(self.chunks)} chunks")


class SearchR1Engine:
    """
    Search-R1: Retrieval-focused reasoning for enhanced search.
    
    Theory:
    Search-R1 extends traditional RAG with multi-step reasoning:
    1. Initial retrieval based on query
    2. Reasoning about retrieved content
    3. Iterative refinement of search queries
    4. Reranking based on reasoning paths
    
    Algorithm:
    - Generate multiple search perspectives for a query
    - Perform parallel retrieval for each perspective
    - Use LLM to reason about relevance and connections
    - Rerank results based on reasoning quality
    
    This implements a simplified version focusing on query expansion and reranking.
    """
    
    def generate_search_queries(self, original_query: str) -> List[str]:
        """
        Generate multiple search perspectives for better retrieval.
        
        Uses query expansion techniques:
        1. Decompose complex queries into subqueries
        2. Generate alternative phrasings
        3. Add domain-specific context
        """
        queries = [original_query]  # Always include original
        
        # Simple query expansion (can be enhanced with LLM)
        words = original_query.lower().split()
        
        # Add questions format
        if not original_query.startswith(('what', 'how', 'why', 'when', 'where', 'who')):
            queries.extend([
                f"What is {original_query}?",
                f"How does {original_query} work?",
                f"Why is {original_query} important?"
            ])
        
        # Add keyword-focused variants
        if len(words) > 2:
            # Focus on key terms
            for i in range(len(words)):
                if len(words[i]) > 3:  # Skip short words
                    focused_query = " ".join([w for j, w in enumerate(words) if j == i or len(w) > 3])
                    if focused_query != original_query:
                        queries.append(focused_query)
        
        return list(set(queries))  # Remove duplicates
    
    def rerank_results(self, query: str, results: List[Tuple[DocumentChunk, float]]) -> List[SearchResult]:
        """
        Rerank search results using reasoning about relevance.
        
        Simple heuristic-based reranking (can be enhanced with LLM):
        1. Boost results with query terms in multiple positions
        2. Consider chunk metadata (length, source quality)
        3. Penalize very similar chunks (diversity)
        """
        if not results:
            return []
        
        query_words = set(query.lower().split())
        reranked = []
        
        for chunk, similarity_score in results:
            # Calculate relevance score
            chunk_words = set(chunk.text.lower().split())
            
            # Term frequency score
            term_overlap = len(query_words.intersection(chunk_words)) / len(query_words) if query_words else 0
            
            # Length penalty (prefer moderate length chunks)
            length_score = 1.0
            chunk_length = len(chunk.text.split())
            if chunk_length < 50:  # Too short
                length_score = 0.8
            elif chunk_length > 500:  # Too long
                length_score = 0.9
            
            # Combine scores
            relevance_score = (
                0.6 * (1.0 - similarity_score / 2.0) +  # Convert L2 distance to similarity
                0.3 * term_overlap +
                0.1 * length_score
            )
            
            reasoning_path = [
                f"Similarity score: {1.0 - similarity_score / 2.0:.3f}",
                f"Term overlap: {term_overlap:.3f}",
                f"Length score: {length_score:.3f}",
                f"Final relevance: {relevance_score:.3f}"
            ]
            
            reranked.append(SearchResult(
                chunk=chunk,
                similarity_score=similarity_score,
                relevance_score=relevance_score,
                reasoning_path=reasoning_path
            ))
        
        # Sort by relevance score (descending)
        reranked.sort(key=lambda x: x.relevance_score, reverse=True)
        
        return reranked


class RAGPipeline:
    """
    Unified RAG pipeline: ingestion, retrieval, and LLM-based answer generation.
    """
    def __init__(self, chunk_size: int = 400, overlap: int = 50, embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        print('!!! --- RAGPipeline initialized --- !!!')
        self.extractor_factory = ContentExtractorFactory()
        self.chunker = TextChunker(chunk_size=chunk_size, overlap=overlap)
        self.embedding_engine = EmbeddingEngine(model_name=embedding_model)
        self.search_engine = SearchR1Engine()
        self.vector_store = None
        try:
            self.load_index()
        except Exception:
            pass

    def ingest_single(self, src: Union[str, DocumentChunk]):
        """
        Ingest a document from a file path or URL, extract content, chunk, embed, and add to vector store.
        """
        text = self.extractor_factory.extract(src).content
        if not text:
            raise ValueError(f"No text content extracted from {src}")
        chunks = self.chunker.chunk_text(text, source=src)
        embeddings = self.embedding_engine.embed_texts([c.text for c in chunks])
        if self.vector_store is None:
            self.vector_store = VectorStore(self.embedding_engine.dimension)
        self.vector_store.add_chunks(chunks, embeddings)
        self.save_index()
        logger.info(f"Added document ({len(chunks)} chunks) to knowledge base")
        print('\n', self.get_stats(), '\n')

    def ingest_multiple(self, documents: List[str]):
        """
        Add multiple documents (list of paths) to the knowledge base.
        """
        all_chunks = []
        documents = [self.extractor_factory.extract(doc) for doc in documents]
        for doc in documents:
            text = doc.content
            source = doc.metadata['source']
            chunks = self.chunker.chunk_text(text, source=source)
            self.ingest_single(chunks)
        logger.info(f"Added {len(documents)} documents ({len(all_chunks)} chunks) to knowledge base")

    def search(self, query: str, k: int = 5) -> List[SearchResult]:
        if self.vector_store is None:
            raise RuntimeError("Vector store is empty. Ingest documents first.")
        queries = self.search_engine.generate_search_queries(query)
        all_results = []
        for q in queries:
            q_emb = self.embedding_engine.embed_single(q)
            results = self.vector_store.search(q_emb, k=k)
            all_results.extend(results)
        seen = set()
        deduped = []
        for chunk, score in all_results:
            chunk_id = chunk.chunk_id
            if chunk_id not in seen:
                deduped.append((chunk, score))
                seen.add(chunk_id)
        reranked = self.search_engine.rerank_results(query, deduped)
        return reranked

    def save_index(self, filepath: str = "data/vector_index"):
        if self.vector_store:
            self.vector_store.save(filepath)
            logger.info(f"Vector index saved to {filepath}")

    def load_index(self, filepath: str = "data/vector_index"):
        if os.path.exists(f"{filepath}.faiss") and os.path.exists(f"{filepath}.pkl"):
            if self.vector_store is None:
                self.vector_store = VectorStore(self.embedding_engine.dimension)
            self.vector_store.load(filepath)
            logger.info(f"Vector index loaded from {filepath}")
        else:
            logger.warning(f"Vector index not found at {filepath}")

    def get_stats(self) -> Dict[str, Any]:
        if not self.vector_store or not self.vector_store.chunks:
            return {"total_chunks": 0, "embedding_dimension": self.embedding_engine.dimension, "sources": [], "avg_chunk_length": 0}
        return {
            "total_chunks": len(self.vector_store.chunks),
            "embedding_dimension": self.embedding_engine.dimension,
            "sources": list(set(chunk.source for chunk in self.vector_store.chunks)),
            "avg_chunk_length": np.mean([len(chunk.text.split()) for chunk in self.vector_store.chunks])
        }
