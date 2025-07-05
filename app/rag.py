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

import os
import json
import pickle
import logging
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
import numpy as np

# Core ML libraries
from sentence_transformers import SentenceTransformer
import faiss

# Text processing
import re
from collections import defaultdict

# Google Gemini integration
import google.generativeai as genai

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DocumentChunk:
    """Represents a chunk of text with metadata."""
    text: str
    source: str
    chunk_id: int
    start_char: int
    end_char: int
    metadata: Dict[str, Any]


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
        
        Args:
            text: Input text to chunk
            source: Source identifier
            
        Returns:
            List of DocumentChunk objects
        """
        sentences = self._split_by_sentences(text)
        chunks = []
        chunk_id = 0
        
        current_chunk = ""
        current_sentences = []
        start_char = 0
        
        for sentence in sentences:
            # Check if adding this sentence exceeds chunk size
            test_chunk = current_chunk + " " + sentence if current_chunk else sentence
            
            if self._estimate_tokens(test_chunk) <= self.chunk_size:
                current_chunk = test_chunk
                current_sentences.append(sentence)
            else:
                # Save current chunk if it has content
                if current_chunk:
                    end_char = start_char + len(current_chunk)
                    chunks.append(DocumentChunk(
                        text=current_chunk.strip(),
                        source=source,
                        chunk_id=chunk_id,
                        start_char=start_char,
                        end_char=end_char,
                        metadata={"sentence_count": len(current_sentences)}
                    ))
                    chunk_id += 1
                
                # Start new chunk with overlap
                if self.overlap > 0 and current_sentences:
                    # Calculate how many sentences to keep for overlap
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
        
        # Don't forget the last chunk
        if current_chunk:
            end_char = start_char + len(current_chunk)
            chunks.append(DocumentChunk(
                text=current_chunk.strip(),
                source=source,
                chunk_id=chunk_id,
                start_char=start_char,
                end_char=end_char,
                metadata={"sentence_count": len(current_sentences)}
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
    
    def __init__(self, llm_client=None):
        self.llm_client = llm_client
    
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
    
    def rerank_results(self, query: str, results: List[Tuple[DocumentChunk, float]], 
                      reasoning_context: str = "") -> List[SearchResult]:
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


class RAGService:
    """
    Complete RAG service with Search-R1 integration.
    
    Architecture:
    1. Document Processing: TextChunker → EmbeddingEngine
    2. Storage: VectorStore (FAISS + metadata)
    3. Retrieval: SearchR1Engine → Multi-query search + reranking
    4. Generation: Gemini with retrieved context
    """
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """Initialize the RAG service."""
        self.chunker = TextChunker(chunk_size=400, overlap=50)
        self.embedding_engine = EmbeddingEngine(model_name)
        self.vector_store = VectorStore(self.embedding_engine.dimension)
        self.search_engine = SearchR1Engine()
        
        # Initialize Gemini client
        self.gemini_client = None
        if os.getenv('GOOGLE_API_KEY'):
            genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
            self.gemini_client = genai.GenerativeModel('gemini-pro')
            logger.info("Gemini client initialized")
    
    def add_documents(self, documents: List[Dict[str, str]]):
        """
        Add documents to the knowledge base.
        
        Args:
            documents: List of dicts with 'text' and 'source' keys
        """
        all_chunks = []
        
        for doc in documents:
            chunks = self.chunker.chunk_text(doc['content'], doc.get('source', 'unknown'))
            all_chunks.extend(chunks)
        
        print("done adding chunks")

        if all_chunks:
            # Generate embeddings
            texts = [chunk.text for chunk in all_chunks]
            embeddings = self.embedding_engine.embed_texts(texts)
            
            print("done embeddings")

            # Add to vector store
            self.vector_store.add_chunks(all_chunks, embeddings)
            
            print("done adding to vector store")
            
            logger.info(f"Added {len(documents)} documents ({len(all_chunks)} chunks) to knowledge base")
    
    def retrieve(self, query: str, k: int = 5) -> List[SearchResult]:
        """
        Retrieve relevant context using Search-R1 algorithm.
        
        Args:
            query: User query
            k: Number of results to return
            
        Returns:
            List of SearchResult objects with reasoning
        """
        if len(self.vector_store.chunks) == 0:
            logger.warning("No documents in knowledge base")
            return []
        
        # Step 1: Generate multiple search queries (Search-R1)
        search_queries = self.search_engine.generate_search_queries(query)
        logger.info(f"Generated {len(search_queries)} search queries: {search_queries}")
        
        # Step 2: Perform retrieval for each query
        all_results = {}
        for search_query in search_queries:
            query_embedding = self.embedding_engine.embed_single(search_query)
            results = self.vector_store.search(query_embedding, k=k*2)  # Get more for diversity
            
            for chunk, score in results:
                chunk_id = id(chunk)
                if chunk_id not in all_results or score < all_results[chunk_id][1]:
                    all_results[chunk_id] = (chunk, score)
        
        # Step 3: Rerank using Search-R1 reasoning
        combined_results = list(all_results.values())
        reranked_results = self.search_engine.rerank_results(
            query, combined_results[:k*3]  # Limit for efficiency
        )
        
        return reranked_results[:k]
    
    def generate_response(self, query: str, context: Optional[List[SearchResult]] = None) -> str:
        """
        Generate response using Gemini with retrieved context.
        
        Args:
            query: User query
            context: Retrieved context (optional)
            
        Returns:
            Generated response
        """
        if context is None:
            context = self.retrieve(query)
        
        if not context:
            return "I don't have relevant information to answer your question."
        
        # Prepare context for LLM
        context_parts = []
        for i, result in enumerate(context, 1):
            context_parts.append(
                f"Context {i} (relevance: {result.relevance_score:.3f}):\n"
                f"Source: {result.chunk.source}\n"
                f"Text: {result.chunk.text}\n"
            )
        
        context_str = "\n---\n".join(context_parts)
        
        # Generate response
        if self.gemini_client:
            prompt = f"""
Based on the following context, please answer the user's question. Be accurate and cite the relevant sources.

Context:
{context_str}

Question: {query}

Answer:"""
            
            try:
                response = self.gemini_client.generate_content(prompt)
                return response.text
            except Exception as e:
                logger.error(f"Error generating response with Gemini: {e}")
                return f"Error generating response: {e}"
        else:
            # Fallback response without LLM
            return f"Based on the retrieved context:\n\n{context_str}\n\nQuestion: {query}\n\nI found relevant information but need an API key to generate a detailed response."
    
    def answer(self, query: str, k: int = 5) -> Tuple[str, float, List[str]]:
        """
        Answer a query using retrieved context and Gemini.
        
        Workflow:
        1. Retrieve top-K chunks using Search-R1 algorithm
        2. Build prompt with system persona and citations
        3. Call Gemini API with context and query
        4. Return answer, confidence score, and sources list
        
        Args:
            query: User query
            k: Number of chunks to retrieve
            
        Returns:
            A tuple of (answer, confidence, sources)
        """
        # Step 1: Retrieve top-K chunks
        retrieved_context = self.retrieve(query, k)
        sources = [result.chunk.source for result in retrieved_context]
        
        # If no context found, return early
        if not retrieved_context:
            return ("I don't have relevant information to answer your query.", 0.0, [])
        
        try:
            if self.gemini_client:
                # Step 2: Build prompt with system persona and citations
                context_str = "\n\n".join(
                    f"[Source: {result.chunk.source}]\n{result.chunk.text}"
                    for result in retrieved_context
                )
                
                system_persona = """You are a knowledgeable AI assistant that provides accurate, well-sourced answers. 
Always cite your sources when providing information. Be concise but thorough in your responses."""
                
                prompt = f"""{system_persona}

Based on the following context sources, please answer the user's question. 
Cite the relevant sources in your answer using [Source: filename] format.

Context:
{context_str}

Question: {query}

Answer:"""
                
                # Step 3: Call Gemini API
                response = self.gemini_client.generate_content(prompt)
                
                # Step 4: Calculate confidence based on retrieval scores and response quality
                avg_relevance = sum(result.relevance_score for result in retrieved_context) / len(retrieved_context)
                
                # Simple confidence estimation (can be enhanced with actual logprobs if available)
                confidence = min(0.95, avg_relevance * 0.8 + 0.2)  # Scale to 0.2-0.95 range
                
                return (response.text, confidence, sources)
            else:
                # Fallback when Gemini client is not configured
                fallback_answer = f"""Based on the retrieved context, I found relevant information from the following sources: {', '.join(set(sources))}.

However, I need a Google API key (GOOGLE_API_KEY environment variable) to provide a detailed generated response.

Retrieved context:
{chr(10).join(f'- {result.chunk.text[:200]}...' for result in retrieved_context[:3])}"""
                
                avg_relevance = sum(result.relevance_score for result in retrieved_context) / len(retrieved_context)
                confidence = avg_relevance * 0.5  # Lower confidence without LLM generation
                
                return (fallback_answer, confidence, sources)
                
        except Exception as e:
            logger.error(f"Error in answer generation: {e}")
            return (f"Error generating answer: {str(e)}", 0.0, sources)
    
    def save_index(self, filepath: str = "data/vector.index"):
        """Save the vector index to disk."""
        self.vector_store.save(filepath)
        logger.info(f"Vector index saved to {filepath}")
    
    def load_index(self, filepath: str = "data/vector.index"):
        """Load the vector index from disk."""
        if os.path.exists(f"{filepath}.faiss") and os.path.exists(f"{filepath}.pkl"):
            self.vector_store.load(filepath)
            logger.info(f"Vector index loaded from {filepath}")
        else:
            logger.warning(f"Vector index not found at {filepath}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the knowledge base."""
        return {
            "total_chunks": len(self.vector_store.chunks),
            "embedding_dimension": self.embedding_engine.dimension,
            "sources": list(set(chunk.source for chunk in self.vector_store.chunks)),
            "avg_chunk_length": np.mean([len(chunk.text.split()) for chunk in self.vector_store.chunks]) if self.vector_store.chunks else 0
        }


# Convenience functions for backward compatibility
def retrieve(query: str, k: int = 5, rag_service: Optional[RAGService] = None) -> List[SearchResult]:
    """Convenience function for retrieval."""
    if rag_service is None:
        # Create a default service and try to load existing index
        rag_service = RAGService()
        rag_service.load_index()
    
    return rag_service.retrieve(query, k)


# Example usage and testing functions
if __name__ == "__main__":
    # Example usage
    rag = RAGService()
    
    # Add some sample documents
    sample_docs = [
        {
            "text": "Machine learning is a subset of artificial intelligence that enables computers to learn without being explicitly programmed. It uses algorithms to analyze data, identify patterns, and make predictions or decisions.",
            "source": "ml_basics.txt"
        },
        {
            "text": "Natural language processing (NLP) is a branch of AI that helps computers understand, interpret and manipulate human language. It combines computational linguistics with statistical and machine learning models.",
            "source": "nlp_intro.txt"
        }
    ]
    
    rag.add_documents(sample_docs)
    rag.save_index()
    
    # Test retrieval
    results = rag.retrieve("What is machine learning?", k=3)
    for result in results:
        print(f"Score: {result.relevance_score:.3f}")
        print(f"Text: {result.chunk.text[:100]}...")
        print(f"Reasoning: {result.reasoning_path}")
        print("---")
