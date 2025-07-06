#!/usr/bin/env python3
"""
Demonstration of the RAG system with Search-R1 integration.

This script shows:
1. Document ingestion and chunking
2. Vector embedding generation
3. FAISS index creation and storage
4. Search-R1 enhanced retrieval
5. Response generation
"""

import os
import sys
from pathlib import Path

# Add the app directory to the Python path
sys.path.append(str(Path(__file__).parent / "app"))

from rag import RAGService, TextChunker, EmbeddingEngine, VectorStore, SearchR1Engine

def demonstrate_text_chunking():
    """Demonstrate the text chunking process."""
    print("=" * 60)
    print("DEMONSTRATION 1: TEXT CHUNKING")
    print("=" * 60)
    
    chunker = TextChunker(chunk_size=100, overlap=20)  # Smaller for demo
    
    sample_text = """
    Machine learning is a subset of artificial intelligence (AI) that enables computers 
    to learn and make decisions without being explicitly programmed. It involves 
    algorithms that can analyze data, identify patterns, and make predictions or 
    decisions based on the patterns they discover.
    
    There are three main types of machine learning: supervised learning, unsupervised 
    learning, and reinforcement learning. Supervised learning uses labeled training 
    data to learn a mapping from input to output. Unsupervised learning finds hidden 
    patterns in data without labeled examples. Reinforcement learning learns through 
    interaction with an environment, receiving rewards or penalties for actions.
    
    Deep learning is a subset of machine learning that uses artificial neural networks 
    with multiple layers (hence "deep") to model and understand complex patterns in data. 
    It has been particularly successful in areas like image recognition, natural language 
    processing, and speech recognition.
    """
    
    chunks = chunker.chunk_text(sample_text, "ml_overview.txt")
    
    print(f"Original text length: {len(sample_text.split())} words")
    print(f"Number of chunks created: {len(chunks)}")
    print("\nChunks:")
    
    for i, chunk in enumerate(chunks):
        print(f"\nChunk {i+1} (ID: {chunk.chunk_id}):")
        print(f"  Text: {chunk.text[:100]}...")
        print(f"  Source: {chunk.source}")
        print(f"  Start char: {chunk.start_char}, End char: {chunk.end_char}")
        print(f"  Metadata: {chunk.metadata}")

def demonstrate_embeddings():
    """Demonstrate the embedding generation process."""
    print("\n" + "=" * 60)
    print("DEMONSTRATION 2: EMBEDDING GENERATION")
    print("=" * 60)
    
    embedding_engine = EmbeddingEngine()
    
    sample_texts = [
        "Machine learning enables computers to learn without explicit programming.",
        "Deep learning uses neural networks with multiple layers.",
        "Natural language processing helps computers understand human language.",
        "Computer vision allows machines to interpret visual information."
    ]
    
    print(f"Generating embeddings for {len(sample_texts)} texts...")
    embeddings = embedding_engine.embed_texts(sample_texts)
    
    print(f"Embedding shape: {embeddings.shape}")
    print(f"Embedding dimension: {embedding_engine.dimension}")
    
    # Show similarity between first two texts
    import numpy as np
    similarity = np.dot(embeddings[0], embeddings[1]) / (
        np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
    )
    print(f"\nCosine similarity between first two texts: {similarity:.4f}")
    print(f"Text 1: {sample_texts[0]}")
    print(f"Text 2: {sample_texts[1]}")

def demonstrate_vector_store():
    """Demonstrate vector storage and search."""
    print("\n" + "=" * 60)
    print("DEMONSTRATION 3: VECTOR STORE & SEARCH")
    print("=" * 60)
    
    # Create sample documents
    documents = [
        {
            "text": "Machine learning is a method of data analysis that automates analytical model building. It is a branch of artificial intelligence based on the idea that systems can learn from data, identify patterns and make decisions with minimal human intervention.",
            "source": "ml_definition.txt"
        },
        {
            "text": "Deep learning is part of a broader family of machine learning methods based on artificial neural networks with representation learning. Learning can be supervised, semi-supervised or unsupervised.",
            "source": "deep_learning.txt"
        },
        {
            "text": "Natural language processing (NLP) is a subfield of linguistics, computer science, and artificial intelligence concerned with the interactions between computers and human language.",
            "source": "nlp_definition.txt"
        },
        {
            "text": "Computer vision is an interdisciplinary scientific field that deals with how computers can gain high-level understanding from digital images or videos.",
            "source": "computer_vision.txt"
        }
    ]
    
    # Initialize RAG service
    rag = RAGService()
    rag.add_documents(documents)
    
    # Test queries
    queries = [
        "What is machine learning?",
        "How does deep learning work?",
        "artificial intelligence applications"
    ]
    
    for query in queries:
        print(f"\nQuery: '{query}'")
        results = rag.retrieve(query, k=3)
        
        for i, result in enumerate(results, 1):
            print(f"\nResult {i}:")
            print(f"  Relevance Score: {result.relevance_score:.4f}")
            print(f"  Similarity Score: {result.similarity_score:.4f}")
            print(f"  Source: {result.chunk.source}")
            print(f"  Text: {result.chunk.text[:100]}...")
            print(f"  Reasoning: {' | '.join(result.reasoning_path)}")

def demonstrate_search_r1():
    """Demonstrate Search-R1 query expansion."""
    print("\n" + "=" * 60)
    print("DEMONSTRATION 4: SEARCH-R1 QUERY EXPANSION")
    print("=" * 60)
    
    search_engine = SearchR1Engine()
    
    test_queries = [
        "machine learning algorithms",
        "neural networks",
        "AI applications",
        "data science techniques"
    ]
    
    for query in test_queries:
        expanded = search_engine.generate_search_queries(query)
        print(f"\nOriginal query: '{query}'")
        print("Expanded queries:")
        for i, expanded_query in enumerate(expanded, 1):
            print(f"  {i}. {expanded_query}")

def demonstrate_full_rag_pipeline():
    """Demonstrate the complete RAG pipeline."""
    print("\n" + "=" * 60)
    print("DEMONSTRATION 5: COMPLETE RAG PIPELINE")
    print("=" * 60)
    
    # Create comprehensive knowledge base
    knowledge_base = [
        {
            "text": """
            Artificial Intelligence (AI) is a broad field of computer science focused on building smart machines 
            capable of performing tasks that typically require human intelligence. AI systems can learn, reason, 
            perceive, and in some cases, interact naturally with humans. The field includes various subfields 
            such as machine learning, natural language processing, computer vision, and robotics.
            """,
            "source": "ai_overview.txt"
        },
        {
            "text": """
            Machine Learning is a subset of AI that enables computers to learn and improve from experience 
            without being explicitly programmed. Instead of following pre-programmed instructions, ML systems 
            build mathematical models based on training data to make predictions or decisions. Common types 
            include supervised learning, unsupervised learning, and reinforcement learning.
            """,
            "source": "ml_detailed.txt"
        },
        {
            "text": """
            Deep Learning is a specialized area of machine learning that uses artificial neural networks 
            with multiple layers to model and understand complex patterns in data. These deep neural networks 
            can automatically learn hierarchical representations of data, making them particularly effective 
            for tasks like image recognition, speech processing, and natural language understanding.
            """,
            "source": "deep_learning_detailed.txt"
        },
        {
            "text": """
            Natural Language Processing (NLP) combines computational linguistics with machine learning and 
            deep learning models to help computers understand, interpret, and generate human language. 
            Applications include machine translation, sentiment analysis, chatbots, text summarization, 
            and question-answering systems.
            """,
            "source": "nlp_detailed.txt"
        }
    ]
    
    # Initialize and populate RAG system
    rag = RAGService()
    rag.add_documents(knowledge_base)
    
    # Save the index
    rag.save_index("data/vector.index")
    print(f"Vector index saved to data/vector.index")
    
    # Show statistics
    stats = rag.get_stats()
    print(f"\nKnowledge Base Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Test complex queries
    test_queries = [
        "What's the difference between AI and machine learning?",
        "How do neural networks work in deep learning?",
        "What are the applications of natural language processing?",
        "Explain supervised vs unsupervised learning"
    ]
    
    for query in test_queries:
        print(f"\n{'='*40}")
        print(f"Query: {query}")
        print('='*40)
        
        # Get response
        response = rag.generate_response(query)
        print(f"Response: {response}")
        
        # Show detailed retrieval results
        results = rag.retrieve(query, k=2)
        print(f"\nDetailed Retrieval Results:")
        for i, result in enumerate(results, 1):
            print(f"\nSource {i}: {result.chunk.source}")
            print(f"Relevance: {result.relevance_score:.4f}")
            print(f"Content: {result.chunk.text.strip()[:200]}...")

def main():
    """Run all demonstrations."""
    print("RAG SYSTEM WITH SEARCH-R1 DEMONSTRATION")
    print("This demo shows the complete document ingestion and retrieval pipeline")
    
    try:
        demonstrate_text_chunking()
        demonstrate_embeddings()
        demonstrate_vector_store()
        demonstrate_search_r1()
        demonstrate_full_rag_pipeline()
        
        print("\n" + "=" * 60)
        print("DEMONSTRATION COMPLETE!")
        print("=" * 60)
        print("\nKey Features Demonstrated:")
        print("✓ Token-based text chunking with overlap")
        print("✓ Sentence-transformer embeddings (all-MiniLM-L6-v2)")
        print("✓ FAISS vector storage with L2 distance")
        print("✓ Search-R1 query expansion and reranking")
        print("✓ Complete RAG pipeline with Gemini integration")
        print("✓ Vector index persistence")
        
        print(f"\nFiles created:")
        print(f"✓ data/vector.index.faiss - FAISS index file")
        print(f"✓ data/vector.index.pkl - Metadata and chunks")
        
    except Exception as e:
        print(f"Error during demonstration: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
