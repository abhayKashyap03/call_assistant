#!/usr/bin/env python3
"""
Demo script for RAG answer method with Gemini integration.

This script demonstrates the complete workflow:
1. Retrieve top-K chunks using Search-R1 algorithm
2. Build prompt with system persona and citations
3. Call Gemini API with context and query
4. Return answer, confidence score, and sources list

Usage:
    python demo_rag_answer.py

Requirements:
    - Set GOOGLE_API_KEY environment variable for full functionality
    - Or run without API key to see fallback behavior
"""

import os
import sys
from typing import List, Dict
from app.rag import RAGService


def setup_sample_knowledge_base() -> RAGService:
    """Set up a sample knowledge base for demonstration."""
    print("🔧 Setting up RAG service...")
    
    rag = RAGService()
    
    # Sample documents covering various topics
    sample_documents = [
        {
            "text": """
            Python is a high-level, interpreted programming language with dynamic semantics. 
            Its high-level built-in data structures, combined with dynamic typing and dynamic binding, 
            make it very attractive for Rapid Application Development, as well as for use as a 
            scripting or glue language to connect existing components together. Python's simple, 
            easy to learn syntax emphasizes readability and therefore reduces the cost of program 
            maintenance. Python supports modules and packages, which encourages program modularity 
            and code reuse. The Python interpreter and the extensive standard library are available 
            in source or binary form without charge for all major platforms.
            """,
            "source": "python_official_docs.md"
        },
        {
            "text": """
            Machine Learning is a subset of artificial intelligence (AI) that provides systems 
            the ability to automatically learn and improve from experience without being explicitly 
            programmed. Machine learning focuses on the development of computer programs that can 
            access data and use it to learn for themselves. The process of learning begins with 
            observations or data, such as examples, direct experience, or instruction, in order 
            to look for patterns in data and make better decisions in the future based on the 
            examples that we provide. The primary aim is to allow the computers to learn automatically 
            without human intervention or assistance and adjust actions accordingly.
            """,
            "source": "ml_introduction.txt"
        },
        {
            "text": """
            Natural Language Processing (NLP) is a branch of artificial intelligence that helps 
            computers understand, interpret and manipulate human language. NLP draws from many 
            disciplines, including computer science and computational linguistics, in its pursuit 
            to fill the gap between human communication and computer understanding. Common NLP 
            tasks include text classification, sentiment analysis, language translation, named 
            entity recognition, question answering, and text summarization. Modern NLP relies 
            heavily on machine learning and deep learning techniques, particularly transformer 
            models like BERT and GPT.
            """,
            "source": "nlp_overview.pdf"
        },
        {
            "text": """
            Flask is a micro web framework written in Python. It is classified as a microframework 
            because it does not require particular tools or libraries. It has no database abstraction 
            layer, form validation, or any other components where pre-existing third-party libraries 
            provide common functions. However, Flask supports extensions that can add application 
            features as if they were implemented in Flask itself. Extensions exist for object-relational 
            mappers, form validation, upload handling, various open authentication technologies and 
            several common framework related tools.
            """,
            "source": "flask_documentation.rst"
        },
        {
            "text": """
            Deep learning is part of a broader family of machine learning methods based on 
            artificial neural networks with representation learning. Learning can be supervised, 
            semi-supervised or unsupervised. Deep learning architectures such as deep neural 
            networks, deep belief networks, deep reinforcement learning, recurrent neural networks, 
            convolutional neural networks and transformers have been applied to fields including 
            computer vision, speech recognition, natural language processing, machine translation, 
            bioinformatics, drug design, medical image analysis, climate science, and board game programs.
            """,
            "source": "deep_learning_handbook.pdf"
        }
    ]
    
    print(f"📚 Adding {len(sample_documents)} documents to knowledge base...")
    rag.add_documents(sample_documents)
    
    stats = rag.get_stats()
    print(f"✅ Knowledge base ready:")
    print(f"   - Total chunks: {stats['total_chunks']}")
    print(f"   - Sources: {len(stats['sources'])}")
    print(f"   - Embedding dimension: {stats['embedding_dimension']}")
    print()
    
    return rag


def demonstrate_answer_method(rag: RAGService):
    """Demonstrate the answer method with various queries."""
    
    # Check if Gemini API key is available
    api_key_available = bool(os.getenv('GOOGLE_API_KEY'))
    print(f"🔑 Google API Key: {'✅ Available' if api_key_available else '❌ Not set (will use fallback)'}")
    print()
    
    # Test queries covering different topics
    test_queries = [
        "What is Python and why is it popular?",
        "Explain machine learning in simple terms",
        "What are the main applications of NLP?",
        "How does Flask differ from other web frameworks?",
        "What is the relationship between deep learning and neural networks?"
    ]
    
    print("🚀 Testing RAG Answer Method")
    print("=" * 60)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n📝 Query {i}: {query}")
        print("-" * 50)
        
        try:
            # Call the answer method
            answer, confidence, sources = rag.answer(query, k=3)
            
            # Display results
            print(f"🤖 Answer:")
            print(f"   {answer}\n")
            
            print(f"📊 Confidence Score: {confidence:.3f}")
            
            print(f"📚 Sources ({len(sources)}):")
            for j, source in enumerate(set(sources), 1):
                print(f"   {j}. {source}")
            
        except Exception as e:
            print(f"❌ Error: {e}")
        
        print("\n" + "=" * 60)


def interactive_mode(rag: RAGService):
    """Interactive mode for testing custom queries."""
    print("\n🎯 Interactive Mode")
    print("Enter your queries (type 'quit' to exit, 'help' for commands)")
    print("-" * 50)
    
    while True:
        try:
            query = input("\n💬 Your question: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            elif query.lower() == 'help':
                print("""
Available commands:
- help: Show this help message
- stats: Show knowledge base statistics
- quit/exit/q: Exit interactive mode
- Any other input: Ask a question to the RAG system
                """)
                continue
            elif query.lower() == 'stats':
                stats = rag.get_stats()
                print(f"""
📊 Knowledge Base Statistics:
- Total chunks: {stats['total_chunks']}
- Sources: {len(stats['sources'])}
- Average chunk length: {stats['avg_chunk_length']:.1f} words
- Embedding dimension: {stats['embedding_dimension']}
- Available sources: {', '.join(stats['sources'])}
                """)
                continue
            elif not query:
                continue
            
            # Process the query
            print("🔍 Processing your query...")
            answer, confidence, sources = rag.answer(query, k=5)
            
            print(f"\n🤖 Answer:")
            print(f"   {answer}")
            print(f"\n📊 Confidence: {confidence:.3f}")
            print(f"📚 Sources: {', '.join(set(sources))}")
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


def main():
    """Main demo function."""
    print("🔥 RAG Answer Method Demo")
    print("=" * 60)
    print("This demo showcases the Gemini-powered RAG response module")
    print("with the following workflow:")
    print("1. Retrieve top-K chunks using Search-R1 algorithm")
    print("2. Build prompt with system persona and citations")
    print("3. Call Gemini API with context and query")
    print("4. Return answer, confidence score, and sources list")
    print()
    
    try:
        # Setup RAG service
        rag = setup_sample_knowledge_base()
        
        # Run demonstrations
        demonstrate_answer_method(rag)
        
        # Optional interactive mode
        if len(sys.argv) > 1 and sys.argv[1] == '--interactive':
            interactive_mode(rag)
        else:
            print("\n💡 Tip: Run with --interactive flag for interactive mode")
            print("   python demo_rag_answer.py --interactive")
    
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
