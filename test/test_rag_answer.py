"""
Unit tests for RAG answer method with mocked Gemini.

Tests cover:
1. Basic answer workflow
2. Confidence scoring
3. Sources extraction
4. Error handling
5. Fallback without API key
"""

import os
import pytest
from unittest.mock import Mock, patch, MagicMock
import numpy as np
from typing import List, Tuple

# Import the RAG module
from app.rag import RAGService, DocumentChunk, SearchResult


class MockGeminiResponse:
    """Mock response from Gemini API."""
    def __init__(self, text: str, confidence: float = 0.85):
        self.text = text
        self.confidence = confidence


class TestRAGAnswer:
    """Test suite for RAG answer method."""
    
    @pytest.fixture
    def rag_service(self):
        """Create a RAG service for testing."""
        with patch('app.rag.genai.GenerativeModel') as mock_model:
            rag = RAGService()
            # Add some test documents
            test_docs = [
                {
                    "text": "Python is a high-level programming language. It was created by Guido van Rossum and first released in 1991. Python emphasizes code readability and simplicity.",
                    "source": "python_intro.txt"
                },
                {
                    "text": "Machine learning is a subset of artificial intelligence. It enables computers to learn and improve from experience without explicit programming. Common algorithms include linear regression, decision trees, and neural networks.",
                    "source": "ml_basics.txt"
                },
                {
                    "text": "Natural language processing (NLP) deals with the interaction between computers and human language. It includes tasks like text classification, sentiment analysis, and language translation.",
                    "source": "nlp_overview.txt"
                }
            ]
            rag.add_documents(test_docs)
            return rag
    
    @pytest.fixture
    def mock_gemini_client(self):
        """Create a mock Gemini client."""
        mock_client = Mock()
        mock_response = MockGeminiResponse(
            "Based on the provided context, Python is a high-level programming language created by Guido van Rossum in 1991. [Source: python_intro.txt] It emphasizes code readability and simplicity, making it popular for various applications including web development and data science."
        )
        mock_client.generate_content.return_value = mock_response
        return mock_client
    
    def test_answer_basic_workflow(self, rag_service, mock_gemini_client):
        """Test basic answer workflow with mocked Gemini."""
        # Set up the mock
        rag_service.gemini_client = mock_gemini_client
        
        # Test the answer method
        query = "What is Python?"
        answer, confidence, sources = rag_service.answer(query, k=3)
        
        # Assertions
        assert isinstance(answer, str)
        assert len(answer) > 0
        assert isinstance(confidence, float)
        assert 0.0 <= confidence <= 1.0
        assert isinstance(sources, list)
        assert len(sources) > 0
        assert "python_intro.txt" in sources
        
        # Verify Gemini was called
        mock_gemini_client.generate_content.assert_called_once()
        call_args = mock_gemini_client.generate_content.call_args[0][0]
        assert "Python" in call_args
        assert "system persona" in call_args.lower() or "knowledgeable ai assistant" in call_args.lower()
    
    def test_answer_confidence_calculation(self, rag_service, mock_gemini_client):
        """Test confidence score calculation."""
        rag_service.gemini_client = mock_gemini_client
        
        query = "What is machine learning?"
        answer, confidence, sources = rag_service.answer(query, k=2)
        
        # Confidence should be between 0.2 and 0.95 (based on implementation)
        assert 0.2 <= confidence <= 0.95
        assert isinstance(confidence, float)
    
    def test_answer_sources_extraction(self, rag_service, mock_gemini_client):
        """Test that sources are correctly extracted."""
        rag_service.gemini_client = mock_gemini_client
        
        query = "Explain natural language processing"
        answer, confidence, sources = rag_service.answer(query, k=5)
        
        # Check sources format
        assert isinstance(sources, list)
        assert all(isinstance(source, str) for source in sources)
        assert len(sources) <= 5  # Should not exceed k
        
        # Should contain relevant sources
        source_names = set(sources)
        expected_sources = {"python_intro.txt", "ml_basics.txt", "nlp_overview.txt"}
        assert len(source_names.intersection(expected_sources)) > 0
    
    def test_answer_with_gemini_error(self, rag_service, mock_gemini_client):
        """Test error handling when Gemini API fails."""
        # Make Gemini client raise an exception
        mock_gemini_client.generate_content.side_effect = Exception("API Error")
        rag_service.gemini_client = mock_gemini_client
        
        query = "What is Python?"
        answer, confidence, sources = rag_service.answer(query)
        
        # Should handle error gracefully
        assert "Error generating answer" in answer
        assert confidence == 0.0
        assert isinstance(sources, list)
    
    def test_answer_without_gemini_client(self, rag_service):
        """Test fallback behavior when Gemini client is not configured."""
        # Ensure no Gemini client
        rag_service.gemini_client = None
        
        query = "What is Python?"
        answer, confidence, sources = rag_service.answer(query)
        
        # Should provide fallback response
        assert "Google API key" in answer or "GOOGLE_API_KEY" in answer
        assert "Retrieved context" in answer or "relevant information" in answer
        assert isinstance(confidence, float)
        assert confidence < 1.0  # Should be lower without LLM
        assert isinstance(sources, list)
        assert len(sources) > 0
    
    def test_answer_no_relevant_context(self, rag_service, mock_gemini_client):
        """Test behavior when no relevant context is found."""
        # Create a RAG service with no documents
        empty_rag = RAGService()
        empty_rag.gemini_client = mock_gemini_client
        
        query = "What is quantum computing?"
        answer, confidence, sources = empty_rag.answer(query)
        
        # Should indicate no relevant information
        assert "don't have relevant information" in answer
        assert confidence == 0.0
        assert sources == []
    
    def test_answer_prompt_construction(self, rag_service, mock_gemini_client):
        """Test that the prompt is constructed correctly."""
        rag_service.gemini_client = mock_gemini_client
        
        query = "What is Python?"
        rag_service.answer(query, k=2)
        
        # Check the prompt construction
        call_args = mock_gemini_client.generate_content.call_args[0][0]
        
        # Should contain system persona
        assert "knowledgeable AI assistant" in call_args
        assert "accurate" in call_args
        assert "cite" in call_args.lower()
        
        # Should contain context with sources
        assert "[Source:" in call_args
        assert "Context:" in call_args
        assert "Question:" in call_args
        assert query in call_args
    
    def test_answer_k_parameter(self, rag_service, mock_gemini_client):
        """Test that k parameter controls number of retrieved chunks."""
        rag_service.gemini_client = mock_gemini_client
        
        query = "What is programming?"
        
        # Test with different k values
        for k in [1, 3, 5]:
            answer, confidence, sources = rag_service.answer(query, k=k)
            assert len(sources) <= k
    
    @patch('app.rag.logger')
    def test_answer_logging(self, mock_logger, rag_service, mock_gemini_client):
        """Test that errors are properly logged."""
        # Make Gemini client raise an exception
        mock_gemini_client.generate_content.side_effect = Exception("Test error")
        rag_service.gemini_client = mock_gemini_client
        
        query = "What is Python?"
        rag_service.answer(query)
        
        # Check that error was logged
        mock_logger.error.assert_called()
        error_call = mock_logger.error.call_args[0][0]
        assert "Error in answer generation" in error_call
    
    def test_answer_return_type(self, rag_service, mock_gemini_client):
        """Test that answer method returns correct types."""
        rag_service.gemini_client = mock_gemini_client
        
        query = "What is Python?"
        result = rag_service.answer(query)
        
        # Should return a tuple of (str, float, list)
        assert isinstance(result, tuple)
        assert len(result) == 3
        
        answer, confidence, sources = result
        assert isinstance(answer, str)
        assert isinstance(confidence, float)
        assert isinstance(sources, list)
    
    @pytest.mark.parametrize("query,expected_relevance", [
        ("Python programming language", True),
        ("machine learning algorithms", True),
        ("natural language processing", True),
        ("quantum physics theories", False),  # Should have lower relevance
    ])
    def test_answer_relevance_scoring(self, rag_service, mock_gemini_client, query, expected_relevance):
        """Test that confidence scores reflect relevance appropriately."""
        rag_service.gemini_client = mock_gemini_client
        
        answer, confidence, sources = rag_service.answer(query, k=3)
        
        if expected_relevance:
            # Should have reasonable confidence for relevant queries
            assert confidence > 0.3
            assert len(sources) > 0
        # Note: For irrelevant queries, the system should still try to find best matches
        # but confidence might be lower based on retrieval scores


class TestRAGAnswerIntegration:
    """Integration tests for the answer method."""
    
    @pytest.fixture
    def sample_documents(self):
        """Sample documents for testing."""
        return [
            {
                "text": "Flask is a micro web framework written in Python. It is classified as a microframework because it does not require particular tools or libraries. Flask provides tools and libraries to build web applications quickly.",
                "source": "flask_intro.md"
            },
            {
                "text": "Django is a high-level Python web framework that encourages rapid development and clean, pragmatic design. It follows the model-template-view architectural pattern and includes an ORM for database operations.",
                "source": "django_basics.md"  
            },
            {
                "text": "FastAPI is a modern web framework for building APIs with Python. It provides automatic API documentation, data validation, and is built on standard Python type hints. FastAPI is known for its high performance.",
                "source": "fastapi_guide.md"
            }
        ]
    
    def test_end_to_end_answer_workflow(self, sample_documents):
        """Test the complete end-to-end workflow."""
        # Create RAG service
        rag = RAGService()
        
        # Add documents
        rag.add_documents(sample_documents)
        
        # Mock Gemini client
        mock_client = Mock()
        mock_response = MockGeminiResponse(
            "Based on the provided sources, Flask is a micro web framework for Python. [Source: flask_intro.md] It's lightweight and doesn't require specific tools, making it ideal for quick web application development."
        )
        mock_client.generate_content.return_value = mock_response
        rag.gemini_client = mock_client
        
        # Test query
        query = "What is Flask?"
        answer, confidence, sources = rag.answer(query, k=2)
        
        # Verify results
        assert "Flask" in answer
        assert "micro web framework" in answer
        assert confidence > 0.0
        assert "flask_intro.md" in sources
        
        # Verify Gemini was called with proper context
        call_args = mock_client.generate_content.call_args[0][0]
        assert "Flask is a micro web framework" in call_args
        assert "[Source: flask_intro.md]" in call_args


# Test utilities
def create_mock_search_result(text: str, source: str, relevance_score: float = 0.8) -> SearchResult:
    """Helper function to create mock search results."""
    chunk = DocumentChunk(
        text=text,
        source=source,
        chunk_id=0,
        start_char=0,
        end_char=len(text),
        metadata={}
    )
    
    return SearchResult(
        chunk=chunk,
        similarity_score=0.3,  # L2 distance
        relevance_score=relevance_score,
        reasoning_path=["Mock reasoning"]
    )


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
