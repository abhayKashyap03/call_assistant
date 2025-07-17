# RAG Answer Method Documentation

## Overview

The `rag.answer(query)` method implements a complete Gemini-powered RAG response workflow that follows these steps:

1. **Retrieve top-K chunks** using the Search-R1 algorithm
2. **Build prompt** with system persona and citations
3. **Call Gemini API** with context and query
4. **Return answer, confidence score, and sources list**

## Method Signature

```python
def answer(self, query: str, k: int = 5) -> Tuple[str, float, List[str]]:
    """
    Answer a query using retrieved context and Gemini.
    
    Args:
        query: User query
        k: Number of chunks to retrieve
        
    Returns:
        A tuple of (answer, confidence, sources)
    """
```

## Usage Example

```python
from app.backend.rag import RAGService

# Initialize RAG service
rag = RAGService()

# Add documents to knowledge base
rag.add_documents([
    {
        "text": "Python is a high-level programming language...",
        "source": "python_docs.md"
    }
])

# Ask a question
answer, confidence, sources = rag.answer("What is Python?", k=3)

print(f"Answer: {answer}")
print(f"Confidence: {confidence:.3f}")
print(f"Sources: {sources}")
```

## Workflow Details

### 1. Chunk Retrieval (Search-R1)

- Uses the existing `retrieve()` method with Search-R1 algorithm
- Generates multiple search queries for better coverage
- Performs semantic search using FAISS vector store
- Reranks results based on relevance scoring

### 2. Prompt Construction

The method builds a comprehensive prompt including:

- **System Persona**: Defines the AI as a knowledgeable assistant
- **Context Sources**: Retrieved chunks with source attribution
- **Citation Instructions**: Guides the model to cite sources
- **User Query**: The original question

Example prompt structure:
```
You are a knowledgeable AI assistant that provides accurate, well-sourced answers. 
Always cite your sources when providing information. Be concise but thorough in your responses.

Based on the following context sources, please answer the user's question. 
Cite the relevant sources in your answer using [Source: filename] format.

Context:
[Source: python_docs.md]
Python is a high-level programming language...

Question: What is Python?

Answer:
```

### 3. Gemini API Integration

- Uses `google-generativeai` for Google Gemini API
- Requires `GOOGLE_API_KEY` environment variable
- Handles API errors gracefully with fallback responses
- Returns structured response with text content

### 4. Confidence Calculation

The confidence score is calculated using:

```python
avg_relevance = sum(result.relevance_score for result in retrieved_context) / len(retrieved_context)
confidence = min(0.95, avg_relevance * 0.8 + 0.2)  # Scale to 0.2-0.95 range
```

This approach:
- Uses average relevance of retrieved chunks as base score
- Scales to reasonable confidence range (0.2-0.95)
- Can be enhanced with actual logprobs when available from Gemini

## Configuration

### Environment Variables

- `GOOGLE_API_KEY`: Required for Gemini API access
- Without API key, the method provides fallback responses with retrieved context

### Parameters

- `k`: Number of chunks to retrieve (default: 5)
- Higher k values provide more context but may introduce noise
- Recommended range: 3-10 depending on chunk size and query complexity

## Error Handling

The method handles various error scenarios:

1. **No Relevant Context**: Returns informative message with 0.0 confidence
2. **Gemini API Error**: Logs error and returns error message with 0.0 confidence  
3. **No API Key**: Provides fallback response with retrieved context
4. **Empty Knowledge Base**: Returns appropriate message

## Testing

Comprehensive unit tests are available in `test_rag_answer.py`:

```bash
# Run tests
python -m pytest test_rag_answer.py -v

# Run specific test
python -m pytest test_rag_answer.py::TestRAGAnswer::test_answer_basic_workflow -v
```

## Demo Script

Use the demo script to test the functionality:

```bash
# Basic demo
python demo_rag_answer.py

# Interactive mode
python demo_rag_answer.py --interactive
```

## Performance Considerations

1. **Retrieval Speed**: Uses optimized FAISS indexing for fast similarity search
2. **Context Size**: Limits retrieved chunks to avoid token limits
3. **Caching**: Embeddings are cached in memory after initial computation
4. **Batch Processing**: Embedding generation is batched for efficiency

## Future Enhancements

Potential improvements to the answer method:

1. **Better Confidence Scoring**: Use actual logprobs from Gemini when available
2. **Context Optimization**: Smart context compression to fit more relevant chunks
3. **Multi-turn Conversations**: Support for conversation history
4. **Response Streaming**: Stream responses for better user experience
5. **Custom Personas**: Allow different system personas for different domains

## API Reference

### RAGService.answer()

**Parameters:**
- `query` (str): The user's question or query
- `k` (int, optional): Number of chunks to retrieve (default: 5)

**Returns:**
- `Tuple[str, float, List[str]]`: A tuple containing:
  - `answer` (str): The generated response text
  - `confidence` (float): Confidence score between 0.2 and 0.95
  - `sources` (List[str]): List of source filenames used

**Raises:**
- No exceptions are raised; errors are handled gracefully and returned in the response

### Related Methods

- `RAGService.retrieve(query, k)`: Get ranked search results
- `RAGService.generate_response(query, context)`: Generate response with custom context
- `RAGService.add_documents(documents)`: Add documents to knowledge base
- `RAGService.get_stats()`: Get knowledge base statistics
