import google.generativeai as genai
from app.backend.configs.settings import SettingsService
from app.backend.llm_rag.extraction.base_extractor import Document
from app.backend.llm_rag.rag_pipeline import RAGPipeline
from typing import List, Optional, Tuple, Any
import logging

logger = logging.getLogger(__name__)

class LLMClient:
    """
    Handles all LLM (Gemini) prompt construction, API calls, and response formatting.
    """
    def __init__(self, model_name: str = 'gemini-2.0-flash-lite', api_key: Optional[str] = None):
        if api_key is None:
            api_key = SettingsService().get_settings().google_api_key
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        logger.info(f"Initialized Gemini LLMClient with model: {model_name}")
        self.rag_pipeline = RAGPipeline()

    def generate_response(self, query: str, conv_context: Any = None, k: int = 5) -> str:
        """
        Generate a conversational response using the LLM, given a query and a RAGPipeline instance.
        This method retrieves context internally from the pipeline.
        """
        if conv_context is None:
            conv_context = "No previous messages."
        # Retrieve context using the pipeline
        context = self.rag_pipeline.search(query, k=k)
        if not context:
            return "I don't have relevant information to answer your question."
        # Format context for prompt
        context_parts = []
        for i, result in enumerate(context, 1):
            context_parts.append(
                f"Context {i} (relevance: {getattr(result, 'relevance_score', 0):.3f}):\n"
                f"Source: {getattr(result.chunk, 'source', '')}\n"
                f"Text: {getattr(result.chunk, 'text', '')}\n"
            )
        context_str = "\n---\n".join(context_parts)
        prompt = f"""
            Based on the following context, and previous conversation messages, if any, please answer the user's question. Be accurate and cite the relevant sources.

            Previous conversation:
            {conv_context}
            
            Context:
            {context_str}

            Question: {query}

            Answer:"""
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            logger.error(f"Error generating response with Gemini: {e}")
            return f"Error generating response: {e}"

    def answer(self, query: str, context: Optional[List[Any]] = None) -> Tuple[str, float, List[str]]:
        """
        Generate a direct answer using the LLM, with confidence and sources.
        Returns (answer, confidence, sources)
        """
        if context is None or not context:
            return ("I don't have relevant information to answer your query.", 0.0, [])
        sources = [getattr(result.chunk, 'source', '') for result in context]
        context_str = "\n\n".join(
            f"[Source: {getattr(result.chunk, 'source', '')}]\n{getattr(result.chunk, 'text', '')}"
            for result in context
        )
        system_persona = """You are a knowledgeable AI assistant that provides accurate, well-sourced answers. Always cite your sources when providing information. Be concise but thorough in your responses."""
        prompt = f"""{system_persona}

            Based on the following context sources, please answer the user's question. 
            Cite the relevant sources in your answer using [Source: filename] format.

            Context:
            {context_str}

            Question: {query}

            Answer:"""
        try:
            response = self.model.generate_content(prompt)
            avg_relevance = sum(getattr(result, 'relevance_score', 0) for result in context) / len(context)
            confidence = min(0.95, avg_relevance * 0.8 + 0.2)
            return (response.text, confidence, sources)
        except Exception as e:
            logger.error(f"Error generating answer with Gemini: {e}")
            return (f"Error generating answer: {str(e)}", 0.0, sources)


