# app/backend/services.py
from flask import g
from app.backend.configs.settings import SettingsService
from app.backend.voice.ngrok_control import NgrokManager
from app.backend.voice.convo import ConversationController
from app.backend.llm_rag.rag_pipeline import RAGPipeline
from app.backend.llm_rag.llm import LLMClient


def get_settings_service() -> SettingsService:
    if 'settings_service' not in g:
        g.settings_service = SettingsService()
    return g.settings_service

def get_ngrok_manager() -> NgrokManager:
    if 'ngrok_manager' not in g:
        g.ngrok_manager = NgrokManager()
    return g.ngrok_manager

def get_rag_pipeline() -> RAGPipeline:
    if 'rag_pipeline' not in g:
        g.rag_pipeline = RAGPipeline()
    return g.rag_pipeline

def get_llm_client() -> LLMClient:
    if 'llm_client' not in g:
        g.llm_client = LLMClient()
    return g.llm_client

def get_conversation_controller() -> ConversationController:
    if 'conversation_controller' not in g:
        g.conversation_controller = ConversationController()
    return g.conversation_controller