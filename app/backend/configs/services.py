from app.backend.configs.settings import SettingsService
from app.backend.voice.ngrok_control import NgrokManager
from app.backend.voice.convo import ConversationController
from app.backend.llm_rag.rag_pipeline import RAGPipeline
from app.backend.llm_rag.llm import LLMClient

class ServiceProvider:
    """
    A Singleton class to manage and provide a single instance
    of each heavy service for the entire application worker process.
    """
    def __init__(self):
        self._instances = {}

    def get(self, service_class):
        """Gets a service, creating it if it doesn't exist."""
        # Use the class itself as the key
        if service_class not in self._instances:
            print(f"--- LAZY LOADING (ONCE PER WORKER): {service_class.__name__} ---")
            if service_class == RAGPipeline:
                instance = RAGPipeline()
                instance.load_index() # Load index on creation
                self._instances[service_class] = instance
            else:
                self._instances[service_class] = service_class()
        
        return self._instances[service_class]

# --- Create a SINGLE, GLOBAL instance of the provider ---
# This is safe because the provider itself is very lightweight.
service_provider = ServiceProvider()

# --- Create simple getter functions for the rest of the app to use ---
# These functions are now much simpler.

def get_settings_service() -> SettingsService:
    return service_provider.get(SettingsService)

def get_ngrok_manager() -> NgrokManager:
    return service_provider.get(NgrokManager)

def get_rag_pipeline() -> RAGPipeline:
    return service_provider.get(RAGPipeline)

def get_llm_client() -> LLMClient:
    return service_provider.get(LLMClient)

def get_conversation_controller() -> ConversationController:
    return service_provider.get(ConversationController)