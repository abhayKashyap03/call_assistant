"""Conversation Controller to orchestrate the AI call flow."""

from app.stt import SpeechToTextService
from app.tts import TextToSpeechService
from app.rag import RAGService


class ConversationController:
    """Controls the conversation flow for AI voice calls."""
    
    def __init__(self):
        """Initialize the conversation controller."""
        self.stt_service = SpeechToTextService()
        self.tts_service = TextToSpeechService()
        self.rag_service = RAGService()
        self.active_conversations = {}
    
    def handle_call(self, call_sid, speech_result=None):
        """
        Handle incoming call and manage conversation flow.
        
        Args:
            call_sid: Twilio call SID
            speech_result: Transcribed speech from Twilio
            
        Returns:
            str: TwiML response
        """
        # Initialize conversation if new call
        if call_sid not in self.active_conversations:
            self.active_conversations[call_sid] = {
                'messages': [],
                'state': 'greeting'
            }
        
        conversation = self.active_conversations[call_sid]
        
        if speech_result:
            # Add user message to conversation
            conversation['messages'].append({
                'role': 'user',
                'content': speech_result
            })
            
            # Generate AI response
            ai_response = self.generate_response(speech_result, conversation)
            
            # Add AI response to conversation
            conversation['messages'].append({
                'role': 'assistant',
                'content': ai_response
            })
        
            print(f"\n\nconversation: {conversation['messages']}\n\n")
            
            # Generate TwiML with AI response
            return self.create_twiml_response(ai_response)
        else:
            # Initial call - send greeting
            greeting = "Hello! I'm your AI assistant. How can I help you today?"
            conversation['messages'].append({
                'role': 'assistant',
                'content': greeting
            })

            print(f"\n\nconversation: {conversation['messages']}\n\n")

            return self.create_twiml_response(greeting)
    
    def generate_response(self, user_input, conversation):
        """
        Generate AI response using RAG service.
        
        Args:
            user_input: User's spoken input
            conversation: Current conversation context
            
        Returns:
            str: AI response
        """
        # Use RAG service to generate contextual response
        response = self.rag_service.generate_response(user_input)
        return response
    
    def create_twiml_response(self, message):
        """
        Create TwiML response for Twilio.
        
        Args:
            message: Message to speak
            
        Returns:
            str: TwiML XML response
        """
        # Basic TwiML response with Gather for speech input
        twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Gather input="speech" action="/voice" method="POST" speechTimeout="auto">
        <Say>{message}</Say>
    </Gather>
    <Say>I didn't hear anything. Please try again.</Say>
    <Redirect>/voice</Redirect>
</Response>"""
        return twiml
    
    def end_conversation(self, call_sid):
        """
        End and cleanup conversation.
        
        Args:
            call_sid: Twilio call SID
        """
        if call_sid in self.active_conversations:
            del self.active_conversations[call_sid]
