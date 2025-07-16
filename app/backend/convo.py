"""Conversation Controller to orchestrate the AI call flow."""

from twilio.twiml.voice_response import VoiceResponse, Gather, Say, Redirect
from twilio.rest import Client
from app.backend.rag import RAGService
from dotenv import load_dotenv
import os


load_dotenv()

def update_webhook_url(url):
    """Update the Twilio webhook URL for voice calls."""
    try:
        # Load environment variables
        account_sid = os.getenv('TWILIO_ACCOUNT_SID')
        auth_token = os.getenv('TWILIO_AUTH_TOKEN')
        phone_sid = os.getenv('TWILIO_PHONE_SID')

        # Initialize Twilio client
        client = Client(account_sid, auth_token)

        # Update the webhook URL
        client.incoming_phone_numbers(phone_sid).update(
            voice_url=url
        )
        
        print("Webhook URL updated successfully.")
    except Exception as e:
        return e


class ConversationController:
    """Controls the conversation flow for AI voice calls."""
    
    def __init__(self):
        """Initialize the conversation controller."""
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
        response = VoiceResponse()
        gather = Gather(input='speech', action='/voice', method='POST', speech_timeout='auto')
        gather.say(message)
        response.append(gather)
        response.say("I didn't hear anything. Please try again.")
        response.redirect('/voice')
        
        return str(response)
    
    def end_conversation(self, call_sid):
        """
        End and cleanup conversation.
        
        Args:
            call_sid: Twilio call SID
        """
        if call_sid in self.active_conversations:
            del self.active_conversations[call_sid]
