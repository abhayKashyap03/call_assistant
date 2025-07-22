"""Conversation Controller to orchestrate the AI call flow."""

from flask import g
from twilio.twiml.voice_response import VoiceResponse, Gather
from app.backend.llm_rag.llm import LLMClient
from app.backend.llm_rag.rag_pipeline import RAGPipeline
from dotenv import load_dotenv


load_dotenv()

class ConversationController:
    """Controls the conversation flow for AI voice calls."""
    
    def __init__(self):
        """Initialize the conversation controller."""
        print('!!! --- ConversationController initialized --- !!!')
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
        if g.rag_pipeline is None:
            g.rag_pipeline = RAGPipeline()
        rag_pipeline = g.rag_pipeline
        
        if g.llm_client is None:
            g.llm_client = LLMClient()
        llm_client = g.llm_client
        
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
            
            context = rag_pipeline.search(speech_result, k=5)
            
            # Generate AI response
            ai_response = llm_client.generate_response(speech_result, context=context, conv_context=conversation['messages'])
            
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
