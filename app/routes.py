from flask import Blueprint, request, jsonify, Response
from app.convo import ConversationController

bp = Blueprint('main', __name__)
conversation_controller = ConversationController()


@bp.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({'status': 'healthy', 'service': 'ai_call_mvp'})


@bp.route('/voice', methods=['POST'])
def twilio_webhook():
    """Handle Twilio webhook for voice calls."""
    try:
        # Get Twilio request data
        call_sid = request.form.get('CallSid')
        speech_result = request.form.get('SpeechResult')
        
        if not call_sid:
            return Response('Missing CallSid', status=400)
        
        # Process the conversation
        twiml_response = conversation_controller.handle_call(call_sid, speech_result)
        
        return Response(twiml_response, mimetype='text/xml')
    
    except Exception as e:
        # Log error in production
        print(f"Error in twilio_webhook: {e}")
        return Response('Internal Server Error', status=500)
