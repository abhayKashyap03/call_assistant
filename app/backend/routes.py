from flask import Blueprint, request, jsonify, Response
from app.backend.convo import ConversationController, update_webhook_url
from app.backend.ngrok_control import NgrokManager


bp = Blueprint('main', __name__)
conversation_controller = ConversationController()
ngrok_manager = NgrokManager()


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

@bp.route('/ngrok/start', methods=['POST'])
def start_ngrok():
    try:
        public_url = ngrok_manager.start_ngrok_tunnel()
        update_webhook_url(f"{public_url}/voice")
        return jsonify({'status': 'success', 'public_url': public_url})
    except Exception as e:
        print(f"Error starting ngrok: {e}")
        return jsonify({'status': 'error', 'message': str(e)})

@bp.route('/ngrok/stop', methods=['POST'])
def stop_ngrok():
    try:
        ngrok_manager.stop_ngrok_tunnel()
        return jsonify({'status': 'success', 'message': 'Ngrok tunnel stopped'})
    except Exception as e:
        print(f"Error stopping ngrok: {e}")
        return jsonify({'status': 'error', 'message': str(e)})

@bp.route('/ngrok/status', methods=['GET'])
def get_ngrok_status():
    return jsonify(ngrok_manager.get_tunnel_status())