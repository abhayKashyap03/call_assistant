import os
import tempfile
from flask import Blueprint, request, jsonify, Response, g
from app.backend.configs import services


bp = Blueprint('main', __name__)

@bp.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({'status': 'healthy', 'service': 'ai_call_mvp'})


@bp.route('/voice', methods=['POST'])
def twilio_webhook():
    """Handle Twilio webhook for voice calls."""
    conversation_controller = services.get_conversation_controller()
    
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
    ngrok_manager = services.get_ngrok_manager()
    try:
        public_url = ngrok_manager.start_ngrok_tunnel()
        return jsonify({'status': 'success', 'public_url': public_url})
    except Exception as e:
        print(f"Error starting ngrok: {e}")
        return jsonify({'status': 'error', 'message': str(e)})

@bp.route('/ngrok/stop', methods=['POST'])
def stop_ngrok():
    ngrok_manager = services.get_ngrok_manager()
    try:
        ngrok_manager.stop_ngrok_tunnel()
        return jsonify({'status': 'success', 'message': 'Ngrok tunnel stopped'})
    except Exception as e:
        print(f"Error stopping ngrok: {e}")
        return jsonify({'status': 'error', 'message': str(e)})

@bp.route('/ngrok/status', methods=['GET'])
def get_ngrok_status():
    ngrok_manager = services.get_ngrok_manager()
    return jsonify(ngrok_manager.get_tunnel_status())

@bp.route('/env', methods=['GET', 'POST'])
def update_env():
    """Update environment variables."""
    settings_service = services.get_settings_service()
    if request.method == 'POST':
        settings_service.save(request.json)
        return jsonify({'status': 'success', 'message': 'Environment variables updated'}), 200
    
    # Mask sensitive data
    current_settings = settings_service.get_settings()
    safe_settings = current_settings.model_dump(exclude_unset=True)
    safe_settings['twilio_auth_token'] = '***' if current_settings.twilio_auth_token else None
    safe_settings['twilio_account_sid'] = '***' if current_settings.twilio_account_sid else None
    safe_settings['twilio_phone_sid'] = '***' if current_settings.twilio_phone_sid else None
    safe_settings['ngrok_authtoken'] = '***' if current_settings.ngrok_authtoken else None
    safe_settings['google_api_key'] = '***' if current_settings.google_api_key else None
    return jsonify(safe_settings), 200

@bp.route('/doc_in', methods=['POST'])
def doc_in():
    """Handle document upload or URL ingestion."""
    try:
        rag_pipeline = services.get_rag_pipeline()
        uploaded_files = request.files.getlist('file')
        url = request.form.get('url')

        if uploaded_files:
            for file_storage in uploaded_files:
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file_storage.filename)[1])
                file_storage.save(tmp.name)
                tmp.close()  # Ensure file is written and closed before ingestion
                # Debug: check file exists and size
                if not os.path.exists(tmp.name):
                    print(f"Temp file does not exist: {tmp.name}")
                else:
                    size = os.path.getsize(tmp.name)
                    print(f"Temp file {tmp.name} size: {size} bytes")
                rag_pipeline.ingest_single(tmp.name)
                os.remove(tmp.name)
            return jsonify({'status': 'success', 'message': 'Document(s) uploaded successfully'})
        elif url:
            rag_pipeline.ingest_single(url)
            return jsonify({'status': 'success', 'message': 'URL ingested successfully'})
        else:
            return jsonify({'status': 'error', 'message': 'No file or URL provided'}), 400
    except Exception as e:
        print(f"Error in doc_in: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500