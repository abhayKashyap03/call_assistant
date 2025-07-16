from flask import Flask
from flask_cors import CORS
from app.backend.routes import bp as routes_bp


def create_app():
    """Create and configure the Flask application."""
    app = Flask(__name__)
    
    # Configuration
    app.config['SECRET_KEY'] = 'dev-secret-key-change-in-production'

    CORS(app)
    
    # Register blueprints
    app.register_blueprint(routes_bp)

    
    return app
