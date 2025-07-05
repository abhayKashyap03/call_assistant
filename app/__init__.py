from flask import Flask


def create_app():
    """Create and configure the Flask application."""
    app = Flask(__name__)
    
    # Configuration
    app.config['SECRET_KEY'] = 'dev-secret-key-change-in-production'
    
    # Register blueprints
    from app.routes import bp as routes_bp
    app.register_blueprint(routes_bp)
    
    return app
