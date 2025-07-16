"""Flask application runner."""

from app import create_app
from dotenv import load_dotenv


load_dotenv()

# Create the Flask application
app = create_app()

if __name__ == '__main__':
    # Run the application in development mode
    app.run(debug=True, host='0.0.0.0', port=5000)
