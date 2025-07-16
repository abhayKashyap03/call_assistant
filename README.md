# AI Call MVP

A minimal Flask service for handling AI-powered voice calls via Twilio.

## Project Structure

```
app/
  __init__.py  # create_app() factory
  backend/
    __init__.py
    ngrok_control.py   # Utility functions for managing ngrok server
    routes.py          # health + /voice endpoints
    rag.py             # retrieval + Gemini integration
    convo.py           # ConversationController
    scraper.py         # Web scraper for online knowledge base
  frontend/
    ...        # React app files
cli.py         # helper commands
run.py         # Flask app launcher
```

## Setup

1. Install dependencies and set up the project:
```bash
pip install -e .
cd app/frontend && npm install
```

2. Run the Flask application:
```bash
# Option 1: Using run.py
python run.py

# Option 2: Using Flask CLI
flask --app app run

# Option 3: With debug mode
flask --app app run --debug
```

3. Start React app from inside the frontend folder
```bash
npm run dev
```

## Endpoints

- `GET /health` - Health check endpoint
- `POST /voice` - Twilio webhook for voice calls
- `POST /ngrok/start` - Start a new ngrok server
- `POST /ngrok/stop`  - Stop the current ngrok server
- `GET /ngrok/status` - Get ngrok server status

## CLI Commands

```bash
# Test call service
python cli.py calltest

# Scrape a website
python cli.py scrape <url>

# Ingest a document
python cli.py ingest

# Start ngrok service
python cli.py ngrok

# Show all configurations
python cli.py --help
```

## Environment Variables

Create a `.env` file with the necessary API keys:
```
GOOGLE_API_KEY=<your-gemini-api-key>
TWILIO_ACCOUNT_SID=<your-twilio-account-sid>
TWILIO_AUTH_TOKEN=<your-twilio-auth-token>
TWILIO_PHONE_SID=<your-twilio-number-sid>
NGROK_AUTH_TOKEN=<your-ngrok-auth-token>
```
