# AI Call MVP

A minimal Flask service for handling AI-powered voice calls via Twilio.

## Project Structure

```
app/
  __init__.py  # create_app() factory
  routes.py    # health + /voice endpoints
  stt.py       # Speech-to-Text wrapper
  tts.py       # Text-to-Speech wrapper
  rag.py       # retrieval + Gemini integration
  convo.py     # ConversationController
cli.py         # helper commands
run.py         # Flask app launcher
```

## Setup

1. Install dependencies and setup project:
```bash
pip install -e .
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

## Endpoints

- `GET /health` - Health check endpoint
- `POST /voice` - Twilio webhook for voice calls

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

Create a `.env` file with necessary API keys:
```
TWILIO_ACCOUNT_SID=your_account_sid
TWILIO_AUTH_TOKEN=your_auth_token
GOOGLE_API_KEY=your_gemini_api_key
```
