"""CLI utilities for AI Call MVP with argparse sub-commands."""

import argparse
import os
import sys
import requests
from bs4 import BeautifulSoup
import subprocess
import json
import markdown
from pathlib import Path
from twilio.rest import Client
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def scrape_website(url, max_depth=2, delay=1.0):
    """Advanced crawl website and store as markdown with metadata."""
    try:
        print(f"Starting advanced scrape of: {url}")
        print(f"Max depth: {max_depth}, Delay: {delay}s")
        
        # Import the advanced scraper
        from app.scraper import scrape_website as advanced_scrape
        
        # Run the advanced scraper
        advanced_scrape(url, max_depth=max_depth, delay=delay)
        
        print("\nScraping completed successfully!")
        print("Content saved to: data/raw/")
        print("Each page saved as .md file with corresponding .json metadata")
        
    except Exception as e:
        print(f"Error scraping website: {e}")
        sys.exit(1)


def ingest_documents():
    """Chunk documents and build/refresh vector store."""
    try:
        print("Ingesting documents and building vector store...")
        
        # Import here to avoid circular imports
        from app.rag import RAGService
        
        rag_service = RAGService()
        
        # Find all markdown files in data/raw directory and current directory
        data_raw_path = Path('data/raw')
        current_path = Path('.')
        
        md_files = []
        if data_raw_path.exists():
            md_files.extend(list(data_raw_path.glob('*.md')))
        md_files.extend([f for f in current_path.glob('*.md') if f.name != 'README.md' and f.name != 'SCRAPER_README.md'])
        
        if not md_files:
            print("No markdown files found in data/raw/ or current directory")
            print("Run 'python cli.py scrape <url>' first to generate content")
            return
        
        documents = []
        for md_file in md_files:
            with open(md_file, 'r', encoding='utf-8') as f:
                content = f.read()
                documents.append({
                    'content': content,
                    'source': str(md_file),
                    'title': md_file.stem
                })
        
        # Add documents to RAG service
        rag_service.add_documents(documents)
        rag_service.save_index()
        
        print(f"Successfully ingested {len(documents)} documents")
        
    except Exception as e:
        print(f"Error ingesting documents: {e}")
        sys.exit(1)


def open_ngrok_tunnel():
    """Open ngrok tunnel and print public URL."""
    try:
        print("Opening ngrok tunnel...")
        
        # Start ngrok tunnel for port 5000 (Flask default)
        process = subprocess.Popen(
            ['ngrok', 'http', '5000', '--log=stdout'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Wait a bit for ngrok to start
        import time
        time.sleep(3)
        
        # Get tunnel information from ngrok API
        try:
            response = requests.get('http://localhost:4040/api/tunnels')
            tunnels = response.json()['tunnels']
            
            if tunnels:
                public_url = tunnels[0]['public_url']
                print(f"Ngrok tunnel active!")
                print(f"Public URL: {public_url}")
                print(f"Local URL: http://localhost:4040")
                print("Press Ctrl+C to stop the tunnel")
                
                # Keep the process running
                try:
                    process.wait()
                except KeyboardInterrupt:
                    print("\nStopping ngrok tunnel...")
                    process.terminate()
            else:
                print("No active tunnels found")
                
        except requests.exceptions.RequestException:
            print("Could not connect to ngrok API. Make sure ngrok is installed and running.")
            
    except FileNotFoundError:
        print("Error: ngrok not found. Please install ngrok first.")
        print("Visit: https://ngrok.com/download")
        sys.exit(1)
    except Exception as e:
        print(f"Error opening ngrok tunnel: {e}")
        sys.exit(1)


def place_test_call():
    """Place outbound test call using Twilio REST API."""
    try:
        print("Placing test call via Twilio...")
        
        # Get Twilio credentials from environment
        account_sid = os.getenv('TWILIO_ACCOUNT_SID')
        auth_token = os.getenv('TWILIO_AUTH_TOKEN')
        from_number = os.getenv('TWILIO_PHONE_NUMBER')
        to_number = os.getenv('TEST_PHONE_NUMBER')
        
        if not all([account_sid, auth_token, from_number, to_number]):
            print("Error: Missing Twilio configuration in environment variables")
            print("Required: TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_PHONE_NUMBER, TEST_PHONE_NUMBER")
            sys.exit(1)
        
        client = Client(account_sid, auth_token)
        
        # Create a test call
        call = client.calls.create(
            to=to_number,
            from_=from_number,
            url='http://demo.twilio.com/docs/voice.xml'  # Default Twilio test message
        )
        
        print(f"Test call placed successfully!")
        print(f"Call SID: {call.sid}")
        print(f"From: {from_number}")
        print(f"To: {to_number}")
        print(f"Status: {call.status}")
        
    except Exception as e:
        print(f"Error placing test call: {e}")
        sys.exit(1)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='CLI utilities for AI Call MVP development and operations'
    )
    
    subparsers = parser.add_subparsers(
        dest='command',
        help='Available commands',
        metavar='<command>'
    )
    
    # Sub-command: scrape
    scrape_parser = subparsers.add_parser(
        'scrape',
        help='Advanced crawl website and store content as markdown with metadata'
    )
    scrape_parser.add_argument(
        'url',
        help='The URL to scrape'
    )
    scrape_parser.add_argument(
        '--max-depth',
        type=int,
        default=2,
        help='Maximum crawl depth (default: 2)'
    )
    scrape_parser.add_argument(
        '--delay',
        type=float,
        default=1.0,
        help='Delay between requests in seconds (default: 1.0)'
    )
    
    # Sub-command: ingest
    ingest_parser = subparsers.add_parser(
        'ingest',
        help='Chunk documents and build/refresh vector store'
    )
    
    # Sub-command: ngrok
    ngrok_parser = subparsers.add_parser(
        'ngrok',
        help='Open ngrok tunnel and print public URL'
    )
    
    # Sub-command: calltest
    calltest_parser = subparsers.add_parser(
        'calltest',
        help='Place outbound test call using Twilio REST API'
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    if args.command == 'scrape':
        scrape_website(args.url, max_depth=args.max_depth, delay=args.delay)
    elif args.command == 'ingest':
        ingest_documents()
    elif args.command == 'ngrok':
        open_ngrok_tunnel()
    elif args.command == 'calltest':
        place_test_call()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
