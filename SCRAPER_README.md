# Web Scraper for Knowledge Base

This project includes an advanced web scraper designed to build a knowledge base by crawling websites and extracting clean content.

## Features

- **Depth-first crawling**: Crawls websites in a depth-first manner within domain boundaries
- **Binary file detection**: Automatically skips binary files (PDFs, images, videos, etc.)
- **Content cleaning**: Uses `trafilatura` library for intelligent HTML content extraction
- **Robots.txt compliance**: Respects robots.txt rules for ethical crawling
- **Duplicate handling**: Tracks visited URLs to avoid duplicates
- **Rate limiting**: Configurable delay between requests to be respectful to servers
- **Metadata extraction**: Extracts titles, descriptions, authors, and dates when available
- **Output formats**: Saves content as `.md` files with corresponding `.json` metadata

## Usage

### Basic Scraping

```bash
python cli.py scrape https://example.com/docs
```

### Advanced Options

```bash
python cli.py scrape https://example.com/docs --max-depth 3 --delay 2.0
```

### Command Line Options

- `url`: The starting URL to scrape (required)
- `--max-depth`: Maximum crawl depth (default: 2)
- `--delay`: Delay between requests in seconds (default: 1.0)

## Output Structure

The scraper saves files to `data/raw/` directory:

- **Markdown files** (`.md`): Clean text content with headers and source attribution
- **JSON files** (`.json`): Metadata including title, URL, fetch timestamp, description, author, and date

### Example Output

**data/raw/example.com_page.md:**
```markdown
# Page Title

*Brief description if available*

**Source:** https://example.com/page

---

Clean extracted content from the page...
```

**data/raw/example.com_page.json:**
```json
{
  "title": "Page Title",
  "url": "https://example.com/page",
  "fetched_at": "2025-06-29T19:15:41.445940",
  "description": "Brief description if available",
  "author": "",
  "date": "2023-10-29",
  "markdown_file": "data/raw/example.com_page.md"
}
```

## Technical Details

### Dependencies

- `requests`: HTTP client for web requests
- `beautifulsoup4`: HTML parsing for link extraction
- `trafilatura`: Advanced content extraction and cleaning
- `urllib3`: URL utilities and robots.txt parsing

### Key Features

1. **Smart Content Extraction**: Uses trafilatura for better content quality than basic HTML parsing
2. **Domain Restriction**: Only follows links within the target domain
3. **Binary File Filtering**: Checks file extensions and MIME types to skip non-text content
4. **URL Normalization**: Removes fragments and normalizes URLs to avoid duplicates
5. **Error Handling**: Graceful handling of network errors, timeouts, and parsing issues
6. **Logging**: Comprehensive logging for debugging and monitoring

### Ethics and Best Practices

- Respects robots.txt by default
- Implements rate limiting to avoid overwhelming servers
- Uses appropriate User-Agent string
- Only crawls within specified domain boundaries
- Provides configurable delays between requests

## Example Test

Test the scraper with httpbin.org:

```bash
python cli.py scrape https://httpbin.org/html --max-depth 1 --delay 0.5
```

This will create:
- `data/raw/httpbin.org_html.md`
- `data/raw/httpbin.org_html.json`

## Integration with Knowledge Base

The scraped content can be processed for the knowledge base by:

1. Running the scraper to collect raw content
2. Using the `ingest` command to process markdown files into the vector store
3. The RAG system can then query against the scraped content

```bash
# Scrape content
python cli.py scrape https://target.com/docs

# Process into knowledge base
python cli.py ingest
```
