from scrapfly import ScrapflyClient, ExtractionConfig, ScrapeConfig
import json
from typing import Dict, Optional, Any
import re
import requests
from io import BytesIO, StringIO
from PIL import Image
import pytesseract
from bs4 import BeautifulSoup
import pandas as pd
import logging
from dotenv import load_dotenv
import os
import time

# Set up logger
logger = logging.getLogger(__name__)

load_dotenv()

SCRAPFLY_KEY = os.getenv("SCRAPFLY_KEY")
scrapfly = ScrapflyClient(key=SCRAPFLY_KEY)

def clean_rag_content(html: str) -> str:
    """Remove all HTML tags and Markdown, return only visible, clean text content."""
    soup = BeautifulSoup(html, "html.parser")
    # Remove scripts, styles, and head/meta elements
    for tag in soup(["script", "style", "noscript", "head", "meta", "link", "title"]):
        tag.decompose()
    # Get all visible text
    text = soup.get_text(separator="\n", strip=True)
    # Remove excessive blank lines
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    text = "\n".join(lines)
    # Remove Markdown formatting
    # Remove headings, bold, italics, inline code, blockquotes, lists, links, images, horizontal rules
    text = re.sub(r'(^|\n)[#>*\-+]+\s?', '\n', text)  # Headings, blockquotes, lists
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)    # Bold
    text = re.sub(r'\*(.*?)\*', r'\1', text)          # Italics
    text = re.sub(r'_(.*?)_', r'\1', text)             # Italics
    text = re.sub(r'`([^`]*)`', r'\1', text)           # Inline code
    text = re.sub(r'!\[.*?\]\(.*?\)', '', text)     # Images
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)  # Links
    text = re.sub(r'^---$', '', text, flags=re.MULTILINE)         # Horizontal rules
    text = re.sub(r'^\s*$', '', text, flags=re.MULTILINE)        # Empty lines
    return text.strip()

def extract_tables(html_content: str) -> str:
    """Extract tables from HTML and convert to Markdown."""
    soup = BeautifulSoup(html_content, "html.parser")
    tables = soup.find_all("table")
    table_texts = []
    for table in tables:
        try:
            # Wrap table HTML in StringIO
            table_html = StringIO(str(table))
            df = pd.read_html(table_html)[0]
            table_texts.append(df.to_markdown(index=False))
        except Exception as e:
            continue
    return "\n\n".join(table_texts)

def extract_images_and_ocr(html_content: str, base_url: str) -> str:
    """Extract image URLs and perform OCR to get text from images."""
    soup = BeautifulSoup(html_content, "html.parser")
    texts = []
    for img_tag in soup.find_all("img"):
        img_url = img_tag.get("src") or ""
        # Handle relative URLs
        if img_url and not img_url.startswith("http"):
            if img_url.startswith("/"):
                from urllib.parse import urljoin
                img_url = urljoin(base_url, img_url)
            else:
                continue
        # Attempt OCR
        try:
            img_response = requests.get(img_url, timeout=5)
            img = Image.open(BytesIO(img_response.content))
            ocr_text = pytesseract.image_to_string(img)
            ocr_text = clean_rag_content(ocr_text)
            if ocr_text:
                texts.append(f"[Image OCR]: {ocr_text}")
        except Exception as e:
            continue
        # Also extract alt text
        alt_text = img_tag.get("alt", "")
        if alt_text:
            texts.append(f"[Image Alt]: {alt_text}")
    return "\n".join(texts)

async def extract_rag_data(url: str):
    """Extract RAG data from a URL using Scrapfly, with fallback to requests if needed."""
    USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"
    html_content = None
    try:
        scrape_config = ScrapeConfig(
            url=url,
            asp=True,
            render_js=True,
            headers={"User-Agent": USER_AGENT}
        )
        scrape_response = scrapfly.scrape(scrape_config)
        html_content = scrape_response.scrape_result['content']
        if not html_content or len(html_content.strip()) < 50:
            raise ValueError("Empty or too short content from Scrapfly")
    except Exception as e:
        logger.warning(f"Scrapfly failed for {url}, trying requests fallback: {e}")
        try:
            response = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=10)
            html_content = response.text
            logger.info("Fallback HTML content (first 500 chars): %s", html_content[:500])
        except Exception as e2:
            logger.error(f"Requests fallback also failed for {url}: {e2}")
            return None

    # Main article extraction via Scrapfly AI
    extraction_prompt = (
        "Extract ALL meaningful human-readable content from this web page, including the main article, headings, subheadings, lists, and any important sections. "
        "Ignore navigation, ads, and unrelated metadata. "
        "Return as JSON with fields: title, content. "
        "The 'content' field should be a single string containing all relevant text, preserving section order and hierarchy as much as possible."
    )
    try:
        extraction_response = scrapfly.extract(
            ExtractionConfig(
                body=html_content,
                content_type="text/html",
                url=url,
                extraction_prompt=extraction_prompt
            )
        )
        result = extraction_response.result
        data = result['result'].get('data', {})
        # logger.info("Title before clean: %s", data.get('title', ''))
        # logger.info("Content before clean: %s", data.get('content', ''))
        title = clean_rag_content(data.get('title', ''))
        main_content = clean_rag_content(data.get('content', ''))
        if not main_content or len(main_content) < 100:
            logger.warning("AI extraction incomplete or too short, falling back to full HTML cleaning.")
            main_content = clean_rag_content(html_content)
        # logger.info("Title after clean: %s", title)
        # logger.info("Content after clean: %s", main_content)
    except Exception as e:
        logger.warning(f"Scrapfly extract failed, falling back to full HTML cleaning: {e}")
        title = ""
        main_content = clean_rag_content(html_content)
        logger.info("Content after clean (fallback): %s", main_content)

    # Extract tables as Markdown
    table_content = extract_tables(html_content)
    # Extract images and OCR text
    image_content = extract_images_and_ocr(html_content, url)
    # Combine all content
    full_content = "\n\n".join([
        main_content,
        "[Tables]\n" + table_content if table_content else "",
        "[Images]\n" + image_content if image_content else ""
    ]).strip()
    if not full_content:
        return None
    return {
        "title": title,
        "content": full_content,
        "source_url": url
    }

async def scrape_search_results(search_data: Dict[str, Any]) -> Dict[str, Any]:
    """Scrape content from search results."""
    if not search_data or not search_data.get('data'):
        return search_data

    scraped_data = []
    for result in search_data['data']:
        url = result.get('url')
        if not url:
            continue

        rag_data = await extract_rag_data(url)
        if rag_data:
            scraped_data.append({
                'title': rag_data['title'],
                'content': rag_data['content'],
                'url': rag_data['source_url']
            })

    if scraped_data:
        search_data['data'] = scraped_data
        search_data['scraped'] = True

    return search_data 