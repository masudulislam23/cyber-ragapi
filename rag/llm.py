from vllm import LLM
from vllm.sampling_params import SamplingParams
from pathlib import Path
import shutil
from transformers import AutoModelForVision2Seq, AutoTokenizer, AutoProcessor
from huggingface_hub import hf_hub_download
from contextlib import asynccontextmanager
from safetensors.torch import load_file
from datetime import datetime
from serpapi import GoogleSearch
from datetime import datetime
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import pytz
from dotenv import load_dotenv
import os
import logging
from langchain_core.messages import HumanMessage

# Define Eastern Time Zone
eastern = pytz.timezone('US/Eastern')
current_data = datetime.now(eastern)

formatted_date = current_data.strftime('%B %d, %Y')  # e.g., "June 15, 2025"
formatted_time = current_data.strftime('%I:%M %p %Z')  # e.g., "09:45 AM EDT"

load_dotenv()

def needs_realtime_data(query: str) -> bool:
    """Check if the query requires real-time data"""
    time_sensitive_keywords = [
        'current', 'latest', 'now', 'today', 'recent', 'population',
        'weather', 'stock', 'price', 'rate', 'news', 'update'
    ]
    query_lower = query.lower()
    return any(keyword in query_lower for keyword in time_sensitive_keywords)

SERPAPI_KEY = os.getenv("SERPAPI_KEY")  # <-- Insert your SerpApi key here

async def fetch_realtime_data(query: str) -> Dict[str, Any]:
    """Fetch real-time data using SerpApi"""
    try:
        params = {
            "engine": "google",
            "q": query,
            "api_key": SERPAPI_KEY,
            "num": 6,
            "hl": "en"
        }
        search = GoogleSearch(params)
        results = search.get_dict()

        # Extract relevant information from search results
        search_data = {
            'type': 'search_results',
            'data': [],
            'source': 'SerpApi',
            'timestamp': datetime.now().isoformat()
        }

        for result in results.get('organic_results', [])[:5]:
            search_data['data'].append({
                'title': result.get('title', ''),
                'content': result.get('snippet', ''),
                'url': result.get('link', '')
            })

        return search_data

    except Exception as e:
        print(f"Error fetching real-time data from SerpApi: {str(e)}")
        return {
            'type': 'error',
            'data': None,
            'source': 'SerpApi',
            'timestamp': datetime.now().isoformat()
        }

async def needs_web_search_with_gpt(llm, query: str, sampling_params=None) -> bool:
    prompt = f"""
    if query matches any of the Search Trigger Conditions answer "YES", else answer "NO"

    Search Trigger Conditions (when to include real-time data):
    - The query involves real-time, recent, or dynamic information  
    _e.g., 'What's the score?', 'Bitcoin price today', 'Latest tech news'_
    - The query references uncommon or newly released tools, models, or events  
    _e.g., 'What is RealVisXL v6.5?', 'New AI models in 2025'_
    - The user explicitly requests a web search or external sources  
    _e.g., 'Search for...', 'Look this up online', 'I'm looking for...', 'Find info about...'_
    - The user asks to refine, expand, or continue a previous search  
    _e.g., 'Give me more links', 'Search deeper', 'Find better options', 'Expand that'_
    - The query asks about current or updated population statistics  
    _e.g., 'What is the population of the USA?', 'Population of Tokyo in 2025'_
    - The query involves rankings, top-N lists, or comparisons likely to change over time  
    _e.g., 'Top 10 richest countries', 'Top 10 football clubs', 'Best universities in 2025'_
    - The query is about people or topics in politics, science, or public affairs where current relevance matters
    _e.g., 'Who is the current Prime Minister of the UK?', 'Latest research on fusion energy', 'Top climate scientists today'_
    - The query involves purchasing or comparing goods or services (especially where availability, prices, or options change over time)  
      _e.g., 'I'm looking for some laptop to buy', 'Best laptop deals now', 'Where to buy fresh salmon?', 'Top phones under $500 in 2025'_

    Query: "{query}"
    Answer:
    """

    # Use ChatOpenAI with LangChain message format
    response = await llm.ainvoke([HumanMessage(content=prompt)])
    answer = response.content
    clean_answer = answer.strip('"\'')
    logging.info(f"needs_web_search_with_gpt called with query: {query!r}, result: {answer}")

    if 'YES' in clean_answer:
        return True
    else:
        return False

async def query2keywords(llm, query, sampling_params=None):
    prompt = f"""
    You are a smart search query optimizer.
    Your goal is to extract the **core intent** of a user's natural language input and expand it into **short, high-relevance search keyword phrases** that would improve the quality of search results.

    ### Instructions:
    - Remove irrelevant filler like "search for", "can you find", etc.
    - Understand the **real intent** behind the request.
    - Expand the query into **5-7 short keyword phrases** that a smart search engine or news engine would understand.
    - Only include keywords that are directly relevant and meaningful.
    - It is okay to return fewer than 5 phrases if necessary to maintain precision.
    - Return a **comma-separated list** of keyword search phrases only.
    - Today's current date and time is **{formatted_date}** **{formatted_time}**(Eastern Daylight Time) and reason about time-sensitive information accordingly. 

    ### Examples:
    **User:** search for new york news
    **Output:** new york news, nyc local news, manhattan headlines, brooklyn news today, breaking news new york

    **User:** find latest laptop prices
    **Output:** laptop prices, new laptops 2025, affordable laptops, budget laptops, best laptops under $1000, gaming laptop deals

    **User:** what's happening in tokyo
    **Output:** tokyo events, tokyo news, tokyo festivals, tokyo weather, current events tokyo

    **User:** news about apple stock
    **Output:** apple stock news, aapl stock forecast, apple earnings report, apple share price, tech stock 
    
    Optimize this query: {query}
    """

    # Use ChatOpenAI with LangChain message format
    response = await llm.ainvoke([HumanMessage(content=prompt)])
    result = response.content
    return result