# data_ingestion/api_loader.py

import requests
import os
from dotenv import load_dotenv
from typing import Dict, List, Optional
import logging

# Load environment variables (can also be loaded in the agent/orchestrator startup)
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get API Key from environment
ALPHAVANTAGE_API_KEY = os.getenv("ALPHAVANTAGE_API_KEY")
if not ALPHAVANTAGE_API_KEY:
    logger.warning("ALPHAVANTAGE_API_KEY not found. AlphaVantage calls will fail.")


ALPHA_VANTAGE_BASE_URL = "https://www.alphavantage.co/query"


class AlphaVantageError(Exception):
    """Custom exception for AlphaVantage API errors."""

    pass


def get_daily_adjusted_prices(
    ticker: str, outputsize: str = "full"
) -> Dict[str, Dict[str, str]]:
    """
    Fetches daily adjusted market data for a single ticker from AlphaVantage.
    Returns the raw 'Time Series (Daily)' part of the response.
    Raises AlphaVantageError on API-specific issues.
    Raises requests.RequestException on network issues.
    """
    if not ALPHAVANTAGE_API_KEY:
        raise AlphaVantageError("AlphaVantage API Key not configured.")

    params = {
        "function": "TIME_SERIES_DAILY_ADJUSTED",
        "symbol": ticker,
        "outputsize": outputsize,  # 'compact' or 'full'
        "apikey": ALPHAVANTAGE_API_KEY,
    }

    logger.info(f"Fetching daily adjusted data for {ticker} from AlphaVantage.")
    response = requests.get(
        ALPHA_VANTAGE_BASE_URL, params=params, timeout=30
    )  # Add timeout
    response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
    data = response.json()

    if "Time Series (Daily)" in data:
        return data["Time Series (Daily)"]
    elif "Note" in data:
        # Handle API limit notes or other messages
        msg = data.get("Note", "Unknown API message")
        raise AlphaVantageError(f"API returned message for {ticker}: {msg}")
    elif "Error Message" in data:
        # Handle specific error messages from AV
        msg = data["Error Message"]
        raise AlphaVantageError(f"API returned error for {ticker}: {msg}")
    else:
        # Handle unexpected response structure
        logger.error(f"Unexpected AlphaVantage response structure for {ticker}: {data}")
        raise AlphaVantageError(f"Unexpected API response structure for {ticker}.")


# You could add other loader functions here (e.g., get_intraday_prices, get_company_overview)
