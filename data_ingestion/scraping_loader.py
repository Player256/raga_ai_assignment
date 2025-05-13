# data_ingestion/scraping_loader.py

import requests
import os
from dotenv import load_dotenv
from typing import List, Dict, Any
import logging

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get API Key from environment
FMP_API_KEY = os.getenv("FMP_API_KEY")
if not FMP_API_KEY:
    logger.warning("FMP_API_KEY not found. FMP calls will fail.")


FMP_BASE_URL = "https://financialmodelingprep.com/api/v3"


class FMPError(Exception):
    """Custom exception for FMP API errors."""

    pass


def get_earnings_surprises(ticker: str) -> List[Dict[str, Any]]:
    """
    Fetches earnings surprise data for a single ticker from Financial Modeling Prep.
    Returns a list of earnings surprise records.
    Raises FMPError on API-specific issues.
    Raises requests.RequestException on network issues.
    """
    if not FMP_API_KEY:
        raise FMPError("FMP API Key not configured.")

    # FMP endpoint for earnings surprises: /earning_surprise/{symbol}
    endpoint = f"{FMP_BASE_URL}/earning_surprise/{ticker}"
    params = {"apikey": FMP_API_KEY}

    logger.info(f"Fetching earnings surprise data for {ticker} from FMP.")
    response = requests.get(endpoint, params=params, timeout=30)  # Add timeout
    response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
    data = response.json()

    # FMP earnings surprise endpoint returns a list of historical surprises
    if isinstance(data, list):
        return data  # Return the list of records
    else:
        # Handle unexpected response structure (e.g., empty list, error object)
        logger.error(f"Unexpected FMP response structure for {ticker}: {data}")
        # Check for common FMP error messages in the response body if it's not a list
        if isinstance(data, dict) and data.get("error"):
            raise FMPError(f"FMP API returned error for {ticker}: {data['error']}")
        # Check for empty dict which sometimes means no data
        if isinstance(data, dict) and not data:
            logger.warning(
                f"FMP API returned empty response for {ticker}, potentially no data."
            )
            return []  # Return empty list if response is empty dict
        raise FMPError(f"Unexpected API response structure for {ticker}.")


# You could add other loader functions here (e.g., get_company_profile, get_financial_statements)
