# data_ingestion/api_loader.py

import requests
import os
from dotenv import load_dotenv
from typing import Dict, List, Optional, Any
import logging

# Load environment variables (can also be loaded in the agent/orchestrator startup)
load_dotenv()

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Get FMP API Key from environment (same key used by scraping_loader)
FMP_API_KEY = os.getenv("FMP_API_KEY")
if not FMP_API_KEY:
    logger.warning("FMP_API_KEY not found. FMP historical price calls will fail.")


FMP_BASE_URL = "https://financialmodelingprep.com/api/v3"


class DataIngestionError(Exception):
    """Custom exception for data ingestion API errors."""

    pass


# We'll keep the function name get_daily_adjusted_prices for now for compatibility
# with the agent, but its implementation changes to use FMP.
def get_daily_adjusted_prices(ticker: str) -> Dict[str, Dict[str, Any]]:
    """
    Fetches historical daily adjusted prices for a single ticker from FMP.
    Returns a dictionary mapping date strings to price dictionaries.
    Raises DataIngestionError on API-specific issues.
    Raises requests.RequestException on network issues.
    """
    if not FMP_API_KEY:
        raise DataIngestionError("FMP API Key not configured.")

    # FMP endpoint for historical prices: /historical-price/{symbol}
    # They also have /historical-price/day/{symbol} for more recent data, might be faster.
    # Let's use the basic /historical-price/{symbol} first.
    endpoint = f"{FMP_BASE_URL}/historical-price/{ticker}"
    # FMP's historical endpoint parameters often include 'apikey'
    # Free tier might be limited in date range or data points.
    params = {
        "apikey": FMP_API_KEY
    }  # Add start/end date params if needed and supported by tier

    logger.info(f"Fetching historical daily adjusted data for {ticker} from FMP.")
    response = requests.get(endpoint, params=params, timeout=30)  # Add timeout
    response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
    data = response.json()

    # FMP historical endpoint structure:
    # { "symbol": "TSM", "historical": [ {...date and price data...}, ... ] }
    # The daily prices are typically in the 'historical' list.
    # Each item in the list is like:
    # {"date": "2023-12-29", "open": 100.1, "high": 100.5, "low": 99.8, "close": 100.3, "adjClose": 100.3, "volume": 12345}

    historical_data_list = data.get("historical")

    if isinstance(historical_data_list, list):
        # Convert the list of dicts into the dictionary format expected by the agent
        # Dict[date_str, Dict[price_type, value]]
        prices_dict: Dict[str, Dict[str, Any]] = {}
        for record in historical_data_list:
            if "date" in record:
                prices_dict[record["date"]] = record
        logger.info(
            f"Successfully fetched and formatted {len(prices_dict)} historical records for {ticker} from FMP."
        )
        return prices_dict
    else:
        # Handle unexpected response structure (e.g., empty list, error object)
        logger.error(
            f"Unexpected FMP historical price response structure for {ticker}: {data}"
        )
        # Check for common FMP error messages in the response body
        if isinstance(data, dict) and data.get("error"):
            raise DataIngestionError(
                f"FMP API returned error for {ticker}: {data['error']}"
            )
        if isinstance(data, dict) and not data:
            logger.warning(
                f"FMP API returned empty response for {ticker}, potentially no data."
            )
            return {}  # Return empty dict if response is empty dict

        raise DataIngestionError(f"Unexpected API response structure for {ticker}.")


# You could add other loader functions here using FMP (e.g., company profile, etc.)
