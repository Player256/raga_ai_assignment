# agents/scraping_agent/main.py (Modified)
import requests
from fastapi import FastAPI, HTTPException, Query, status
from pydantic import BaseModel
from typing import List, Optional, Dict, Any

# Import the loader function
from data_ingestion.scraping_loader import (
    get_earnings_surprises,
    FMPError,
)  # Import custom error
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Scraping Agent (FMP Earnings)")


class FilingRequest(BaseModel):
    ticker: str
    filing_type: Optional[str] = "earnings_surprise"  # Matches supported type
    start_date: Optional[str] = None  # Not directly used by current loader
    end_date: Optional[str] = None  # Not directly used by current loader


@app.post("/get_filings")
def get_filings(request: FilingRequest):
    """
    Fetches filings (earnings surprise) by calling the data_ingestion layer.
    """
    if request.filing_type != "earnings_surprise":
        raise HTTPException(
            status_code=400,
            detail=f"Only 'earnings_surprise' filing_type supported in demo, received '{request.filing_type}'.",
        )

    try:
        # Call the loader function from data_ingestion
        earnings_data_list = get_earnings_surprises(request.ticker)

        # The loader handles API calls and basic error checking.
        # The agent now just formats the response structure.
        return {
            "ticker": request.ticker,
            "filing_type": request.filing_type,
            "data": earnings_data_list,  # This is the list returned by the loader
        }

    except (requests.exceptions.RequestException, FMPError) as e:
        # Catch exceptions raised by the loader
        error_msg = f"Error fetching filings for {request.ticker}: {e}"
        logger.error(error_msg)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=error_msg
        )
    except Exception as e:
        error_msg = f"An unexpected error occurred processing {request.ticker}: {e}"
        logger.error(error_msg)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=error_msg
        )
