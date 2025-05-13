# agents/api_agent/main.py (Modified)

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from typing import List, Dict, Optional, Any

# Import the loader function
from data_ingestion.api_loader import (
    get_daily_adjusted_prices,
    AlphaVantageError,
)  # Import custom error
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="API Agent")


class MarketDataRequest(BaseModel):
    tickers: List[str]
    start_date: Optional[str] = (
        None  # Not directly used by current loader, but kept for model
    )
    end_date: Optional[str] = (
        None  # Not directly used by current loader, but kept for model
    )
    data_type: Optional[str] = (
        "close"  # Not directly used by current loader, but kept for model
    )


@app.post("/get_market_data")
def get_market_data(request: MarketDataRequest):
    """
    Fetches daily adjusted market data by calling the data_ingestion layer.
    Returns adjusted close prices per ticker per date.
    """
    market_data_results: Dict[str, Dict[str, float]] = {}
    errors: Dict[str, str] = {}
    warnings: Dict[str, str] = {}  # Added warnings field

    for ticker in request.tickers:
        try:
            # Call the loader function from data_ingestion
            # Use outputsize='compact' for demo to avoid huge responses & hit limits less often
            raw_time_series_data = get_daily_adjusted_prices(
                ticker, outputsize="compact"
            )

            # Process the raw data returned by the loader
            ticker_prices: Dict[str, float] = {}
            if raw_time_series_data:
                # Assuming loader returns the 'Time Series (Daily)' part
                for date_str, values in raw_time_series_data.items():
                    # AlphaVantage keys are standardized, '5. adjusted close' for TIME_SERIES_DAILY_ADJUSTED
                    close_price_key = "5. adjusted close"
                    if close_price_key in values:
                        try:
                            # Convert to float and store. Loader returned strings, agent converts.
                            ticker_prices[date_str] = float(values[close_price_key])
                        except (ValueError, TypeError):
                            logger.warning(
                                f"Could not convert close price to float for {ticker} on {date_str}: {values.get(close_price_key)}"
                            )
                            warnings[ticker] = (
                                warnings.get(ticker, "")
                                + f"Invalid price for {date_str}. "
                            )
                            continue  # Skip this date if value is invalid

                # Optional: Apply date filtering here if needed based on request.start_date/end_date
                # This logic would filter the `ticker_prices` dictionary

            market_data_results[ticker] = ticker_prices
            logger.info(
                f"Successfully processed data for {ticker}. {len(ticker_prices)} data points."
            )

        except (requests.exceptions.RequestException, AlphaVantageError) as e:
            # Catch exceptions raised by the loader
            errors[ticker] = f"Error fetching data for {ticker}: {e}"
            logger.error(errors[ticker])
        except Exception as e:
            errors[ticker] = f"An unexpected error occurred processing {ticker}: {e}"
            logger.error(errors[ticker])

    # Decide if we raise an HTTPException if *all* tickers failed
    if not market_data_results and errors:
        # If no data was successfully fetched for any ticker and there were errors
        detail_msg = "Failed to fetch market data for all requested tickers."
        # Optionally include specific errors in the detail or response body
        # detail_msg += " Specific issues: " + "; ".join(errors.values())
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=detail_msg
        )

    return {
        "result": market_data_results,
        "errors": errors,
        "warnings": warnings,
    }  # Return errors/warnings
