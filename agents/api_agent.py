# agents/api_agent.py (Modified again for processing FMP list output)

import requests  # Still used implicitly by the loader
from fastapi import FastAPI, HTTPException, Query, status
from pydantic import BaseModel
from typing import List, Dict, Optional, Any

# Import the loader function and custom error
from data_ingestion.api_loader import get_daily_adjusted_prices, DataIngestionError
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI(title="API Agent")


class MarketDataRequest(BaseModel):
    tickers: List[str]
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    data_type: Optional[str] = "adjClose"


@app.post("/get_market_data")
def get_market_data(request: MarketDataRequest):
    """
    Fetches daily adjusted market data by calling the data_ingestion layer (FMP).
    Returns adjusted close prices per ticker per date.
    """
    market_data_results: Dict[str, Dict[str, float]] = {}
    errors: Dict[str, str] = {}
    warnings: Dict[str, str] = {}

    # Use the requested data_type, defaulting to 'adjClose'
    requested_data_type_key = (
        request.data_type
        if request.data_type in ["open", "high", "low", "close", "adjClose", "volume"]
        else "adjClose"
    )

    for ticker in request.tickers:
        try:
            # Call the loader function from data_ingestion
            # Loader returns Dict[date_str, Dict[price_type, value]]
            # NOTE: The FMP loader *actually* returns the inner dict directly.
            # Let's make the loader return Dict[date_str, Dict[price_type, value]] as initially planned
            # OR adjust the agent to handle the list.
            # Let's stick to the plan: The loader reformats the FMP list into the desired dict structure.
            # Re-checking data_ingestion/api_loader.py -> Yes, it returns `prices_dict`. The 'list' error suggests the loader returned something else.
            # Let's look at the loader code again carefully.

            # --- RE-EXAMINE data_ingestion/api_loader.py ---
            # Ah, I see the potential confusion. The FMP loader returns a Dict[date, Dict[value]],
            # but the *original* AlphaVantage code structure in the agent iterated like
            # `for date_str, values in raw_time_series_data.items():`
            # and then used `values.get(...)`. This structure *is* correct for the FMP loader's output.
            # The error "'list' object has no attribute 'get'" must mean that `raw_time_series_data`
            # *itself* was a list. This happens if the FMP endpoint returns just a list
            # of historical records without the outer {"symbol": ..., "historical": [...]}.
            # Some FMP endpoints *do* this! Let's adjust the loader slightly to be robust.

            raw_data_from_loader = get_daily_adjusted_prices(ticker)

            ticker_prices: Dict[str, float] = {}
            # The loader *should* return Dict[date, Dict[values]], but sometimes FMP just returns the list.
            # Let's add a check here.
            if isinstance(raw_data_from_loader, dict):
                # This is the expected format Dict[date, Dict[values]]
                for date_str, values_dict in raw_data_from_loader.items():
                    if requested_data_type_key in values_dict:
                        try:
                            ticker_prices[date_str] = float(
                                values_dict[requested_data_type_key]
                            )
                        except (ValueError, TypeError):
                            logger.warning(
                                f"Could not convert price to float for {ticker} on {date_str}: {values_dict.get(requested_data_type_key)}"
                            )
                            warnings[ticker] = (
                                warnings.get(ticker, "")
                                + f"Invalid price for {date_str}. "
                            )
                            continue
                    else:
                        logger.warning(
                            f"Data type '{requested_data_type_key}' not found for {ticker} on {date_str}."
                        )
                        warnings[ticker] = (
                            warnings.get(ticker, "")
                            + f"Missing data type {requested_data_type_key} for {date_str}. "
                        )

            # Also handle the case where the loader might return an empty dict {}
            elif not raw_data_from_loader:
                logger.warning(f"Loader returned no data for {ticker}.")
                warnings[ticker] = (
                    warnings.get(ticker, "") + "No data returned by loader. "
                )

            # The error "'list' object has no attribute 'get'" strongly suggests the loader might have returned a list.
            # Let's check the loader code again.
            # In data_ingestion/api_loader.py, the line `return prices_dict` seems correct.
            # The only way it would return a list is if the FMP response `data.get("historical")` *itself*
            # was the final response structure, rather than being nested.
            # Let's assume the FMP endpoint /historical-price/{symbol} *always* returns {"symbol": "...", "historical": [...]}.
            # The error might be in the *previous* AlphaVantage attempt logs polluting the state, or a very strange edge case.
            # Let's refine the loop based on the EXPECTED FMP loader output (Dict[date, Dict[values]]).
            # The previous loop logic was actually mostly correct for this expected output.
            # The list error is confusing if the loader returned a dict.
            # Could the loader sometimes return a list directly? Let's make the loader explicitly raise error if not dict.

            # --- REFINING data_ingestion/api_loader.py parser ---
            # Ensure the loader *always* returns a dictionary, even if empty or error occurred within it.

            # Let's assume the loader *is* returning Dict[date_str, Dict[str, Any]] as intended.
            # The original loop *should* have worked on that.
            # The 'list' object error indicates `raw_time_series_data` was a list.
            # This shouldn't happen if `get_daily_adjusted_prices` is working as intended.
            # Could it be an older version of the loader code running? Or a caching issue?
            # Or did I misunderstand the FMP historical response structure?

            # Let's re-verify the FMP historical price structure.
            # Example: https://financialmodelingprep.com/api/v3/historical-price-full/AAPL?apikey=YOUR_KEY
            # This endpoint returns: {"symbol":"AAPL","historical":[{...}, {...}]}
            # My loader is using /historical-price/{symbol}, which I believe returns similar.

            # Possibility: Is the loader function `get_daily_adjusted_prices` itself defined as returning a List?
            # No, the type hint is Dict[str, Dict[str, Any]].

            # Let's add a more explicit check in the agent for the loader's output type
            if not isinstance(raw_data_from_loader, dict):
                logger.error(
                    f"Loader returned unexpected type {type(raw_data_from_loader)} for {ticker}. Expected dict."
                )
                errors[ticker] = (
                    errors.get(ticker, "") + f"Loader returned unexpected data format."
                )
                # Continue to next ticker rather than crashing the agent
                continue

            # If it IS a dict, the loop should work. Let's add more specific logging inside the loop.
            for date_str, values_dict in raw_data_from_loader.items():
                if not isinstance(values_dict, dict):
                    logger.warning(
                        f"Value for date {date_str} for {ticker} is not a dict (is {type(values_dict)}). Skipping."
                    )
                    warnings[ticker] = (
                        warnings.get(ticker, "")
                        + f"Invalid data format for date {date_str}. "
                    )
                    continue  # Skip this date

                if requested_data_type_key in values_dict:
                    try:
                        # Convert to float and store
                        ticker_prices[date_str] = float(
                            values_dict[requested_data_type_key]
                        )
                    except (ValueError, TypeError):
                        logger.warning(
                            f"Could not convert price to float for {ticker} on {date_str}: {values_dict.get(requested_data_type_key)}"
                        )
                        warnings[ticker] = (
                            warnings.get(ticker, "") + f"Invalid price for {date_str}. "
                        )
                        continue  # Skip this date if value is invalid
                else:
                    logger.warning(
                        f"Data type '{requested_data_type_key}' not found for {ticker} on {date_str}."
                    )
                    warnings[ticker] = (
                        warnings.get(ticker, "")
                        + f"Missing data type {requested_data_type_key} for {date_str}. "
                    )

            # Apply date filtering here if needed

            market_data_results[ticker] = ticker_prices
            logger.info(
                f"Successfully processed data for {ticker}. {len(ticker_prices)} data points extracted for '{requested_data_type_key}'."
            )

        except (requests.exceptions.RequestException, DataIngestionError) as e:
            errors[ticker] = f"Error fetching data for {ticker}: {e}"
            logger.error(errors[ticker])
        except Exception as e:
            # Catch any other unexpected errors during processing *after* loader call
            errors[ticker] = f"An unexpected error occurred processing {ticker}: {e}"
            logger.error(errors[ticker])

    # Decide if we raise an HTTPException if *all* tickers failed
    if not market_data_results and errors:
        detail_msg = "Failed to fetch market data for all requested tickers."
        logger.error(f"All tickers failed. Specific errors: {errors}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=detail_msg
        )
    # If some data was fetched, return 200 even if there were errors/warnings for other tickers
    return {
        "result": market_data_results,
        "errors": errors,
        "warnings": warnings,
    }
