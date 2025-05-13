import requests
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel
from typing import List, Dict, Optional, Any


from data_ingestion.api_loader import get_daily_adjusted_prices, DataIngestionError
import logging


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
    result: Dict[str, Dict[str, float]] = {}
    errors: Dict[str, str] = {}
    warnings: Dict[str, str] = {}

    key = (
        request.data_type
        if request.data_type in ["open", "high", "low", "close", "adjClose", "volume"]
        else "adjClose"
    )

    for ticker in request.tickers:
        try:
            raw = get_daily_adjusted_prices(ticker)

            time_series: Dict[str, Any] = {}
            if isinstance(raw, dict):
                time_series = raw
            elif isinstance(raw, list):
                logger.warning(
                    f"Loader returned list for {ticker}; filtering dict entries."
                )
                for rec in raw:
                    if isinstance(rec, dict) and "date" in rec:
                        date_val = rec["date"]
                        time_series[date_val] = rec
                    else:
                        logger.warning(
                            f"Skipping non-dict or missing-date entry for {ticker}: {rec}"
                        )
            else:
                raise DataIngestionError(
                    f"Unexpected format from loader for {ticker}: {type(raw)}"
                )

            ticker_prices: Dict[str, float] = {}
            for date_str, values in time_series.items():
                if not isinstance(values, dict):
                    warnings.setdefault(ticker, "")
                    warnings[ticker] += f" Non-dict for {date_str}; skipped."
                    continue
                if key not in values:
                    warnings.setdefault(ticker, "")
                    warnings[ticker] += f" Missing '{key}' on {date_str}."
                    continue
                try:
                    ticker_prices[date_str] = float(values[key])
                except (TypeError, ValueError):
                    warnings.setdefault(ticker, "")
                    warnings[ticker] += f" Invalid '{key}' value on {date_str}."

            if ticker_prices:
                result[ticker] = ticker_prices
                logger.info(f"Fetched {len(ticker_prices)} points for {ticker}.")
            else:
                warnings.setdefault(ticker, "")
                warnings[ticker] += " No valid data points found."

        except (requests.RequestException, DataIngestionError) as err:
            errors[ticker] = str(err)
            logger.error(f"Error fetching {ticker}: {err}")
        except Exception as err:
            errors[ticker] = f"Unexpected error for {ticker}: {err}"
            logger.error(errors[ticker])

    if not result and errors:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch market data for all tickers.",
        )

    return {"result": result, "errors": errors, "warnings": warnings}
