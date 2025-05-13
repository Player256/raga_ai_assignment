# agents/analysis_agent/main.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, validator
from typing import Dict, List, Optional, Any, Union
import logging
from datetime import datetime, timedelta, date

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


app = FastAPI(title="Analysis Agent")


# Refined Pydantic models for input data structure expected from APIs
class EarningsSurpriseRecord(BaseModel):
    date: str
    symbol: str
    actual: Union[float, int, str, None] = (
        None  # Allow None/str in case API sends weird data
    )
    estimate: Union[float, int, str, None] = None
    difference: Union[float, int, str, None] = None
    surprisePercentage: Union[float, int, str, None] = None

    # Add a validator to ensure numeric fields can be converted or are None
    @validator("actual", "estimate", "difference", "surprisePercentage", pre=True)
    def parse_numeric(cls, v):
        if v is None or v == "":
            return None
        try:
            return float(v)
        except (ValueError, TypeError):
            logger.warning(f"Could not parse value '{v}' to float.")
            return None  # Or raise a validation error


class AnalysisRequest(BaseModel):
    portfolio: Dict[str, float]  # e.g., {"TSMC": 0.12, "SAMSUNG": 0.10}
    # Market data is now Dict[ticker, Dict[date, price]]
    market_data: Dict[str, Dict[str, float]]
    # Earnings data is now Dict[ticker, List[EarningsSurpriseRecord]] from FMP API
    earnings_data: Dict[str, List[EarningsSurpriseRecord]]
    asia_tech_tickers: Optional[List[str]] = (
        None  # Optionally specify which tickers to include
    )

    @validator("portfolio", "market_data", "earnings_data")
    def check_not_empty(cls, v):
        if not v:
            raise ValueError(
                "Input data (portfolio, market_data, earnings_data) cannot be empty."
            )
        return v


@app.post("/analyze")
def analyze(request: AnalysisRequest):
    """
    Analyzes portfolio allocation and earnings data.
    """
    logger.info("Received analysis request.")

    portfolio = request.portfolio
    market_data = request.market_data
    earnings_data = request.earnings_data
    asia_tech = (
        request.asia_tech_tickers
        if request.asia_tech_tickers is not None
        else list(portfolio.keys())
    )

    # --- 1. Calculate Asia tech allocation ---
    asia_tech_alloc = sum([portfolio.get(t, 0) for t in asia_tech])
    logger.info(f"Calculated Asia tech allocation: {asia_tech_alloc}")

    # --- 2. Estimate Yesterday's Allocation (More Realistic Placeholder) ---
    # This is still a placeholder, as true yesterday's allocation depends on
    # yesterday's AUM and position values. But we can at least TRY to get
    # yesterday's closing prices from market_data to show using the input.
    yesterday = (datetime.now() - timedelta(days=1)).strftime(
        "%Y-%m-%d"
    )  # Simple date format
    total_portfolio_value_yesterday = 0  # This would be needed for a real calculation
    yesterday_asia_tech_alloc = 0

    # In a real scenario, you'd need total AUM yesterday and yesterday's share counts
    # For this demo, we can simulate the *change* based on today's allocation
    # and a hypothetical price change based on market data, or keep the dummy.
    # Let's keep the dummy logic for now as per the original code, but acknowledge
    # this is where market_data *could* be used to derive this more realistically
    # if you had historical portfolio state (share counts).

    # Original dummy logic:
    yesterday_alloc_dummy = max(0, asia_tech_alloc - 0.04)  # Still dummy
    logger.info(f"Simulated yesterday's allocation (dummy): {yesterday_alloc_dummy}")

    # --- 3. Detect Earnings Surprises ---
    surprises = []
    for ticker, records in earnings_data.items():
        logger.info(f"Processing earnings data for {ticker}")
        # FMP provides a list. Find the most recent one with both actual and estimate.
        latest_record = None
        # Sort records by date (descending) to find the latest easily
        records.sort(key=lambda x: x.date, reverse=True)

        for record in records:
            if record.actual is not None and record.estimate is not None:
                latest_record = record
                break  # Found the latest relevant record

        if latest_record:
            # Use the surprise percentage directly from FMP if available and valid
            if latest_record.surprisePercentage is not None:
                surprise_pct = round(latest_record.surprisePercentage, 1)
                surprises.append({"ticker": ticker, "surprise_pct": surprise_pct})
                logger.info(f"{ticker}: Latest surprise data found, pct={surprise_pct}")
            else:
                # Fallback to calculating if percentage is missing but actual/estimate are present
                if latest_record.estimate != 0:  # Avoid division by zero
                    surprise_pct = round(
                        100
                        * (latest_record.actual - latest_record.estimate)
                        / latest_record.estimate,
                        1,
                    )
                    surprises.append({"ticker": ticker, "surprise_pct": surprise_pct})
                    logger.info(f"{ticker}: Calculated surprise pct={surprise_pct}")
                else:
                    logger.warning(
                        f"{ticker}: Estimate is zero, cannot calculate surprise percentage."
                    )
        else:
            logger.warning(
                f"No recent earnings record with both actual and estimate found for {ticker}."
            )

    logger.info(f"Detected earnings surprises: {surprises}")

    # --- Add potential uses for market_data here ---
    # Example: Get today's and yesterday's closing price for a ticker
    # This isn't used in the *output* of this agent in the original code,
    # but demonstrates accessing the data.
    # for ticker in asia_tech:
    #     ticker_market_data = market_data.get(ticker, {})
    #     # Need to find latest and second latest date from keys in ticker_market_data
    #     dates = sorted(ticker_market_data.keys(), reverse=True)
    #     today_close = ticker_market_data.get(dates[0]) if dates else None
    #     yesterday_close = ticker_market_data.get(dates[1]) if len(dates) > 1 else None
    #     if today_close is not None and yesterday_close is not None:
    #         price_change_pct = ((today_close - yesterday_close) / yesterday_close) * 100 if yesterday_close else 0
    #         logger.info(f"{ticker}: Today's Close={today_close}, Yesterday's Close={yesterday_close}, Change={price_change_pct:.2f}%")
    # This data could be used to inform the overall market sentiment aspect
    # or potentially refine the allocation change estimate if share counts were available.

    return {
        "asia_tech_alloc": asia_tech_alloc,
        # Returning the dummy value as in original code structure
        "yesterday_alloc": yesterday_alloc_dummy,
        "earnings_surprises": surprises,
        # You could add more analysis results here, e.g.,
        # "recent_performance": {...},
        # "volatility": {...},
        # "sentiment_indicators": {...} # If derived from market/news data
    }
