from fastapi import FastAPI, Query
from pydantic import BaseModel
from typing import List, Optional
import yfinance as yf

app = FastAPI(title="API Agent")

class MarketData(BaseModel):
    tickers: List[str]
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    data_type: Optional[str] = "close"


@app.post("/get_market_data")
def get_market_data(request: MarketData):
    data = {}
    for ticker in request.tickers:
        ticker_data = yf.Ticker(ticker)
        hist = ticker_data.history(start=request.start_date, end=request.end_date)
        if request.data_type in hist.columns:
            data[ticker] = hist[request.data_type].to_dict()
        else:
            data[ticker] = hist.to_dict()
    return {"result": data}
