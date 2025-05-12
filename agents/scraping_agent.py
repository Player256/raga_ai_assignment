from fastapi import FastAPI, Query
from pydantic import BaseModel
from typing import List, Optional
import requests
from bs4 import BeautifulSoup

app = FastAPI(title="Scraping Agent")


class FilingRequest(BaseModel):
    ticker: str
    filing_type: Optional[str] = "earnings"
    start_date: Optional[str] = None
    end_date: Optional[str] = None


def fetch_yahoo_earnings(ticker):
    url = f"https://finance.yahoo.com/quote/{ticker}/financials"
    resp = requests.get(url)
    soup = BeautifulSoup(resp.text, "html.parser")
    # This is a placeholder: Actual parsing will depend on Yahoo's HTML structure
    tables = soup.find_all("table")
    if tables:
        return tables[0].text
    return "No data found"


@app.post("/get_filings")
def get_filings(request: FilingRequest):
    # For demo: Only supports Yahoo Finance earnings
    if request.filing_type == "earnings":
        data = fetch_yahoo_earnings(request.ticker)
        return {
            "ticker": request.ticker,
            "filing_type": request.filing_type,
            "data": data,
        }
    else:
        return {"error": "Only earnings filings supported in demo."}
