# agents/analysis_agent/main.py

from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict, List, Optional

app = FastAPI(title="Analysis Agent")

class AnalysisRequest(BaseModel):
    portfolio: Dict[str, float]  # e.g., {"TSMC": 0.12, "SAMSUNG": 0.10}
    market_data: Dict[str, dict]  # Output from API Agent
    earnings_data: Dict[str, dict]  # Output from Scraping Agent
    asia_tech_tickers: Optional[List[str]] = None  # Optionally specify which tickers to include

@app.post("/analyze")
def analyze(request: AnalysisRequest):
    # Calculate Asia tech allocation
    asia_tech = request.asia_tech_tickers or list(request.portfolio.keys())
    asia_tech_alloc = sum([request.portfolio.get(t, 0) for t in asia_tech])
    
    # Dummy: yesterday's allocation (for demo, subtract 0.04)
    yesterday_alloc = max(0, asia_tech_alloc - 0.04)
    
    # Detect earnings surprises
    surprises = []
    for ticker, earnings in request.earnings_data.items():
        reported = earnings.get("reported", None)
        estimate = earnings.get("estimate", None)
        if reported is not None and estimate is not None and estimate != 0:
            surprise_pct = round(100 * (reported - estimate) / estimate, 1)
            surprises.append({
                "ticker": ticker,
                "surprise_pct": surprise_pct
            })
    
    return {
        "asia_tech_alloc": asia_tech_alloc,
        "yesterday_alloc": yesterday_alloc,
        "earnings_surprises": surprises
    }
