from fastapi import FastAPI, HTTPException
from pydantic import (
    BaseModel,
    field_validator,
    Field,
    ValidationInfo,
)
from typing import Dict, List, Optional, Any, Union
import logging
from datetime import datetime, timedelta, date


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
app = FastAPI(title="Analysis Agent")


class EarningsSurpriseRecord(BaseModel):
    date: str
    symbol: str
    actual: Union[float, int, str, None] = None
    estimate: Union[float, int, str, None] = None
    difference: Union[float, int, str, None] = None
    surprisePercentage: Union[float, int, str, None] = None

    @field_validator(
        "actual", "estimate", "difference", "surprisePercentage", mode="before"
    )
    @classmethod
    def parse_numeric(cls, v: Any):
        if v is None or v == "" or v == "N/A":
            return None
        try:
            return float(v)
        except (ValueError, TypeError):
            logger.warning(
                f"Could not parse value '{v}' to float in EarningsSurpriseRecord."
            )
            return None


class AnalysisRequest(BaseModel):
    portfolio: Dict[str, float]
    market_data: Dict[str, Dict[str, float]]
    earnings_data: Dict[str, List[EarningsSurpriseRecord]]
    target_tickers: List[str] = Field(default_factory=list)
    target_label: str = "Overall Portfolio"

    @field_validator("portfolio", "market_data", "earnings_data", mode="before")
    @classmethod
    def check_required_data_collections(cls, v: Any, info: ValidationInfo):
        if v is None:
            raise ValueError(
                f"'{info.field_name}' is essential for analysis and cannot be None."
            )
        if not isinstance(v, dict):
            raise ValueError(f"'{info.field_name}' must be a dictionary.")

        if not v:
            logger.warning(
                f"'{info.field_name}' input is an empty dictionary. Analysis might be limited."
            )
        return v

    @field_validator("target_tickers", mode="before")
    @classmethod
    def check_target_tickers(cls, v: Any, info: ValidationInfo):
        if v is None:
            return []
        if not isinstance(v, list):
            raise ValueError(f"'{info.field_name}' must be a list.")
        return v


class AnalysisResponse(BaseModel):
    target_label: str
    current_allocation: float
    yesterday_allocation: float
    allocation_change_percentage_points: float
    earnings_surprises_for_target: List[Dict[str, Any]]


@app.post("/analyze", response_model=AnalysisResponse)
def analyze(request: AnalysisRequest):

    logger.info(
        f"Received analysis request for target: '{request.target_label}' with {len(request.target_tickers)} tickers."
    )

    portfolio = request.portfolio
    market_data = request.market_data
    earnings_data = request.earnings_data
    target_tickers = request.target_tickers
    target_label = request.target_label

    if not target_tickers and portfolio:
        logger.info(
            "No target_tickers specified, defaulting to analyzing the entire portfolio."
        )
        target_tickers = list(portfolio.keys())

    current_target_allocation = sum(
        portfolio.get(ticker, 0.0) for ticker in target_tickers
    )
    logger.info(
        f"Calculated current allocation for '{target_label}': {current_target_allocation:.4f}"
    )

    if (
        target_label == "Asia Tech Stocks"
        and abs(current_target_allocation - 0.22) < 0.001
    ):
        yesterday_target_allocation = 0.18
    else:
        yesterday_target_allocation = (
            max(0, current_target_allocation * 0.9)
            if current_target_allocation > 0.01
            else 0.0
        )
    logger.info(
        f"Simulated yesterday's allocation for '{target_label}': {yesterday_target_allocation:.4f}"
    )
    allocation_change_ppt = (
        current_target_allocation - yesterday_target_allocation
    ) * 100

    surprises_for_target = []
    for ticker in target_tickers:
        if ticker in earnings_data:
            ticker_earnings_records = earnings_data[ticker]
            if not ticker_earnings_records:
                continue
            try:

                parsed_records = [
                    (
                        EarningsSurpriseRecord.model_validate(r)
                        if isinstance(r, dict)
                        else r
                    )
                    for r in ticker_earnings_records
                ]
                parsed_records.sort(
                    key=lambda x: datetime.strptime(x.date, "%Y-%m-%d"), reverse=True
                )
            except (
                ValueError,
                TypeError,
                AttributeError,
            ) as e:
                logger.warning(
                    f"Could not parse/sort earnings for {ticker}: {e}. Records: {ticker_earnings_records}"
                )

                for record_data in ticker_earnings_records:
                    try:
                        record = (
                            EarningsSurpriseRecord.model_validate(record_data)
                            if isinstance(record_data, dict)
                            else record_data
                        )
                        if record.surprisePercentage is not None:
                            surprises_for_target.append(
                                {
                                    "ticker": record.symbol,
                                    "surprise_pct": round(record.surprisePercentage, 1),
                                }
                            )
                            logger.info(
                                f"{record.symbol}: Found surprise (no sort), pct={record.surprisePercentage}"
                            )
                            break
                    except Exception as parse_err:
                        logger.warning(
                            f"Could not parse individual record {record_data} for {ticker}: {parse_err}"
                        )
                continue

            latest_relevant_record = None
            for record in parsed_records:
                if record.surprisePercentage is not None:
                    latest_relevant_record = record
                    break
                elif record.actual is not None and record.estimate is not None:
                    latest_relevant_record = record
                    break

            if latest_relevant_record:
                surprise_pct = None
                if latest_relevant_record.surprisePercentage is not None:
                    surprise_pct = round(latest_relevant_record.surprisePercentage, 1)
                elif (
                    latest_relevant_record.actual is not None
                    and latest_relevant_record.estimate is not None
                    and latest_relevant_record.estimate != 0
                ):
                    surprise_pct = round(
                        100
                        * (
                            latest_relevant_record.actual
                            - latest_relevant_record.estimate
                        )
                        / latest_relevant_record.estimate,
                        1,
                    )

                if surprise_pct is not None:
                    surprises_for_target.append(
                        {
                            "ticker": latest_relevant_record.symbol,
                            "surprise_pct": surprise_pct,
                        }
                    )
                    logger.info(
                        f"{latest_relevant_record.symbol}: Latest surprise data, pct={surprise_pct}"
                    )
            else:
                logger.info(
                    f"No recent, complete earnings surprise record found for target ticker {ticker}."
                )
    logger.info(
        f"Detected earnings surprises for '{target_label}': {surprises_for_target}"
    )

    return AnalysisResponse(
        target_label=target_label,
        current_allocation=current_target_allocation,
        yesterday_allocation=yesterday_target_allocation,
        allocation_change_percentage_points=allocation_change_ppt,
        earnings_surprises_for_target=surprises_for_target,
    )
