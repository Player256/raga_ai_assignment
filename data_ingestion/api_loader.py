import requests
import os
from dotenv import load_dotenv
from typing import Dict, List, Optional, Any
import logging

load_dotenv()

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

FMP_API_KEY = os.getenv("FMP_API_KEY")
ALPHAVANTAGE_API_KEY = os.getenv("ALPHAVANTAGE_API_KEY")

FMP_BASE_URL = "https://financialmodelingprep.com/api/v3"
ALPHAVANTAGE_BASE_URL = "https://www.alphavantage.co/query"


class DataIngestionError(Exception):
    """Custom exception for data ingestion API errors."""

    pass


class FMPFetchError(DataIngestionError):
    """Specific error for FMP fetching issues."""

    pass


class AVFetchError(DataIngestionError):
    """Specific error for AlphaVantage fetching issues."""

    pass


def _fetch_from_fmp(ticker: str, api_key: str) -> Dict[str, Dict[str, Any]]:
    """Internal function to fetch data from FMP. Uses /historical-price-full/ as recommended."""

    endpoint = f"{FMP_BASE_URL}/historical-price-full/{ticker}"
    params = {"apikey": api_key}
    logger.info(
        f"Fetching historical daily data for {ticker} from FMP (using /historical-price-full/)."
    )
    try:
        response = requests.get(endpoint, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()

        if isinstance(data, dict):

            if "Error Message" in data:
                raise FMPFetchError(
                    f"FMP API returned error for {ticker}: {data['Error Message']}"
                )
            if data.get("symbol") and "historical" in data:
                historical_data_list = data.get("historical")

                if isinstance(historical_data_list, list):
                    if not historical_data_list:
                        logger.warning(
                            f"FMP API returned empty historical data list for {ticker} (from /historical-price-full/)."
                        )
                        return {}

                    prices_dict: Dict[str, Dict[str, Any]] = {}
                    for record in historical_data_list:
                        if isinstance(record, dict) and "date" in record:
                            prices_dict[record["date"]] = record
                        else:
                            logger.warning(
                                f"Skipping invalid FMP record format for {ticker}: {record}"
                            )
                    logger.info(
                        f"Successfully fetched and formatted {len(prices_dict)} historical records for {ticker} from FMP."
                    )
                    return prices_dict
                else:
                    raise FMPFetchError(
                        f"FMP API historical data for {ticker} has unexpected 'historical' type: {type(historical_data_list)}"
                    )
            else:
                raise FMPFetchError(
                    f"FMP API response for {ticker} (from /historical-price-full/) missing expected structure (symbol/historical keys). Response: {str(data)[:200]}"
                )

        elif isinstance(data, list):
            if not data:
                logger.warning(
                    f"FMP API returned empty list for {ticker} (from /historical-price-full/)."
                )
                return {}
            if isinstance(data[0], dict) and (
                "Error Message" in data[0] or "error" in data[0]
            ):
                error_msg = data[0].get(
                    "Error Message", data[0].get("error", "Unknown error in list")
                )
                raise FMPFetchError(
                    f"FMP API returned error list for {ticker}: {error_msg}"
                )
            else:
                raise FMPFetchError(
                    f"FMP API returned unexpected top-level list structure for {ticker} (from /historical-price-full/). Response: {str(data)[:200]}"
                )
        else:
            raise FMPFetchError(
                f"FMP API returned unexpected response type for {ticker} (from /historical-price-full/): {type(data)}. Response: {str(data)[:200]}"
            )

    except requests.exceptions.RequestException as e:
        raise FMPFetchError(f"FMP data fetch (network) failed for {ticker}: {e}")
    except Exception as e:
        raise FMPFetchError(
            f"FMP data fetch (processing) failed for {ticker}: {e}. Response: {str(locals().get('data', 'N/A'))[:200]}"
        )


def _fetch_from_alphavantage(ticker: str, api_key: str) -> Dict[str, Dict[str, Any]]:
    """Internal function to fetch data from AlphaVantage."""
    endpoint = f"{ALPHAVANTAGE_BASE_URL}/query"
    params = {
        "function": "TIME_SERIES_DAILY_ADJUSTED",
        "symbol": ticker,
        "apikey": api_key,
        "outputsize": "compact",
    }
    logger.info(f"Fetching historical daily data for {ticker} from AlphaVantage.")
    try:
        response = requests.get(endpoint, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()

        if not isinstance(data, dict):
            raise AVFetchError(
                f"AlphaVantage API returned unexpected response type for {ticker}: {type(data)}. Expected dict. Response: {str(data)[:200]}"
            )

        if "Error Message" in data:
            raise AVFetchError(
                f"AlphaVantage API returned error for {ticker}: {data['Error Message']}"
            )
        if "Note" in data:
            logger.warning(
                f"AlphaVantage API returned note for {ticker}: {data['Note']} - treating as no data."
            )

            return {}

        time_series_data = data.get("Time Series (Daily)")

        if time_series_data is None:

            if not data:
                logger.warning(
                    f"AlphaVantage API returned an empty dictionary for {ticker}."
                )
                return {}
            else:
                raise AVFetchError(
                    f"AlphaVantage API response for {ticker} missing 'Time Series (Daily)' key. Response: {str(data)[:200]}"
                )

        if not isinstance(time_series_data, dict):
            raise AVFetchError(
                f"AlphaVantage API 'Time Series (Daily)' for {ticker} is not a dictionary. Type: {type(time_series_data)}. Response: {str(data)[:200]}"
            )

        if not time_series_data:
            logger.warning(
                f"AlphaVantage API returned empty time series data for {ticker}."
            )
            return {}

        prices_dict: Dict[str, Dict[str, Any]] = {}
        for date_str, values_dict in time_series_data.items():
            if isinstance(values_dict, dict):
                cleaned_values: Dict[str, Any] = {}
                if "1. open" in values_dict:
                    cleaned_values["open"] = values_dict["1. open"]
                if "2. high" in values_dict:
                    cleaned_values["high"] = values_dict["2. high"]
                if "3. low" in values_dict:
                    cleaned_values["low"] = values_dict["3. low"]
                if "4. close" in values_dict:
                    cleaned_values["close"] = values_dict["4. close"]
                if "5. adjusted close" in values_dict:
                    cleaned_values["adjClose"] = values_dict["5. adjusted close"]
                if "6. volume" in values_dict:
                    cleaned_values["volume"] = values_dict["6. volume"]

                if cleaned_values:
                    prices_dict[date_str] = cleaned_values
                else:
                    logger.warning(
                        f"AlphaVantage data for {ticker} on {date_str} missing expected price keys within daily record."
                    )
            else:
                logger.warning(
                    f"Skipping invalid AlphaVantage daily record (not a dict) for {ticker} on {date_str}: {values_dict}"
                )
        logger.info(
            f"Successfully fetched and formatted {len(prices_dict)} historical records for {ticker} from AlphaVantage."
        )
        return prices_dict

    except requests.exceptions.RequestException as e:
        raise AVFetchError(
            f"AlphaVantage data fetch (network) failed for {ticker}: {e}"
        )
    except Exception as e:
        raise AVFetchError(
            f"AlphaVantage data fetch (processing) failed for {ticker}: {e}. Response: {str(locals().get('data', 'N/A'))[:200]}"
        )


def get_daily_adjusted_prices(ticker: str) -> Dict[str, Dict[str, Any]]:
    """
    Fetches historical daily adjusted prices for a single ticker.
    Tries FMP first if key is available. If FMP fails, tries AlphaVantage if key is available.
    Returns a dictionary mapping date strings to price dictionaries.
    Raises DataIngestionError if no keys are configured or if both APIs fail.
    """
    fmp_key_available = bool(FMP_API_KEY)
    av_key_available = bool(ALPHAVANTAGE_API_KEY)

    if not fmp_key_available and not av_key_available:
        raise DataIngestionError(
            "No API keys configured for historical price data (FMP, AlphaVantage)."
        )

    fmp_error_detail = None
    av_error_detail = None
    data_from_fmp = {}
    data_from_av = {}

    if fmp_key_available:
        try:
            data_from_fmp = _fetch_from_fmp(ticker, FMP_API_KEY)
            if data_from_fmp:
                return data_from_fmp
            else:

                fmp_error_detail = f"FMP API returned no data for {ticker}."
                logger.warning(fmp_error_detail)
        except FMPFetchError as e:
            fmp_error_detail = str(e)
            logger.error(f"FMPFetchError for {ticker}: {fmp_error_detail}")
        except Exception as e:
            fmp_error_detail = (
                f"An unexpected error occurred during FMP fetch for {ticker}: {e}"
            )
            logger.error(fmp_error_detail)

    if av_key_available:
        try:
            data_from_av = _fetch_from_alphavantage(ticker, ALPHAVANTAGE_API_KEY)
            if data_from_av:
                return data_from_av
            else:

                av_error_detail = f"AlphaVantage API returned no data for {ticker}."
                logger.warning(av_error_detail)
        except AVFetchError as e:
            av_error_detail = str(e)
            logger.error(f"AVFetchError for {ticker}: {av_error_detail}")
        except Exception as e:
            av_error_detail = f"An unexpected error occurred during AlphaVantage fetch for {ticker}: {e}"
            logger.error(av_error_detail)

    error_messages = []
    if fmp_key_available:
        if fmp_error_detail:
            error_messages.append(f"FMP: {fmp_error_detail}")
        elif not data_from_fmp:
            error_messages.append(f"FMP: Returned no data for {ticker}.")

    if av_key_available:
        if av_error_detail:
            error_messages.append(f"AlphaVantage: {av_error_detail}")
        elif not data_from_av:
            error_messages.append(f"AlphaVantage: Returned no data for {ticker}.")

    providers_tried = []
    if fmp_key_available:
        providers_tried.append("FMP")
    if av_key_available:
        providers_tried.append("AlphaVantage")

    final_message = f"Failed to fetch historical data for {ticker} after trying {', '.join(providers_tried) if providers_tried else 'available providers'}."
    if error_messages:
        final_message += " Details: " + "; ".join(error_messages)
    else:
        final_message += " No data was returned from any attempted source."

    raise DataIngestionError(final_message)
