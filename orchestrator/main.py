from fastapi import FastAPI, UploadFile, File, HTTPException, status
from pydantic import BaseModel
import httpx
import os
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from typing import Dict, List, Optional, Any, Union
import logging
import json

load_dotenv()
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Orchestrator (Generalized)")

AGENT_API_URL = os.getenv("AGENT_API_URL", "http://localhost:8001")
AGENT_SCRAPING_URL = os.getenv("AGENT_SCRAPING_URL", "http://localhost:8002")
AGENT_RETRIEVER_URL = os.getenv("AGENT_RETRIEVER_URL", "http://localhost:8003")
AGENT_ANALYSIS_URL = os.getenv("AGENT_ANALYSIS_URL", "http://localhost:8004")
AGENT_LANGUAGE_URL = os.getenv("AGENT_LANGUAGE_URL", "http://localhost:8005")
AGENT_VOICE_URL = os.getenv("AGENT_VOICE_URL", "http://localhost:8006")


class EarningsSurpriseRecordState(BaseModel):
    date: str
    symbol: str
    actual: Union[float, int, str, None] = None
    estimate: Union[float, int, str, None] = None
    difference: Union[float, int, str, None] = None
    surprisePercentage: Union[float, int, str, None] = None


class MarketBriefState(BaseModel):
    audio_input: Optional[bytes] = None
    user_text: Optional[str] = None
    nlu_results: Optional[Dict[str, str]] = None

    target_tickers_for_data_fetch: List[str] = []
    market_data: Optional[Dict[str, Dict[str, float]]] = None
    filings: Optional[Dict[str, List[EarningsSurpriseRecordState]]] = None

    indexed: bool = False
    retrieved_docs: Optional[List[str]] = None
    analysis: Optional[Dict[str, Any]] = None
    brief: Optional[str] = None
    audio_output: Optional[bytes] = None
    errors: List[str] = []
    warnings: List[str] = []

    class Config:
        arbitrary_types_allowed = True


EXAMPLE_PORTFOLIO_FILE = "example_portfolio.json"
EXAMPLE_METADATA_FILE = "example_metadata.json"


def load_example_data(file_path: str, default_data: Dict) -> Dict:
    if os.path.exists(file_path):
        try:
            with open(file_path, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load {file_path}: {e}. Using default.")
    return default_data


EXAMPLE_PORTFOLIO = load_example_data(
    EXAMPLE_PORTFOLIO_FILE,
    {
        "TSM": {
            "weight": 0.22,
            "country": "Taiwan",
            "sector": "Technology",
        },
        "AAPL": {"weight": 0.15, "country": "USA", "sector": "Technology"},
        "MSFT": {"weight": 0.10, "country": "USA", "sector": "Technology"},
        "JNJ": {"weight": 0.08, "country": "USA", "sector": "Healthcare"},
        "BABA": {
            "weight": 0.05,
            "country": "China",
            "sector": "Technology",
        },
    },
)


async def call_agent(
    client: httpx.AsyncClient,
    url: str,
    method: str = "post",
    json_payload: Optional[Dict] = None,
    files_payload: Optional[Dict] = None,
    timeout: float = 60.0,
) -> Dict:
    try:
        logger.info(
            f"Calling agent at {url} with payload keys: {list(json_payload.keys()) if json_payload else 'N/A'}"
        )
        if method == "post":
            if json_payload:
                response = await client.post(url, json=json_payload, timeout=timeout)
            elif files_payload:
                response = await client.post(url, files=files_payload, timeout=timeout)
            else:
                raise ValueError("POST request requires json_payload or files_payload.")
        elif method == "get":
            response = await client.get(url, params=json_payload, timeout=timeout)
        else:
            raise ValueError(f"Unsupported method: {method}")

        response.raise_for_status()
        logger.info(f"Agent at {url} returned status {response.status_code}.")
        return response.json()

    except httpx.ConnectError as e:
        error_msg = f"Connection error calling agent at {url}: {e}"
        logger.error(error_msg)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=error_msg
        )
    except httpx.RequestError as e:
        error_msg = f"Request error calling agent at {url}: {e}"
        logger.error(error_msg)
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT, detail=error_msg
        )
    except httpx.HTTPStatusError as e:
        error_msg = f"HTTP error calling agent at {url}: {e.response.status_code} - {e.response.text[:200]}"
        logger.error(error_msg)
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
    except Exception as e:
        error_msg = f"An unexpected error occurred calling agent at {url}: {e}"
        logger.error(error_msg, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=error_msg
        )


async def stt_node(state: MarketBriefState) -> MarketBriefState:

    async with httpx.AsyncClient() as client:
        if not state.audio_input:
            state.errors.append("STT Node: No audio input provided.")
            logger.error(state.errors[-1])
            state.user_text = "Error: No audio provided for STT."
            return state
        files = {"audio": ("input.wav", state.audio_input, "audio/wav")}
        try:
            response_data = await call_agent(
                client, f"{AGENT_VOICE_URL}/stt", files_payload=files
            )
            if "transcript" in response_data:
                state.user_text = response_data["transcript"]
                logger.info(f"STT successful. Transcript: {state.user_text[:50]}...")
            else:
                error_msg = f"STT agent response missing 'transcript': {response_data}"
                logger.error(error_msg)
                state.errors.append(error_msg)
                state.user_text = "Error: STT failed."
        except HTTPException as e:
            state.errors.append(f"STT Node failed: {e.detail}")
            state.user_text = "Error: STT service unavailable or failed."
    return state


async def nlu_node(state: MarketBriefState) -> MarketBriefState:
    """(NEW) Calls an NLU process (simulated here) to extract intent."""
    if not state.user_text or "Error:" in state.user_text:
        state.warnings.append(
            "NLU Node: Skipping due to missing or error in user_text."
        )
        state.nlu_results = {
            "region": "Global",
            "sector": "Overall Portfolio",
        }
        return state

    logger.info(f"NLU Node: Processing query: '{state.user_text}'")

    query_lower = state.user_text.lower()
    region = "Global"
    sector = "Overall Portfolio"

    if "asia" in query_lower and "tech" in query_lower:
        region = "Asia"
        sector = "Technology"
        logger.info("NLU Simulation: Detected 'Asia' and 'Tech'.")
    elif "us" in query_lower or "usa" in query_lower or "america" in query_lower:
        region = "USA"
        if "tech" in query_lower:
            sector = "Technology"
        elif "health" in query_lower:
            sector = "Healthcare"
        logger.info(f"NLU Simulation: Detected Region '{region}', Sector '{sector}'.")

    state.nlu_results = {"region": region, "sector": sector}
    logger.info(f"NLU Node: Results: {state.nlu_results}")

    target_tickers = []
    portfolio_keys = list(EXAMPLE_PORTFOLIO.keys())

    if region == "Global" and (
        sector == "Overall Portfolio" or sector == "Overall Market"
    ):
        target_tickers = portfolio_keys
    else:
        for ticker, details in EXAMPLE_PORTFOLIO.items():
            matches_region = region == "Global"
            if region == "Asia" and details.get("country") in [
                "Taiwan",
                "China",
                "Korea",
                "Japan",
                "India",
            ]:
                matches_region = True
            elif region == "USA" and details.get("country") == "USA":
                matches_region = True

            matches_sector = sector == "Overall Portfolio" or sector == "Overall Market"
            if sector.lower() == details.get("sector", "").lower():
                matches_sector = True

            if matches_region and matches_sector:
                target_tickers.append(ticker)

    if not target_tickers and portfolio_keys:
        logger.warning(
            f"NLU filtering yielded no specific tickers for {region}/{sector}, defaulting to all portfolio tickers."
        )
        target_tickers = portfolio_keys
        state.nlu_results["region_effective"] = "Global"
        state.nlu_results["sector_effective"] = "Overall Portfolio"

    state.target_tickers_for_data_fetch = list(set(target_tickers))
    logger.info(
        f"NLU Node: Target tickers for data fetch: {state.target_tickers_for_data_fetch}"
    )
    if not state.target_tickers_for_data_fetch:
        state.warnings.append(
            "NLU Node: No target tickers identified for data fetching based on query and portfolio."
        )

    return state


async def api_agent_node(state: MarketBriefState) -> MarketBriefState:
    if not state.target_tickers_for_data_fetch:
        state.warnings.append(
            "API Agent Node: No target tickers to fetch market data for. Skipping."
        )
        state.market_data = {}
        return state

    async with httpx.AsyncClient() as client:
        payload = {
            "tickers": state.target_tickers_for_data_fetch,
            "data_type": "adjClose",
        }
        try:
            response_data = await call_agent(
                client, f"{AGENT_API_URL}/get_market_data", json_payload=payload
            )
            state.market_data = response_data.get("result", {})
            logger.info(
                f"API Agent successful. Fetched data for tickers: {list(state.market_data.keys()) if state.market_data else 'None'}"
            )
            if response_data.get("errors"):
                state.warnings.append(
                    f"API Agent reported errors: {response_data['errors']}"
                )
            if response_data.get("warnings"):
                state.warnings.extend(response_data.get("warnings", []))

        except HTTPException as e:
            state.errors.append(
                f"API Agent Node failed for tickers {state.target_tickers_for_data_fetch}: {e.detail}"
            )
            state.market_data = {}
    return state


async def scraping_agent_node(state: MarketBriefState) -> MarketBriefState:
    if not state.target_tickers_for_data_fetch:
        state.warnings.append(
            "Scraping Agent Node: No target tickers to fetch earnings for. Skipping."
        )
        state.filings = {}
        return state

    async with httpx.AsyncClient() as client:
        filings_data: Dict[str, List[Dict[str, Any]]] = {}
        for ticker in state.target_tickers_for_data_fetch:
            payload = {"ticker": ticker, "filing_type": "earnings_surprise"}
            try:
                response_data = await call_agent(
                    client, f"{AGENT_SCRAPING_URL}/get_filings", json_payload=payload
                )

                if "data" in response_data and isinstance(response_data["data"], list):

                    filings_data[ticker] = response_data["data"]
                    logger.info(
                        f"Scraping Agent got {len(response_data['data'])} records for {ticker}."
                    )
                    if not response_data["data"]:
                        logger.info(
                            f"Scraping Agent for {ticker} returned 0 earnings surprise records."
                        )
                else:
                    filings_data[ticker] = []
                    state.errors.append(
                        f"Scraping agent for {ticker} returned malformed data: {str(response_data)[:100]}"
                    )
            except HTTPException as e:
                state.errors.append(
                    f"Scraping Agent Node failed for {ticker}: {e.detail}"
                )
                filings_data[ticker] = []
        state.filings = filings_data
    return state


async def retriever_agent_node(state: MarketBriefState) -> MarketBriefState:

    async with httpx.AsyncClient() as client:
        docs_to_index = []
        if state.filings:
            for (
                ticker,
                records_list,
            ) in state.filings.items():
                if records_list:
                    doc_content = f"Earnings surprise data for {ticker}:\n" + "\n".join(
                        [
                            f"Date: {r.get('date', 'N/A')}, Symbol: {r.get('symbol', 'N/A')}, "
                            f"Actual: {r.get('actual', 'N/A')}, Estimate: {r.get('estimate', 'N/A')}, "
                            f"Surprise%: {r.get('surprisePercentage', 'N/A')}"
                            for r in records_list
                        ]
                    )
                    docs_to_index.append(doc_content)

        if docs_to_index:
            try:

                pass
            except Exception as e:
                state.errors.append(f"Retriever indexing failed: {e}")
                state.indexed = False
        else:
            state.indexed = False
            logger.info("Retriever: No new documents to index.")

        if state.user_text:
            try:

                pass
            except Exception as e:
                state.errors.append(f"Retriever retrieval failed: {e}")
                state.retrieved_docs = []
        else:
            state.retrieved_docs = []
    return state


async def analysis_agent_node(state: MarketBriefState) -> MarketBriefState:
    if not state.market_data and not state.filings:
        state.warnings.append(
            "Analysis Agent Node: No market data or filings available. Skipping analysis."
        )
        state.analysis = None
        return state

    async with httpx.AsyncClient() as client:

        nlu_res = state.nlu_results if state.nlu_results else {}
        region_label = nlu_res.get("region_effective", nlu_res.get("region", "Global"))
        sector_label = nlu_res.get(
            "sector_effective", nlu_res.get("sector", "Overall Portfolio")
        )

        if region_label == "Global" and (
            sector_label == "Overall Portfolio" or sector_label == "Overall Market"
        ):
            target_label_for_analysis = "Overall Portfolio"
        else:
            target_label_for_analysis = (
                f"{region_label.replace('USA', 'US')} {sector_label} Stocks".strip()
            )

        analysis_target_tickers = state.target_tickers_for_data_fetch

        current_portfolio_weights = {
            ticker: details["weight"] for ticker, details in EXAMPLE_PORTFOLIO.items()
        }

        payload = {
            "portfolio": current_portfolio_weights,
            "market_data": state.market_data if state.market_data else {},
            "earnings_data": (state.filings if state.filings else {}),
            "target_tickers": analysis_target_tickers,
            "target_label": target_label_for_analysis,
        }
        try:
            response_data = await call_agent(
                client, f"{AGENT_ANALYSIS_URL}/analyze", json_payload=payload
            )

            state.analysis = response_data
            logger.info(
                f"Analysis Agent successful for '{response_data.get('target_label')}'."
            )
        except HTTPException as e:
            state.errors.append(f"Analysis Agent Node failed: {e.detail}")
            state.analysis = None
    return state


async def language_agent_node(state: MarketBriefState) -> MarketBriefState:

    async with httpx.AsyncClient() as client:
        if not state.user_text or "Error:" in state.user_text:
            state.errors.append("Language Agent: Skipping due to no valid user text.")
            state.brief = (
                "I could not understand your query or there was an earlier error."
            )
            return state

        analysis_payload_for_llm: Dict[str, Any]
        if state.analysis and isinstance(state.analysis, dict):

            analysis_payload_for_llm = {
                "target_label": state.analysis.get("target_label", "the portfolio"),
                "current_allocation": state.analysis.get("current_allocation", 0.0),
                "yesterday_allocation": state.analysis.get("yesterday_allocation", 0.0),
                "allocation_change_percentage_points": state.analysis.get(
                    "allocation_change_percentage_points", 0.0
                ),
                "earnings_surprises_for_target": state.analysis.get(
                    "earnings_surprises_for_target", []
                ),
            }
        else:
            logger.warning(
                "Language Agent: Analysis data is missing or not a dict. Using defaults."
            )
            state.warnings.append(
                "Language Agent: Analysis data unavailable, brief will be general."
            )
            analysis_payload_for_llm = {
                "target_label": "the portfolio (analysis data missing)",
                "current_allocation": 0.0,
                "yesterday_allocation": 0.0,
                "allocation_change_percentage_points": 0.0,
                "earnings_surprises_for_target": [],
            }

        payload = {
            "user_query": state.user_text,
            "analysis": analysis_payload_for_llm,
            "retrieved_docs": state.retrieved_docs if state.retrieved_docs else [],
        }
        try:
            response_data = await call_agent(
                client, f"{AGENT_LANGUAGE_URL}/generate_brief", json_payload=payload
            )
            state.brief = response_data.get("brief")
            logger.info(f"Language Agent successful. Brief: {state.brief[:70]}...")
        except HTTPException as e:
            state.errors.append(f"Language Agent Node failed: {e.detail}")
            state.brief = "Sorry, I couldn't generate the brief at this time due to an internal error."
    return state


async def tts_node(state: MarketBriefState) -> MarketBriefState:

    brief_text_for_tts = state.brief
    if state.errors and (
        not state.brief
        or "sorry" in state.brief.lower()
        or "error" in state.brief.lower()
    ):

        error_count = len(state.errors)
        brief_text_for_tts = f"I encountered {error_count} error{'s' if error_count > 1 else ''} while processing your request. Please check the detailed report."
        logger.warning(
            f"TTS Node: Generating audio for error summary due to {error_count} errors in state."
        )
    elif not state.brief:
        brief_text_for_tts = "The market brief could not be generated."
        logger.warning("TTS Node: No brief text available from language agent.")
        state.warnings.append("TTS Node: No brief content to synthesize.")

    if not brief_text_for_tts:
        state.audio_output = None
        return state

    async with httpx.AsyncClient() as client:
        payload = {"text": brief_text_for_tts, "lang": "en"}
        try:
            response_data = await call_agent(
                client, f"{AGENT_VOICE_URL}/tts", json_payload=payload
            )
            if "audio" in response_data and isinstance(response_data["audio"], str):
                state.audio_output = bytes.fromhex(response_data["audio"])
                logger.info("TTS successful. Audio bytes received.")
            else:
                state.errors.append(
                    f"TTS Agent response missing or invalid 'audio': {str(response_data)[:100]}"
                )
                state.audio_output = None
        except HTTPException as e:
            state.errors.append(f"TTS Node failed: {e.detail}")
            state.audio_output = None
    return state


def build_market_brief_graph():
    builder = StateGraph(MarketBriefState)
    builder.add_node("stt", stt_node)
    builder.add_node("nlu", nlu_node)
    builder.add_node("api_agent", api_agent_node)
    builder.add_node("scraping_agent", scraping_agent_node)
    builder.add_node("retriever_agent", retriever_agent_node)
    builder.add_node("analysis_agent", analysis_agent_node)
    builder.add_node("language_agent", language_agent_node)
    builder.add_node("tts", tts_node)

    builder.set_entry_point("stt")
    builder.add_edge("stt", "nlu")
    builder.add_edge("nlu", "api_agent")
    builder.add_edge("api_agent", "scraping_agent")
    builder.add_edge("scraping_agent", "retriever_agent")
    builder.add_edge("retriever_agent", "analysis_agent")
    builder.add_edge("analysis_agent", "language_agent")
    builder.add_edge("language_agent", "tts")
    builder.add_edge("tts", END)
    return builder.compile()


graph = build_market_brief_graph()


@app.post("/market_brief")
async def market_brief(audio: UploadFile = File(...)):

    logger.info("Received request to /market_brief")
    if not audio.content_type or not audio.content_type.startswith("audio/"):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Invalid file type.",
        )

    current_run_state = MarketBriefState()
    try:
        current_run_state.audio_input = await audio.read()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to read audio: {e}",
        )

    processed_state: MarketBriefState = current_run_state

    try:
        logger.info("Invoking LangGraph workflow...")

        initial_state_dict = current_run_state.model_dump(exclude_none=True)
        invocation_result = await graph.ainvoke(initial_state_dict)

        if isinstance(invocation_result, dict):

            processed_state = MarketBriefState(**invocation_result)
            logger.info("LangGraph execution finished. State updated.")
        else:
            logger.error(
                f"LangGraph ainvoke returned unexpected type: {type(invocation_result)}. Using partially updated state."
            )

            processed_state.errors.append(
                f"Internal graph error: result type {type(invocation_result)}"
            )

    except HTTPException as e:
        logger.error(
            f"Graph execution stopped due to HTTPException from an agent: {e.detail}"
        )
        processed_state.errors.append(f"Agent call failed: {e.detail}")
    except Exception as e:
        error_msg = f"An unexpected error occurred during graph execution: {e}"
        logger.error(error_msg, exc_info=True)
        processed_state.errors.append(error_msg)

    response_payload = {
        "transcript": processed_state.user_text,
        "brief": processed_state.brief,
        "audio": (
            processed_state.audio_output.hex() if processed_state.audio_output else None
        ),
        "errors": processed_state.errors,
        "warnings": processed_state.warnings,
        "status": "success" if not processed_state.errors else "failed",
        "message": "Market brief process completed."
        + (" With errors." if processed_state.errors else " Successfully."),
        "nlu_detected": processed_state.nlu_results,
        "analysis_details": processed_state.analysis,
    }
    logger.info(
        f"Request finished. Status: {response_payload['status']}. Errors: {len(response_payload['errors'])}. Warnings: {len(response_payload['warnings'])}."
    )
    return response_payload
