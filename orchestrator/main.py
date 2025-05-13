from fastapi import FastAPI, UploadFile, File, HTTPException, status
from pydantic import BaseModel
import httpx  # Use async http client
import os
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END  # Use END for clarity
from typing import Dict, List, Optional, Any
import logging

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Orchestrator")

# --- Configuration: Agent URLs from Environment Variables ---
AGENT_API_URL = os.getenv("AGENT_API_URL", "http://localhost:8001")
AGENT_SCRAPING_URL = os.getenv("AGENT_SCRAPING_URL", "http://localhost:8002")
AGENT_RETRIEVER_URL = os.getenv("AGENT_RETRIEVER_URL", "http://localhost:8003")
AGENT_ANALYSIS_URL = os.getenv("AGENT_ANALYSIS_URL", "http://localhost:8004")
AGENT_LANGUAGE_URL = os.getenv("AGENT_LANGUAGE_URL", "http://localhost:8005")
AGENT_VOICE_URL = os.getenv("AGENT_VOICE_URL", "http://localhost:8006")


# --- State Definition ---
# Added error handling fields to the state
class MarketBriefState(BaseModel):
    audio_input: Optional[bytes] = None
    user_text: Optional[str] = None
    market_data: Optional[Dict[str, Dict[str, float]]] = (
        None  # Matches refactored API Agent output
    )
    filings: Optional[Dict[str, List[Dict[str, Any]]]] = (
        None  # Matches refactored Scraping Agent output (list of records)
    )
    indexed: bool = False
    retrieved_docs: Optional[List[str]] = None  # List of document content strings
    analysis: Optional[Dict[str, Any]] = (
        None  # Matches refactored Analysis Agent output structure
    )
    brief: Optional[str] = None
    audio_output: Optional[bytes] = None
    # Error tracking fields
    errors: List[str] = []
    warnings: List[str] = []

    class Config:
        arbitrary_types_allowed = True  # Allow bytes in state


# Helper function to call agents and handle errors
async def call_agent(
    client: httpx.AsyncClient,
    url: str,
    method: str = "post",
    json_payload: Optional[Dict] = None,
    files_payload: Optional[Dict] = None,
) -> Dict:
    """Makes an asynchronous call to an agent and handles HTTP errors."""
    try:
        logger.info(f"Calling agent at {url}...")
        if method == "post":
            if json_payload:
                response = await client.post(
                    url, json=json_payload, timeout=60.0
                )  # Add timeout
            elif files_payload:
                response = await client.post(
                    url, files=files_payload, timeout=60.0
                )  # Add timeout
            else:
                raise ValueError(
                    "Must provide either json_payload or files_payload for POST."
                )
        elif method == "get":
            response = await client.get(url, timeout=60.0)  # Add timeout
        else:
            raise ValueError(f"Unsupported method: {method}")

        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
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
        # Catch potential client-side errors like timeouts
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=error_msg
        )
    except httpx.HTTPStatusError as e:
        error_msg = f"HTTP error calling agent at {url}: {e.response.status_code} - {e.response.text}"
        logger.error(error_msg)
        # Catch bad status codes from the agent itself
        raise HTTPException(status_code=e.response.status_code, detail=error_msg)
    except Exception as e:
        error_msg = f"An unexpected error occurred calling agent at {url}: {e}"
        logger.error(error_msg)
        # Catch any other unexpected errors
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=error_msg
        )


# --- Agent Node Functions (Async) ---


async def stt_node(state: MarketBriefState) -> MarketBriefState:
    """Calls the Voice Agent for Speech-to-Text."""
    async with httpx.AsyncClient() as client:
        files = {"audio": ("input.wav", state.audio_input, "audio/wav")}
        # Use the helper function for robust calls
        response_data = await call_agent(
            client, f"{AGENT_VOICE_URL}/stt", files_payload=files
        )

        # Validate and update state
        if "transcript" in response_data:
            state.user_text = response_data["transcript"]
            logger.info(f"STT successful. Transcript: {state.user_text[:50]}...")
        else:
            error_msg = f"STT agent response missing 'transcript': {response_data}"
            logger.error(error_msg)
            state.errors.append(error_msg)
            # Optionally, raise HTTPException if transcript is essential for next steps
            # raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="STT failed to return transcript.")

        return state


async def api_agent_node(state: MarketBriefState) -> MarketBriefState:
    """Calls the API Agent to fetch market data."""
    async with httpx.AsyncClient() as client:
        # Example tickers - ideally these might come from user query analysis or config
        payload = {
            "tickers": [
                "TSM",
            ],  # Example tickers (TSM for TSMC US ADR, 005930.KS for Samsung Korea)
            "start_date": None,  # Handled by API agent implementation
            "end_date": None,  # Handled by API agent implementation
            "data_type": "close",  # Handled by API agent implementation
        }
        response_data = await call_agent(
            client, f"{AGENT_API_URL}/get_market_data", json_payload=payload
        )

        # Validate and update state
        if "result" in response_data:
            state.market_data = response_data["result"]
            logger.info(
                f"API Agent successful. Fetched data for tickers: {list(state.market_data.keys())}"
            )
            # Check for potential errors reported by the agent itself
            if response_data.get("errors"):
                state.warnings.append(
                    f"API Agent reported errors: {response_data['errors']}"
                )
                logger.warning(state.warnings[-1])
        else:
            error_msg = f"API Agent response missing 'result': {response_data}"
            logger.error(error_msg)
            state.errors.append(error_msg)
            # Optionally, raise HTTPException if market data is essential
            # raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="API Agent failed to return market data.")

        return state


async def scraping_agent_node(state: MarketBriefState) -> MarketBriefState:
    """Calls the Scraping Agent to fetch filings (earnings)."""
    async with httpx.AsyncClient() as client:
        # Fetch filings for example tickers - match API agent tickers
        tickers = ["TSM"]
        filings_data = {}
        for ticker in tickers:
            payload = {
                "ticker": ticker,
                "filing_type": "earnings_surprise",
            }  # Matches refactored Scraping Agent
            try:
                response_data = await call_agent(
                    client, f"{AGENT_SCRAPING_URL}/get_filings", json_payload=payload
                )
                # The refactored scraping agent returns a dict with 'ticker', 'filing_type', 'data' (list), and optionally 'error'
                if "data" in response_data:
                    filings_data[ticker] = (
                        response_data  # Store the whole response for the ticker
                    )
                    logger.info(
                        f"Scraping Agent successful for {ticker}. Records: {len(response_data.get('data', []))}"
                    )
                    if response_data.get("error"):
                        state.warnings.append(
                            f"Scraping Agent reported error for {ticker}: {response_data['error']}"
                        )
                        logger.warning(state.warnings[-1])
                else:
                    error_msg = f"Scraping Agent response for {ticker} missing 'data': {response_data}"
                    logger.error(error_msg)
                    state.errors.append(error_msg)
            except (HTTPException, Exception) as e:
                # Catch exceptions raised by call_agent for this specific ticker
                error_msg = f"Error fetching filings for {ticker}: {e.detail if isinstance(e, HTTPException) else str(e)}"
                logger.error(error_msg)
                state.errors.append(error_msg)
                # Don't necessarily stop if one ticker fails, try others

        state.filings = filings_data
        # Decide if failure to get *any* filings is critical
        if not state.filings:
            state.errors.append("Failed to retrieve filings for any ticker.")
            # Optionally, raise HTTPException here if needed
            # raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to retrieve any filings.")

        return state


async def retriever_agent_node(state: MarketBriefState) -> MarketBriefState:
    """Calls the Retriever Agent to index and retrieve documents."""
    async with httpx.AsyncClient() as client:
        # --- Index Filings ---
        # Extract document content from the refactored scraping agent output
        # Need to format this nicely for indexing. Let's just use the raw 'data' string from the old agent
        # OR format the structured data from the new agent into a string.
        # Let's assume the new scraping agent structure: Dict[ticker, List[record_dict]]
        # We need to decide what 'documents' to index. The raw FMP data isn't ideal text.
        # A better approach would be to use a DocumentLoader or format the relevant parts.
        # For this demo, let's just stringify the relevant parts of the earnings data.
        docs_to_index = []
        if state.filings:
            for ticker, filing_response in state.filings.items():
                # Use the 'data' list from the filing response
                records = filing_response.get("data", [])
                if records:
                    # Simple formatting: include ticker and relevant record data
                    # You might want more sophisticated document creation here
                    doc_content = f"Earnings data for {ticker}:\n" + "\n".join(
                        [
                            f"Date: {r.get('date')}, Actual: {r.get('actual')}, Estimate: {r.get('estimate')}, Surprise%: {r.get('surprisePercentage')}"
                            for r in records
                        ]
                    )
                    docs_to_index.append(doc_content)
                else:
                    logger.warning(
                        f"No data records found in filings for {ticker} to index."
                    )
                    state.warnings.append(
                        f"No data records found in filings for {ticker} to index."
                    )

        if docs_to_index:
            try:
                index_response = await call_agent(
                    client,
                    f"{AGENT_RETRIEVER_URL}/index",
                    json_payload={"docs": docs_to_index},
                )
                if index_response.get("status") == "indexed":
                    state.indexed = True
                    logger.info(
                        f"Retriever Agent indexing successful. Indexed {index_response.get('num_docs')} documents."
                    )
                else:
                    error_msg = f"Retriever Agent indexing reported non-indexed status: {index_response}"
                    logger.error(error_msg)
                    state.errors.append(error_msg)
            except (HTTPException, Exception) as e:
                error_msg = f"Error during Retriever Agent indexing: {e.detail if isinstance(e, HTTPException) else str(e)}"
                logger.error(error_msg)
                state.errors.append(error_msg)
                # Proceed to retrieval even if indexing failed, the agent might use a cached index

        # --- Retrieve Relevant Docs ---
        # Retrieval requires the user query. If STT failed, we can't retrieve.
        if (
            state.user_text and state.indexed
        ):  # Only retrieve if we have a query and indexing was attempted/successful
            try:
                payload = {"query": state.user_text, "top_k": 2}
                retrieve_response = await call_agent(
                    client, f"{AGENT_RETRIEVER_URL}/retrieve", json_payload=payload
                )

                if "results" in retrieve_response:
                    # Extract just the document content strings
                    state.retrieved_docs = [
                        r["doc"] for r in retrieve_response["results"]
                    ]
                    logger.info(
                        f"Retriever Agent retrieval successful. Retrieved {len(state.retrieved_docs)} documents."
                    )
                elif (
                    "error" in retrieve_response
                ):  # Handle agent-specific error like "No documents indexed"
                    error_msg = f"Retriever Agent retrieval returned error: {retrieve_response['error']}"
                    logger.warning(error_msg)
                    state.warnings.append(error_msg)
                    state.retrieved_docs = (
                        []
                    )  # Ensure it's an empty list on retrieval error
                else:
                    error_msg = f"Retriever Agent retrieval response missing 'results' or 'error': {retrieve_response}"
                    logger.error(error_msg)
                    state.errors.append(error_msg)
                    state.retrieved_docs = (
                        []
                    )  # Ensure it's an empty list on unexpected response
            except (HTTPException, Exception) as e:
                error_msg = f"Error during Retriever Agent retrieval: {e.detail if isinstance(e, HTTPException) else str(e)}"
                logger.error(error_msg)
                state.errors.append(error_msg)
                state.retrieved_docs = (
                    []
                )  # Ensure it's an empty list on retrieval error
        else:
            logger.warning(
                "Skipping retrieval: No user text or indexing was skipped/failed."
            )
            state.retrieved_docs = (
                []
            )  # Ensure it's an empty list if retrieval is skipped

        return state


async def analysis_agent_node(state: MarketBriefState) -> MarketBriefState:
    """Calls the Analysis Agent to compute portfolio/earnings analysis."""
    async with httpx.AsyncClient() as client:
        # Analysis agent requires market_data, filings, and portfolio.
        # If market_data or filings are missing due to upstream errors,
        # the analysis agent might fail or produce partial results.
        # We should pass what we have and let the analysis agent handle its validation.
        if not state.market_data or not state.filings:
            warning_msg = "Skipping Analysis Agent: Missing market data or filings from previous steps."
            logger.warning(warning_msg)
            state.warnings.append(warning_msg)
            state.analysis = None  # Ensure analysis is None if skipped
            return state  # Return state without calling agent

        # Example portfolio - ideally from config or user context
        portfolio_example = {
            "TSM": 0.12,
        }  # Use tickers matching API agent
        asia_tech_tickers_example = ["TSM"]

        # Prepare earnings data for analysis agent - need to map ticker -> List[record_dict]
        earnings_for_analysis = {}
        for ticker, filing_response in state.filings.items():
            # The analysis agent expects the list of records directly as the value for the ticker
            earnings_for_analysis[ticker] = filing_response.get("data", [])

        payload = {
            "portfolio": portfolio_example,
            "market_data": state.market_data,
            "earnings_data": earnings_for_analysis,  # Pass the list of records
            "asia_tech_tickers": asia_tech_tickers_example,
        }

        try:
            response_data = await call_agent(
                client, f"{AGENT_ANALYSIS_URL}/analyze", json_payload=payload
            )

            # Analysis agent returns dict with "asia_tech_alloc", "yesterday_alloc", "earnings_surprises"
            if (
                "asia_tech_alloc" in response_data
                and "earnings_surprises" in response_data
            ):
                state.analysis = response_data
                logger.info("Analysis Agent successful.")
            else:
                error_msg = (
                    f"Analysis Agent response missing expected keys: {response_data}"
                )
                logger.error(error_msg)
                state.errors.append(error_msg)
                # Optionally, raise if analysis is critical
                # raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Analysis Agent failed.")

        except (HTTPException, Exception) as e:
            error_msg = f"Error calling Analysis Agent: {e.detail if isinstance(e, HTTPException) else str(e)}"
            logger.error(error_msg)
            state.errors.append(error_msg)
            # Optionally, raise if analysis is critical
            # raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=error_msg)

        return state


async def language_agent_node(state: MarketBriefState) -> MarketBriefState:
    """Calls the Language Agent to generate the brief."""
    async with httpx.AsyncClient() as client:
        # Language agent requires user_text, analysis, and retrieved_docs.
        # It can likely proceed even if docs or analysis are missing/empty,
        # but user_text is critical.
        if not state.user_text:
            error_msg = "Skipping Language Agent: No user text available from STT."
            logger.error(error_msg)
            state.errors.append(error_msg)
            state.brief = "Could not process your request as speech-to-text failed."
            return state  # Return state with failure message

        # Ensure analysis and retrieved_docs are not None, even if empty lists/dicts
        analysis_data = state.analysis if state.analysis is not None else {}
        retrieved_docs = (
            state.retrieved_docs if state.retrieved_docs is not None else []
        )

        # The Language Agent expects the analysis data in a specific structure
        # Use the refined Pydantic model structure from the Language Agent's `BriefRequest`
        payload = {
            "user_query": state.user_text,
            "analysis": analysis_data,  # Ensure this matches Language Agent's AnalysisData model
            "retrieved_docs": retrieved_docs,
        }

        try:
            response_data = await call_agent(
                client, f"{AGENT_LANGUAGE_URL}/generate_brief", json_payload=payload
            )

            if "brief" in response_data:
                state.brief = response_data["brief"]
                logger.info(
                    f"Language Agent successful. Brief generated: {state.brief[:100]}..."
                )
            else:
                error_msg = f"Language Agent response missing 'brief': {response_data}"
                logger.error(error_msg)
                state.errors.append(error_msg)
                state.brief = (
                    state.brief or "Failed to generate brief text."
                )  # Keep previous brief if exists or set fallback

        except (HTTPException, Exception) as e:
            error_msg = f"Error calling Language Agent: {e.detail if isinstance(e, HTTPException) else str(e)}"
            logger.error(error_msg)
            state.errors.append(error_msg)
            state.brief = (
                state.brief or "Failed to generate brief text due to an error."
            )

        return state


async def tts_node(state: MarketBriefState) -> MarketBriefState:
    """Calls the Voice Agent for Text-to-Speech."""
    async with httpx.AsyncClient() as client:
        # TTS requires the brief. If language agent failed, we can't generate speech.
        brief_text = state.brief
        if (
            not brief_text or len(state.errors) > 0
        ):  # Also check if there are errors that might make the brief unreliable
            warning_msg = (
                "Skipping TTS: No brief text available or previous errors exist."
            )
            logger.warning(warning_msg)
            state.warnings.append(warning_msg)
            # Optionally, generate a generic error brief TTS
            if len(state.errors) > 0:
                error_brief = (
                    "An error occurred during the market brief generation process."
                )
                try:
                    payload = {"text": error_brief, "lang": "en"}
                    response_data = await call_agent(
                        client, f"{AGENT_VOICE_URL}/tts", json_payload=payload
                    )
                    if "audio" in response_data:
                        state.audio_output = bytes.fromhex(response_data["audio"])
                        logger.info("Generated TTS for error message.")
                    else:
                        logger.error(
                            "TTS agent failed to return audio for error message."
                        )
                except (HTTPException, Exception) as e:
                    logger.error(f"Failed to generate error brief TTS: {e}")
                # Regardless of error brief success, mark main audio as None
                state.audio_output = None
            else:
                state.audio_output = None  # No brief and no specific error brief
            return state

        payload = {"text": brief_text, "lang": "en"}
        try:
            response_data = await call_agent(
                client, f"{AGENT_VOICE_URL}/tts", json_payload=payload
            )

            # The TTS agent returns {"audio": hex_string}
            if "audio" in response_data and isinstance(response_data["audio"], str):
                try:
                    state.audio_output = bytes.fromhex(response_data["audio"])
                    logger.info("TTS successful. Audio bytes received.")
                except ValueError:
                    error_msg = "TTS Agent returned invalid hex string."
                    logger.error(error_msg)
                    state.errors.append(error_msg)
                    state.audio_output = None  # Ensure audio is None on hex error
            else:
                error_msg = (
                    f"TTS Agent response missing or invalid 'audio': {response_data}"
                )
                logger.error(error_msg)
                state.errors.append(error_msg)
                state.audio_output = (
                    None  # Ensure audio is None on response structure error
                )

        except (HTTPException, Exception) as e:
            error_msg = f"Error calling TTS Agent: {e.detail if isinstance(e, HTTPException) else str(e)}"
            logger.error(error_msg)
            state.errors.append(error_msg)
            state.audio_output = None  # Ensure audio is None on call error

        return state


# ---- LangGraph Workflow ----


def build_market_brief_graph():
    """Builds the LangGraph StateGraph for the market brief workflow."""
    builder = StateGraph(MarketBriefState)

    # Add nodes
    builder.add_node("stt", stt_node)
    builder.add_node("api_agent", api_agent_node)
    builder.add_node("scraping_agent", scraping_agent_node)
    builder.add_node("retriever_agent", retriever_agent_node)
    builder.add_node("analysis_agent", analysis_agent_node)
    builder.add_node("language_agent", language_agent_node)
    builder.add_node("tts", tts_node)

    # Define the sequence
    builder.set_entry_point("stt")
    builder.add_edge("stt", "api_agent")
    builder.add_edge("api_agent", "scraping_agent")
    # Note: Retriever depends on scraping, analysis depends on api & scraping.
    # Langgraph allows this structure implicitly.
    builder.add_edge("scraping_agent", "retriever_agent")
    builder.add_edge("retriever_agent", "analysis_agent")
    builder.add_edge("analysis_agent", "language_agent")
    builder.add_edge("language_agent", "tts")
    builder.add_edge("tts", END)  # Use END constant

    # No explicit error handling edges defined here, nodes raise exceptions or add errors to state.
    # LangGraph default behavior is to propagate exceptions, stopping the graph run.
    # The FastAPI endpoint will catch the HTTPException raised by call_agent or nodes.

    return builder.compile()


# Build the graph instance
graph = build_market_brief_graph()

# ---- FastAPI Endpoint ----


@app.post("/market_brief")
async def market_brief(audio: UploadFile = File(...)):
    """
    Processes an audio input to generate a spoken market brief.
    """
    logger.info("Received request to /market_brief")
    if not audio.content_type or not audio.content_type.startswith("audio/"):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Invalid file type. Please upload an audio file.",
        )

    # Initialize state with audio input
    initial_state = MarketBriefState()
    try:
        initial_state.audio_input = await audio.read()
        logger.info(f"Read {len(initial_state.audio_input)} bytes of audio input.")
    except Exception as e:
        error_msg = f"Failed to read audio file: {e}"
        logger.error(error_msg)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=error_msg
        )

    # Invoke the LangGraph workflow
    final_state = (
        initial_state  # Initialize final_state in case graph invocation fails early
    )
    try:
        # The graph will process the state through the nodes
        final_state = await graph.ainvoke(initial_state)
        logger.info("LangGraph execution finished.")

    except HTTPException as e:
        # Catch HTTPExceptions specifically raised by call_agent or nodes
        logger.error(f"Graph execution stopped due to HTTPException: {e.detail}")
        # You might want to return a generic error response here
        raise e  # Re-raise the caught HTTPException

    except Exception as e:
        # Catch any other unexpected errors during graph execution
        error_msg = f"An unexpected error occurred during graph execution: {e}"
        logger.error(error_msg)
        # Return a generic server error
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=error_msg
        )

    # Prepare the final response based on the final state
    response_payload = {
        "transcript": final_state.user_text,
        "brief": final_state.brief,
        "audio": (
            final_state.audio_output.hex() if final_state.audio_output else None
        ),  # Return hex string or None
        "errors": final_state.errors,
        "warnings": final_state.warnings,
        "status": "success" if not final_state.errors else "failed",
        "message": (
            "Market brief generated successfully."
            if not final_state.errors
            else "Market brief generation failed with errors."
        ),
    }

    # Return 500 if there were errors accumulated in the state
    if final_state.errors:
        logger.error("Request finished with errors.")
        return response_payload  # Return payload even with errors for debugging

    logger.info("Request finished successfully.")
    return response_payload
