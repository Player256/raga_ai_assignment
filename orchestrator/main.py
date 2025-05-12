# orchestrator/main.py

from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import requests
from langgraph.graph import StateGraph

app = FastAPI(title="Orchestrator")


# ---- State Definition ----
class MarketBriefState(BaseModel):
    audio_input: bytes = None
    user_text: str = None
    market_data: dict = None
    filings: dict = None
    indexed: bool = False
    retrieved_docs: list = None
    analysis: dict = None
    brief: str = None
    audio_output: bytes = None


# ---- Agent Node Functions ----


def stt_node(state: MarketBriefState) -> MarketBriefState:
    # Send audio to Voice Agent (STT)
    files = {"audio": ("input.wav", state.audio_input, "audio/wav")}
    resp = requests.post("http://localhost:8006/stt", files=files)
    state.user_text = resp.json()["transcript"]
    return state


def api_agent_node(state: MarketBriefState) -> MarketBriefState:
    # Fetch market data
    payload = {
        "tickers": ["TSMC", "SAMSUNG"],  # Example tickers
        "start_date": None,
        "end_date": None,
        "data_type": "close",
    }
    resp = requests.post("http://localhost:8001/get_market_data", json=payload)
    state.market_data = resp.json()["result"]
    return state


def scraping_agent_node(state: MarketBriefState) -> MarketBriefState:
    # Fetch earnings filings
    filings = {}
    for ticker in ["TSMC", "SAMSUNG"]:
        payload = {"ticker": ticker, "filing_type": "earnings"}
        resp = requests.post("http://localhost:8002/get_filings", json=payload)
        filings[ticker] = resp.json()
    state.filings = filings
    return state


def retriever_agent_node(state: MarketBriefState) -> MarketBriefState:
    # Index filings
    docs = [f["data"] for f in state.filings.values()]
    requests.post("http://localhost:8003/index", json={"docs": docs})
    state.indexed = True
    # Retrieve relevant docs
    resp = requests.post(
        "http://localhost:8003/retrieve", json={"query": state.user_text, "top_k": 2}
    )
    state.retrieved_docs = [r["doc"] for r in resp.json()["results"]]
    return state


def analysis_agent_node(state: MarketBriefState) -> MarketBriefState:
    # Compute risk/earnings surprises
    payload = {
        "portfolio": {"TSMC": 0.12, "SAMSUNG": 0.10},
        "market_data": state.market_data,
        "earnings_data": {k: v for k, v in state.filings.items()},
        "asia_tech_tickers": ["TSMC", "SAMSUNG"],
    }
    resp = requests.post("http://localhost:8004/analyze", json=payload)
    state.analysis = resp.json()
    return state


def language_agent_node(state: MarketBriefState) -> MarketBriefState:
    # Generate market brief
    payload = {
        "user_query": state.user_text,
        "analysis": state.analysis,
        "retrieved_docs": state.retrieved_docs,
    }
    resp = requests.post("http://localhost:8005/generate_brief", json=payload)
    state.brief = resp.json()["brief"]
    return state


def tts_node(state: MarketBriefState) -> MarketBriefState:
    # Convert brief to speech
    payload = {"text": state.brief, "lang": "en"}
    resp = requests.post("http://localhost:8006/tts", json=payload)
    state.audio_output = bytes.fromhex(resp.json()["audio"])
    return state


# ---- LangGraph Workflow ----


def build_market_brief_graph():
    builder = StateGraph(MarketBriefState)
    builder.add_edge("start", "stt")
    builder.add_node("stt", stt_node)
    builder.add_edge("stt", "api_agent")
    builder.add_node("api_agent", api_agent_node)
    builder.add_edge("api_agent", "scraping_agent")
    builder.add_node("scraping_agent", scraping_agent_node)
    builder.add_edge("scraping_agent", "retriever_agent")
    builder.add_node("retriever_agent", retriever_agent_node)
    builder.add_edge("retriever_agent", "analysis_agent")
    builder.add_node("analysis_agent", analysis_agent_node)
    builder.add_edge("analysis_agent", "language_agent")
    builder.add_node("language_agent", language_agent_node)
    builder.add_edge("language_agent", "tts")
    builder.add_node("tts", tts_node)
    builder.add_edge("tts", "end")
    return builder.compile()


graph = build_market_brief_graph()

# ---- FastAPI Endpoint ----


@app.post("/market_brief")
async def market_brief(audio: UploadFile = File(...)):
    audio_bytes = await audio.read()
    state = MarketBriefState(audio_input=audio_bytes)
    final_state = graph.invoke(state)
    return {
        "transcript": final_state.user_text,
        "brief": final_state.brief,
        "audio": final_state.audio_output.hex(),  # Return as hex string
    }
