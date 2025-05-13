# agents/language_agent/main.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, validator
from typing import List, Dict, Any, Union
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import (
    StrOutputParser,
)  # Good practice to use parsers
import os
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Language Agent")

# Get OpenAI API Key and model name from environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "gpt-4o-mini")  # Default model
# Ensure API key is set, but raise error only when needed (on endpoint call)
if not OPENAI_API_KEY:
    logger.warning("OPENAI_API_KEY not found. Language agent will fail if called.")


# Refined Pydantic model for analysis data structure
class EarningsSummary(BaseModel):
    ticker: str
    surprise_pct: float


# Analysis model adjusted to match potentially refined analysis agent output
class AnalysisData(BaseModel):
    asia_tech_alloc: float
    yesterday_alloc: float  # Still dummy from analysis agent, but model expects float
    earnings_surprises: List[EarningsSummary]
    # Add other potential analysis fields here if needed later


class BriefRequest(BaseModel):
    user_query: str
    analysis: AnalysisData  # Use the refined model
    retrieved_docs: List[str]  # List of document *content* strings

    @validator("analysis", "retrieved_docs")
    def check_not_none(cls, v):
        if v is None:
            raise ValueError("analysis and retrieved_docs cannot be None")
        return v


# Define prompt template using LangChain's ChatPromptTemplate
# This is more structured than f-strings for complex prompts
PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a professional financial assistant providing concise, spoken market briefs.",
        ),
        (
            "human",
            (
                "Synthesize a morning market brief based on the following information.\n"
                "User Query: {user_query}\n"
                "Portfolio Analysis:\n{analysis_summary}\n\n"
                "Relevant Filings Context:\n{docs_context}\n\n"
                "Provide a concise, spoken-style brief, suitable for a portfolio manager at 8 AM."
            ),
        ),
    ]
)


@app.post("/generate_brief")
def generate_brief(request: BriefRequest):
    """
    Generates a spoken-style market brief based on analysis and retrieved documents.
    """
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="OpenAI API Key not configured.")

    logger.info(f"Generating brief for query: '{request.user_query}'")

    # Prepare analysis summary string
    analysis_data = request.analysis
    analysis_summary = (
        f"Asia tech allocation: {analysis_data.asia_tech_alloc*100:.1f}% of AUM, "
        f"up from {analysis_data.yesterday_alloc*100:.1f}% yesterday.\n"
    )
    if analysis_data.earnings_surprises:
        earnings_summary_parts = []
        for e in analysis_data.earnings_surprises:
            part = (
                f"{e.ticker} beat estimates by {e.surprise_pct:.1f}%"
                if e.surprise_pct >= 0
                else f"{e.ticker} missed by {abs(e.surprise_pct):.1f}%"
            )
            earnings_summary_parts.append(part)
        analysis_summary += (
            "Earnings updates: " + ", ".join(earnings_summary_parts) + "."
        )
    else:
        analysis_summary += "No significant earnings surprises reported recently."

    # Use top 2 docs for context, similar to original logic
    # In a real RAG, you might pass all relevant chunks subject to token limits
    docs_context = (
        "\n".join(request.retrieved_docs[:2])
        if request.retrieved_docs
        else "No relevant filings found."
    )

    # Initialize LLM
    try:
        llm = ChatOpenAI(api_key=OPENAI_API_KEY, model=LLM_MODEL_NAME, temperature=0.7)
        # Chain the prompt and LLM with a simple output parser
        chain = PROMPT | llm | StrOutputParser()
        logger.info("LLM chain created.")

        # Invoke the chain
        response = chain.invoke(
            {
                "user_query": request.user_query,
                "analysis_summary": analysis_summary,
                "docs_context": docs_context,
            }
        )
        logger.info("LLM invoked successfully.")

        return {"brief": response}

    except Exception as e:
        logger.error(f"Error during brief generation: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate brief: {e}")
