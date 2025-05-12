# agents/language_agent/main.py

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, HumanMessage, SystemMessage

app = FastAPI(title="Language Agent")

# Define prompt template for the market brief
SYSTEM_PROMPT = (
    "You are a professional financial assistant. "
    "Given the portfolio analysis and relevant filings, synthesize a concise, spoken morning market brief for a portfolio manager."
)


class BriefRequest(BaseModel):
    user_query: str
    analysis: Dict[str, Any]
    retrieved_docs: List[str]


@app.post("/generate_brief")
def generate_brief(request: BriefRequest):
    # Compose prompt
    analysis_summary = (
        f"Asia tech allocation: {request.analysis.get('asia_tech_alloc', 'N/A')*100:.1f}% of AUM, "
        f"up from {request.analysis.get('yesterday_alloc', 'N/A')*100:.1f}% yesterday.\n"
    )
    earnings = request.analysis.get("earnings_surprises", [])
    earnings_summary = " ".join(
        [
            (
                f"{e['ticker']} beat estimates by {e['surprise_pct']}%"
                if e["surprise_pct"] > 0
                else f"{e['ticker']} missed by {abs(e['surprise_pct'])}%"
            )
            for e in earnings
        ]
    )
    docs_context = "\n".join(request.retrieved_docs[:2])  # Use top 2 docs for brevity

    prompt = (
        f"{SYSTEM_PROMPT}\n"
        f"User Query: {request.user_query}\n"
        f"Portfolio Analysis: {analysis_summary} {earnings_summary}\n"
        f"Relevant Filings: {docs_context}\n"
        f"Respond with a spoken-style, concise market brief."
    )

    # Call LLM (e.g., OpenAI GPT-4o-mini or local model)
    llm = ChatOpenAI(api_key="YOUR_API_KEY", model="gpt-4o-mini")
    messages = [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=prompt)]
    response = llm.invoke(messages)
    return {"brief": response.content}
