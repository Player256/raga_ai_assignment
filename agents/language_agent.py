from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, validator, Field
from typing import List, Dict, Any, Union
import google.generativeai as genai
import os
from dotenv import load_dotenv
import logging
import time

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Language Agent (Gemini Pro - Generalized)")

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-1.5-flash-latest")

if not GOOGLE_API_KEY:
    logger.warning("GOOGLE_API_KEY not found.")
else:
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
        logger.info(f"Google Generative AI configured for model {GEMINI_MODEL_NAME}.")
    except Exception as e:
        logger.error(f"Failed to configure Google Generative AI: {e}")


class EarningsSummaryLLM(BaseModel):
    ticker: str
    surprise_pct: float


class AnalysisDataLLM(BaseModel):
    target_label: str = "the portfolio"
    current_allocation: float = 0.0
    yesterday_allocation: float = 0.0
    allocation_change_percentage_points: float = 0.0

    earnings_surprises: List[EarningsSummaryLLM] = Field(
        default_factory=list, alias="earnings_surprises_for_target"
    )


class BriefRequest(BaseModel):
    user_query: str
    analysis: AnalysisDataLLM
    retrieved_docs: List[str] = Field(default_factory=list)


def construct_gemini_prompt(
    user_query: str, analysis_data: AnalysisDataLLM, docs_context: str
) -> str:

    alloc_change_str = ""
    if analysis_data.allocation_change_percentage_points > 0.01:
        alloc_change_str = f"up by {analysis_data.allocation_change_percentage_points:.1f} percentage points from yesterday (approx. {analysis_data.yesterday_allocation*100:.0f}%)."
    elif analysis_data.allocation_change_percentage_points < -0.01:
        alloc_change_str = f"down by {abs(analysis_data.allocation_change_percentage_points):.1f} percentage points from yesterday (approx. {analysis_data.yesterday_allocation*100:.0f}%)."
    else:
        alloc_change_str = f"remaining stable around {analysis_data.yesterday_allocation*100:.0f}% yesterday."

    analysis_summary_str = f"For {analysis_data.target_label}, the current allocation is {analysis_data.current_allocation*100:.0f}% of AUM, {alloc_change_str}\n"

    if analysis_data.earnings_surprises:
        earnings_parts = []
        for e in analysis_data.earnings_surprises:
            direction = (
                "beat estimates by" if e.surprise_pct >= 0 else "missed estimates by"
            )
            earnings_parts.append(f"{e.ticker} {direction} {abs(e.surprise_pct):.1f}%")
        if earnings_parts:
            analysis_summary_str += (
                "Key earnings updates: " + ", ".join(earnings_parts) + "."
            )
        else:
            analysis_summary_str += (
                "No specific earnings surprises to highlight for this segment."
            )
    else:
        analysis_summary_str += (
            "No notable earnings surprises reported for this segment."
        )

    prompt = (
        f"You are a professional financial assistant. Based on the user's query and the provided data, "
        f"deliver a concise, spoken-style morning market brief for a portfolio manager. "
        f"The brief should start with 'Good morning.'\n\n"
        f"User Query: {user_query}\n\n"
        f"Key Portfolio and Market Analysis:\n{analysis_summary_str}\n\n"
        f"Relevant Filings Context (if any):\n{docs_context}\n\n"
        f"If the user's query mentions a specific region or sector not covered by the 'Key Portfolio and Market Analysis', "
        f"you can state that specific data for that exact query aspect was not available in the analysis provided. "
        f"Mention any specific company earnings surprises from the analysis clearly (e.g., 'TSMC beat estimates by X%, Samsung missed by Y%')."
        f"If there's information about broad regional sentiment or rising yields in the 'docs_context', incorporate it naturally. Otherwise, focus on the provided analysis."
    )
    return prompt


generation_config = genai.types.GenerationConfig(
    temperature=0.6, max_output_tokens=1024
)
safety_settings = [
    {"category": c, "threshold": "BLOCK_MEDIUM_AND_ABOVE"}
    for c in [
        "HARM_CATEGORY_HARASSMENT",
        "HARM_CATEGORY_HATE_SPEECH",
        "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "HARM_CATEGORY_DANGEROUS_CONTENT",
    ]
]


@app.post("/generate_brief")
async def generate_brief(request: BriefRequest):
    if not GOOGLE_API_KEY:
        raise HTTPException(status_code=500, detail="Google API Key not configured.")
    logger.info(
        f"Generating brief for query: '{request.user_query}' using Gemini model {GEMINI_MODEL_NAME}"
    )

    docs_context = (
        "\n".join(request.retrieved_docs[:2])
        if request.retrieved_docs
        else "No relevant context from documents found."
    )

    full_prompt = construct_gemini_prompt(
        user_query=request.user_query,
        analysis_data=request.analysis,
        docs_context=docs_context,
    )
    logger.debug(f"Full prompt for Gemini:\n{full_prompt}")

    try:
        model = genai.GenerativeModel(
            model_name=GEMINI_MODEL_NAME,
            generation_config=generation_config,
            safety_settings=safety_settings,
        )
        max_retries = 1
        retry_delay_seconds = 10
        for attempt in range(max_retries + 1):
            try:
                response = await model.generate_content_async(full_prompt)

                if not response.parts:
                    if (
                        response.prompt_feedback
                        and response.prompt_feedback.block_reason
                    ):
                        block_reason_message = (
                            response.prompt_feedback.block_reason_message
                            or "Unknown safety block"
                        )
                        logger.error(
                            f"Gemini content generation blocked. Reason: {block_reason_message}"
                        )
                        raise HTTPException(
                            status_code=400,
                            detail=f"Content generation blocked: {block_reason_message}",
                        )
                    else:
                        logger.error("Gemini response has no parts (empty content).")

                        if attempt == max_retries:
                            raise HTTPException(
                                status_code=500,
                                detail="Gemini returned empty content after retries.",
                            )
                        else:
                            logger.warning(
                                f"Gemini returned empty content, attempt {attempt+1}/{max_retries+1}. Retrying..."
                            )
                            await asyncio.sleep(retry_delay_seconds)
                            continue

                brief_text = response.text
                logger.info("Gemini content generated successfully.")
                return {"brief": brief_text}

            except (
                genai.types.generation_types.BlockedPromptException,
                genai.types.generation_types.StopCandidateException,
            ) as sce_bpe:
                logger.error(
                    f"Gemini generation issue on attempt {attempt+1}: {sce_bpe}"
                )
                raise HTTPException(
                    status_code=400, detail=f"Gemini generation issue: {sce_bpe}"
                )
            except Exception as e:
                logger.error(
                    f"Error during Gemini generation on attempt {attempt+1}: {type(e).__name__} - {e}"
                )
                if (
                    "rate limit" in str(e).lower()
                    or "quota" in str(e).lower()
                    or "429" in str(e)
                    or "resource_exhausted" in str(e).lower()
                ):
                    if attempt < max_retries:
                        wait_time = retry_delay_seconds * (2**attempt)
                        logger.info(f"Rate limit likely. Retrying in {wait_time}s...")
                        await asyncio.sleep(wait_time)
                        continue
                    else:
                        logger.error("Max retries reached for rate limit.")
                        raise HTTPException(
                            status_code=429,
                            detail=f"Gemini API rate limit/quota exceeded: {e}",
                        )
                elif attempt < max_retries:
                    await asyncio.sleep(retry_delay_seconds)
                    continue
                else:
                    raise HTTPException(
                        status_code=500,
                        detail=f"Failed to generate brief with Gemini: {e}",
                    )

        raise HTTPException(
            status_code=500, detail="Brief generation failed after all attempts."
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Critical error in /generate_brief: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Critical failure in brief generation: {e}"
        )
