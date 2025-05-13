# agents/voice_agent/main.py

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import Response  # Import Response for returning audio bytes
from pydantic import BaseModel
from gtts import gTTS
import tempfile
import os
import logging
from faster_whisper import WhisperModel  # For STT
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Voice Agent")

# Get Whisper model size from environment
WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "small")  # Default to 'small'
# Initialize Whisper model once on startup
try:
    # Using cpu is generally safer for deployment unless you have a specific GPU setup
    whisper_model = WhisperModel(WHISPER_MODEL_SIZE, device="cpu")
    logger.info(f"Whisper model '{WHISPER_MODEL_SIZE}' loaded successfully on CPU.")
except Exception as e:
    logger.error(f"Error loading Whisper model '{WHISPER_MODEL_SIZE}': {e}")
    # Depending on criticality, you might raise here or handle gracefully
    whisper_model = None  # Set to None if loading failed


class TTSRequest(BaseModel):
    text: str
    lang: str = "en"


@app.post("/stt")
async def stt(audio: UploadFile = File(...)):
    """
    Performs Speech-to-Text on an uploaded audio file.
    """
    if whisper_model is None:
        raise HTTPException(status_code=503, detail="STT model not loaded.")

    logger.info(f"Received audio file for STT: {audio.filename}")

    # Save uploaded audio file to a temporary location
    # Use .with_suffix('.wav') explicitly if needed, although whisper handles formats
    suffix = os.path.splitext(audio.filename)[1] if audio.filename else ".wav"
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            audio_content = await audio.read()
            tmp.write(audio_content)
            tmp_path = tmp.name
        logger.info(f"Audio saved to temporary file: {tmp_path}")

        # Transcribe using faster-whisper
        # max_int16 ensures compatibility, adjust as needed
        segments, info = whisper_model.transcribe(
            tmp_path, language=info.language if "info" in locals() else None
        )
        transcript = " ".join([seg.text for seg in segments]).strip()
        logger.info(f"Transcription complete. Transcript: '{transcript}'")

        return {"transcript": transcript}

    except Exception as e:
        logger.error(f"Error during STT processing: {e}")
        raise HTTPException(status_code=500, detail=f"STT processing failed: {e}")
    finally:
        # Clean up temporary file
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
            logger.info(f"Temporary file removed: {tmp_path}")


@app.post("/tts")
def tts(request: TTSRequest):
    """
    Performs Text-to-Speech using gTTS.
    Returns the audio data as a hex string (to match original orchestrator expectation).
    NOTE: Returning raw bytes with media_type='audio/mpeg' is more standard for APIs.
          This implementation keeps the hex encoding to avoid changing the orchestrator.
    """
    logger.info(
        f"Generating TTS for text (lang={request.lang}): '{request.text[:50]}...'"
    )
    tmp_path = None
    try:
        # Create gTTS object
        tts_obj = gTTS(text=request.text, lang=request.lang, slow=False)

        # Save to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
            tts_obj.save(tmp.name)
            tmp_path = tmp.name
        logger.info(f"TTS audio saved to temporary file: {tmp_path}")

        # Read the audio file bytes
        with open(tmp_path, "rb") as f:
            audio_bytes = f.read()
        logger.info(f"Read {len(audio_bytes)} bytes from temporary file.")

        # Return as hex string as per original orchestrator expectation
        audio_hex = audio_bytes.hex()
        logger.info("Audio bytes converted to hex.")

        return {"audio": audio_hex}

        # --- Alternative (More standard API practice - requires orchestrator change) ---
        # return Response(content=audio_bytes, media_type="audio/mpeg")
        # ---------------------------------------------------------------------------

    except Exception as e:
        logger.error(f"Error during TTS processing: {e}")
        raise HTTPException(status_code=500, detail=f"TTS processing failed: {e}")
    finally:
        # Clean up temporary file
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
            logger.info(f"Temporary file removed: {tmp_path}")
