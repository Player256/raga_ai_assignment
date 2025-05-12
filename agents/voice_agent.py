# agents/voice_agent/main.py

from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from gtts import gTTS
import tempfile
import os

# For STT, using faster-whisper
from faster_whisper import WhisperModel

app = FastAPI(title="Voice Agent")

# Initialize Whisper model (small for demo, use large for prod)
whisper_model = WhisperModel("small")


class TTSRequest(BaseModel):
    text: str
    lang: str = "en"


@app.post("/stt")
async def stt(audio: UploadFile = File(...)):
    # Save uploaded audio file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(await audio.read())
        tmp_path = tmp.name
    # Transcribe
    segments, info = whisper_model.transcribe(tmp_path)
    transcript = " ".join([seg.text for seg in segments])
    os.remove(tmp_path)
    return {"transcript": transcript}


@app.post("/tts")
def tts(request: TTSRequest):
    tts = gTTS(text=request.text, lang=request.lang)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
        tts.save(tmp.name)
        tmp_path = tmp.name
    # Return audio file path or stream (for API, you may want to stream or send as bytes)
    with open(tmp_path, "rb") as f:
        audio_bytes = f.read()
    os.remove(tmp_path)
    return {"audio": audio_bytes.hex()}  # Or use StreamingResponse for real audio
