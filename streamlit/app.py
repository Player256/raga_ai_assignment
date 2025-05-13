# streamlit_app/app.py

import streamlit as st
import httpx
import os
import io
import base64
from dotenv import load_dotenv
import logging
import asyncio  # Import asyncio to explicitly run async tasks if needed (though Streamlit handles top-level async)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Get orchestrator URL from environment
ORCHESTRATOR_URL = os.getenv("ORCHESTRATOR_URL")
if not ORCHESTRATOR_URL:
    st.error(
        "ORCHESTRATOR_URL environment variable not set! Please add it to your .env file."
    )
    st.stop()  # Stop the app if the URL is not configured

# Initialize session state variables if they don't exist
if "processing_state" not in st.session_state:
    st.session_state.processing_state = (
        "initial"  # 'initial', 'processing', 'completed', 'error'
    )
if "orchestrator_response" not in st.session_state:
    st.session_state.orchestrator_response = None
if "audio_bytes_input" not in st.session_state:
    st.session_state.audio_bytes_input = None
if "audio_filename" not in st.session_state:
    st.session_state.audio_filename = None
if "audio_filetype" not in st.session_state:
    st.session_state.audio_filetype = None


# --- Async Helper Function to Call Orchestrator ---


async def call_orchestrator(audio_bytes: bytes, filename: str, content_type: str):
    """Calls the orchestrator's /market_brief endpoint with audio data."""
    url = f"{ORCHESTRATOR_URL}/market_brief"
    files = {"audio": (filename, audio_bytes, content_type)}

    logger.info(f"Calling orchestrator at {url} with audio file: {filename}")

    try:
        # Use httpx.AsyncClient for making async requests
        async with httpx.AsyncClient() as client:
            response = await client.post(
                url, files=files, timeout=180.0
            )  # Increased timeout for potential long workflows
            response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
            logger.info(f"Orchestrator returned status {response.status_code}.")
            return response.json()

    except httpx.RequestError as e:
        error_msg = f"HTTP Request failed: {e}"
        logger.error(error_msg)
        # Return a structured error response compatible with the expected output
        return {
            "status": "error",
            "message": "Error communicating with the orchestrator.",
            "errors": [error_msg],
            "warnings": [],
            "transcript": None,
            "brief": None,
            "audio": None,
        }
    except Exception as e:
        error_msg = f"An unexpected error occurred: {e}"
        logger.error(error_msg)
        # Return a structured error response
        return {
            "status": "error",
            "message": "An unexpected error occurred.",
            "errors": [error_msg],
            "warnings": [],
            "transcript": None,
            "brief": None,
            "audio": None,
        }


# --- Main Streamlit App Logic ---

st.title("📈 AI Financial Assistant - Morning Market Brief")
st.markdown(
    "Upload an audio file with your query (e.g., 'What's our risk exposure in Asia tech stocks today, and highlight any earnings surprises?')"
)

# File uploader - clears previous results when a new file is uploaded
uploaded_file = st.file_uploader(
    "Upload Audio File",
    type=["wav", "mp3", "m4a", "ogg"],
    help="Supported formats: WAV, MP3, M4A, OGG",
    on_change=lambda: st.session_state.update(
        processing_state="initial", orchestrator_response=None
    ),  # Reset state on new upload
)

# Store uploaded file info in session state
if uploaded_file is not None:
    st.session_state.audio_bytes_input = uploaded_file.getvalue()
    st.session_state.audio_filename = uploaded_file.name
    st.session_state.audio_filetype = uploaded_file.type
    # Display file details and a preview if needed
    # st.write("File uploaded:", uploaded_file.name)
    # st.audio(st.session_state.audio_bytes_input, format=st.session_state.audio_filetype) # Optional preview

# Button to trigger the workflow
# Disable button if no file is uploaded or if currently processing
button_disabled = (
    st.session_state.audio_bytes_input is None
    or st.session_state.processing_state == "processing"
)

if st.button("Generate Market Brief", disabled=button_disabled):
    # Button click initiates the processing state
    st.session_state.processing_state = "processing"
    st.session_state.orchestrator_response = None  # Clear previous response
    st.rerun()  # Rerun the script to enter the processing block


# --- Processing Logic based on State ---

if st.session_state.processing_state == "processing":
    st.info("Processing your request...")
    # Perform the async call and await it
    # Streamlit handles awaiting top-level async calls in the script.
    # We call a sync wrapper around our async function here to make it work directly in this part of the script.
    # A more robust way in complex apps is to use separate async tasks or components.
    # For a simple demo, directly awaiting here often works in recent Streamlit versions.
    # Let's simplify and rely on Streamlit's async handling here.
    try:
        response = asyncio.run(
            call_orchestrator(
                st.session_state.audio_bytes_input,
                st.session_state.audio_filename,
                st.session_state.audio_filetype,
            )
        )
        st.session_state.orchestrator_response = response
        st.session_state.processing_state = (
            "completed" if response and response.get("status") != "error" else "error"
        )

    except Exception as e:
        logger.error(f"Error during orchestrator call in Streamlit: {e}")
        st.session_state.orchestrator_response = {
            "status": "error",
            "message": f"Streamlit failed to call orchestrator: {e}",
            "errors": [str(e)],
            "warnings": [],
            "transcript": None,
            "brief": None,
            "audio": None,
        }
        st.session_state.processing_state = "error"
    # Rerun the script to display results based on the new state
    st.rerun()


# --- Display Results based on State ---

if st.session_state.processing_state in ["completed", "error"]:
    response = st.session_state.orchestrator_response
    st.subheader("Results")

    if response is None:
        st.error("No response received from the orchestrator.")
    elif response.get("status") == "error":
        st.error(f"Workflow Failed: {response.get('message', 'Unknown error')}")
        if response.get("errors"):
            st.warning("Details:")
            for i, err in enumerate(response["errors"]):
                st.write(f"- Error {i+1}: {err}")
        if response.get("warnings"):
            st.warning("Warnings:")
            for i, warn in enumerate(response["warnings"]):
                st.write(f"- Warning {i+1}: {warn}")

    else:  # status is likely 'success' or similar
        # Display Transcript
        st.subheader("Transcript")
        transcript = response.get("transcript", "N/A")
        st.write(transcript)

        # Display Brief Text
        st.subheader("Generated Brief")
        brief_text = response.get("brief", "N/A")
        st.write(brief_text)

        # Play Audio Brief
        audio_hex = response.get("audio")
        if audio_hex:
            st.subheader("Audio Brief")
            try:
                # Ensure the hex string is valid
                if not isinstance(audio_hex, str) or not all(
                    c in "0123456789abcdefABCDEF" for c in audio_hex
                ):
                    raise ValueError("Invalid hex string received.")

                audio_bytes_output = bytes.fromhex(audio_hex)
                audio_io = io.BytesIO(audio_bytes_output)
                # Assuming gTTS outputs MP3
                st.audio(audio_io, format="audio/mpeg")
                logger.info("Displayed audio player.")
            except ValueError as ve:
                st.error(f"Failed to decode audio data: {ve}")
                logger.error(f"Failed to decode audio data from hex string: {ve}")
            except Exception as e:
                st.error(f"Failed to play audio: {e}")
                logger.error(f"Error playing audio: {e}")
        else:
            st.warning("No audio brief generated (or audio data missing in response).")

        # Display Errors and Warnings if any were accumulated in the state
        errors = response.get("errors", [])
        warnings = response.get("warnings", [])

        if errors:
            st.subheader("Errors Reported During Processing")
            for i, err in enumerate(errors):
                st.error(f"Error {i+1}: {err}")

        if warnings:
            st.subheader("Warnings Reported During Processing")
            for i, warn in enumerate(warnings):
                st.warning(f"Warning {i+1}: {warn}")


# Optional: Add instructions on how to run the app
st.markdown("---")
st.markdown("#### To Run This Application Locally:")
st.markdown("1. Ensure your `.env` file is in the project root and configured.")
st.markdown(
    "2. Start all agent services (api, scraping, retriever, analysis, language, voice) in separate terminals using the correct `uvicorn agents.<module_name>:app --reload --port <port>` commands from the project root."
)
st.markdown(
    "3. Start the orchestrator service using `uvicorn orchestrator.main:app --reload --port 8000` (or your configured port)."
)
st.markdown("4. Run the Streamlit app from the project root:")
st.code("streamlit run streamlit_app/app.py")
st.markdown("5. Open the provided URL in your browser.")

# Optional: Add a simple session state clear button for easy re-runs during development
if st.button("Clear Session State & Results"):
    # Reset all session state variables
    st.session_state.processing_state = "initial"
    st.session_state.orchestrator_response = None
    st.session_state.audio_bytes_input = None
    st.session_state.audio_filename = None
    st.session_state.audio_filetype = None
    st.rerun()  # Rerun the app to clear the display
