# streamlit_app/app.py

import streamlit as st
import httpx
import os
import io
import base64  # Not strictly needed if using bytes.fromhex, but good to have
from dotenv import load_dotenv
import logging

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
    st.error("ORCHESTRATOR_URL environment variable not set!")
    st.stop()  # Stop the app if the URL is not configured

# --- Helper Function to Call Orchestrator ---


async def call_orchestrator(audio_bytes: bytes, filename: str, content_type: str):
    """Calls the orchestrator's /market_brief endpoint with audio data."""
    url = f"{ORCHESTRATOR_URL}/market_brief"
    files = {"audio": (filename, audio_bytes, content_type)}

    logger.info(f"Calling orchestrator at {url} with audio file: {filename}")

    # Use httpx.AsyncClient for making async requests
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                url, files=files, timeout=120.0
            )  # Increased timeout
            response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
            logger.info(f"Orchestrator returned status {response.status_code}.")
            return response.json()

        except httpx.RequestError as e:
            error_msg = f"HTTP Request failed: {e}"
            logger.error(error_msg)
            return {
                "status": "error",
                "message": "Error communicating with the orchestrator.",
                "errors": [error_msg],
                "warnings": [],
            }
        except Exception as e:
            error_msg = f"An unexpected error occurred: {e}"
            logger.error(error_msg)
            return {
                "status": "error",
                "message": "An unexpected error occurred.",
                "errors": [error_msg],
                "warnings": [],
            }


# --- Streamlit App Layout ---

st.title("📈 AI Financial Assistant - Morning Market Brief")
st.markdown(
    "Upload an audio file with your query (e.g., 'What's our risk exposure in Asia tech stocks today, and highlight any earnings surprises?')"
)

uploaded_file = st.file_uploader(
    "Upload Audio File",
    type=["wav", "mp3", "m4a", "ogg"],  # Specify allowed audio types
    help="Supported formats: WAV, MP3, M4A, OGG",
)

# Button to trigger the workflow
if st.button("Generate Market Brief"):
    if uploaded_file is None:
        st.warning("Please upload an audio file first.")
    else:
        # Read audio file bytes
        audio_bytes = uploaded_file.getvalue()
        file_details = {"filename": uploaded_file.name, "filetype": uploaded_file.type}
        logger.info(f"Uploaded file details: {file_details}")

        # Use st.spinner to show processing status
        with st.spinner("Processing your request..."):
            # Call the async orchestrator function
            # Streamlit's modern versions handle awaiting async functions directly
            response = st.session_state.get("orchestrator_response")
            if response is None:  # Prevent re-running if state already has response
                response = call_orchestrator(
                    audio_bytes, file_details["filename"], file_details["filetype"]
                )
                st.session_state["orchestrator_response"] = (
                    response  # Store response in state
                )

        # --- Display Results ---

        st.subheader("Results")

        if response and response.get("status") == "error":
            st.error(f"Workflow Failed: {response.get('message', 'Unknown error')}")
            if response.get("errors"):
                st.json(
                    {"Errors": response["errors"]}
                )  # Display raw errors for debugging
            if response.get("warnings"):
                st.json({"Warnings": response["warnings"]})  # Display raw warnings

        elif response:
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
                    audio_bytes_output = bytes.fromhex(audio_hex)
                    audio_io = io.BytesIO(audio_bytes_output)
                    # Assuming gTTS outputs MP3
                    st.audio(audio_io, format="audio/mpeg")
                    logger.info("Displayed audio player.")
                except ValueError:
                    st.error("Failed to decode audio data.")
                    logger.error("Failed to decode audio data from hex string.")
                except Exception as e:
                    st.error(f"Failed to play audio: {e}")
                    logger.error(f"Error playing audio: {e}")

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

        else:
            st.error("Received an empty or unexpected response from the orchestrator.")

        # Clear the cached response after displaying
        # del st.session_state['orchestrator_response'] # Can clear state to allow rerunning easily

# Optional: Add instructions on how to run the app
st.markdown("---")
st.markdown("#### To Run This Application:")
st.markdown("1. Ensure all agents and the orchestrator are running.")
st.markdown("2. Update the `.env` file with the correct `ORCHESTRATOR_URL`.")
st.markdown("3. Run the Streamlit app from your terminal:")
st.code("streamlit run streamlit_app/app.py")
st.markdown("4. Open the provided URL in your browser.")

# Optional: Add a simple session state clear button for easy re-runs during development
if st.button("Clear Results"):
    if "orchestrator_response" in st.session_state:
        del st.session_state["orchestrator_response"]
    st.experimental_rerun()  # Rerun the app to clear the display
