import streamlit as st
import httpx
import os
import io
from dotenv import load_dotenv
import logging
import asyncio
from streamlit_mic_recorder import mic_recorder


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
load_dotenv()

ORCHESTRATOR_URL = os.getenv("ORCHESTRATOR_URL")


if "processing_state" not in st.session_state:
    st.session_state.processing_state = "initial"
if "orchestrator_response" not in st.session_state:
    st.session_state.orchestrator_response = None
if "audio_bytes_input" not in st.session_state:
    st.session_state.audio_bytes_input = None
if "audio_filename" not in st.session_state:
    st.session_state.audio_filename = None
if "audio_filetype" not in st.session_state:
    st.session_state.audio_filetype = None
if "last_audio_source" not in st.session_state:
    st.session_state.last_audio_source = None
if "current_recording_id" not in st.session_state:
    st.session_state.current_recording_id = None


async def call_orchestrator(audio_bytes: bytes, filename: str, content_type: str):

    url = f"{ORCHESTRATOR_URL}/market_brief"
    files = {"audio": (filename, audio_bytes, content_type)}
    logger.info(
        f"Calling orchestrator at {url} with audio file: {filename} ({content_type})"
    )
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(url, files=files, timeout=180.0)
            response.raise_for_status()
            logger.info(f"Orchestrator returned status {response.status_code}.")
            return response.json()
    except httpx.RequestError as e:
        error_msg = f"HTTP Request failed: {e}"
        logger.error(error_msg)
        return {
            "status": "error",
            "message": "Error communicating with orchestrator.",
            "errors": [error_msg],
            "transcript": None,
            "brief": None,
            "audio": None,
        }
    except Exception as e:
        error_msg = f"An unexpected error occurred: {e}"
        logger.error(error_msg)
        return {
            "status": "error",
            "message": "An unexpected error occurred.",
            "errors": [error_msg],
            "transcript": None,
            "brief": None,
            "audio": None,
        }


st.set_page_config(layout="wide")
st.title("📈 AI Financial Assistant - Morning Market Brief")
st.markdown(
    "Ask your query verbally (e.g., 'What's our risk exposure in Asia tech stocks today, and highlight any earnings surprises?') "
    "or upload an audio file."
)

input_method = st.radio(
    "Choose input method:",
    ("Record Audio", "Upload File"),
    horizontal=True,
    index=0,
    key="input_method_radio",
)

audio_data_ready = False


if st.session_state.audio_bytes_input is not None:
    audio_data_ready = True


if input_method == "Record Audio":
    st.subheader("Record Your Query")

    if st.session_state.last_audio_source == "uploader":
        st.session_state.audio_bytes_input = None
        st.session_state.audio_filename = None
        st.session_state.audio_filetype = None
        st.session_state.last_audio_source = "recorder"
        audio_data_ready = False

    audio_info = mic_recorder(
        start_prompt="⏺️ Start Recording",
        stop_prompt="⏹️ Stop Recording",
        just_once=False,
        use_container_width=True,
        format="wav",
        key="mic_recorder_widget",
    )

    if audio_info and audio_info.get("bytes"):

        if st.session_state.current_recording_id != audio_info.get("id"):
            st.session_state.current_recording_id = audio_info.get("id")
            st.success("Recording complete! Click 'Generate Market Brief' below.")
            st.session_state.audio_bytes_input = audio_info["bytes"]
            st.session_state.audio_filename = f"live_recording_{audio_info['id']}.wav"
            st.session_state.audio_filetype = "audio/wav"
            st.session_state.last_audio_source = "recorder"
            audio_data_ready = True
            st.session_state.processing_state = "initial"
            st.session_state.orchestrator_response = None
            st.audio(audio_info["bytes"])

        elif st.session_state.audio_bytes_input:
            audio_data_ready = True
            st.audio(st.session_state.audio_bytes_input)

    elif (
        st.session_state.last_audio_source == "recorder"
        and st.session_state.audio_bytes_input
    ):
        st.markdown("Using last recording:")
        st.audio(st.session_state.audio_bytes_input)
        audio_data_ready = True


elif input_method == "Upload File":
    st.subheader("Upload Audio File")

    if st.session_state.last_audio_source == "recorder":
        st.session_state.audio_bytes_input = None
        st.session_state.audio_filename = None
        st.session_state.audio_filetype = None
        st.session_state.last_audio_source = "uploader"
        st.session_state.current_recording_id = None
        audio_data_ready = False

    if "uploaded_file_state" not in st.session_state:
        st.session_state.uploaded_file_state = None

    uploaded_file = st.file_uploader(
        "Select Audio File",
        type=["wav", "mp3", "m4a", "ogg", "flac"],
        key="file_uploader_key",
    )

    if uploaded_file is not None:
        if st.session_state.uploaded_file_state != uploaded_file:
            st.session_state.uploaded_file_state = uploaded_file
            st.session_state.audio_bytes_input = uploaded_file.getvalue()
            st.session_state.audio_filename = uploaded_file.name
            st.session_state.audio_filetype = uploaded_file.type
            st.session_state.last_audio_source = "uploader"
            audio_data_ready = True
            st.session_state.processing_state = "initial"
            st.session_state.orchestrator_response = None
            st.success(f"File '{uploaded_file.name}' ready.")
            st.audio(
                st.session_state.audio_bytes_input,
                format=st.session_state.audio_filetype,
            )
        elif st.session_state.audio_bytes_input:
            audio_data_ready = True
            st.audio(
                st.session_state.audio_bytes_input,
                format=st.session_state.audio_filetype,
            )

    elif (
        st.session_state.last_audio_source == "uploader"
        and st.session_state.audio_bytes_input
    ):
        st.markdown("Using last uploaded file:")
        st.audio(
            st.session_state.audio_bytes_input, format=st.session_state.audio_filetype
        )
        audio_data_ready = True


st.divider()
button_disabled = (
    not audio_data_ready or st.session_state.processing_state == "processing"
)

if st.button(
    "Generate Market Brief",
    disabled=button_disabled,
    type="primary",
    use_container_width=True,
    key="generate_button",
):
    if st.session_state.audio_bytes_input:
        st.session_state.processing_state = "processing"
        st.session_state.orchestrator_response = None
        logger.info(
            f"Generate Market Brief button clicked. Source: {st.session_state.last_audio_source}, Filename: {st.session_state.audio_filename}"
        )
        st.rerun()
    else:
        st.warning("Please record or upload an audio query first.")


if st.session_state.processing_state == "processing":
    if (
        st.session_state.audio_bytes_input
        and st.session_state.audio_filename
        and st.session_state.audio_filetype
    ):
        with st.spinner("Processing your request... This may take a moment. 🤖"):

            logger.info(
                f"Calling orchestrator with filename: {st.session_state.audio_filename}, type: {st.session_state.audio_filetype}, bytes: {len(st.session_state.audio_bytes_input)}"
            )
            try:
                response = asyncio.run(
                    call_orchestrator(
                        st.session_state.audio_bytes_input,
                        st.session_state.audio_filename,
                        st.session_state.audio_filetype,
                    )
                )
                st.session_state.orchestrator_response = response

                is_successful_response = True
                if not response:
                    is_successful_response = False
                elif (
                    response.get("status") == "error"
                    or response.get("status") == "failed"
                ):
                    is_successful_response = False
                elif response.get("errors") and len(response.get("errors")) > 0:
                    is_successful_response = False

                st.session_state.processing_state = (
                    "completed" if is_successful_response else "error"
                )

            except Exception as e:
                logger.error(
                    f"Error during orchestrator call in Streamlit: {e}", exc_info=True
                )
                st.session_state.orchestrator_response = {
                    "status": "error",
                    "message": f"Streamlit failed to call orchestrator: {str(e)}",
                    "errors": [str(e)],
                    "transcript": None,
                    "brief": None,
                    "audio": None,
                }
                st.session_state.processing_state = "error"
        st.rerun()
    else:
        st.error("Audio data is missing for processing. Please record or upload again.")
        st.session_state.processing_state = "initial"


if st.session_state.processing_state in ["completed", "error"]:

    response = st.session_state.orchestrator_response
    st.subheader("📝 Results")

    if response is None:
        st.error("No response received from the orchestrator.")

    elif (
        response.get("status") == "failed"
        or response.get("status") == "error"
        or (response.get("errors") and len(response.get("errors")) > 0)
    ):
        st.error(
            f"Workflow {response.get('status', 'failed')}: {response.get('message', 'Check errors below.')}"
        )
        if response.get("errors"):
            st.warning("Details of Errors:")
            for i, err in enumerate(response["errors"]):
                st.markdown(f"`Error {i+1}`: {err}")
        if response.get("warnings"):
            st.warning("Details of Warnings:")
            for i, warn in enumerate(response["warnings"]):
                st.markdown(f"`Warning {i+1}`: {warn}")

        if response.get("transcript"):
            st.markdown("---")
            st.markdown("Transcript (despite errors):")
            st.caption(response.get("transcript"))
        if response.get("brief"):
            st.markdown("---")
            st.markdown("Generated Brief (despite errors):")
            st.caption(response.get("brief"))
    else:
        st.success(response.get("message", "Market brief generated successfully!"))
        if response.get("transcript"):
            st.markdown("---")
            st.markdown("Your Query (Transcript):")
            st.caption(response.get("transcript"))
        else:
            st.info("Transcript not available.")

        if response.get("brief"):
            st.markdown("---")
            st.markdown("Generated Brief:")
            st.write(response.get("brief"))
        else:
            st.info("Brief text not available.")

        audio_hex = response.get("audio")
        if audio_hex:
            st.markdown("---")
            st.markdown("Audio Brief:")
            try:
                if not isinstance(audio_hex, str) or not all(
                    c in "0123456789abcdefABCDEF" for c in audio_hex
                ):
                    raise ValueError("Invalid hex string for audio.")
                audio_bytes_output = bytes.fromhex(audio_hex)
                st.audio(audio_bytes_output, format="audio/mpeg")
            except ValueError as ve:
                st.error(f"⚠️ Failed to decode audio data: {ve}")
            except Exception as e:
                st.error(f"⚠️ Failed to play audio: {e}")
        else:
            st.info("Audio brief not available.")

        if response.get("warnings"):
            st.markdown("---")
            st.warning("Process Warnings:")
            for i, warn in enumerate(response["warnings"]):
                st.markdown(f"`Warning {i+1}`: {warn}")
