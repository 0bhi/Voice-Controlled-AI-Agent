import assemblyai as aai
from config import ASSEMBLYAI_API_KEY


aai.settings.api_key = ASSEMBLYAI_API_KEY


def transcribe(audio_path: str) -> str:
    """Transcribe an audio file using AssemblyAI.

    Returns the transcribed text, or an error message string prefixed with
    "ERROR:" if transcription fails.
    """
    if not ASSEMBLYAI_API_KEY:
        return "ERROR: ASSEMBLYAI_API_KEY is not set. Please add it to your .env file."

    try:
        config = aai.TranscriptionConfig(
            speech_models=["universal-3-pro", "universal-2"],
            language_code="en",
        )
        transcriber = aai.Transcriber()
        transcript = transcriber.transcribe(audio_path, config=config)

        if transcript.status == aai.TranscriptStatus.error:
            return f"ERROR: Transcription failed — {transcript.error}"

        text = (transcript.text or "").strip()
        if not text:
            return "ERROR: No speech detected in the audio. Please try again with clearer audio."

        return text

    except Exception as exc:
        return f"ERROR: Could not transcribe audio — {exc}"
