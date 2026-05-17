from pathlib import Path

PROJECT_ROOT = Path(r"C:\Users\Rohit\Desktop\heidelberg lectures\sem 3\practical\JArvis\Jarvis_Main")

USERNAME = "Rohit"

WAKE_WORDS = {
    "jarvis": str(PROJECT_ROOT / "models/hey_jarvis_v0.1.onnx"),
}

WHISPER_MODEL = "small.en"
WHISPER_DEVICE = "cpu"
WHISPER_COMPUTE = "int8"

PIPER_MODEL = str(PROJECT_ROOT / "models/en_US-lessac-medium.onnx")
PIPER_EXE = str(PROJECT_ROOT / "piper/piper.exe")

SAMPLE_RATE = 16000
RECORD_MAX_SECONDS = 10    # Longer for session commands
SILENCE_SECONDS = 2.0      # Pause in speech ends recording
ENERGY_THRESHOLD = 150

OWW_THRESHOLD = 0.5

INPUT_DEVICE = 1
OUTPUT_DEVICE = 4

# Session timeout in seconds (150 = 150 seconds of silence ends session)
SESSION_TIMEOUT = 150