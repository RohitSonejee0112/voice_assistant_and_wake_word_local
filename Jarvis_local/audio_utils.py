"""Microphone input, WAV recording, and speaker output."""

import wave
import tempfile
import subprocess
import numpy as np
import sounddevice as sd
from pathlib import Path


def play_wav(path: str, device=None):
    """Play a WAV file through the specified output device."""
    with wave.open(path, "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        raw = wf.readframes(wf.getnframes())

        if sampwidth == 2:
            audio = np.frombuffer(raw, dtype=np.int16)
        else:
            audio = np.frombuffer(raw, dtype=np.int32)

        # Normalize to float32 [-1.0, 1.0] for sounddevice
        audio = audio.astype(np.float32) / np.iinfo(audio.dtype).max
        sd.play(audio, framerate, device=device)
        sd.wait()


def tts(text: str, model_path: str, piper_exe: str) -> str:
    """Synthesize speech with Piper. Returns path to temporary WAV."""
    out_path = tempfile.mktemp(suffix=".wav")
    cmd = [str(Path(piper_exe).resolve()), "-m", str(Path(model_path).resolve()), "-f", out_path]
    subprocess.run(cmd, input=text.encode(), check=True, capture_output=True)
    return out_path


def record_until_silence(
    out_path: str,
    samplerate: int = 16000,
    max_seconds: float = 5.0,
    silence_seconds: float = 1.2,
    energy_threshold: float = 500,
):
    """
    Record audio until silence is detected or max duration reached.
    Saves to a 16-bit mono WAV file.
    """
    chunk_samples = int(samplerate * 0.1)          # 100 ms chunks
    max_chunks = int(max_seconds / 0.1)
    silence_chunks = int(silence_seconds / 0.1)

    buffer = []
    silent_count = 0

    stream = sd.RawInputStream(
        samplerate=samplerate,
        blocksize=chunk_samples,
        dtype="int16",
        channels=1,
    )
    stream.start()

    for _ in range(max_chunks):
        chunk, _ = stream.read(chunk_samples)
        chunk = np.frombuffer(chunk, dtype=np.int16)
        buffer.append(chunk.copy())

        rms = np.sqrt(np.mean(chunk.astype(np.float32) ** 2))
        if rms < energy_threshold:
            silent_count += 1
            if silent_count >= silence_chunks:
                break
        else:
            silent_count = 0

    stream.stop()
    stream.close()

    audio = np.concatenate(buffer)
    with wave.open(out_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(samplerate)
        wf.writeframes(audio.tobytes())