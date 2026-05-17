"""Streaming STT using chunked Whisper."""

import numpy as np
import sounddevice as sd
import tempfile
import wave
from faster_whisper import WhisperModel
from collections import deque


class StreamingWhisperSTT:
    def __init__(self, model_name: str = "small.en", device: str = "cpu", compute_type: str = "int8"):
        print("Loading Whisper for streaming…")
        self.model = WhisperModel(model_name, device=device, compute_type=compute_type)
        self.sample_rate = 16000
        self.chunk_duration = 0.5  # 500ms chunks
        self.chunk_samples = int(self.sample_rate * self.chunk_duration)
        self.silence_threshold = 300
        self.silence_chunks_needed = 4  # 2 seconds of silence = done
        self.max_chunks = 20  # 10 seconds max
        
        # Buffer for accumulating audio
        self.audio_buffer = []
        
    def _save_buffer_to_wav(self, buffer: list, path: str):
        """Save numpy audio chunks to WAV file."""
        audio = np.concatenate(buffer)
        with wave.open(path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.sample_rate)
            wf.writeframes(audio.astype(np.int16).tobytes())
    
    def stream_transcribe(self) -> str:
        """
        Stream audio from mic, transcribe in chunks, return final text.
        Stops when silence detected or max duration reached.
        """
        print("[Streaming] Speak now…")
        
        buffer = []
        silent_count = 0
        last_text = ""
        
        stream = sd.RawInputStream(
            samplerate=self.sample_rate,
            blocksize=self.chunk_samples,
            dtype="int16",
            channels=1,
        )
        stream.start()
        
        try:
            for i in range(self.max_chunks):
                chunk, _ = stream.read(self.chunk_samples)
                chunk_np = np.frombuffer(chunk, dtype=np.int16)
                buffer.append(chunk_np.copy())
                
                # Check energy
                rms = np.sqrt(np.mean(chunk_np.astype(np.float32) ** 2))
                
                if rms < self.silence_threshold:
                    silent_count += 1
                    if silent_count >= self.silence_chunks_needed and len(buffer) > 2:
                        print("[Silence detected, finalizing…]")
                        break
                else:
                    silent_count = 0
                
                # Periodic transcription for feedback (optional)
                if i > 0 and i % 2 == 0:  # Every 1 second
                    temp_path = tempfile.mktemp(suffix=".wav")
                    self._save_buffer_to_wav(buffer, temp_path)
                    
                    segments, _ = self.model.transcribe(
                        temp_path,
                        beam_size=1,
                        condition_on_previous_text=True,
                        initial_prompt="Jarvis, turn on the speaker. Play Good Vibes. Volume up. Pause.",
                    )
                    current_text = " ".join(s.text for s in segments).strip()
                    
                    if current_text and current_text != last_text:
                        print(f"[Partial: {current_text}]")
                        last_text = current_text
                    
                    import os
                    os.remove(temp_path)
            
            # Final transcription on full audio
            print("[Finalizing transcription…]")
            final_path = tempfile.mktemp(suffix=".wav")
            self._save_buffer_to_wav(buffer, final_path)
            
            segments, _ = self.model.transcribe(
                final_path,
                beam_size=1,
                condition_on_previous_text=False,
                initial_prompt="Jarvis, turn on the speaker. Play Good Vibes. Volume up. Pause.",
            )
            final_text = " ".join(s.text for s in segments).strip()
            
            import os
            os.remove(final_path)
            
            return final_text
            
        finally:
            stream.stop()
            stream.close()