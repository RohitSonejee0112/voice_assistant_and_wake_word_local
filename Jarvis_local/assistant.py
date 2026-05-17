"""Session-based voice assistant with latency tracking."""

import os
import datetime
import time
import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
from openwakeword.model import Model

import config
from audio_utils import play_wav, tts, record_until_silence
from intent_parser import IntentParser
from device_controller import DeviceController


class JarvisAssistant:
    def __init__(self):
        self.username = config.USERNAME
        self.controller = DeviceController()
        self.parser = IntentParser()

        self.wake_word_paths = {}
        for friendly_name, abs_path in config.WAKE_WORDS.items():
            if not os.path.exists(abs_path):
                raise FileNotFoundError(f"Wake word model not found: {abs_path}")
            self.wake_word_paths[friendly_name] = abs_path

        print("Loading openWakeWord models…")
        self.oww = Model(
            wakeword_models=list(self.wake_word_paths.values()),
            inference_framework="onnx",
        )
        self.oww_threshold = config.OWW_THRESHOLD
        self.wake_names = list(self.wake_word_paths.keys())

        print("Loading Whisper STT…")
        self.whisper = WhisperModel(
            config.WHISPER_MODEL,
            device=config.WHISPER_DEVICE,
            compute_type=config.WHISPER_COMPUTE,
        )

        self.piper_model = config.PIPER_MODEL
        self.piper_exe = config.PIPER_EXE
        
        # Session state
        self.session_active = False
        self.session_timeout = 300.0  # 5 minutes
        self.last_command_time = 0

    def _time_greeting(self) -> str:
        hour = datetime.datetime.now().hour
        if 5 <= hour < 12:
            return "Good Morning"
        if 12 <= hour < 17:
            return "Good Afternoon"
        if 17 <= hour < 21:
            return "Good Evening"
        return "Hello"

    def _speak(self, text: str, latency_ms: int = None):
        """Speak with optional latency display."""
        if latency_ms is not None:
            print(f"J.A.R.V.I.S.: {text} (latency: {latency_ms}ms)")
        else:
            print(f"J.A.R.V.I.S.: {text}")
        wav_path = tts(text, self.piper_model, self.piper_exe)
        play_wav(wav_path, device=config.OUTPUT_DEVICE)

    def _listen_for_wake(self) -> str | None:
        """Block until wake word detected."""
        print("Listening for wake word…")

        with sd.RawInputStream(
            samplerate=16000,
            blocksize=1280,
            dtype="int16",
            channels=1,
            device=config.INPUT_DEVICE,
        ) as stream:
            while True:
                chunk, _ = stream.read(1280)
                chunk = np.frombuffer(chunk, dtype=np.int16)
                predictions = self.oww.predict(chunk)

                for model_key, score in predictions.items():
                    if score > self.oww_threshold:
                        for friendly_name in self.wake_names:
                            path = self.wake_word_paths[friendly_name]
                            if (friendly_name.lower() in model_key.lower() or
                                os.path.basename(path).lower() in model_key.lower()):
                                print(f"[Wake word detected: {friendly_name}]")
                                return friendly_name
                        return self.wake_names[0]

    def _listen_for_command(self, timeout_sec: float = None) -> str | None:
        """Listen for a command during active session."""
        if timeout_sec is None:
            timeout_sec = self.session_timeout

        print(f"[Session active. Listening for command... {timeout_sec:.0f}s timeout]")
        
        tmp_wav = "temp_command.wav"
        
        record_until_silence(
            tmp_wav,
            samplerate=config.SAMPLE_RATE,
            max_seconds=10,
            silence_seconds=2.0,
            energy_threshold=config.ENERGY_THRESHOLD,
        )

        # Check if silence
        import wave
        with wave.open(tmp_wav, "rb") as wf:
            n_frames = wf.getnframes()
            if n_frames == 0:
                return None
            
            raw = wf.readframes(n_frames)
            audio = np.frombuffer(raw, dtype=np.int16)
            rms = np.sqrt(np.mean(audio.astype(np.float32) ** 2))
            
            if rms < 100:
                return None

        # Transcribe
        command_text = self._transcribe(tmp_wav)
        print(f"User: {command_text}")
        
        if not command_text.strip():
            return None
            
        return command_text.strip()

    def _transcribe(self, wav_path: str) -> str:
        segments, _ = self.whisper.transcribe(
            wav_path,
            initial_prompt="Jarvis, turn on the speaker. Play the playlist Good Vibes. Volume up. Pause. Goodbye.",
            condition_on_previous_text=False,
        )
        return " ".join(s.text for s in segments).strip()

    def _execute(self, intent):
        a = intent.action
        
        if a == "pause":
            self.controller.pause()
            return None
        
        if a == "power_on":
            return self.controller.turn_on(intent.target or "speaker")
        if a == "power_off":
            return self.controller.turn_off(intent.target or "speaker")
        if a == "power_toggle":
            return self.controller.toggle(intent.target or "speaker")
        if a == "play_playlist":
            return self.controller.play_playlist(intent.target)
        if a == "play_song":
            result = self.controller.play_song(intent.target, intent.artist)
            return result if result else "Unable to find song"
        if a == "vol_up":
            return self.controller.volume_up()
        if a == "vol_down":
            return self.controller.volume_down()
        if a == "goodbye":
            return f"Goodbye {self.username}"
        return "I didn't understand that command"

    def _run_session(self):
        """Active session: listen for commands without wake word."""
        self.session_active = True
        self.last_command_time = time.time()
        
        # Initial greeting (no latency to track here)
        greeting = self._time_greeting()
        self._speak(f"{greeting} {self.username}, what can I do for you today")

        while self.session_active:
            # Check session timeout
            time_since_last = time.time() - self.last_command_time
            remaining = self.session_timeout - time_since_last
            
            if remaining <= 0:
                print("[Session timeout - no commands heard]")
                self._speak("Going to sleep")
                self.session_active = False
                break

            # === START LATENCY TIMER ===
            # Timer starts after recording finishes (user stopped speaking)
            recording_start = time.time()
            
            command_text = self._listen_for_command(timeout_sec=remaining)
            
            if command_text is None:
                if time.time() - self.last_command_time >= self.session_timeout:
                    print("[Session expired due to silence]")
                    self._speak("Goodbye")
                    self.session_active = False
                    break
                continue

            # === MEASURE LATENCY ===
            # From end of recording to now (before parsing/execution)
            parse_start = time.time()
            
            intent = self.parser.parse(command_text)
            response = self._execute(intent)
            
            parse_end = time.time()
            
            # Total latency = STT time + parse/execute time
            # We approximate: from when recording finished to response ready
            total_latency_ms = int((parse_end - recording_start) * 1000)
            
            # Reset timeout
            self.last_command_time = time.time()

            # Speak response with latency
            if response:
                self._speak(response, latency_ms=total_latency_ms)

            if intent.action == "goodbye":
                print("Shutting down session.")
                self.session_active = False
                break

            time.sleep(0.3)

    def run(self):
        """Main loop: wake word → session → wake word..."""
        try:
            while True:
                wake_name = self._listen_for_wake()
                if wake_name is None:
                    continue

                print(f"[Activated by: {wake_name}]")

                self._run_session()

                print("[Session ended. Returning to wake word listening...]")
                time.sleep(1.0)

        except KeyboardInterrupt:
            print("\nInterrupted by user.")
        finally:
            if os.path.exists("temp_command.wav"):
                os.remove("temp_command.wav")