"""LLM-based intent parser using local Ollama."""

import json
import ollama
from dataclasses import dataclass
from typing import Optional


@dataclass
class Intent:
    action: str
    target: Optional[str] = None
    artist: Optional[str] = None


class OllamaIntentParser:
    def __init__(self, model: str = "qwen2.5:3b"):
        self.model = model
        self.system_prompt = """You are a strict command parser for a smart home voice assistant named J.A.R.V.I.S.

The user speaks to control a speaker and music player. They may have any accent (German, Indian, etc.) or use unusual phrasing.

You MUST output ONLY a JSON object with this exact format:
{"action": "ACTION_NAME", "target": "optional_value", "artist": "optional_artist"}

Valid actions and examples:
- power_on: "turn on the speaker", "switch on", "make it loud", "activate"
- power_off: "turn off the speaker", "switch off", "deactivate", "shut down"
- play_playlist: "play good vibes", "playlist workout", "start my morning mix"
- play_song: "play sugar by maroon 5", "song bohemian rhapsody"
- vol_up: "volume up", "louder", "increase volume", "more sound"
- vol_down: "volume down", "quieter", "decrease volume", "less sound"
- pause: "pause", "stop music", "halt"
- goodbye: "goodbye", "bye", "exit", "sleep", "see you"

Rules:
1. Output ONLY valid JSON. No explanations, no markdown.
2. Interpret intent generously — "make louder" = vol_up, "I want music" = play_playlist if context suggests.
3. Correct obvious misheard words: "but bye" → goodbye, "poz" → pause, "jarviss" → ignore (it's the wake word).
4. If completely unclear, output {"action": "unknown"}.
5. The target field should contain device names, playlist names, or song names.
"""

    def parse(self, text: str) -> Intent:
        """Send text to Ollama and get structured intent."""
        try:
            response = ollama.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": f'User said: "{text}"\nOutput JSON only:'}
                ],
                options={
                    "temperature": 0.0,      # Deterministic
                    "num_predict": 80,       # Short output
                    "stop": ["\n\n", "}"]    # Stop early if model rambles
                }
            )
            
            raw = response["message"]["content"].strip()
            
            # Extract JSON from markdown blocks if present
            if "```json" in raw:
                raw = raw.split("```json")[1].split("```")[0].strip()
            elif "```" in raw:
                raw = raw.split("```")[1].split("```")[0].strip()
            
            # Ensure valid JSON (sometimes model adds trailing text)
            if "}" in raw and not raw.endswith("}"):
                raw = raw[:raw.rfind("}") + 1]
            
            data = json.loads(raw)
            
            return Intent(
                action=data.get("action", "unknown"),
                target=data.get("target"),
                artist=data.get("artist")
            )
            
        except json.JSONDecodeError as e:
            print(f"[Ollama JSON parse error: {e}]")
            print(f"[Raw output: {raw}]")
            return Intent(action="unknown")
            
        except Exception as e:
            print(f"[Ollama error: {e}]")
            return Intent(action="unknown")


class HybridIntentParser:
    """Fast regex for common commands, Ollama for unclear ones."""
    
    def __init__(self, ollama_model: str = "qwen2.5:3b"):
        self.regex = _RegexFallback()
        self.llm = OllamaIntentParser(model=ollama_model)
        self.use_llm = True
    
    def parse(self, text: str) -> Intent:
        # Try fast regex first
        result = self.regex.parse(text)
        if result.action != "unknown":
            return result
        
        # Fall back to Ollama for complex/unclear commands
        if self.use_llm:
            return self.llm.parse(text)
        
        return Intent(action="unknown")


class _RegexFallback:
    """Lightweight regex for instant matching of common phrases."""
    
    def __init__(self):
        import re
        self.patterns = [
            (r"turn\s+(on|off)\s+(?:the\s+)?(.+)", "power"),
            (r"(?:switch|toggle)\s+(?:the\s+)?(.+)", "power_toggle"),
            (r"play\s+(?:the\s+)?(?:playlist\s+)?(.+)", "play_playlist"),
            (r"play\s+(?:the\s+)?song\s+(.+?)(?:\s+by\s+(.+))?", "play_song"),
            (r"volume\s+(up|higher|increase|more)", "vol_up"),
            (r"volume\s+(down|lower|decrease|less)", "vol_down"),
            (r"^pause$|stop\s+(?:the\s+)?(?:music|song)", "pause"),
            (r"good\s*bye|goodbye|bye|exit|sleep", "goodbye"),
        ]
    
    def parse(self, text: str) -> Intent:
        import re
        text = text.lower().strip()
        
        # Strip wake word residue
        for ww in ["jarvis", "alexa"]:
            if text.startswith(ww):
                text = text[len(ww):].strip()
        
        for pattern, intent_type in self.patterns:
            match = re.search(pattern, text)
            if match:
                if intent_type == "power":
                    return Intent(f"power_{match.group(1)}", match.group(2).strip())
                if intent_type == "power_toggle":
                    return Intent("power_toggle", match.group(1).strip())
                if intent_type == "play_playlist":
                    return Intent("play_playlist", match.group(1).strip())
                if intent_type == "play_song":
                    return Intent("play_song", match.group(1).strip(), match.group(2).strip() if match.group(2) else None)
                if intent_type == "vol_up":
                    return Intent("vol_up")
                if intent_type == "vol_down":
                    return Intent("vol_down")
                if intent_type == "pause":
                    return Intent("pause")
                if intent_type == "goodbye":
                    return Intent("goodbye")
        
        return Intent(action="unknown")