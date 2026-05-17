"""Deterministic command parser — zero latency, zero hallucination."""

import re
from dataclasses import dataclass
from typing import Optional


@dataclass
class Intent:
    action: str
    target: Optional[str] = None
    artist: Optional[str] = None


class IntentParser:
    def __init__(self):
        self.patterns = [
            # Power
            (r"turn\s+(on|off)\s+(?:the\s+)?(.+)", "power"),
            (r"(?:switch|toggle)\s+(?:the\s+)?(.+)", "power_toggle"),
            # Playlist
            (r"(?:play\s+(?:the\s+)?playlist\s+|playlist\s+)(.+)", "play_playlist"),
            (r"play\s+(?:my\s+)?playlist\s+(.+)", "play_playlist"),
            # Volume
            (r"(?:turn\s+)?(?:volume\s+)?(up|higher|increase)", "vol_up"),
            (r"(?:turn\s+)?(?:volume\s+)?(down|lower|decrease)", "vol_down"),
            # Pause / Stop / Play (music control)
            (r"^pause$", "pause"),
            (r"pause\s+(?:the\s+)?(?:music|song|playback)", "pause"),
            (r"stop\s+(?:the\s+)?(?:music|song|playback)", "pause"),
            (r"^play$", "play_resume"),
            (r"resume\s+(?:the\s+)?(?:music|song)", "play_resume"),
            (r"resume", "play_resume"),
            # Song
            (r"play\s+(?:the\s+)?song\s+(.+)", "play_song"),
            (r"play\s+(.+?)\s+by\s+(.+)", "play_song_artist"),
            # Goodbye
            (r"good\s*bye|goodbye|sleep|shut\s+down|exit|bye\s+jarvis|bye\s+alexa|bye$", "goodbye"),
        ]

    def parse(self, text: str) -> Intent:
        text = text.lower().strip()
        
        # Strip trailing punctuation
        text = text.rstrip(".!?,")
        
        # Strip wake word residue if STT captured it
        for ww in ["jarvis", "alexa"]:
            if text.startswith(ww):
                text = text[len(ww):].strip()

        for pattern, intent_type in self.patterns:
            match = re.search(pattern, text)
            if match:
                if intent_type == "power":
                    state = match.group(1)
                    device = match.group(2).strip()
                    return Intent(action=f"power_{state}", target=device)

                if intent_type == "power_toggle":
                    device = match.group(1).strip()
                    return Intent(action="power_toggle", target=device)

                if intent_type == "play_playlist":
                    return Intent(action="play_playlist", target=match.group(1).strip())

                if intent_type == "vol_up":
                    return Intent(action="vol_up")

                if intent_type == "vol_down":
                    return Intent(action="vol_down")

                if intent_type == "pause":
                    return Intent(action="pause")

                if intent_type == "play_resume":
                    return Intent(action="power_on", target="speaker")

                if intent_type == "play_song":
                    return Intent(action="play_song", target=match.group(1).strip())

                if intent_type == "play_song_artist":
                    return Intent(action="play_song", target=match.group(1).strip(), artist=match.group(2).strip())

                if intent_type == "goodbye":
                    return Intent(action="goodbye")

        return Intent(action="unknown")