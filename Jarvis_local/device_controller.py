"""Mock smart speaker controller."""

class DeviceController:
    def __init__(self):
        self.is_on = False
        self.volume = 5
        self.is_playing = False
        self.current_playlist = None

        self.songs = {
            "sugar": "Sugar by Maroon 5",
            "blinding lights": "Blinding Lights by The Weeknd",
            "shape of you": "Shape of You by Ed Sheeran",
            "bohemian rhapsody": "Bohemian Rhapsody by Queen",
        }
        self.playlists = {
            "good vibes": ["sugar", "blinding lights"],
            "workout": ["shape of you", "blinding lights"],
            "relax": ["bohemian rhapsody"],
        }

    def turn_on(self, device: str) -> str:
        self.is_on = True
        self.is_playing = True  # Auto-play when turned on
        return f"Turning on {device}"

    def turn_off(self, device: str) -> str:
        self.is_on = False
        self.is_playing = False
        return f"Turning off {device}"

    def toggle(self, device: str) -> str:
        self.is_on = not self.is_on
        state = "on" if self.is_on else "off"
        return f"Turning {state} {device}"

    def play_playlist(self, name: str) -> str:
        key = name.lower().strip().strip("'\"")
        if key in self.playlists:
            self.current_playlist = key
            self.is_playing = True
            return f"Playing the playlist '{name}'"
        return f"Playlist '{name}' not found"

    def play_song(self, query: str, artist: str = None) -> str | None:
        key = query.lower().strip().strip("'\"")

        if key in self.songs:
            self.is_playing = True
            return f"Playing {self.songs[key]}"

        for k, v in self.songs.items():
            if key in k or k in key:
                self.is_playing = True
                return f"Playing {v}"

        return None

    def volume_up(self) -> str:
        if self.volume < 10:
            self.volume += 1
        return f"Increasing the volume to {self.volume}"

    def volume_down(self) -> str:
        if self.volume > 0:
            self.volume -= 1
        return f"Decreasing the volume to {self.volume}"

    def pause(self):
        self.is_playing = False
        # Returns None → silent