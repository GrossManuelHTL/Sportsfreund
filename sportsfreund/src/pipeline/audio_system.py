import threading
import tempfile
import os
from gtts import gTTS
from playsound import playsound


class AudioSystem:
    """Simple TTS using Google (gTTS) for German language output.

    This replaces the previous Coqui TTS implementation with gTTS (Google-backed).
    Note: gTTS requires internet access.
    """

    def __init__(self, lang: str = "de"):
        self.lang = lang

    def listen_for_command(self, timeout: int = 5) -> str:
        """Placeholder for voice command recognition."""
        return ""

    def speak(self, text: str, async_play: bool = False) -> bool:
        """Speak text using gTTS and playsound.

        :param text: Text to speak (German)
        :param async_play: If True, play in background thread
        """
        if not text:
            return False

        def _play(tmp_mp3: str):
            try:
                playsound(tmp_mp3)
            except Exception:
                pass
            try:
                os.remove(tmp_mp3)
            except Exception:
                pass

        try:
            # create temporary mp3 file
            fd, tmp_path = tempfile.mkstemp(suffix=".mp3")
            os.close(fd)

            tts = gTTS(text=text, lang=self.lang)
            tts.save(tmp_path)

            if async_play:
                threading.Thread(target=_play, args=(tmp_path,), daemon=True).start()
            else:
                _play(tmp_path)

            return True
        except Exception as e:
            print(f"[AudioSystem Error] {e}")
            return False
