import threading
import tempfile
import os
import sounddevice as sd
import soundfile as sf
from TTS.api import TTS

class AudioSystem:
    def __init__(self, model_name: str = "tts_models/de/thorsten/vits"):
        """
        Loads a German TTS model (offline).
        """
        self.tts = TTS(model_name)

    def listen_for_command(self, timeout: int = 5) -> str:
        """
        Listens for a voice command and returns the recognized text.
        This is a placeholder method and should be implemented with actual voice recognition logic.
        """

        return "This is a placeholder for voice command recognition."

    def speak(self, text: str, async_play: bool = False) -> bool:
        """
        Speaks the given text.
        :param text: German text (no IPA).
        :param async_play: If True, playback runs in a thread.
        """
        try:
            def _play():
                # Generate own temp filename
                tmp_path = tempfile.mktemp(suffix=".wav")

                # Generate speech
                self.tts.tts_to_file(text=text, file_path=tmp_path)

                # Load and play
                data, samplerate = sf.read(tmp_path)
                sd.play(data, samplerate)
                sd.wait()

                # Delete file afterwards
                os.remove(tmp_path)

            if async_play:
                threading.Thread(target=_play, daemon=True).start()
            else:
                _play()

            return True
        except Exception as e:
            print(f"[AudioSystem Error] {e}")
            return False
