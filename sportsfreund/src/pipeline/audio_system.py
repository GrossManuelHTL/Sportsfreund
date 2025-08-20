import threading
import tempfile
import os
import sounddevice as sd
import soundfile as sf
from TTS.api import TTS

class AudioSystem:
    def __init__(self, model_name: str = "tts_models/de/thorsten/vits"):
        """
        Lädt ein deutsches TTS-Modell (offline).
        """
        self.tts = TTS(model_name)

    def speak(self, text: str, async_play: bool = False) -> bool:
        """
        Spricht den gegebenen Text.
        :param text: Deutscher Text (kein IPA).
        :param async_play: Wenn True, läuft die Wiedergabe in einem Thread.
        """
        try:
            def _play():
                # eigenen Temp-Dateinamen generieren
                tmp_path = tempfile.mktemp(suffix=".wav")

                # Sprache generieren
                self.tts.tts_to_file(text=text, file_path=tmp_path)

                # laden und abspielen
                data, samplerate = sf.read(tmp_path)
                sd.play(data, samplerate)
                sd.wait()

                # Datei danach löschen
                os.remove(tmp_path)

            if async_play:
                threading.Thread(target=_play, daemon=True).start()
            else:
                _play()

            return True
        except Exception as e:
            print(f"[AudioSystem Fehler] {e}")
            return False
